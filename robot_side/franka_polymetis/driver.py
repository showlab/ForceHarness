"""Polymetis driver for the robot-side worker: state, wrench, IK and target updates around the stock controller.

Runs on the real-time computer (Python 3.8, Polymetis environment). It instantiates the installed
torchcontrol HybridJointImpedanceControl unchanged and only moves its joint target; every check here runs
outside the server's real-time loop. If this process dies, the stock controller holds its last target: it has
no command heartbeat watchdog of its own (the guard's lease applies only while this worker runs).
"""
import time

import numpy as np

from guard import LOW,HIGH,angle,vec,workspace
from wrench import Kinematics,check_poses,solve_wrench


class DeadlineStub:
    def __init__(self,stub):
        self.stub = stub

    def __getattr__(self,name):
        rpc = getattr(self.stub,name)
        def bounded(*args,**kwargs):
            # Loading and server-side JIT warmup precede activation.
            # This is NOT the runtime step/feedback deadline.
            kwargs.setdefault('timeout',3. if name=='SetController' else .10)
            return rpc(*args,**kwargs)
        return bounded


class PolymetisDriver:
    def __init__(self,hardware_config,robot_port=50051,gripper_port=50052):
        import tempfile
        import yaml
        import torch
        import grpc
        import polymetis_pb2_grpc
        from polymetis import RobotInterface
        from polymetis_pb2 import Empty
        self.torch,self.Empty = torch,Empty
        self.robot = RobotInterface(ip_address='127.0.0.1',port=robot_port)
        self.robot.grpc_connection = DeadlineStub(self.robot.grpc_connection)
        self.gripper_channel = grpc.insecure_channel('127.0.0.1:%d' % gripper_port)
        self.gripper = DeadlineStub(polymetis_pb2_grpc.GripperServerStub(self.gripper_channel))
        meta = self.robot.metadata
        if meta.ee_link_name!='panda_link8' or not 500<=meta.hz<=2000:
            raise RuntimeError('Unexpected end-effector link or controller frequency')
        with open(hardware_config) as f:
            cfg = yaml.safe_load(f)['robot_client']['executable_cfg']['limits']
        self.low = np.maximum(LOW,np.array(cfg['joint_pos_lower']))
        self.high = np.minimum(HIGH,np.array(cfg['joint_pos_upper']))
        with tempfile.NamedTemporaryFile('w',suffix='.urdf') as f:
            f.write(meta.urdf_file); f.flush()
            self.kin = Kinematics(f.name,meta.ee_link_name)
        frame,self.frame_errors = self.kin.calibrate(check_poses(meta.rest_pose))
        if frame!='world':
            raise RuntimeError('Cartesian policy requires verified world-aligned Jacobian')
        self.own_start = None
        self.start_attempted = False
        self.last_target = self.last_joint_target = None
        self.start_q = self.last_q = self.last_flange = None
        self.kp_max = np.array(meta.default_Kx)
        self.kd_max = np.array(meta.default_Kxd)
        self.active_kp = self.active_kd = None
        self.last_update_index = None
        self.telemetry = lambda row: None

    def metadata(self):
        meta = self.robot.metadata
        return dict(default_Kx=list(meta.default_Kx),default_Kxd=list(meta.default_Kxd),
                    joint_low=self.low.tolist(),joint_high=self.high.tolist(),hz=meta.hz,
                    frame='world',reference='panda_link8',frame_errors=self.frame_errors,
                    default_Kq=list(meta.default_Kq),default_Kqd=list(meta.default_Kqd),
                    controller='torchcontrol.policies.HybridJointImpedanceControl',
                    implementation='continuous_compensated_v1',use_grav_comp=self.robot.use_grav_comp,
                    joint_gain_scales=[4,4,4,4,4,4,8],
                    joint_gain_rule='legacy scales times minimum normalized Cartesian gain',
                    kp_semantics='hybrid gains, not independently calibrated TCP stiffness',
                    gain_changes='between controller sessions only',
                    normal_stop='rate-limited target release (nominal 0.3 s), then stock terminate',
                    update_schema='joint_pos_desired/1',controller_watchdog=False,
                    process_failure_behavior='stock controller retains last joint target',
                    lead_limit_is_physical_stop_guarantee=False,
                    diagnostics='stock_tracking/1',
                    target_feedback_semantics='last RPC-accepted target, not actual joint position')

    def read(self):
        read_started = time.monotonic()
        s = self.robot.get_robot_state()
        interval = self.robot.grpc_connection.GetEpisodeInterval(self.Empty())
        running = interval.start!=-1 and interval.end==-1
        if self.own_start is not None and running and interval.start!=self.own_start:
            raise RuntimeError('Controller ownership changed; will not stop another controller')
        q = np.array(s.joint_positions)
        self.last_q = q.copy()
        p,r = self.kin.fk(q)
        self.last_flange = np.array(p,dtype=float).copy()
        w,residual,smin = solve_wrench(self.kin.jacobian(q),s.motor_torques_external)
        grip = self.gripper.GetState(self.Empty())
        if grip.error_code:
            raise RuntimeError('Gripper hardware error')
        stamp = int(s.timestamp.seconds*1000000000+s.timestamp.nanos)
        grip_stamp = int(grip.timestamp.seconds*1000000000+grip.timestamp.nanos)
        if grip_stamp<=0 or abs(stamp-grip_stamp)>350000000:
            raise RuntimeError('Gripper/arm state timestamp skew exceeds 350 ms')
        self.telemetry(dict(type='timing',operation='read',elapsed_s=time.monotonic()-read_started,
                            timestamp_ns=stamp))
        # These are fields from the SAME RobotState already fetched above.
        # No extra RPC, IK, dynamics computation or control update is added.
        torques = {name:list(getattr(s,name,())) for name in (
            'joint_torques_computed','prev_joint_torques_computed',
            'prev_joint_torques_computed_safened','motor_torques_measured',
            'motor_torques_external','motor_torques_desired')}
        return dict(timestamp_ns=stamp,
                    flange_pos=p.tolist(),flange_quat=r.tolist(),joints=q.tolist(),
                    velocities=list(s.joint_velocities),wrench=w.tolist(),smin=smin,residual=residual,
                    policy_running=running,error_code=s.error_code,gripper_width=grip.width,
                    controller_latency_ms=s.prev_controller_latency_ms,
                    command_successful=s.prev_command_successful,
                    episode_start=int(interval.start),episode_end=int(interval.end),
                    rpc_accepted_joint_target=(self.last_joint_target.tolist()
                                              if self.last_joint_target is not None else None),
                    last_update_index=self.last_update_index,
                    # A retained target after termination is history, not the
                    # new default controller's target. Interval/state RPCs are
                    # separate snapshots, so this is not cycle-level alignment.
                    target_controller_active=bool(running and self.own_start==interval.start),
                    **torques)

    def _validate_gains(self,kp,kd):
        kp,kd = vec(kp,6,'kp'),vec(kd,6,'kd')
        if (np.any(kp<=0) or np.any(kp>self.kp_max)
            or np.any(kd<=0) or np.any(kd>self.kd_max)):
            raise RuntimeError('Invalid hybrid gains')
        return kp,kd

    def make_policy(self,kp=None,kd=None):
        import torchcontrol as toco
        torch = self.torch
        kp,kd = self._validate_gains(self.kp_max if kp is None else kp,
                                    self.kd_max if kd is None else kd)
        ratio = float(np.min(kp/self.kp_max))
        scales = np.array([4.,4.,4.,4.,4.,4.,8.])
        q = self.robot.get_joint_positions()
        joints = vec(q.detach().numpy(),7,'joint state')
        if np.any(joints<self.low+.03) or np.any(joints>self.high-.03):
            raise RuntimeError('Start joint margin exceeded')
        # Installed class itself: no subclass, custom forward/update or patch.
        return toco.policies.HybridJointImpedanceControl(
            joint_pos_current=q,
            Kq=torch.tensor(np.array(self.robot.metadata.default_Kq)*scales*ratio,dtype=torch.float32),
            Kqd=torch.tensor(np.array(self.robot.metadata.default_Kqd)*np.sqrt(scales*ratio),dtype=torch.float32),
            Kx=torch.tensor(kp,dtype=torch.float32),Kxd=torch.tensor(kd,dtype=torch.float32),
            robot_model=self.robot.robot_model,ignore_gravity=self.robot.use_grav_comp)

    def start(self,kp,kd):
        if self.robot.is_running_policy():
            raise RuntimeError('Another controller active')
        kp,kd = self._validate_gains(kp,kd)
        policy = self.make_policy(kp,kd)
        self.start_q = policy.joint_pos_desired.detach().numpy().copy()
        self.last_joint_target = self.start_q.copy()
        self.last_update_index = None
        p,r = self.robot.robot_model.forward_kinematics(policy.joint_pos_desired.detach())
        self.start_pose = np.r_[p.detach().numpy(),r.detach().numpy()]
        workspace(self.start_pose[:3])
        self.last_target = self.start_pose.copy()
        self.active_kp,self.active_kd = kp.copy(),kd.copy()
        # Compile BEFORE arming ownership. Compilation is not a robot command.
        self.torch.jit.script(policy)
        self.start_attempted = True
        self.robot.send_torch_policy(policy,blocking=False)
        deadline = time.monotonic()+.2
        while time.monotonic()<deadline:
            interval = self.robot.grpc_connection.GetEpisodeInterval(self.Empty())
            if interval.start!=-1 and interval.end==-1:
                self.own_start = interval.start
                self.telemetry(dict(type='controller_start',episode_start=int(self.own_start),
                                    joint_target=self.start_q.tolist(),
                                    flange_target=self.start_pose.tolist(),
                                    kp=kp.tolist(),kd=kd.tolist()))
                return
            time.sleep(.005)
        raise RuntimeError('Controller did not start')

    def reset_primitive(self,state):
        """Rebase only per-primitive bounds; preserve policy, gains and IK target."""
        self.start_pose=np.r_[vec(state['flange_pos'],3,'pose'),vec(state['flange_quat'],4,'quaternion')]
        self.start_q=vec(state['joints'],7,'joints')

    def update(self,pose,kp,kd,heartbeat):
        pose,kp,kd = vec(pose,7,'pose'),vec(kp,6,'kp'),vec(kd,6,'kd')
        torch = self.torch
        self._validate_gains(kp,kd)
        if (self.active_kp is None or not np.array_equal(kp,self.active_kp)
            or not np.array_equal(kd,self.active_kd)):
            raise RuntimeError('Stock controller gains are fixed; stop and start a new session')
        workspace(pose[:3])
        # Same per-motion envelope as guard.Limits: 120 mm of target travel and 0.60 rad per primitive.
        if (abs(np.linalg.norm(pose[3:])-1.)>1e-4
            or np.linalg.norm(pose[:3]-self.start_pose[:3])>.120001
            or angle(pose[3:],self.start_pose[3:])>.60):
            raise RuntimeError('Target pose outside guarded envelope')
        if self.last_q is None or self.last_joint_target is None:
            raise RuntimeError('Missing checked joint state')
        if self.last_target is None or not np.allclose(pose,self.last_target,rtol=0,atol=1e-9):
            started = time.monotonic()
            q_goal = self.robot.robot_model.inverse_kinematics(
                torch.tensor(pose[:3],dtype=torch.float32),torch.tensor(pose[3:],dtype=torch.float32),
                rest_pose=torch.tensor(self.last_joint_target,dtype=torch.float32),max_iters=100)
            ik_elapsed = time.monotonic()-started
            if ik_elapsed>.10:
                raise RuntimeError('IK exceeded 100 ms; target not sent')
            q_new = q_goal.detach().numpy()
            p,r = self.robot.robot_model.forward_kinematics(q_goal)
            # The guard lets a pressing target lead the flange by up to 25 mm instead of 8 mm; the check
            # against the MEASURED joints grows with that distance (at most by 25/8). The IK jump check
            # against the previous joint target stays at 0.05 rad.
            lead_ratio = (1. if self.last_flange is None else
                          min(.025/.008,max(1.,float(np.linalg.norm(pose[:3]-self.last_flange))/.008)))
            if (not np.isfinite(q_new).all() or np.any(q_new<self.low+.03) or np.any(q_new>self.high-.03)
                or np.max(np.abs(q_new-self.start_q))>.9     # joint travel per motion (guard joint_delta)
                or np.max(np.abs(q_new-self.last_joint_target))>.05
                or np.max(np.abs(q_new-self.last_q))>.05*lead_ratio
                or np.linalg.norm(p.detach().numpy()-pose[:3])>.0002
                or abs(float(np.dot(r.detach().numpy(),pose[3:])))<.9999995):
                raise RuntimeError('IK target failed bounds/continuity/FK checks')
        else:
            # Re-sending unchanged targets cannot provide a stock watchdog.
            return
        started = time.monotonic()
        index = self.robot.update_desired_joint_positions(torch.tensor(q_new,dtype=torch.float32))
        self.telemetry(dict(type='timing',operation='update_rpc',elapsed_s=time.monotonic()-started))
        if index<0:
            raise RuntimeError('Controller target update rejected')
        self.telemetry(dict(type='timing',operation='ik',elapsed_s=ik_elapsed,heartbeat=heartbeat))
        self.last_target,self.last_joint_target = np.array(pose).copy(),np.array(q_new).copy()
        self.last_update_index = int(index)
        self.telemetry(dict(type='joint_target_accepted',heartbeat=heartbeat,
                            update_index=int(index),episode_start=int(self.own_start),
                            time=time.monotonic(),joint_target=self.last_joint_target.tolist(),
                            flange_target=self.last_target.tolist(),
                            fk_pos=p.detach().numpy().tolist(),fk_quat=r.detach().numpy().tolist(),
                            checked_joint_state=self.last_q.tolist()))

    def stop(self):
        interval = self.robot.grpc_connection.GetEpisodeInterval(self.Empty())
        if interval.start!=-1 and interval.end==-1:
            if self.own_start is not None and interval.start!=self.own_start:
                raise RuntimeError('Controller ownership changed; refusing foreign termination')
            if self.start_attempted:
                self.robot.terminate_current_policy(return_log=False)
                deadline = time.monotonic()+.5
                while self.robot.is_running_policy():
                    if time.monotonic()>deadline:
                        raise RuntimeError('Controller termination not confirmed')
                    time.sleep(.01)
        self.own_start = None
        self.start_attempted = False
