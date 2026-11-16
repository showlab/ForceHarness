"""Offline self-test of the robot-side bundle: guard lifecycle, stock controller class, wrench solve. No robot RPC.

Run on the robot PC with the Polymetis Python after deploying (scripts/real/deploy_robot_side.py --selftest):
it blocks every network connection, exercises the guarded session on a portable fixture (hold, resume, idle
timeout, stall), checks that the Jacobian frame is world-aligned, and checks that the installed
HybridJointImpedanceControl, built by our driver, computes the same torques as its reference equation.
"""
import argparse
import copy
import io
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
import torchcontrol as toco

from driver import PolymetisDriver
from guard import LOW,HIGH,GuardedSession
from wrench import Kinematics,REST_POSE,check_poses,default_urdf


def check_hold_resume():
    """Portable lifecycle fixture; no RobotInterface or network construction."""
    class Clock:
        value=1.
        def __call__(self):return self.value
        def sleep(self,dt):self.value+=dt
    clock=Clock()
    class Driver:
        def __init__(self):
            self.pose=np.array([.4,0,.35,1.,0,0,0]);self.running=False
            self.starts=self.stops=0
        def metadata(self):
            return dict(default_Kx=[750.]*3+[15.]*3,default_Kxd=[37.]*3+[2.]*3,
                        joint_low=LOW.tolist(),joint_high=HIGH.tolist())
        def read(self):
            return dict(timestamp_ns=int(clock()*1e9),flange_pos=self.pose[:3].tolist(),
                flange_quat=self.pose[3:].tolist(),joints=[-.03,-.48,.03,-2.6,.03,2.08,.75],
                velocities=[0.]*7,wrench=[0.]*6,gripper_width=.08,smin=.1,residual=0.,
                policy_running=self.running,error_code=0)
        def start(self,kp,kd):self.starts+=1;self.running=True
        def stop(self):self.stops+=1;self.running=False
        def update(self,pose,kp,kd,heartbeat):self.pose=pose.copy()
        def reset_primitive(self,state):
            PolymetisDriver.reset_primitive(self,state)
    driver=Driver();events=[]
    s=GuardedSession(driver,clock=clock,emit=events.append);s.release_sleep=clock.sleep
    s.begin(2.)
    for i in range(3):
        if i:s.resume(2.,[150.]*6)
        s.command([.001,0,0],[0,0,0],[0,0,0],[0,0,0,1])
        clock.sleep(.05);s.tick();s.hold(sleep=clock.sleep)
        for _ in range(25):clock.sleep(.02);s.tick()
    assert driver.starts==1 and driver.stops==0 and s.holding
    while s.holding:clock.sleep(.02);s.tick()
    assert driver.stops==1 and s.fault is None
    timeout=next(e for e in events if e['type']=='hold_timeout')
    assert 60.<=timeout['elapsed_s']<=60.03
    s.begin(2.);s.finish(sleep=clock.sleep)
    assert driver.starts==driver.stops==2
    # A bounded, static offset of the target (the hand supported while a load is read) is legitimate;
    # keep exercising the production stall guard without constructing a RobotInterface.
    fixed=Driver();fixed.update=lambda *args:None
    guard=GuardedSession(fixed,clock=clock);guard.release_sleep=clock.sleep
    guard.begin(30.)
    def step(x):
        guard.command([x,0,0],[0,0,0],[0,0,0],[0,0,0,1])
        clock.sleep(.05);guard.tick()
    for _ in range(30):step(.004)
    for i in range(220):step(.004+(i%2)*.0000048)
    assert guard.active and guard.fault is None
    assert not guard.target_moving
    faults=[]
    guard.emit=lambda row:faults.append(clock()) if row['type']=='fault_release_start' else None
    moving_at=clock()
    try:
        for i in range(120):step(.004+.00003*(i+1))
    except RuntimeError as exc:
        assert 'Cartesian tracking stalled' in str(exc)
    else:raise AssertionError('Moving blocked target did not stop')
    # 0.6 mm/s takes ~0.83 s to qualify as moving; then 4 s, excluding unload.
    assert len(faults)==1 and 4.8-1e-9<=faults[0]-moving_at<=4.95 and not fixed.running
    assert guard.limits.stall_seconds==4. and guard.limits.lead==.008 and guard.limits.contact_lead==.025
    return dict(hold=True,resume=True,three_primitives_one_start=True,
                idle_timeout_s=timeout['elapsed_s'],restart_after_timeout=True,
                static_support_10s=True,micrometre_jitter_static=True,
                moving_blocked_stall_4s=True,target_window_s=1.,target_path_threshold_m=.0005)


@torch.no_grad()
def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--urdf',default=None,help='Panda URDF (default: the one that ships with Polymetis, or $FH_URDF)')
    args=ap.parse_args(argv)
    urdf=args.urdf or default_urdf()
    if not urdf:
        raise SystemExit('No Panda URDF found: pass --urdf or set FH_URDF')
    import grpc
    import socket
    def reject_network(*args,**kwargs):
        raise RuntimeError('Zero-RPC selftest attempted a network connection')
    grpc.insecure_channel=grpc.secure_channel=socket.create_connection=reject_network
    lifecycle=check_hold_resume()
    kin = Kinematics(urdf,'panda_link8')
    frame,errors = kin.calibrate(check_poses(REST_POSE))
    if frame!='world':
        raise RuntimeError('Unexpected Jacobian frame')
    q = torch.tensor([-.03,-.48,.03,-2.6,.03,2.08,.75])
    kp,kd = torch.tensor([750.]*3+[15.]*3),torch.tensor([37.]*3+[2.]*3)
    ff = toco.modules.feedforward.InverseDynamics(kin.model,ignore_gravity=True)
    kq,kqd = torch.tensor([40.,30.,50.,25.,35.,25.,10.]),torch.tensor([4.,6.,5.,5.,3.,2.,1.])
    driver = PolymetisDriver.__new__(PolymetisDriver)
    driver.torch,driver.low,driver.high = torch,LOW,HIGH
    driver.kp_max,driver.kd_max = kp.numpy(),kd.numpy()
    driver.robot = SimpleNamespace(robot_model=kin.model,use_grav_comp=True,
        get_joint_positions=lambda:q.clone(),
        metadata=SimpleNamespace(default_Kq=kq.tolist(),default_Kqd=kqd.tolist()))
    start = time.monotonic()
    policy = driver.make_policy()
    if type(policy) is not toco.policies.HybridJointImpedanceControl:
        raise RuntimeError('Must use the installed stock class without a subclass')
    scripted = torch.jit.script(policy)
    output = io.BytesIO(); torch.jit.save(scripted,output); output.seek(0)
    scripted = torch.jit.load(output)
    state = dict(joint_positions=q,joint_velocities=torch.zeros(7))
    # Match the installed server: backup, five forward calls, restore.
    # update() is deliberately left cold, just as it is during activation.
    backup = copy.deepcopy(scripted)
    warm_started = time.monotonic()
    for _ in range(5):
        scripted(state)
    warm_elapsed = time.monotonic()-warm_started
    scripted = backup
    p,r = kin.model.forward_kinematics(q)
    target = p+torch.tensor([.001,0,0])
    q_goal = kin.model.inverse_kinematics(target,r,rest_pose=q,max_iters=100)
    cold_updates = []
    for i in range(2):
        tick = time.monotonic()
        scripted.update({'joint_pos_desired':q_goal})
        cold_updates.append(time.monotonic()-tick)
    before = scripted(state)['joint_torques']
    scales = torch.tensor([4.,4.,4.,4.,4.,4.,8.])
    reference = toco.modules.feedback.HybridJointSpacePD(kq*scales,kqd*torch.sqrt(scales),kp,kd)
    expected = reference(q,torch.zeros(7),q_goal,torch.zeros(7),kin.model.compute_jacobian(q))+ff(q,torch.zeros(7),torch.zeros(7))
    difference = float(torch.max(torch.abs(before-expected)))
    if difference>1e-5:
        raise RuntimeError('Hybrid output differs from installed legacy equation')
    # Gains change by constructing a NEW installed controller.
    soft = torch.jit.script(driver.make_policy(kp.numpy()*.5,kd.numpy()*np.sqrt(.5)))
    soft.update({'joint_pos_desired':q_goal})
    after = soft(state)['joint_torques']
    soft_reference = toco.modules.feedback.HybridJointSpacePD(
        kq*scales*.5,kqd*torch.sqrt(scales*.5),kp*.5,kd*np.sqrt(.5))
    expected_soft = soft_reference(q,torch.zeros(7),q_goal,torch.zeros(7),kin.model.compute_jacobian(q))+ff(q,torch.zeros(7),torch.zeros(7))
    soft_difference = float(torch.max(torch.abs(after-expected_soft)))
    if soft_difference>1e-5 or float(torch.norm(before-after))<1e-5:
        raise RuntimeError('New-controller gain selection check failed')
    for _ in range(301):
        scripted(state)
    if scripted.is_terminated():
        raise RuntimeError('Unexpected stock controller termination without a stop')
    print(json.dumps(dict(type='offline_compile_check',ok=True,hardware_connected=False,
        network_calls=0,protocol='fh_continuous/2',lifecycle=lifecycle,
        torch_version=torch.__version__,model=urdf,frame_errors=errors,
        serialization=True,forward=True,target_update=True,stock_class=True,
        runtime_gain_update=False,new_controller_gains=True,controller_watchdog=False,
        hybrid_reference_max_error_Nm=difference,soft_reference_max_error_Nm=soft_difference,
        update_schema='joint_pos_desired/1',server_warmup_s=warm_elapsed,
        cold_update_s=cold_updates,timing_is_realtime_certification=False,
        torque_change=float(torch.norm(before-after)),elapsed_s=time.monotonic()-start)))


if __name__=='__main__':
    main()
