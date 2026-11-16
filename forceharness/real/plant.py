"""Synthetic plant for offline tests and demos: a deterministic contract fixture, NOT a physics simulator.

The real robot-side GuardedSession (robot_side/franka_polymetis/guard.py) is reused unchanged, so every
guard, lease and release rule runs exactly as on the robot. The fixture supplies an ideal position follower,
an optional horizontal contact plane, a configurable payload and a fixed sensor bias (Plant, DemoPort), and
DemoWorld adds two synthetic bottles that the runner's demo mode can grasp, lift and place.
"""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R

from forceharness.real.config import BackendConfig
from forceharness.real.backend import ContinuousBackend
from forceharness.real.types import BackendError

_spec = importlib.util.spec_from_file_location('fh_fixture_core',Path(__file__).resolve().parents[2]/'robot_side/franka_polymetis/guard.py')
core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(core)
G = 9.81


class Clock:
    def __init__(self):
        self.t = 1.
    def __call__(self):
        return self.t
    def sleep(self,t):
        self.t += t


class Plant:
    def __init__(self,clock,*,plane=None,payload=.6):
        self.clock = clock
        self.p = np.array([.4,0.,.35])
        self.r = np.array([1.,0.,0.,0.])
        self.q = np.array([-.03,-.48,.03,-2.60,.03,2.08,.75])
        self.width = .08
        self.running = False
        self.held = False
        self.plane,self.payload = plane,payload
        self.bias = np.array([.4,-.3,.2,.01,-.01,.01])
        self.w = self.bias.copy()
        self.starts = self.stops = 0
        self.start_gains = []
        self.updates,self.grips = [],[]
        self.frozen = False
        self.override = {}

    def metadata(self):
        return dict(default_Kx=[750]*3+[15]*3,default_Kxd=[37]*3+[2]*3,
                    joint_low=core.LOW.tolist(),joint_high=core.HIGH.tolist(),
                    hz=1000,frame='world',reference='panda_link8',controller='offline_fixture')

    def read(self):
        state = dict(timestamp_ns=int(self.clock()*1e9),flange_pos=self.p.tolist(),flange_quat=self.r.tolist(),
                    joints=self.q.tolist(),velocities=[0]*7,wrench=self.w.tolist(),smin=.1,residual=.05,
                    policy_running=self.running,error_code=0,gripper_width=self.width)
        state.update(self.override)
        return state

    def start(self,kp,kd):
        self.starts += 1; self.running = True
        self.start_gains.append(kp.tolist())

    def stop(self):
        self.stops += 1; self.running = False

    def reset_primitive(self,state):
        pass  # The fixture has no second IK envelope; core checks real state.

    def update(self,pose,kp,kd,heartbeat):
        self.updates.append(dict(pose=pose.tolist(),Kx=kp.tolist(),Kxd=kd.tolist(),heartbeat=heartbeat))
        if not self.frozen:
            self.p,self.r = pose[:3].copy(),pose[3:].copy()
        self.w = self.bias.copy()
        if self.plane is not None and self.p[2]<self.plane:
            self.w[2] += (self.plane-self.p[2])*kp[2]
            self.p[2] = self.plane
        if self.held:
            self.w[2] -= self.payload*9.81


class DemoPort:
    def __init__(self,config=None,*,plane=None,payload=.6,emit=None):
        self.config = config or BackendConfig(motion_enabled=True)
        self.clock = Clock()
        self.driver = Plant(self.clock,plane=plane,payload=payload)
        self.session = core.GuardedSession(self.driver,clock=self.clock,emit=emit,
            limits=core.Limits(speed=self.config.speed_m_s,f_cap=self.config.f_cap,t_cap=self.config.t_cap))
        self.session.release_sleep=self.clock.sleep
        self.metadata,self.limits = self.driver.metadata(),self.session.limits.as_dict()

    def open(self):
        self.session.read_checked()

    def close(self):
        self.session.end()

    def sleep(self,seconds):
        until=self.clock()+seconds
        while self.session.holding and self.clock()<until-1e-10:
            self.clock.sleep(min(.02,until-self.clock()));self.session.tick()
        if self.clock()<until:self.clock.sleep(until-self.clock())

    def request(self,op,**args):
        try:
            if op=='state':
                return dict(state=self.session.read_checked(),session=self.session.status())
            if op=='end':
                self.session.end()
                return dict(state=self.session.read_checked())
            if not self.config.motion_enabled:
                raise RuntimeError('Read-only fixture')
            if op=='begin':
                state=self.session.begin(args['budget'])
                return dict(state=state,session=self.session.status())
            if op=='resume':
                state=self.session.resume(args['budget'],args['kp'])
                return dict(state=state,session=self.session.status())
            if op in ('finish','hold'):
                getattr(self.session,op)(sleep=self.clock.sleep)
                return dict(state=self.session.read_checked(),session=self.session.status())
            if op=='set_kp':
                return self.session.set_kp(args['kp'])
            if op=='step':
                self.session.command(args['dpos'],args['drot'],args['tcp_translation'],args['tcp_quaternion'],
                                     args.get('lead'))
                for _ in range(5):
                    self.clock.sleep(.01); self.session.tick()
                return dict(state=self.session.read_checked())
            raise ValueError('Unknown operation')
        except BaseException as exc:
            self.session.fail(exc)
            raise

    def grip(self,closed):
        self.driver.grips.append(closed)
        self.driver.width = .05 if closed else .08
        self.driver.held = closed
        self.sleep(3.1) if self.session.holding else self.clock.sleep(3.1)
        self.driver.w = self.driver.bias.copy()
        if closed:
            self.driver.w[2] -= self.driver.payload*9.81

    def calibration(self):
        return dict(verified=True,force_sign=1,bias_force=self.driver.bias[:3].tolist(),
                    bias_torque=self.driver.bias[3:].tolist(),position_low=[.30,-.10,.20],
                    position_high=[.50,.10,.49],flange_quaternion=[1,0,0,0],
                    orientation_radius_rad=.1,noise_std=.03,bias_std=.3,torque_verified=False,
                    gripper_verified=True,gripper_delta_wrench=[0]*6,closed_width_m=.05,source='offline fixture only')

    def backend(self,*,keep_controller=False):
        return ContinuousBackend(self,config=self.config,calibration=self.calibration(),
                                 clock=self.clock,sleep=self.sleep if keep_controller else self.clock.sleep,
                                 keep_controller=keep_controller)


class DemoWorld:
    """Two standing bottles on a synthetic table: positions for the object list, a payload while one is
    held and lifted, placement where it is released. A contract fixture, not physics."""
    TABLE_Z = .005

    def __init__(self, port, tcp_translation):
        self.port, self.tcp_translation = port, tcp_translation
        self.objects = [dict(name='A', xy=[.45, .12], height=.20, diameter=.065, mass=.32),
                        dict(name='B', xy=[.45, -.10], height=.20, diameter=.065, mass=.15)]
        self.held, self.grasp_z = None, None

    def tcp(self):
        d = self.port.driver
        return d.p+R.from_quat(d.r).apply(self.tcp_translation)

    def at_hand(self):
        tcp = self.tcp()
        for o in self.objects:
            if (np.linalg.norm(tcp[:2]-np.asarray(o['xy'])) < .02 and
                    self.TABLE_Z+.02 < tcp[2] < self.TABLE_Z+o['height']+.005):
                return o
        return None

    def install(self):
        """Grip like the runner's real RobotPort: one close per trial, no trial budget, a new trial only
        after a confirmed release, and a close below the minimum width reported as an error."""
        port, driver = self.port, self.port.driver
        original_grip, original_update = port.grip, driver.update
        port.trial, port.released, port.attempts, port.max_trials = 0, False, 0, None

        def next_weigh_trial(trial):
            if trial != port.trial+1 or trial < 1:
                raise BackendError('Sequential six-trial gripper budget required')
            if trial > 1 and not port.released:
                raise BackendError('Previous bottle release not confirmed; no retry')
            port.trial, port.released, port.attempts = trial, False, 0

        def grip(closed):
            if not port.trial:
                raise BackendError('Gripper disabled or no gripper trial')
            if closed:
                if port.attempts:
                    raise RuntimeError('Grasp retry prohibited')
                port.attempts = 1
            original_grip(closed)
            driver.held = False                       # the payload is applied below, only when lifted
            driver.w = driver.bias.copy()
            if closed:
                self.held = self.at_hand()
                self.grasp_z = float(self.tcp()[2])
                top = self.held and self.tcp()[2] > self.TABLE_Z+self.held['height']-.03
                driver.width = (.0 if self.held is None else .03 if top else self.held['diameter'])
                if self.held is None:
                    raise RuntimeError('No plausible grasp width; stop without retry')
            else:
                if self.held is not None:
                    self.held['xy'] = self.tcp()[:2].round(4).tolist()
                self.held = None
            port.released = not closed

        def update(*args):
            original_update(*args)
            if self.held is not None:
                self.held['xy'] = self.tcp()[:2].round(4).tolist()
                if self.tcp()[2] > self.grasp_z+.005:      # off the table: the hand carries it
                    driver.w[2] -= self.held['mass']*G
        port.grip, driver.update, port.next_weigh_trial = grip, update, next_weigh_trial
        port.next_weigh_trial(1)

    def found(self):
        tcp = self.tcp()
        objects = []
        for i, o in enumerate(sorted(self.objects, key=lambda o: -o['xy'][1]), 1):
            lifted = 0. if o is not self.held else max(0., float(tcp[2])-self.grasp_z)
            z0 = self.TABLE_Z+lifted
            objects.append(dict(id=i, centre=[*o['xy'], round(z0+o['height']/2, 3)], top=[*o['xy'], round(z0+o['height'], 3)],
                                height=o['height'], length=o['diameter'], width=o['diameter'], axis=[1., 0.], axis_deg=0.,
                                partial=False, label='upright bottle (synthetic)',
                                under_hand=bool(np.linalg.norm(tcp[:2]-np.asarray(o['xy'])) < o['diameter']/2+.01
                                                and tcp[2] < z0+o['height']+.03)))
        return dict(table_z=self.TABLE_Z, view=None, objects=objects)
