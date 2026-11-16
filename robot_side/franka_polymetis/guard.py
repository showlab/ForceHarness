"""Robot-side guards (layer 2 safety): every hard limit the arm obeys, independent of the workstation.

Runs on the real-time computer inside worker.py, Python 3.8 with numpy/scipy only and no hardware
imports, so it can also run offline (tests, the synthetic plant). The driver supplies read(), start(),
update(), stop() and metadata(); this module decides whether a state is acceptable, how fast and how
far a target may move, when a controller must be released, and when the session faults.
The limits here can only be tightened at run time (Limits rejects a looser value).
"""
import math
import time
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation as R

LOW = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-.0175,-2.8973])
HIGH = np.array([2.8973,1.7628,2.8973,-.0698,2.8973,3.7525,2.8973])
# Intersect the Panda joint bounds with the FR3 bounds, so that no joint inherits a Panda limit
# that is wider than the FR3 one (FR3 joint 6 has a much higher lower limit).
LOW = np.maximum(LOW,[-2.90074002,-1.83609002,-2.90074002,-3.07702002,-2.87630335,.43982265,-3.05083335])
HIGH = np.minimum(HIGH,[2.90074002,1.83609002,2.90074002,-.11693708,2.87630335,4.62163335,3.05083335])


def vec(value,n,name):
    a = np.asarray(value,dtype=float)
    if a.shape != (n,) or not np.isfinite(a).all():
        raise ValueError('%s must have %s finite values' % (name,n))
    return a.copy()


def quat(value):
    a = vec(value,4,'quaternion')
    if abs(np.linalg.norm(a)-1) > .01:
        raise ValueError('Quaternion is not normalized')
    return a/np.linalg.norm(a)


def angle(a,b):
    return float((R.from_quat(quat(a)).inv()*R.from_quat(quat(b))).magnitude())


def workspace(p):
    x,y,z = vec(p,3,'flange position')
    # Flange box in the base frame. With the fingers pointing down the fingertips sit about 0.10 m below
    # the flange, so the 0.12 m floor keeps them about 2 cm above a table at z = 0.
    if not (.20 < x < .65 and abs(y) < .30 and .12 < z < .55):
        raise ValueError('Flange workspace exceeded')


def norm_clip(a,limit):
    n = float(np.linalg.norm(a))
    return a*min(1.,limit/max(n,1e-12))


# A step may ask for the larger contact lead only once the raw force has risen this much since the
# primitive began, i.e. the hand is already pressing on something. Free motion keeps the 8 mm lead.
CONTACT_LEAD_FORCE_N = 3.
# The target speed bounds (8 mm/s, 0.15 rad/s) apply over the real time since the previous accepted
# target, capped at this window, instead of a nominal 50 ms per command: steps arrive every 60-70 ms,
# and a fixed 50 ms budget per step would hold free motion to about 5 mm/s.
RATE_WINDOW_S = .08


class Limits:
    def __init__(self, **kw):
        # angular_speed: turning about the vertical tool axis at 0.15 rad/s moves the fingertips by
        # about 6 mm/s, below the 8 mm/s translation limit.
        defaults = dict(dt=.05, speed=.008, angular_speed=.15, lead=.008,
                        # contact_lead: a target pressing on something may lead the flange by up to 25 mm
                        # (the pressing stiffness is about 0.75 N/mm, so 25 mm gives about 18 N); the
                        # tracking limit and the driver's measured-joint check grow with the granted lead,
                        # nothing else does.
                        contact_lead=.025,
                        # Per-primitive envelopes: rotation, displacement, joint travel and duration.
                        tracking=.015, rotation=.60, displacement=.15, joint_delta=.9,
                        f_cap=25.,t_cap=3., stale=.15, lease=.30, duration=300.,
                        # residual: the wrench residual grows by about 0.8 N*m per radian of wrist turn
                        # (model error, also at rest), so the absolute ceiling is 3 N*m, and a rise of
                        # 1 N*m within one primitive (something touching the links) stops the arm.
                        stall_seconds=4., joint_margin=.03, smin=.01, residual=3., residual_rise=1.,
                        # width_check: 0 skips the gripper-width range check and the width-change check
                        # in read_checked for this worker (used for a tool clamped by the operator, whose
                        # width reading is not meaningful). A reading that is not a number still stops;
                        # force, torque, motion and workspace guards are unchanged.
                        width_check=1.)
        if set(kw)-set(defaults):
            raise ValueError('Unknown limits')
        for k,maximum in defaults.items():
            value = kw.get(k,maximum)
            if (isinstance(value,bool) or not math.isfinite(value)
                    or (value <= 0 and not (k == 'width_check' and value == 0))):
                raise ValueError('Invalid limit '+k)
            # Conditioning/limit margins may only be made more conservative.
            if k in ('joint_margin','smin'):
                if value < maximum:
                    raise ValueError('Cannot reduce '+k)
            elif value > maximum:
                raise ValueError('Cannot increase '+k)
            setattr(self,k,float(value))
        if self.dt != .05:
            raise ValueError('This version has a fixed requested 20 Hz step interface')
        if self.contact_lead < self.lead:
            raise ValueError('contact_lead must be at least lead')

    def as_dict(self):
        return dict(vars(self))


class GuardedSession:
    """Robot-side session. A tick runs independently of workstation decisions.

    driver: read(), start(kp,kd), update(pose,kp,kd,heartbeat), stop(), metadata().
    read includes matching timestamped pose/joints/wrench from one RobotState.
    """
    def __init__(self,driver,limits=None,clock=time.monotonic,emit=None):
        self.driver,self.limits,self.clock = driver,limits or Limits(),clock
        self.emit = emit or (lambda row:None)
        self.active = self.owned = False
        self.fault = None
        self.initial = self.desired = None
        self.last_stamp = None
        self.advanced_at = self.clock()
        self.last_tick = self.clock()
        self.last_step = self.clock()
        self.rate_from = self.last_step-self.limits.dt
        self.started = self.clock()
        self.token = 0
        self.kp_ratio = np.ones(6)
        meta = driver.metadata()
        self.kp_max,self.kd_max = vec(meta['default_Kx'],6,'Kx'),vec(meta['default_Kxd'],6,'Kxd')
        if np.any(self.kp_max<=0) or np.any(self.kd_max<=0):
            raise ValueError('Invalid hardware default gains')
        # Intersect the fixed joint box with the configured hardware limits.
        self.low = np.maximum(LOW,vec(meta['joint_low'],7,'joint_low'))
        self.high = np.minimum(HIGH,vec(meta['joint_high'],7,'joint_high'))
        if np.any(self.low>=self.high):
            raise ValueError('No common joint limits')
        self.current = None
        self.releasing = False
        self.release_sleep = time.sleep
        self.release_cancel = None
        self.holding = False
        self.controller_starts = 0
        self.holds = []
        self.hold_samples = 0
        self.lead_now = self.limits.lead
        self.initial_force = None
        self.initial_residual = None

    def status(self):
        return dict(mode='holding' if self.holding else 'active' if self.active else 'idle',
                    controller_starts=self.controller_starts,holds=[dict(h) for h in self.holds])

    def _tcp(self,s):
        return np.array(s['flange_pos'])+R.from_quat(s['flange_quat']).apply(self.tcp_translation)

    def _hold_observation(self,s):
        h=self.holds[-1]
        distance=float(np.linalg.norm(self._tcp(s)-self.hold_anchor))
        error=max(float(np.linalg.norm(self._tcp(s)-self.hold_target)),
                  float(np.linalg.norm(np.array(s['flange_pos'])-self.desired[:3])))
        h['max_displacement_m']=max(h['max_displacement_m'],distance)
        h['max_target_error_m']=max(h['max_target_error_m'],error)
        h['elapsed_s']=self.clock()-h['started']
        if error>.005+1e-12:
            raise RuntimeError('hold_unexpected_motion: position deviated >5 mm')

    def _leave_hold(self,reason):
        if self.holding:
            self.holds[-1].update(ended=self.clock(),reason=reason,
                                  elapsed_s=self.clock()-self.holds[-1]['started'])
            self.emit(dict(type='hold_end',**self.holds[-1]))
            self.holding=False

    def read_checked(self):
        s = self.driver.read()
        p = vec(s['flange_pos'],3,'pose'); q = quat(s['flange_quat'])
        joints = vec(s['joints'],7,'joints'); vec(s['velocities'],7,'velocities')
        w = vec(s['wrench'],6,'wrench')
        width = s['gripper_width']
        if not math.isfinite(width) or (self.limits.width_check and not 0<=width<=.085):
            raise RuntimeError('Invalid gripper width')
        workspace(p)
        stamp = s['timestamp_ns']
        if not isinstance(stamp,int) or stamp<=0 or (self.last_stamp is not None and stamp<self.last_stamp):
            raise RuntimeError('Invalid/backwards robot timestamp')
        if stamp != self.last_stamp:
            self.last_stamp,self.advanced_at = stamp,self.clock()
        if self.clock()-self.advanced_at > self.limits.stale:
            raise RuntimeError('Robot timestamp stalled')
        if s.get('error_code',0):
            raise RuntimeError('Robot hardware error')
        if np.any(joints<self.low+self.limits.joint_margin) or np.any(joints>self.high-self.limits.joint_margin):
            raise RuntimeError('Joint margin exceeded')
        if np.linalg.norm(w[:3])>=self.limits.f_cap or np.linalg.norm(w[3:])>=self.limits.t_cap:
            raise RuntimeError('Raw force/torque cap exceeded')
        if not math.isfinite(s['smin']) or not math.isfinite(s['residual']) or s['smin']<self.limits.smin or s['residual']>self.limits.residual:
            raise RuntimeError('Wrench estimate conditioning/residual rejected')
        if self.initial_residual is not None and s['residual']-self.initial_residual>self.limits.residual_rise:
            raise RuntimeError('Wrench estimate conditioning/residual rejected: rose >%.1f N*m in this primitive'
                               % self.limits.residual_rise)
        if self.active:
            if self.limits.width_check and not self.holding and abs(width-self.initial_width)>.003:
                raise RuntimeError('Gripper width changed during arm primitive')
            if not s['policy_running']:
                raise RuntimeError('Owned controller stopped')
            if np.linalg.norm(p-self.initial['flange_pos'])>self.limits.displacement:
                raise RuntimeError('Per-primitive displacement exceeded')
            if angle(q,self.initial['flange_quat'])>self.limits.rotation:
                raise RuntimeError('Per-primitive rotation exceeded')
            if np.max(np.abs(joints-self.initial['joints']))>self.limits.joint_delta:
                raise RuntimeError('Per-primitive joint delta exceeded')
            # Keep the same 7 mm slack over whatever lead is granted right now.
            if np.linalg.norm(p-self.desired[:3])>self.limits.tracking+max(0.,self.lead_now-self.limits.lead):
                raise RuntimeError('Tracking error exceeded')
            if self.releasing and self.clock()>self.release_deadline:
                raise RuntimeError('Release duration exceeded')
            if not self.releasing and not self.holding and self.clock()-self.started>self.budget:
                raise RuntimeError('Primitive duration exceeded')
            if self.holding and not self.releasing:
                self._hold_observation(s)
        self.current = s
        self.emit(dict(type='sample',time=self.clock(),**s))
        return s

    def fail(self,exc):
        original = '%s: %s' % (type(exc).__name__,exc)
        # Motion-budget/stall and rejected-target guards can release the last
        # safe target. Unreliable state, hard force limits and connection faults
        # must terminate immediately. Failed release never recursively retries.
        recoverable = any(reason in str(exc) for reason in (
            'Primitive duration exceeded','Cartesian tracking stalled',
            'Requested step exceeds','Per-primitive desired pose exceeds',
            'Tracking error exceeded','Joint margin exceeded',
            'Per-primitive displacement exceeded','Per-primitive rotation exceeded',
            'Per-primitive joint delta exceeded','Flange workspace exceeded'))
        if recoverable and self.active and not self.releasing and not self.fault:
            # Notify the workstation before a ramp that outlasts a step RPC.
            self.emit(dict(type='fault_release_start',error=original))
            try:
                self.finish(sleep=self.release_sleep,cancel=self.release_cancel)
            except BaseException as release_error:
                original += '; release failed: '+str(release_error)
        self.fault = original
        try:
            self.end()
        except Exception as stop_error:
            self.fault += '; STOP FAILED: '+str(stop_error)
        self.emit(dict(type='fault',error=self.fault))

    def require_ok(self):
        if self.fault:
            raise RuntimeError('Session fault latched: '+self.fault)

    def begin(self,budget=30.):
        self.require_ok()
        if self.active:
            raise RuntimeError('Already active')
        if isinstance(budget,bool) or not math.isfinite(budget) or not 0<budget<=self.limits.duration:
            raise ValueError('Invalid primitive budget')
        try:
            s = self.read_checked()
            if s['policy_running']:
                raise RuntimeError('Another controller active; ownership not acquired')
            if np.max(np.abs(s['velocities']))>.03:
                raise RuntimeError('Start requires stationary arm')
            self.initial = {k:np.array(s[k]).copy() for k in ('flange_pos','flange_quat','joints')}
            self.initial_width = s['gripper_width']
            self.initial_force,self.lead_now = np.array(s['wrench'][:3]),self.limits.lead
            self.initial_residual = float(s['residual'])
            self.desired = np.r_[s['flange_pos'],s['flange_quat']]
            self.tcp_translation=np.zeros(3)
            self.tcp_quaternion=np.array([0.,0.,0.,1.])
            self.budget = budget
            self.started = self.last_step = self.last_tick = self.clock()
            self.progress_pos = np.array(s['flange_pos'])
            self.progress_at = self.clock()
            self.target_history = deque([(self.clock(),self.desired[:3].copy())])
            self.target_moving = False
            # Ownership is armed before sending start: a lost reply may hide a
            # controller that actually started and must be stopped in finally.
            self.owned = True
            self.driver.start(self.kp_max*self.kp_ratio,self.kd_max*np.sqrt(self.kp_ratio))
            self.started = self.last_step = self.last_tick = self.clock()
            self.rate_from = self.started-self.limits.dt     # the first step keeps one nominal 50 ms window
            self.active = True
            self.controller_starts += 1
            self.emit(dict(type='begin',budget=budget))
            return self.read_checked()
        except BaseException as exc:
            self.fail(exc)
            raise

    def end(self):
        self._leave_hold('end')
        if self.owned:
            self.driver.stop()
            self.owned = False
        self.active = False
        self.initial_residual = None
        self.emit(dict(type='end'))

    def hold(self,sleep=time.sleep,cancel=None):
        self.require_ok()
        if not self.active or self.holding:
            raise RuntimeError('hold requires an executing primitive')
        return self.finish(sleep=sleep,cancel=cancel,keep=True)

    def resume(self,budget,kp):
        self.require_ok()
        if not self.holding:
            raise RuntimeError('resume requires holding state')
        if isinstance(budget,bool) or not math.isfinite(budget) or not 0<budget<=self.limits.duration:
            raise ValueError('Invalid primitive budget')
        k=vec(kp,6,'kp')
        if not np.array_equal(k/150.,self.kp_ratio):
            raise ValueError('resume gains differ; finish then begin required')
        s=self.read_checked()
        # Rebase the DRIVER's IK envelopes too, without changing the last target
        # or touching the installed controller or its episode ownership.
        self.driver.reset_primitive(s)
        self.initial={k:np.array(s[k]).copy() for k in ('flange_pos','flange_quat','joints')}
        self.initial_width=s['gripper_width']
        self.initial_force,self.lead_now=np.array(s['wrench'][:3]),self.limits.lead
        self.initial_residual=float(s['residual'])
        self.budget=budget
        self.started=self.last_step=self.last_tick=self.clock()
        self.rate_from=self.started-self.limits.dt
        self.progress_pos=np.array(s['flange_pos']);self.progress_at=self.clock()
        self.target_history=deque([(self.clock(),self.desired[:3].copy())]);self.target_moving=False
        self._leave_hold('resume')
        self.emit(dict(type='resume',budget=budget,state=s))
        return s

    def finish(self,sleep=time.sleep,cancel=None,keep=False):
        """Retract targets outside the servo loop, then terminate stock policy.

        Nominal release is 0.3 s. Larger errors take longer at the unchanged
        target speed; faults and disconnects use immediate end(), never a ramp.
        """
        self.require_ok()
        if not self.active:
            self.end()
            return
        # A 25 mm pressing lead takes about 3.1 s to retract at the unchanged 8 mm/s target speed.
        release_s=max(2.5,.5+self.limits.contact_lead/self.limits.speed)
        try:
            self.releasing=True
            self.release_deadline=(self.clock()+release_s if self.holding else
                                   min(self.clock()+release_s,self.started+self.limits.duration))
            s = self.read_checked()
            if cancel is not None and cancel():
                raise RuntimeError('Connection lost before controller stop')
            began=self.clock()
            r0=R.from_quat(self.desired[3:])
            tcp0=self.desired[:3]+r0.apply(self.tcp_translation)
            release_r=R.from_quat(s['flange_quat'])
            release_p=np.array(s['flange_pos'])
            release_tcp=release_p+release_r.apply(self.tcp_translation)
            self.emit(dict(type='release_start',nominal_s=.3,state=s))
            while True:
                if cancel is not None and cancel():
                    raise RuntimeError('Connection lost during target release')
                elapsed=self.clock()-began
                if elapsed>release_s:
                    raise RuntimeError('Target release did not converge within %.1f s' % release_s)
                if self.clock()-self.last_tick>self.limits.stale:
                    raise RuntimeError('NUC guard deadline missed during release')
                s=self.read_checked()
                r=R.from_quat(s['flange_quat'])
                tcp=np.array(s['flange_pos'])+r.apply(self.tcp_translation)
                remaining=max(0.,1.-(elapsed+self.limits.dt)/.3)
                wanted=release_tcp+(tcp0-release_tcp)*remaining
                wanted_r=R.from_rotvec((r0*release_r.inv()).as_rotvec()*remaining)*release_r
                rotation=(wanted_r*r.inv()).as_rotvec()
                self.command(wanted-tcp,rotation,self.tcp_translation,self.tcp_quaternion)
                self.token+=1
                self.driver.update(self.desired,self.kp_max*self.kp_ratio,
                                   self.kd_max*np.sqrt(self.kp_ratio),self.token)
                self.last_tick=self.clock()
                sleep(self.limits.dt)
                s=self.read_checked()
                if (self.clock()-began>=.3-1e-9
                    and np.linalg.norm(self.desired[:3]-release_p)<=.0001
                    and angle(self.desired[3:],release_r.as_quat())<=.001):
                    break
            if cancel is not None and cancel():
                raise RuntimeError('Connection lost before controller stop')
            if keep:
                self.holding=True
                self.hold_anchor=self._tcp(s)
                self.hold_target=self.desired[:3]+R.from_quat(self.desired[3:]).apply(self.tcp_translation)
                self.hold_force=np.array(s['wrench'][:3]);self.hold_samples=0
                self.holds.append(dict(index=len(self.holds)+1,started=self.clock(),
                    anchor_tcp_m=self.hold_anchor.tolist(),target_tcp_m=self.hold_target.tolist(),
                    max_displacement_m=0.,max_target_error_m=0.,max_force_change_N=0.,elapsed_s=0.))
                self.emit(dict(type='hold_start',release_elapsed_s=self.clock()-began,**self.holds[-1]))
            else:
                self.emit(dict(type='normal_stop',mode='target_release_then_stock_terminate',
                               release_elapsed_s=self.clock()-began,state=s))
                self.end()
        except BaseException as exc:
            self.fail(exc)
            raise
        finally:
            self.releasing=False

    def set_kp(self,value):
        self.require_ok()
        k = np.ones(6)*float(value) if np.isscalar(value) else vec(value,6,'kp')
        if not np.isfinite(k).all() or np.any(k<=0) or np.any(k>150) or isinstance(value,bool):
            raise ValueError('kp is a normalized 0<kp<=150 scalar or six-vector; 150=hardware defaults')
        if self.active and not np.array_equal(k/150.,self.kp_ratio):
            raise RuntimeError('Stock controller gains are fixed; change gains between motion sessions')
        self.kp_ratio = k/150.
        self.emit(dict(type='gains',normalized=k.tolist(),Kx=(self.kp_max*self.kp_ratio).tolist(),
                       joint_gain_ratio=float(np.min(self.kp_ratio))))
        return dict(kp=k.tolist(),Kx=(self.kp_max*self.kp_ratio).tolist(),Kxd=(self.kd_max*np.sqrt(self.kp_ratio)).tolist(),
                    joint_gain_ratio=float(np.min(self.kp_ratio)),kp_semantics='normalized hybrid gains')

    def grant_lead(self,requested,s):
        """The requested lead if it is the normal one, or if the hand is already pressing."""
        if requested is None:
            return self.limits.lead
        if isinstance(requested,bool) or not math.isfinite(requested) or not 0<requested<=self.limits.contact_lead:
            raise ValueError('Requested lead must be in (0, contact_lead]')
        pressing = np.linalg.norm(np.array(s['wrench'][:3])-self.initial_force)>=CONTACT_LEAD_FORCE_N
        return float(requested) if requested<=self.limits.lead or pressing else self.limits.lead

    def command(self,dpos,drot,tcp_translation,tcp_quaternion,lead=None):
        self.require_ok()
        if not self.active or (self.holding and not self.releasing):
            raise RuntimeError('step requires an explicit active primitive')
        dpos = vec(dpos,3,'dpos'); drot = vec(drot,3,'drot')
        offset = vec(tcp_translation,3,'TCP offset'); tool_q = quat(tcp_quaternion)
        if np.linalg.norm(offset)>.25 or np.linalg.norm(dpos)>.06 or np.linalg.norm(drot)>.10:
            raise ValueError('Requested step exceeds initial adaptation envelope')
        s = self.read_checked()
        lead = self.grant_lead(lead,s)
        window = min(max(self.clock()-self.rate_from,0.),RATE_WINDOW_S)
        step_limit = self.limits.speed*window
        measured_r = R.from_quat(s['flange_quat'])
        tcp = np.array(s['flange_pos'])+measured_r.apply(offset)
        tcp_r = measured_r*R.from_quat(tool_q)
        wanted_tcp = tcp+dpos
        wanted_flange_r = R.from_rotvec(drot)*tcp_r*R.from_quat(tool_q).inv()
        rotation_error = (wanted_flange_r*R.from_quat(self.desired[3:]).inv()).as_rotvec()
        new_r = R.from_rotvec(norm_clip(rotation_error,self.limits.angular_speed*window))*R.from_quat(self.desired[3:])
        # Rate-limit both the TCP target and flange target (rotation moves the
        # offset tool even when the caller requests zero TCP translation).
        old_tcp = self.desired[:3]+R.from_quat(self.desired[3:]).apply(offset)
        new_tcp = old_tcp+norm_clip(wanted_tcp-old_tcp,step_limit)
        p = new_tcp-new_r.apply(offset)
        p = np.array(s['flange_pos'])+norm_clip(p-np.array(s['flange_pos']),lead)
        # A common interpolation fraction preserves BOTH speed bounds; clipping
        # translation again after rotation can otherwise accelerate the TCP.
        old_r = R.from_quat(self.desired[3:])
        rv = (new_r*old_r.inv()).as_rotvec()
        delta = p-self.desired[:3]
        def trial(fraction):
            r = R.from_rotvec(rv*fraction)*old_r
            q = self.desired[:3]+delta*fraction
            return q,r,max(np.linalg.norm(delta*fraction),np.linalg.norm(q+r.apply(offset)-old_tcp))<=step_limit+1e-12
        # The largest fraction within both bounds, found by bisection.
        low,high = (1.,1.) if trial(1.)[2] else (0.,1.)
        for _ in range(20 if low<high else 0):
            if trial((low+high)/2)[2]:
                low = (low+high)/2
            else:
                high = (low+high)/2
        p,new_r,_ = trial(low)
        # A larger lead granted earlier shrinks at the target speed, never in one jump.
        # With the normal 8 mm lead throughout this is exactly the plain lead check.
        previous = float(np.linalg.norm(self.desired[:3]-s['flange_pos']))
        if np.linalg.norm(p-s['flange_pos'])>max(lead,min(self.lead_now,previous))+1e-9:
            raise RuntimeError('Cannot preserve target rate and lead simultaneously')
        self.lead_now = max(lead,float(np.linalg.norm(p-s['flange_pos'])))
        workspace(p)
        if np.linalg.norm(p-self.initial['flange_pos'])>.12 or angle(new_r.as_quat(),self.initial['flange_quat'])>self.limits.rotation:
            raise ValueError('Per-primitive desired pose exceeds 120 mm / rotation-limit envelope')
        # Store accepted command targets. The independent tick evaluates the
        # cumulative path over 1 s, not per-command micrometre fluctuations.
        self.target_history.append((self.clock(),p.copy()))
        self.desired = np.r_[p,new_r.as_quat()]
        self.tcp_translation,self.tcp_quaternion=offset.copy(),tool_q.copy()
        self.last_step = self.rate_from = self.clock()
        self.emit(dict(type='target',time=self.last_step,requested_dpos=dpos.tolist(),
                       requested_drot=drot.tolist(),desired_flange=self.desired.tolist(),lead=lead,
                       tcp_translation=offset.tolist(),tcp_quaternion=tool_q.tolist()))
        return self.desired.copy()

    def check_stall(self,actual,now):
        """Tracking-stall guard, also replayable with recorded targets and states.

        Targets are held between commands: include each target change whose
        command timestamp lies in (now-1 s, now]. Keep its preceding anchor.
        """
        cutoff=now-1.
        while len(self.target_history)>1 and self.target_history[1][0]<=cutoff:
            self.target_history.popleft()
        recent=np.array([pose for _,pose in self.target_history])
        self.target_moving=bool(np.linalg.norm(np.diff(recent,axis=0),axis=1).sum()>=.0005)
        actual=np.asarray(actual)
        if not self.target_moving or np.linalg.norm(actual-self.progress_pos)>.0005:
            self.progress_pos,self.progress_at=actual.copy(),now
        error=np.linalg.norm(self.desired[:3]-actual)
        if error<=.002:
            self.progress_at=now
        if now-self.progress_at>self.limits.stall_seconds and error>.002:
            raise RuntimeError('Cartesian tracking stalled')

    def tick(self):
        self.require_ok()
        try:
            now = self.clock()
            if self.active and ((not self.holding and now-self.last_step>self.limits.lease)
                                or now-self.last_tick>self.limits.stale):
                raise RuntimeError('Command lease or NUC guard deadline missed')
            s = self.read_checked()
            self.last_tick = now
            if self.holding:
                change=float(np.linalg.norm(np.array(s['wrench'][:3])-self.hold_force))
                h=self.holds[-1];h['max_force_change_N']=max(h['max_force_change_N'],change)
                self.hold_samples=self.hold_samples+1 if change>8. else 0
                if self.hold_samples>=3:
                    self.finish(sleep=self.release_sleep,cancel=self.release_cancel)
                    raise RuntimeError('hold_contact: force changed >8 N for 3 ticks')
                if now-h['started']>=60.:
                    self.emit(dict(type='hold_timeout',elapsed_s=now-h['started']))
                    self._leave_hold('hold_timeout')
                    # The fixed target was unloaded on hold entry; no new ramp.
                    self.end()
                else:
                    self.token+=1
                    self.driver.update(self.desired,self.kp_max*self.kp_ratio,
                                       self.kd_max*np.sqrt(self.kp_ratio),self.token)
                return s
            if self.active:
                self.check_stall(s['flange_pos'],now)
                self.token += 1
                self.driver.update(self.desired,self.kp_max*self.kp_ratio,self.kd_max*np.sqrt(self.kp_ratio),self.token)
            return s
        except BaseException as exc:
            self.fail(exc)
            raise
