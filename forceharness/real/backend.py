"""ContinuousBackend: the real FR3 as a backend for the shared primitives (layers 1 and 2).

A 20 Hz request/feedback interface over the transport to the robot-side worker, which checks every
deadline and hard limit itself. Constructing or connecting does not start a controller. Use motion()
explicitly or ContinuousPrimitives, which scopes controller ownership to one primitive; with
keep_controller=True the robot side keeps the controller holding between primitives (a stop under load
would sag the arm). wrench() applies the calibration file: fixed bias, closed-gripper correction and the
validity scope; reaction_force() applies force_sign. The stock controller has no independent watchdog if
the worker crashes.
"""
from contextlib import contextmanager
import time
from types import SimpleNamespace

import numpy as np

from forceharness.real.config import BackendConfig
from forceharness.real.geometry import angular_distance,flange_to_tcp,torque_at_tcp,vector
from forceharness.real.types import BackendError,WrenchSample

# A step round trip of 150-170 ms happens occasionally on the network (the robot side itself takes
# <= 6 ms), so the workstation allows 250 ms. The robot side's own 300 ms command lease still stops
# the robot if steps stop arriving; this workstation check stays below it.
STEP_ROUND_TRIP_S=.25


class ContinuousBackend:
    dt = .05
    control_freq = 20
    kp = 150.
    action_dim = 7

    def __init__(self,port,*,config=None,calibration=None,clock=time.monotonic,sleep=time.sleep,keep_controller=False):
        if type(keep_controller) is not bool:
            raise ValueError('keep_controller must be a bool')
        self.port = port
        self.keep_controller=keep_controller
        self.holding=False
        self._session_stats={}
        self._motion_result=None
        self._port_closed=False
        self.config = config or BackendConfig()
        self.calibration = calibration or {}
        self.clock,self.sleep = clock,sleep
        self._state = None
        self._t0 = None
        self.active = False
        self.fault = None
        self._closed = None
        self._grasp_attempted = False
        self._last_grip = -float('inf')
        self._bias_force = vector(self.calibration.get('bias_force',[0,0,0]),3,'bias_force')
        self._bias_torque = vector(self.calibration.get('bias_torque',[0,0,0]),3,'bias_torque')
        self.force_sign = self.calibration.get('force_sign')
        self.force_noise_std = float(self.calibration.get('noise_std',.03))
        self.force_bias_std = float(self.calibration.get('bias_std',.3))
        self.peak_wrench = 0.
        self.peak_contact_force = float('nan')
        self.task = SimpleNamespace(has_gripper=True,extra={},truth_fn=None)
        self.task_state = {'contact_truth_available':False}
        self.kp_vec = np.ones(6)*150.
        # A pressing primitive may ask the robot side for its contact lead on the next step; None = 8 mm.
        self.step_lead = None
        self._fixed_gains_scope = False
        self._pending_kp = None
        self._unsigned_position_guard = False
        self._local_bias_sampled = False
        self._force_operation = None
        if self.calibration.get('force_scope') is not None:
            c = self.calibration
            if (c.get('force_scope')!='empty_downward_contact' or c.get('scope_verified') is not True
                or self.force_sign!=-1 or c.get('verified') or c.get('torque_verified')
                or c.get('gripper_verified')
                or set(c.get('allowed_primitives',[]))!={'push','slide','move_until_contact','probe'}
                or c.get('allowed_normal')!=[0,0,-1]):
                raise ValueError('Invalid K21 empty downward force calibration scope')
        if self.calibration.get('torque_verified') and not self.calibration.get('verified'):
            raise ValueError('Torque certification also requires verified force calibration')
        if self.calibration.get('verified'):
            c = self.calibration
            lo,hi = vector(c['position_low'],3,'lower scope'),vector(c['position_high'],3,'upper scope')
            if np.any(lo>=hi) or self.force_sign not in (-1,1) or isinstance(self.force_sign,bool):
                raise ValueError('Invalid verified calibration')
            from forceharness.real.geometry import quaternion
            quaternion(c['flange_quaternion'])
            # A calibration may declare its orientation scope record-only (relative
            # contact/resistance use while turning a held object); the measured
            # scope is still required to be sane.
            rotation_cap = 2. if c.get('orientation_scope_record_only') is True else .1
            if not np.isfinite(c['orientation_radius_rad']) or not 0<c['orientation_radius_rad']<=rotation_cap:
                raise ValueError('Calibration rotation scope must be measured, within %.1f rad' % rotation_cap)
            for name in ('noise_std','bias_std'):
                if not np.isfinite(c.get(name,0)) or c.get(name,0)<=0:
                    raise ValueError('Verified calibration requires positive '+name)

    def __enter__(self):
        if self._t0 is not None:
            raise BackendError('Sessions are single-use')
        self._t0 = self.clock()
        try:
            self.port.open()
            self._receive(self.port.request('state'))
            meta = self.port.metadata
            limits = self.port.limits
            if (limits['speed']>self.config.speed_m_s or limits['f_cap']>self.config.f_cap
                    or limits['t_cap']>self.config.t_cap):
                raise BackendError('NUC guard limits do not match requested tighter configuration')
            self._arm_ranges = np.column_stack((meta['joint_low'],meta['joint_high']))
            self._closed = False if self._state['gripper_width']>=.078 else None
            return self
        except BaseException:
            self.port.close()
            raise

    def __exit__(self,*args):
        self.close()

    def close(self):
        if self._port_closed:return
        try:
            if self.keep_controller:
                self.finish_episode()
            elif self.active:
                self.port.request('end')
        finally:
            self.active = self.holding = False
            self._port_closed=True
            self.port.close()

    def _receive(self,reply):
        if 'state' in reply:self._state=reply['state']
        if 'session' in reply:
            self._session_stats=reply['session']
            self.holding=self._session_stats['mode']=='holding'
        return reply

    def episode_report(self):
        return dict(keep_controller=self.keep_controller,
                    controller_starts=self._session_stats.get('controller_starts',0),
                    holds=[dict(h) for h in self._session_stats.get('holds',[])])

    def finish_episode(self):
        """Explicit episode boundary; defaults never keep a controller alive."""
        if self.keep_controller and not self._port_closed and self._t0 is not None:
            if self.fault:
                if self.active or self.holding:self._receive(self.port.request('end'))
            else:
                self._request('finish')
            self.active=self.holding=False

    def _wait_still(self,timeout=3.):
        """The robot side refuses to begin while any joint moves faster than 0.03 rad/s. Right after a controller
        stop under load the wrist can still swing (joint 6 at 0.1 rad/s seen after a peeling stroke), so wait first."""
        end=self.clock()+timeout
        while True:
            v=self._state.get('velocities') if self._state else None
            if v is None or np.max(np.abs(v))<=.02 or self.clock()>=end:
                return
            self.sleep(.05)
            self.refresh()

    def motion_result(self,termination,*,normal_contact=False):
        # Do not retain a controller after a force cap, guard or failed primitive.
        # A plain timeout while holding an object keeps it: a stop under load sags the arm, and the
        # next begin can then fault on the lead check (seen with a 0.42 kg bottle far from the base).
        self._motion_result=(termination=='completed' or (termination=='contact' and normal_contact)
                             or (termination=='timeout' and self._closed is True))

    def _request(self,op,**args):
        if self.fault:
            raise BackendError('Session fault latched: '+self.fault)
        try:
            reply = self.port.request(op,**args)
            return self._receive(reply)
        except BaseException as exc:
            self._abort(exc)
            raise

    def _abort(self,exc):
        self.fault = '%s: %s' % (type(exc).__name__,exc)
        try:
            self.port.request('end')
        except BaseException as stop_error:
            self.fault += '; stop not confirmed: '+str(stop_error)
            # EOF can stop a live worker. A crashed robot-side worker has no stock
            # controller watchdog; an unconfirmed stop remains a real fault.
            self.port.close()
        finally:
            self.active = self.holding = False

    def refresh(self):
        self._request('state')
        return self._state

    @contextmanager
    def motion(self,max_time=30.):
        if not self.config.motion_enabled:
            raise BackendError('Motion disabled')
        if self.active:
            raise BackendError('Nested primitives are not allowed')
        if self.keep_controller:
            self.refresh()  # The robot side may already have completed its idle timeout.
            if self.holding and self._session_stats['holds'][-1]['elapsed_s']>55.:
                # Leave a margin before the robot side's 60 s idle timeout.
                # finish is also valid if the timeout has just stopped it.
                self._request('finish')
        if self.keep_controller and self.holding:
            self._request('resume',budget=max_time,kp=self.kp_vec.tolist())
        else:
            self._wait_still()
            self._request('begin',budget=max_time)
        self.active = True
        self._motion_result=None
        completed = False
        try:
            yield self
            completed = True
        except BaseException as exc:
            self.fault = '%s: %s' % (type(exc).__name__,exc)
            raise
        finally:
            if self.active:
                try:
                    op='finish' if completed else 'end'
                    if self.keep_controller and completed:
                        op='hold' if self._motion_result is not False else 'finish'
                    self._receive(self.port.request(op))
                except BaseException as exc:
                    self._abort(exc)
                    raise
                finally:
                    self.active = False

    @property
    def sim_time(self):
        return self.clock()-self._t0

    @property
    def eef_pos(self):
        return flange_to_tcp(self._state['flange_pos'],self._state['flange_quat'],self.config)[0]

    @property
    def eef_quat(self):
        return flange_to_tcp(self._state['flange_pos'],self._state['flange_quat'],self.config)[1]

    @property
    def obs(self):
        return {'robot0_joint_pos':np.array(self._state['joints'])}

    @property
    def wrist_joint_rad(self):
        return float(self._state['joints'][-1])

    @property
    def gripper_opening(self):
        return self._state['gripper_width']

    @property
    def bias_force(self):
        return self._bias_force.copy()

    @bias_force.setter
    def bias_force(self,value):
        # The shared point_tool primitive re-expresses the bias while it turns the hand; in base axes that is an
        # identity. Never let it invent a new bias while a real load may be held.
        if not np.allclose(vector(value,3,'bias'),self._bias_force,atol=1e-9):
            raise BackendError('Implicit bias changes are disabled')

    @property
    def bias_torque(self):
        return self._bias_torque.copy()

    def calibration_valid(self):
        c = self.calibration
        if self._closed is None or not c.get('verified') or self.force_sign not in (-1,1) or isinstance(self.force_sign,bool):
            return False
        p = np.array(self._state['flange_pos'])
        low,high = vector(c.get('position_low',[]),3,'calibration lower'),vector(c.get('position_high',[]),3,'calibration upper')
        within_translation = bool(np.all(p>=low) and np.all(p<=high))
        # A calibration may declare its translation scope record-only: the scope is
        # recorded, orientation validity and every raw force / motion guard remain.
        translation_ok = within_translation or c.get('translation_scope_record_only') is True
        rotation_ok = (angular_distance(self._state['flange_quat'],c['flange_quaternion'])<=c['orientation_radius_rad']
                       or c.get('orientation_scope_record_only') is True)
        return bool(translation_ok and rotation_ok)

    def raw_wrench(self):
        w = np.array(self._state['wrench'])
        tau = torque_at_tcp(w[:3],w[3:],self._state['flange_quat'],self.config)
        return w[:3].copy(),tau,np.eye(3)

    @contextmanager
    def force_calibration_scope(self,name,direction):
        """An empty-downward-contact calibration certifies one empty downward use, never payloads or torque."""
        previous=self._force_operation
        self._force_operation=(name,direction)
        try:
            yield
        finally:
            self._force_operation=previous

    def scoped_force_valid(self):
        c=self.calibration
        if (c.get('force_scope')!='empty_downward_contact' or not c.get('scope_verified')
            or self._force_operation is None or self._closed is not False
            or self.gripper_opening<.078 or not self._local_bias_sampled):
            return False
        name,direction=self._force_operation
        d=np.asarray(direction,dtype=float)
        return bool(name in c['allowed_primitives'] and d.shape==(3,)
                    and np.isfinite(d).all() and np.allclose(d,[0,0,-1],atol=1e-9,rtol=0))

    def wrench(self):
        raw = np.array(self._state['wrench'])
        correction = np.zeros(6)
        valid = self.calibration_valid()
        if self._closed:
            correction = vector(self.calibration.get('gripper_delta_wrench',[0]*6),6,'gripper correction')
            valid &= self.calibration.get('gripper_verified') is True
            valid &= abs(self.gripper_opening-self.calibration.get('closed_width_m',-1))<=.003
        net = raw-np.r_[self._bias_force,self._bias_torque]-correction
        raw_t = torque_at_tcp(raw[:3],raw[3:],self._state['flange_quat'],self.config)
        net_t = torque_at_tcp(net[:3],net[3:],self._state['flange_quat'],self.config)
        if self.calibration.get('torque_verified'):
            # The shared primitives and estimators expect sensor torque opposite to the
            # reaction torque (estimators.TORQUE_SIGN=-1). Certification must cover
            # the SAME wrench sign for force and moment at this TCP.
            net_t = -self.force_sign*net_t
        self.peak_wrench = max(self.peak_wrench,float(np.linalg.norm(raw[:3])))
        return WrenchSample(self.sim_time,raw[:3],raw_t,raw[:3],raw_t,net[:3],net_t,net[:3],net_t,
                            self._state['timestamp_ns'],self.clock(),valid,reference_point='tcp')

    def reaction_force(self,w):
        if not w.bias_valid:
            if self.scoped_force_valid():
                return self.force_sign*w.net_force_world
            if self._unsigned_position_guard and self._local_bias_sampled and self._closed is False:
                # Free empty-gripper motion uses only the FORCE NORM. This
                # preserves sensor sign and does not certify force_sign.
                return w.net_force_world.copy()
            raise BackendError('Verified force calibration is missing or outside its measured pose/width scope')
        return self.force_sign*w.net_force_world

    @contextmanager
    def position_force_guard(self,enabled=False):
        previous=self._unsigned_position_guard
        self._unsigned_position_guard=bool(enabled)
        try:
            yield
        finally:
            self._unsigned_position_guard=previous

    def step(self,dpos,drot=None,grip=None):
        if not self.active:
            raise BackendError('Use motion() or ContinuousPrimitives before step')
        if self._pending_kp is not None:
            raise BackendError('Gain change inside a primitive requires a new controller session')
        dpos = vector(dpos,3,'dpos')
        drot = np.zeros(3) if drot is None else vector(drot,3,'drot')
        if grip is not None:
            if isinstance(grip,bool) or not np.isfinite(grip) or grip not in (-1.,1.):
                raise ValueError('Only binary gripper states -1/+1 are supported')
            if self._closed is None or (grip>0)!=self._closed:
                raise BackendError('Change gripper state explicitly before the continuous primitive')
        lead = {}
        if self.step_lead is not None:
            if not 0<self.step_lead<=self.port.limits.get('contact_lead',0.):
                raise ValueError('step_lead must be within the NUC contact lead')
            lead = dict(lead=float(self.step_lead))
        start = self.clock()
        before = self._state['timestamp_ns']
        self._request('step',dpos=dpos.tolist(),drot=drot.tolist(),
                      tcp_translation=list(self.config.tcp_translation),tcp_quaternion=list(self.config.tcp_quaternion),
                      **lead)
        elapsed = self.clock()-start
        if self._state['timestamp_ns']<=before or elapsed>STEP_ROUND_TRIP_S:
            causes = []
            if self._state['timestamp_ns']<=before:
                causes.append('Robot timestamp did not advance (%d -> %d)' % (before,self._state['timestamp_ns']))
            if elapsed>STEP_ROUND_TRIP_S:
                causes.append('Step round trip exceeded %.0f ms (%.3f ms)' % (STEP_ROUND_TRIP_S*1000,elapsed*1000))
            exc = BackendError('; '.join(causes))
            self._abort(exc)
            raise exc
        return self.wrench()

    def hold(self,n=1,grip=None):
        if type(n) is not int or not 1<=n<=1200:
            raise ValueError('hold expects 1..1200 cycles')
        w = None
        for _ in range(n):
            w = self.step(np.zeros(3),None,grip)
        return w

    def set_kp(self,kp):
        value = np.asarray(kp,dtype=float)
        value = np.repeat(value,6) if value.ndim==0 else value
        if (value.shape!=(6,) or not np.isfinite(value).all() or np.any(value<=0)
            or np.any(value>150) or isinstance(kp,(bool,np.bool_))):
            raise ValueError('kp must be a finite scalar or six-vector in (0,150]')
        if self.active and not np.array_equal(value,self.kp_vec) and self._fixed_gains_scope:
            # The inherited primitive restores its free gains after its last
            # step. Defer that restoration until motion() has stopped. A later
            # step is rejected, so a mid-primitive change cannot be hidden.
            self._pending_kp = value.copy()
            return dict(kp=value.tolist(),applied=False,deferred_until='controller_stopped')
        if self.keep_controller and self.holding and not np.array_equal(value,self.kp_vec):
            self._request('finish')
        reply = self._request('set_kp',kp=value.tolist())
        self.kp_vec = np.array(reply['kp'])
        return reply

    @contextmanager
    def fixed_gains(self,kp):
        """Select gains before a primitive; restore only after its controller ends."""
        if self.active or self._fixed_gains_scope:
            raise BackendError('Fixed-gain scopes cannot overlap')
        self.set_kp(kp)
        self._fixed_gains_scope = True
        completed = False
        try:
            yield
            completed = True
        finally:
            pending = self._pending_kp
            self._fixed_gains_scope = False
            self._pending_kp = None
            if completed and pending is not None and not self.active and not self.fault:
                self.set_kp(pending)

    def rezero_bias(self,steps=20,*,unloaded=False):
        if unloaded is not True or self._closed is not False or self.active:
            raise BackendError('Explicit empty, open, inactive session required for bias sampling')
        if type(steps) is not int or not 10<=steps<=200:
            raise ValueError('Bias requires 10..200 fresh samples')
        initial = self._request('state')['state']
        rows = []
        stamp = initial['timestamp_ns']
        for _ in range(steps):
            self.sleep(self.dt)
            s = self._request('state')['state']
            if (s['timestamp_ns']<=stamp or (s['policy_running'] and not (self.keep_controller and self.holding)) or np.max(np.abs(s['velocities']))>.03
                or np.linalg.norm(np.array(s['flange_pos'])-initial['flange_pos'])>.001
                or angular_distance(s['flange_quat'],initial['flange_quat'])>.01
                or s['gripper_width']<.078):
                raise BackendError('Empty reference was not fresh/stationary/open')
            stamp = s['timestamp_ns']; rows.append(s['wrench'])
        mean = np.mean(rows,axis=0)
        self._bias_force,self._bias_torque = mean[:3],mean[3:]
        self._local_bias_sampled=True
        # Sampling never certifies sign or a larger pose range.
        return mean

    def command_gripper(self,closed,*,confirmed=False):
        if self.fault:
            raise BackendError('Session fault latched: '+self.fault)
        if not self.config.motion_enabled or self.active or type(closed) is not bool:
            raise BackendError('Gripper requires an inactive motion-enabled session')
        if closed == self._closed:
            return
        if self._closed is None:
            raise BackendError('Gripper holding state unknown; no actuation')
        if confirmed is not True:
            raise BackendError('Current alignment/support confirmation required')
        if self.clock()-self._last_grip<3 or (closed and self._grasp_attempted):
            raise BackendError('Gripper interval or one-grasp budget exceeded')
        self._last_grip = self.clock(); self._grasp_attempted |= closed
        self._closed = None  # lost reply => held state unknown, never retry/open
        try:
            self.port.grip(closed)
            self._request('state')
            width = self.gripper_opening
            # A closed width down to 8 mm is a real grasp of a thin part; the lower
            # bound only has to reject "closed on nothing" (about 0 mm).
            if (closed and not .008<width<.075) or (not closed and width<.078):
                raise BackendError('Gripper width acceptance failed')
            self._closed = closed
        except BaseException as exc:
            self.fault = 'Gripper outcome unknown: '+str(exc)
            raise

    def gripper_contact_state(self):
        return dict(available=False,in_contact=False,touching=[],total_force=float('nan'))

    def images(self):
        return {}
