"""ContinuousPrimitives: the shared self-terminating primitives on the real FR3 (layer 2).

Subclasses forceharness.primitives.library.Primitives and keeps its controllers, scoping each primitive to one
explicit controller lifetime on the robot side (or one resume of the held controller). Adds what the real arm
needs: 105 mm reach per primitive, stall-compensated free motion, the contact lead for pressing, the stop check
after each primitive, and three primitives that exist only here: screw (press, then turn with the thread),
wiggle (press while rocking a plug into a port) and the hop mode of search. No implicit construction-time
leveling or tare; no simulated contact truth.
"""
import inspect
import math
from contextlib import contextmanager

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from forceharness.translation.estimators import Quantity
from forceharness.primitives.library import Primitives,_yaw_deg_xyzw
from forceharness.real.types import BackendError
from forceharness.real.geometry import check_flange_workspace
from forceharness.real.targets import StallCompensatedTarget,clipped
from forceharness.real.events import RealEventDetector,FREE_COLLISION_N,FIND_CONTACT_N

# Contact lead: with the normal 8 mm lead and about 1 N/mm of stiffness, a slide or push saturates at
# about 10 N. Once a slide/push is pressing (>= FIND_CONTACT_N), it asks the robot side for its contact
# lead (guard Limits.contact_lead, granted there only after a 3 N rise). Slide splits the lead: the press
# may use at most 80% of its length, so the motion along the surface keeps at least 60% instead of
# whatever the normal-force loop left over.
# Rotation: the robot side turns at up to 0.15 rad/s; workstation motions ask for <= 0.14 rad/s.
ROT_RATE_MAX_RAD_S=.14
# Steps arrive every 60-70 ms, not every 50 ms, so advancing move_to's nominal target by 8 mm/s x 50 ms per
# step gave about 5 mm/s. It advances by the real time between steps instead, capped like the robot side's
# RATE_WINDOW_S, which bounds the target at 8 mm/s over that same time.
STEP_WINDOW_S=.08
CONTACT_LEAD_M=.025         # the robot side's contact_lead
SLIDE_PRESS_SHARE=.8
# screw: rotate_z holds the TCP still, so a thread that pulls the screw down drags the hand against the
# controller (1.5-1.7 mm per 30 deg was observed, then 1-3 mm more once released). screw keeps the target
# push/Kz below the hand instead: the hand presses about push N however the force estimate drifts (2-4 N)
# and follows the thread down.
SCREW_SINK_MAX_M=.01        # more than this in one motion is no thread any more: stop
SCREW_STILL_M=.0002         # below this the screw did not move along its axis
# Press first, then turn: 1-3 N never pressed a screw into the thread. push becomes a lead with the measured
# stiffness of the stock controller at kp 150 (about 1 N/mm), so 15 N asks for the whole contact lead
# (granted by the robot side only after a 3 N rise). The ramp runs at 8 mm/s: 15 mm in about 1.9 s, well
# inside the robot side's 4 s tracking-stall window. The commanded yaw waits for the hand when a pressed
# thread slows the wrist (the robot side's IK continuity check stops at 0.05 rad).
# The first press seats the screw: press with the most the contact lead gives (15 N -> 15 mm, about 15-19 N
# on the FR3) and keep pressing until the screw stops sinking (seated) before turning; a screw that is not
# pressed in on the first press tends to tip over.
SCREW_PUSH_N,SCREW_PUSH_MAX_N=15.,15.
SCREW_KZ_N_PER_M=750.       # pressing stiffness measured on the FR3: about 0.75 N/mm
SCREW_PRESS_SPEED=.008
SCREW_PRESS_HOLD_S=1.
SCREW_PRESS_MAX_S=6.
SCREW_SEAT_WINDOW_S,SCREW_SEAT_STILL_M=.5,.0001    # seated: sank less than this over the window
# Press hold: screw has no workstation stop on the raw force, but a press can drive the raw reading into the
# robot side's absolute 25 N cap, which ends the session. So the push stops growing at SCREW_PRESS_RAW_N and
# backs off at speed_near while above SCREW_BACKOFF_RAW_N (a held reading can still creep by about 4 N); the
# primitive itself never stops for it. A jammed press can rise by about 11 N/s, so the hold watches the
# largest of the untared, tared and robot-side raw readings and acts HOLD_AHEAD_S ahead of the rise.
SCREW_PRESS_RAW_N=23.
SCREW_BACKOFF_RAW_N=24.
HOLD_AHEAD_S,HOLD_RELEASE_N=.3,3.     # for the screw, wiggle and search-thrust holds
SCREW_YAW_LEAD_RAD=.035
# wiggle: a plug can catch on the port mouth (the search sinks a few millimetres, insert advances nothing and a
# plain push does not move it). A tight rectangular fit needs a small yaw and sideways rock while pressing,
# so the chamfer guides it in: press like screw (contact lead, held at SCREW_PRESS_RAW_N raw), rock the yaw
# +-yaw_deg and circle `shift` sideways on incommensurate periods. It cannot fix a plug held upside down.
# The press ramps from WIGGLE_FORCE_START_N to `force` (default 18 N) in WIGGLE_FORCE_RAMP_S; once the hand
# stops following the circle (held sideways: in the port, or jammed) the rocking freezes and it only presses
# until it stops sinking, since circling a plug held by the port walls only trips the stall guard. Depth
# counts from the support surface touched last nearby (move_until_contact or a slide search), else from the start.
WIGGLE_FORCE_START_N,WIGGLE_FORCE_N,WIGGLE_FORCE_MAX_N,WIGGLE_FORCE_RAMP_S=8.,18.,20.,2.
WIGGLE_YAW_DEG,WIGGLE_YAW_MAX_DEG=3.,5.
WIGGLE_SHIFT_M,WIGGLE_SHIFT_MAX_M=.0005,.002
WIGGLE_DEPTH_M,WIGGLE_DEPTH_MAX_M=.008,.02
WIGGLE_TIME_S,WIGGLE_TIME_MAX_S=15.,30.
WIGGLE_YAW_PERIOD_S,WIGGLE_SHIFT_PERIOD_S,WIGGLE_RAMP_S=3.,3.9,1.
WIGGLE_HELD_M,WIGGLE_HELD_PATH_M=.00015,.0005    # held sideways: the hand moved < 0.15 mm in 1 s while the circle ran 0.5 mm
WIGGLE_BOTTOM_M,WIGGLE_BOTTOM_S,WIGGLE_PRESS_MAX_S=.0002,1.,4.   # pressed: sank < 0.2 mm in 1 s (or 4 s) = bottomed
# Two pads pinch a short knob, so a screw pivots in the grip under a few N at its tip about 6 cm below.
# Holding the hand rigidly at its start position sideways built up 1.2 -> 7.2 N there while turning, and the
# press tipped the screw further. So sideways the target follows the hand while turning (the screw decides
# where it goes; the runner's path guard still stops at 10 mm).
SCREW_TURN_PUSH_N=3.
# A constant 3 N while turning let a deepening thread push the screw back up. Each turn starts at
# SCREW_TURN_START_N and rises linearly to push (default 15 N) by the end of the segment, still holding at
# SCREW_PRESS_RAW_N raw (the robot side's absolute 25 N cap also sees the 8-12 N drift of the estimate at
# large wrist angles). Later segments of one call start with a quick press to the turn start instead of
# pressing until seated: the thread is already engaged.
SCREW_TURN_START_N=5.
SCREW_QUICK_HOLD_S=.3
# the push rises in steps every SCREW_RAMP_STEP_S (driven at the press speed, then held): a lead that keeps
# growing against a board that does not move reads as a tracking stall to the robot side (4 s) and the runner (3.5 s)
SCREW_RAMP_STEP_S=2.
# Fully free sideways, a press tips the screw over its tip in the hole mouth like a pushed stick. A screw
# pivoting at its tip about 6 cm below the pads stays up only if the sideways stiffness beats push/6 cm
# (8 N -> about 130 N/m). Pressing therefore holds the start position sideways (full stiffness, about 1 N/mm);
# turning keeps 70% of it (the target follows 30% of the hand's drift).
SCREW_TURN_SIDE_FOLLOW=.3      # 70% sideways stiffness while turning (15 N on a ~3 cm lever needs ~500 N/m)
# Per-mode defaults of the real search (hop tuned on a screw). A first aim from the camera is often 5-10 mm
# off, and friction keeps the hand 2-3 mm inside the spiral target, so the slide default radius is 12 mm
# (about 1 minute at a 2 mm pitch) and radius_max may go up to SEARCH_RADIUS_MAX_M.
SEARCH_RADIUS_MAX_M=.015
SEARCH_DEFAULTS=dict(slide=dict(radius_max=.012,pitch=.002,normal_force=1.,drop=.004,hop=.003),
                     hop=dict(radius_max=.004,pitch=.0015,normal_force=None,drop=.001,hop=.002))   # hop force: HOP_PROBE_N
# hop mode (see ContinuousPrimitives._search_hop): a 3 N touch never pressed the tip into a thread mouth
# (it sinks there only under a sustained press), so every probe stopped within +-0.4 mm and the search timed
# out; a probe presses at least HOP_PROBE_N and keeps pressing for HOP_PRESS_S.
HOP_PROBE_N=6.              # probe contact: the reaction rose this much over its own free-air zero
HOP_PRESS_S=1.              # then keep pressing at that force this long before the depth counts
HOP_ZERO_S=.3
HOP_MOVE_MAX_S=2.           # a point not reached by then (joint friction) is probed where the hand is
HOP_SURFACE_MAX_M=.03       # probing this far past the surface with no contact: no surface
# A search that starts above a port can have its first probe go into the port; with no probe before it, that
# depth became the "surface", and the hops (aimed relative to the start) could not lift the part out and move
# sideways, so it dipped into the same port again and again. So the last nearby support contact judges the
# first probe, each hop rises `hop` above the shallowest probe so far with time for the distance, and a hand
# that stays put through several unreached points at one depth stops as stuck.
HOP_PRIOR_NEAR_M=.015       # a support contact this close sideways to the start judges the first probe
HOP_PRIOR_MARGIN_M=.002     # ... by at least this much (it was touched at 5 N, the probe presses 6 N for 1 s)
HOP_MOVE_SPEED_M_S=.008     # robot-side target speed: a hop move gets HOP_MOVE_MAX_S plus its distance at this speed
HOP_STUCK_POINTS,HOP_STUCK_M=4,.001   # this many unreached points pressed within 1 mm sideways and in depth
HOP_POINT_MIN_S,HOP_POINT_TYPICAL_S,HOP_MAX_TIME_S=1.5,5.,195.   # ~4.8 s a point under friction; 195 s keeps the 75 ms-a-step robot-side budget under 300 s
# Wrist range: turning joint 7 to about 2.7 rad pulls the wrist camera cable tight, the wrist falls behind and
# the robot side's IK continuity check stops the session, so joint 7 stays within -0.8..2.4 rad. The inherited
# simulation clip assumes d(j7) = +d(yaw); on the FR3 a clockwise turn (negative yaw) raises j7. Lower end: the
# same 1.6 rad either side of the 0.785 rad home wrist angle.
WRIST_Q7_RAD=(-.8,2.4)


def wrist_turn_range(j7,q7_range=WRIST_Q7_RAD):
    """Degrees (positive = counter-clockwise from above) a vertical-axis turn may still go from joint 7
    at j7: d(j7) = -d(yaw); a wrist already outside the range may only turn back towards it."""
    lo,hi=min(q7_range[0],j7),max(q7_range[1],j7)
    return math.degrees(j7-hi),math.degrees(j7-lo)


class ContinuousPrimitives(Primitives):
    # A threaded hole never lets the tip fall free (the thread keeps about 3 N of support), and the
    # simulation's "blocked" test needs the target to move > 8 mm in 1 s, which 4 mm/s can never do, so
    # the spiral could drag a screw that had sunk past drop back out. On the real robot sinking past drop
    # stops the search.
    search_sink_is_drop = True
    # A tip caught in a threaded hole mouth (about 1 mm deep) can be held sideways while the spiral target runs
    # on and drags it back out; the thread never lets the support drop, so nothing paused the spiral (flat
    # surfaces sink 0-0.6 mm). Slide lightly, but pause at a dip and press the suspected hole hard.
    search_settle = (.00025,.0008,2.,15.)
    # Once the tip dips, press it in at once with about 15 N (the 25 mm contact lead along the axis, driven at
    # 8 mm/s) and finish the 2 s press before judging: entering the hole is decided by that first press; a
    # later push does not help. After the thrust a hole is >= 1.5 mm below the slid surface (real hole entries
    # read 1.5-2.4 mm, while a 15 N thrust on a flat spot reads about 0.9 mm), whatever drop says.
    search_press = (CONTACT_LEAD_M,23.,.0015)          # stop raising the press at 23 N raw, back off above 24 N
    # A shallow dip (about 0.27 mm below the surface it slid on) must already pause the spiral. On the FR3 the
    # catching side force is friction in the drag direction, so the simulation's guided hops walked the spiral
    # away from the hole, and a 0.6 N "support" (30% of 2 N) was force drift in the air, which armed the search
    # above the board.
    search_guided_hops = 0
    # A held part's force estimate can rise by about 2 N on its own while descending in the air, so arming on a
    # force rise fired above the board; the board stopping the hand is what counts. The simulation's "fell
    # within 1 s while unsupported" rule fired when the hand rode over a 1 mm bump, so it is off here.
    search_arm_rise = None
    search_arm_blocked = (.002,.0002,.3)
    search_fall_rule = False
    screw_phase = None          # 'press' or 'turn' while a screw motion runs

    def __init__(self,backend,*,confirm=None,interface='force',empty_contact_check=False,stop_protocol='r7'):
        if stop_protocol not in ('r7','r9'):raise ValueError('Unknown stop protocol')
        self.stop_protocol=stop_protocol
        self.confirm = confirm or (lambda kind,state:False)
        super().__init__(backend,interface=interface,speed_free=.008,speed_near=.004,
                         kp_contact=150.,f_contact=FIND_CONTACT_N,f_cap=backend.config.f_cap,t_cap=backend.config.t_cap,
                         pos_tol=.003,slow_in_contact=True,extra_tools=('pull','turn','point_tool','screw','wiggle'))
        self.kp_free=150.  # do not inherit a previous gain-diagnostic setting
        self.target_quat = backend.eef_quat.copy()
        self.max_lead_free = self.max_lead_guard = .008
        self.max_lead_contact = .006
        self.grip_closed = backend._closed is True
        self._inside_probe=False
        self.empty_contact_check=empty_contact_check
        self._preserve_bias=False
        self.detector=RealEventDetector(lambda:backend.sim_time,f_contact=self.f_contact,
                                        f_cap=self.f_cap,t_cap=self.t_cap)

    def _level_and_rezero(self,max_steps=40):
        # The shared base class would turn the hand straight down and swallow tare failures here; on the real
        # robot, physical initialization is explicit and recorded by the entry point.
        return

    def _run(self,*args,**kwargs):
        max_time = kwargs.get('max_time',args[3] if len(args)>3 else 30.)
        if not math.isfinite(max_time) or not 0<max_time<=298:   # the robot side's 300 s envelope
            raise ValueError('Primitive duration must fit the 300-second adaptation envelope')
        max_steps=max(1,int(max_time/self.sim.dt))
        # Match the inherited primitive's step count, allowing real RPC time.
        # The two-second margin also covers its final two settling steps.
        nuc_budget=max_steps*.075+2.
        if nuc_budget>300.:
            raise ValueError('Step-based budget exceeds the 300-second limit')
        name,params = args[:2]
        if name in ('slide','push') and kwargs.get('soft_dir') is not None:
            args=(*args[:2],self._pressing(name,params,args[2],kwargs['soft_dir']),*args[3:])
        collision_guard=name in ('move_to','rotate_z','point_tool','turn','lift','hold')
        threshold=FREE_COLLISION_N if collision_guard else float(params.get('f_threshold',self.f_contact))
        self.detector.f_contact=threshold
        self.detector.f_release=threshold*.5
        if collision_guard:
            # These geometry primitives only record contact in the shared library;
            # on the real arm an accidental collision must actually stop them.
            if len(args)>4:
                args=(*args[:4],True,*args[5:])
            else:
                kwargs['stop_on_contact']=True
            kwargs['suppress_contact']=False
        # The robot side rejects a DESIRED pose beyond 120 mm from the motion start, and the
        # desired point leads the hand by up to 8 mm, so the workstation reach must stay
        # below that: 105 mm (+8 mm lead +margin < 120 mm). Speed/lead unchanged.
        for key in ('max_dist','distance','depth','height','radius_max','drop','hop'):
            if key in params and not 0<float(params[key])<=.105:
                raise ValueError(key+' must be positive and <= 105 mm (r30e reach)')
        if name=='move_to':
            goal = np.array([params[k] for k in ('x','y','z')])
            delta = goal-self.sim.eef_pos
            if not np.isfinite(goal).all() or np.linalg.norm(delta)>.105:
                raise ValueError('move_to/move_by must fit a 105 mm primitive')
            check_flange_workspace(np.array(self.sim._state['flange_pos'])+delta)
        self.sim.refresh()
        if not self.sim.config.motion_enabled:
            raise BackendError('Motion disabled')
        if (not self._preserve_bias and self.sim._closed is False and self.sim.gripper_opening>=.078
            and np.max(np.abs(self.sim._state['velocities']))<=.03):
            self.sim.rezero_bias(steps=10,unloaded=True)
        # Without a sign calibration, empty free-position moves may use a freshly
        # tared force NORM only. Force-directed/contact primitives and held
        # payloads still require the measured calibration.
        unsigned=name in ('move_to','hold','rotate_z','point_tool','turn') and self.sim._closed is False
        # The empty contact check may run before sign calibration. Only empty,
        # vertical downward contact finding may use its unsigned force norm.
        # It cannot authorize force tracking, payloads or lateral contact.
        if self.empty_contact_check and name=='move_until_contact':
            if (self.sim._closed is not False or self.sim.gripper_opening<.078
                or not np.allclose(params['direction'],[0,0,-1],atol=1e-9)
                or threshold!=5. or self.f_cap>10.):
                raise BackendError('Empty contact check requires downward 5 N finding with a <=10 N cap')
            unsigned=True
        initial_kp = kwargs.get('kp_vec')
        if initial_kp is None:
            initial_kp = self.kp_contact if kwargs.get('compliant') else self.kp_free
            soft_dir = kwargs.get('soft_dir')
            if kwargs.get('compliant') and soft_dir is not None:
                u = np.abs(np.asarray(soft_dir)/np.linalg.norm(soft_dir))
                initial_kp = np.r_[self.kp_free-(self.kp_free-self.kp_contact)*u,[self.kp_free]*3]
        force_operation='probe' if self._inside_probe and name=='move_until_contact' else name
        force_direction=kwargs.get('soft_dir',params.get('direction'))
        with self.sim.force_calibration_scope(force_operation,force_direction), self.sim.position_force_guard(enabled=unsigned):
            self.sim.reaction_force(self.sim.wrench())
            scoped_force=self.sim.scoped_force_valid()
            with self.sim.fixed_gains(initial_kp):
                with self.sim.motion(nuc_budget):
                    motion_start=self.sim.eef_pos.copy()
                    try:
                        report = super()._run(*args,**kwargs)
                    finally:
                        self.sim.step_lead=None
                    if self.sim.keep_controller:
                        self.sim.motion_result(report.termination,normal_contact=name=='move_until_contact')
                    before_release=self.sim.eef_pos.copy()
                    before_quat=self.sim.eef_quat.copy()
        stop_displacement=float(np.linalg.norm(self.sim.eef_pos-before_release))
        stop_tol=.005
        stop_check=None
        if self.stop_protocol=='r9':
            from forceharness.real.stop_validation import classify_stop
            direction=(np.array([params[k] for k in ('x','y','z')])-motion_start
                       if name=='move_to' else np.array(params.get('direction',before_release-motion_start)))
            stop_check=classify_stop([self.sim.eef_pos-before_release],direction,normal_soft_stop=True)
            stop_tol=stop_check['rebound_tolerance_mm']/1000
        report.derived.update(stop_displacement_m=stop_displacement,stop_tolerance_m=stop_tol,
                              active_end_pos=before_release.tolist(),active_end_quat=before_quat.tolist(),
                              stop_stable=(stop_check['rebound_passed'] if stop_check else stop_displacement<=stop_tol),
                              contact_threshold_N=threshold,
                              collision_threshold_N=FREE_COLLISION_N,
                              find_contact_threshold_N=FIND_CONTACT_N,
                              contact_samples=3,contact_grace_s=.3,
                              contact_sample=self.detector.contact_sample,
                              planned_max_steps=max_steps,nuc_budget_s=nuc_budget,
                              budget_per_step_s=.075,budget_margin_s=2.)
        if not report.derived['stop_stable']:
            report.termination='stop_drift'
        if stop_check:
            report.derived['stop_check']=stop_check
            if stop_check['safety_event']:report.termination='unexpected_motion'
        if not self.sim.calibration_valid():
            report.quantities=[]
            report.derived.update(force_sign_calibrated=False,
                scoped_force_sign_calibrated=scoped_force,
                force_use='K21 empty downward contact only' if scoped_force else 'locally tared norm guard only')
        report.truth = dict(contact_available=False,contact_fraction=None,touching_end=None,
                            peak_object_contact_force=None,grip_closed=self.grip_closed)
        report.derived.update(backend='continuous_real',force_guard='NUC nominal 50 Hz; not hard real time',
                              controller='installed HybridJointImpedanceControl',
                              controller_watchdog=False,gains_fixed_per_primitive=True,
                              wrench_reference='base axes, TCP origin; raw limits use flange wrench',
                              torque_convention=('negative reaction torque, original primitive convention'
                                  if self.sim.calibration.get('torque_verified') else 'uncertified raw torque sign'))
        if self.sim.keep_controller:
            report.derived['episode']=self.sim.episode_report()
        return report

    def _estimate(self,rep):
        if not self.sim.calibration_valid():
            rep.quantities=[]
            return
        super()._estimate(rep)
        # Torque/COM and pose-scope calibration must be independently validated.
        if not self.sim.calibration.get('torque_verified',False):
            rep.quantities = [q for q in rep.quantities if q['name']!='com_offset']

    def move_to(self,x,y,z,stop_on_contact=None,speed=None,max_time=8.):
        goal=np.array([x,y,z],dtype=float)
        start=self.sim.eef_pos.copy()
        v=self.speed_free if speed is None else float(speed)
        if not 0<v<=.008:
            raise ValueError('Real speed must be 0 < speed <= .008 m/s')
        distance=float(np.linalg.norm(goal-start))
        max_time=max(max_time,2.*distance/v+3.)
        target=StallCompensatedTarget(goal,start)
        contact_seen=[False]
        last_time=[None]
        travel=[0.]
        def ctrl(s):
            contact=bool(s['in_contact'])
            defaults=bool(np.allclose(self.sim.kp_vec,150.) and not self._inside_probe)
            if not defaults:
                target.bias[:]=0.
            # Held-load sampling can precede the first controller call. It is
            # preparation, not a compensation interval; later real intervals
            # still pass through the sample/transport guards (robot side 150 ms, step round trip 250 ms).
            dt=self.sim.dt if last_time[0] is None else s['t']-last_time[0]
            last_time[0]=s['t']
            window=min(dt,STEP_WINDOW_S)
            travel[0]+=v*window
            nominal=start+clipped(goal-start,travel[0])
            wanted=target.update(s['pos'],nominal,dt,
                                 enabled=defaults and not contact and np.linalg.norm(nominal-goal)<1e-6,
                                 contact=contact)
            if contact:
                contact_seen[0]=True
                return s['pos']-s['target'],0.,True
            vv=min(v,self.speed_near) if s['in_contact'] else v
            delta=clipped(wanted-s['target'],vv*window)
            tolerance=target.tolerance if defaults else self.pos_tol
            # A held load (for example a 0.42 kg bottle) hangs 6-7 mm below its target, so retracting the
            # compensation at the goal drops the hand out of tolerance and the target dithers until the robot-side
            # stall guard ends the episode. Holding something, reaching the goal is enough; the stop releases the target.
            done=bool((np.linalg.norm(goal-s['pos'])<=tolerance and (target.retracted or self.sim._closed is True))
                      or target.stalled_short)
            return delta,0.,done
        report=self._run('move_to',dict(x=x,y=y,z=z),ctrl,max_time,
                         self._stop_on_contact(stop_on_contact))
        if report.termination=='contact':
            contact_seen[0]=True
            target.bias[:]=0.
        if contact_seen[0] and report.termination=='completed':
            report.termination='contact'
        if target.stalled_short and report.termination=='completed':
            report.termination='stalled_short'
        report.derived.update(position_tolerance_m=self.pos_tol,
                              compensation_tolerance_m=target.tolerance,
                              compensation_contact_stop=contact_seen[0],
                              goal_error_after_stop_m=float(np.linalg.norm(goal-self.sim.eef_pos)),
                              compensation_peak_m=target.peak_bias_m)
        return report

    last_support = None      # where the tool last found support, for the next search or wiggle

    def move_until_contact(self,direction,max_dist=.02,f_threshold=None,speed=None):
        if speed is not None and not 0<float(speed)<=.004:
            raise ValueError('Finding contact requires speed <= 4 mm/s')
        rep=super().move_until_contact(direction,max_dist,f_threshold,speed)
        if rep.termination=='contact':
            d=np.asarray(direction,dtype=float)
            self.last_support=dict(pos=np.array(rep.end_pos,dtype=float),normal=d/np.linalg.norm(d))
        return rep

    def probe(self,direction=(0,0,-1),amplitude=.02,f_threshold=None):
        self._inside_probe=True
        try:
            start=self.sim.eef_pos.copy()
            rep=self.move_until_contact(direction,amplitude,f_threshold)
            outbound=rep.to_dict()
            rep.primitive='probe'
            rep.params=dict(direction=list(direction),amplitude=amplitude)
            rep.derived.update(outbound_report=outbound,contact_found=rep.termination=='contact',
                               contact_pos=(rep.derived['contact_sample'] or {}).get('pos'))
            if rep.termination not in ('contact','no_contact_within_max_dist'):
                return rep  # No automatic recovery after a guard or failed stop.
            # The retreat is free motion, with the same compensation and
            # 8 N collision guard as move_to. Keep the pre-contact force zero.
            self._inside_probe=False
            with self.preserve_contact_bias():
                back=self.move_to(*start,stop_on_contact=True,max_time=4.)
            error=float(np.linalg.norm(np.array(back.derived['active_end_pos'])-start))
            rep.derived.update(return_report=back.to_dict(),return_error_m=error,
                               return_error_basis='before controller stop',
                               return_error_after_stop_m=float(np.linalg.norm(self.sim.eef_pos-start)),
                               return_passed=back.termination=='completed' and error<=.003
                                             and back.derived['stop_stable'])
            rep.end_pos=self.sim.eef_pos.tolist()
            rep.t_end=back.t_end
            rep.actual_disp=(self.sim.eef_pos-start).tolist()
            if not rep.derived['return_passed']:
                rep.termination='probe_return_failed'
            self._estimate(rep)
            return rep
        finally:
            self._inside_probe=False

    @contextmanager
    def preserve_contact_bias(self):
        """Retreat with the pre-contact zero; never tare an applied load away."""
        previous=self._preserve_bias
        self._preserve_bias=True
        try:
            yield
        finally:
            self._preserve_bias=previous

    def _pressing(self,name,params,controller,press_dir):
        """Wrap a slide/push controller to use the contact lead; soft_dir is the press direction (slide normal, push axis)."""
        n=np.asarray(press_dir,dtype=float)/np.linalg.norm(press_dir)
        u=None
        if name=='slide':
            u=np.asarray(params['direction'],dtype=float)
            u=u-(u@n)*n
            u=u/np.linalg.norm(u)

        def ctrl(s):
            dtarget,dyaw,done=controller(s)
            pressing=not done and float(np.dot(s['reaction'],-n))>=self.f_contact
            self.sim.step_lead=CONTACT_LEAD_M if pressing else None
            if u is None or not pressing:
                return dtarget,dyaw,done
            # Re-split the target's lead (relative to the hand): press <= 80%, the pull gets the rest.
            # Not before contact: the press must first reach the threshold that unlocks the lead.
            lead=CONTACT_LEAD_M
            pos=np.asarray(s['pos'],dtype=float)
            rel=np.asarray(s['target'],dtype=float)+np.asarray(dtarget,dtype=float)-pos
            rn,rt=float(rel@n),float(rel@u)
            other=rel-rn*n-rt*u
            rn=min(rn,SLIDE_PRESS_SHARE*lead)
            rt=min(rt,math.sqrt(max(0.,lead**2-rn**2-float(other@other))))
            return pos+rn*n+rt*u+other-np.asarray(s['target'],dtype=float),dyaw,done
        return ctrl

    def lift(self,height=.02,hold_time=1.):
        return super().lift(height,hold_time)

    def slide(self,direction,distance,normal_force=1.,normal=(0,0,-1),speed=None):
        return super().slide(direction,distance,normal_force,normal,speed)

    def pull(self,direction,force=1.,distance=.02,speed=.004,max_time=None):
        u=np.asarray(direction,dtype=float)
        if u.shape!=(3,) or not np.isfinite(u).all() or np.linalg.norm(u)<1e-9:
            raise ValueError('Pull direction requires three finite, nonzero values')
        u=u/np.linalg.norm(u)
        speed=.004 if speed is None else float(speed)
        if not math.isfinite(speed) or not 0<speed<=.004:
            raise ValueError('Pull speed must be 0 < speed <= .004 m/s')
        if abs(u[2])<1e-9:
            # Keep the existing horizontal door/constraint-following primitive.
            return super().pull(u,force,distance,speed,max_time)
        return self._linear_pull(u,force,distance,speed,max_time)

    def _linear_pull(self,u,force,distance,speed,max_time):
        """Straight 3-D pull for vertical/inclined drawers, fixed wrist.

        Regulate the measured reaction along the requested axis. A mechanism's
        existing resistance is not a carried payload to tare away. Distance
        limits measured TCP travel; the desired point may lead it by the
        existing contact lead allowance, also enforced by _run at every step.
        """
        force,distance=float(force),float(distance)
        if not math.isfinite(force) or not 0<force<=self.f_cap*.8:
            raise ValueError('Pull force must be positive and <= 80% of the force cap')
        if not math.isfinite(distance) or not 0<distance<=.105:
            raise ValueError('Pull distance must be positive and <= 105 mm')
        if max_time is None:
            max_time=distance/speed*2.5+3.
        start=self.sim.eef_pos.copy()
        target_distance=distance+self.max_lead_contact
        check_flange_workspace(np.asarray(self.sim._state['flange_pos'])+u*target_distance)
        self.target_quat=self.sim.eef_quat.copy()
        state=dict(history=[],stop=None,target_peak=0.,lead_peak=0.)

        def ctrl(s):
            along=float((s['pos']-start)@u)
            lateral=s['pos']-start-u*along
            resistance=-float(s['reaction_raw']@u)
            requested=float((s['target']-start)@u)
            state['target_peak']=max(state['target_peak'],requested)
            state['lead_peak']=max(state['lead_peak'],float(np.linalg.norm(s['target']-s['pos'])))
            history=state['history']
            history.append((s['t'],along,resistance))
            history[:]=[r for r in history if r[0]>=s['t']-1.]
            if np.linalg.norm(lateral)>self.pos_tol:
                state['stop']='path_deviation'
            elif along>=distance-.0002:
                # Unload the target lead once actual travel is reached. Keeping
                # the advanced target during settling would continue pulling.
                return s['pos']-s['target'],0.,True
            elif (len(history)>=10 and history[-1][0]-history[0][0]>=.9
                  and max(r[1] for r in history)-min(r[1] for r in history)<.0002
                  and np.mean([r[2] for r in history])>=.95*force):
                state['stop']='stalled'
            if state['stop']:
                return s['pos']-s['target'],0.,True
            v=float(np.clip(self.kf*3.*(force-resistance),-speed,speed))
            wanted=start+u*float(np.clip(requested+v*self.sim.dt,0.,target_distance))
            return clipped(wanted-s['target'],speed*self.sim.dt),0.,False

        def derive(rep):
            rows=rep.wrench_array()
            tail=rows[len(rows)//5:,1:4]
            forces=-(tail@u)
            pos=np.asarray(rep.pos_rows,dtype=float).reshape(-1,3)
            delta=pos-start
            return dict(pull_mode='straight_3d',resistance_tared=False,
                        pull_distance_reference='measured_tcp',
                        pull_target_travel_max_m=state['target_peak'],
                        pull_target_lead_max_m=state['lead_peak'],
                        pull_target_lead_limit_m=self.max_lead_contact,
                        pull_force_mean_N=round(float(np.mean(forces)),3),
                        pull_force_std_N=round(float(np.std(forces)),3),
                        travel_along_direction_m=round(float((np.asarray(rep.end_pos)-start)@u),5),
                        lateral_drift_max_m=float(np.max(np.linalg.norm(delta-np.outer(delta@u,u),axis=1))),
                        stalled=state['stop']=='stalled')

        previous_payload,previous_bias=self.payload_max,self.carry_bias.copy()
        self.payload_max=0.  # Disable inherited automatic payload tare for this mechanism probe.
        self.carry_bias=np.zeros(3)
        try:
            rep=self._run('pull',dict(direction=u.tolist(),force=force,distance=distance,speed=speed),
                          ctrl,max_time,stop_on_contact=False,derive=derive,
                          max_lead=self.max_lead_contact,grip=1. if self.grip_closed else None,
                          suppress_contact=self.grip_closed)
            if state['stop'] and rep.termination=='completed':
                rep.termination=state['stop']
            self._estimate(rep)
            return rep
        finally:
            self.payload_max,self.carry_bias=previous_payload,previous_bias

    def search(self,normal=(0,0,-1),radius_max=None,pitch=None,normal_force=None,
               drop=None,speed=None,mode='slide',hop=None,f_catch=6.,guide_step=.003,guess=None):
        # Unset arguments take the defaults of the mode (SEARCH_DEFAULTS); hop may start at a guessed
        # offset of the hole from the hand.
        d=SEARCH_DEFAULTS['hop' if mode=='hop' else 'slide']
        radius_max,pitch,drop,hop=[d[k] if v is None else v for k,v in
                                   (('radius_max',radius_max),('pitch',pitch),('drop',drop),('hop',hop))]
        normal_force=d['normal_force'] if normal_force is None else normal_force
        if guess is not None and mode!='hop':
            raise ValueError('guess works with mode="hop" only')
        if mode=='hop':              # the probe presses at normal_force, at least HOP_PROBE_N
            nrm=np.asarray(normal,dtype=float)/np.linalg.norm(normal)
            rep=self._search_hop(nrm,radius_max,pitch,drop,speed,hop,
                                 probe_n=float(min(max(normal_force or 0.,HOP_PROBE_N),self.f_cap*.8)),guess=guess)
            self._estimate(rep)
            return rep
        rep=super().search(normal,radius_max,pitch,normal_force,drop,speed,mode,hop,f_catch,guide_step)
        adv=rep.derived.get('advance_along_normal_m')
        if adv is not None and rep.termination!='no_surface':   # the surface it slid on, for wiggle's depth
            nrm=np.asarray(normal,dtype=float)/np.linalg.norm(normal)
            self.last_support=dict(pos=np.array(rep.end_pos,dtype=float)-adv*nrm,normal=nrm)
        return rep

    def _search_hop(self,nrm,radius_max,pitch,drop,speed,hop,probe_n=HOP_PROBE_N,guess=None):
        """Hop search: on a slippery surface a slide search drags the tip sideways and tilts a screw held by
        two pads on a short knob, so this mode probes instead: lift `hop` above the start surface, move to
        the next spiral point in the air, zero the force there, then probe down along `nrm` until the
        reaction rises HOP_PROBE_N over that zero. The simulation's hop waits for the hand within 1.5 mm of
        each point; on the FR3 joint friction leaves it 2-3 mm short, and the session tare drifts 2-4 N while
        holding. Here a stall-compensated target pushes the hand on, a point not reached in HOP_MOVE_MAX_S
        is probed where the hand is, and a probe deeper than the typical (median) contact so far by `drop`
        stops at once as dropped."""
        a=np.array([1.,0,0]) if abs(nrm[0])<.9 else np.array([0,1.,0])
        e1=a-np.dot(a,nrm)*nrm
        e1/=np.linalg.norm(e1)
        e2=np.cross(nrm,e1)
        v_n=self.speed_near
        press_lead=min(probe_n/SCREW_KZ_N_PER_M,.008)
        p0=np.array(self.sim.eef_pos,dtype=float).copy()
        pts=[np.zeros(3)]
        theta=r=0.
        while r<radius_max:
            theta+=pitch/max(pitch*theta/(2*math.pi),pitch/2)
            r=pitch*theta/(2*math.pi)
            pts.append(r*(math.cos(theta)*e1+math.sin(theta)*e2))
        g=np.zeros(3)
        if guess is not None:        # probe the start (the surface reference) first, then spiral
            g=np.r_[np.asarray(guess,dtype=float)[:2],0.]      # out from the guessed hole position
            g-=np.dot(g,nrm)*nrm
            pts=[np.zeros(3)]+[g+pt for pt in pts]
        if len(pts)*HOP_POINT_MIN_S>HOP_MAX_TIME_S:
            raise ValueError('hop search has %d points (about %.0f s at best); use a larger pitch or a smaller '
                             'radius_max' % (len(pts),len(pts)*HOP_POINT_MIN_S))
        prior=None                   # depth of the last nearby support contact along nrm, from p0
        sup=self.last_support
        if sup is not None and float(np.dot(sup['normal'],nrm))>.95:
            off=sup['pos']-p0
            if np.linalg.norm(off-np.dot(off,nrm)*nrm)<=HOP_PRIOR_NEAR_M:
                prior=float(np.dot(off,nrm))
        st=dict(i=0,phase='up',comp=None,t_phase=0.,zero=[],base=None,depths=[],dropped=False,no_surface=False,
                late=0,drop_depth=None,move_s=HOP_MOVE_MAX_S,probes=[],stuck=False)

        def surface():
            return float(np.median(st['depths'])) if st['depths'] else prior

        def top():                   # hops rise `hop` above the shallowest probe so far (the start before any);
            return min(st['depths']) if st['depths'] else 0.   # the prior only judges, a table beside may be lower

        def hold(s):
            return np.asarray(s['pos'])-np.asarray(s['target'])

        def ctrl(s):
            if st['i']>=len(pts):
                return np.zeros(3),0.,True
            adv=float(np.dot(s['pos']-p0,nrm))
            f_n=float(np.dot(s['reaction'],-nrm))
            ph=st['phase']
            if ph=='up':                 # in the air, `hop` above the support, over the next point
                goal=p0+pts[st['i']]+nrm*(top()-hop)
                if st['comp'] is None:
                    st['comp'],st['t_phase']=StallCompensatedTarget(goal,s['pos']),s['t']
                    st['move_s']=HOP_MOVE_MAX_S+float(np.linalg.norm(goal-s['pos']))/HOP_MOVE_SPEED_M_S
                err=goal-s['pos']
                lat=err-np.dot(err,nrm)*nrm
                if (np.linalg.norm(lat)<.001 and abs(float(np.dot(err,nrm)))<.001) or s['t']-st['t_phase']>st['move_s']:
                    st['late']+=int(np.linalg.norm(lat)>=.001)
                    st['unreached']=bool(np.linalg.norm(lat)>=.001)
                    st.update(phase='zero',t_phase=s['t'],zero=[],comp=None)
                    return hold(s),0.,False
                return st['comp'].update(s['pos'],goal,self.sim.dt)-s['target'],0.,False
            if ph=='zero':               # still in free air: this probe's own force zero
                st['zero'].append(f_n)
                if s['t']-st['t_phase']>=HOP_ZERO_S:
                    st.update(base=float(np.mean(st['zero'])),phase='down',t_phase=s['t'])
                return hold(s),0.,False
            ref=surface()                # down: probe along the normal
            margin=drop if st['depths'] else max(drop,HOP_PRIOR_MARGIN_M)
            if ref is not None and adv-ref>margin:     # deeper than the surface so far: in a hole, stop now
                st.update(dropped=True,drop_depth=round(adv-ref,4))
                return hold(s),0.,True
            if ph=='press':              # keep pressing at probe_n, following the tip down, sideways held
                if s['t']-st['t_phase']>=HOP_PRESS_S:
                    st['depths'].append(adv)
                    st['probes'].append((st.get('unreached',False),np.asarray(s['pos'],dtype=float).copy()))
                    last=st['probes'][-HOP_STUCK_POINTS:]
                    if len(last)==HOP_STUCK_POINTS and all(u for u,_ in last):
                        at=np.array([q for _,q in last])
                        if np.ptp(at,axis=0).max()<=HOP_STUCK_M:   # held in one place: caught (in a hole?)
                            st['stuck']=True
                            return hold(s),0.,True
                    st.update(i=st['i']+1,phase='up')
                    return hold(s),0.,False
                at=st['press_at']        # sideways: where it touched (a pressed screw tips over if let free)
                want=at+nrm*(float(np.dot(np.asarray(s['pos'])-at,nrm))+press_lead)
                d=want-np.asarray(s['target'])
                dn=float(np.clip(np.dot(d,nrm),-self.speed_near*self.sim.dt,self.speed_near*self.sim.dt))
                return d-np.dot(d,nrm)*nrm+nrm*dn,0.,False
            if f_n-st['base']>=probe_n:
                st.update(phase='press',t_phase=s['t'],press_at=np.asarray(s['pos'],dtype=float).copy())
                return np.zeros(3),0.,False
            if adv>(ref if ref is not None else 0.)+HOP_SURFACE_MAX_M:
                st['no_surface']=True
                return hold(s),0.,True
            return nrm*v_n*self.sim.dt,0.,False

        def derive(rep):
            adv=float(np.dot(np.array(rep.end_pos)-p0,nrm))
            return dict(dropped=st['dropped'],probes=len(st['depths']),advance_along_normal_m=round(adv,4),
                        surface_m=None if surface() is None else round(surface(),4),drop_depth_m=st['drop_depth'],
                        points_not_reached=st['late'],probe_N=round(probe_n,1),
                        prior_support_depth_m=None if prior is None else round(prior,4),
                        held_in_place=st['stuck'],
                        guess_m=None if guess is None else np.round(g,4).tolist(),
                        found_from_guess_m=(np.round((np.array(rep.end_pos)-p0)-adv*nrm-g,4).tolist()
                                            if guess is not None and st['dropped'] else None),
                        radius_searched_m=round(float(np.linalg.norm(pts[min(st['i'],len(pts)-1)])),4),
                        lateral_offset_at_end_m=np.round((np.array(rep.end_pos)-p0)-adv*nrm,4).tolist())

        rep=self._run('search',dict(normal=nrm.round(3).tolist(),radius_max=radius_max,pitch=pitch,drop=drop,mode='hop',
                                    hop=hop),ctrl,min(HOP_MAX_TIME_S,len(pts)*HOP_POINT_TYPICAL_S+10.),
                      stop_on_contact=False,derive=derive,compliant=True,soft_dir=nrm,max_lead=.008)
        if st['dropped']:
            rep.termination='dropped'
        elif st['no_surface']:
            rep.termination='no_surface'
        elif st['stuck']:
            rep.termination='stuck'
        return rep

    def grasp(self,steps=16):
        self.sim.refresh()
        self.sim.reaction_force(self.sim.wrench())
        if self.sim.calibration.get('gripper_verified') is not True:
            raise BackendError('Closed-width gripper bias calibration required before grasp')
        if type(steps) is not int or not 1<=steps<=200:
            raise ValueError('steps must be 1..200')
        if not self.confirm('grasp',self.sim._state):
            raise BackendError('Grasp alignment not confirmed')
        self.sim.command_gripper(True,confirmed=True)
        self.last_support=None       # the tool tip changes with what the hand holds
        return super().grasp(steps)

    def release(self,steps=16):
        self.sim.refresh()
        self.sim.reaction_force(self.sim.wrench())
        if type(steps) is not int or not 1<=steps<=200:
            raise ValueError('steps must be 1..200')
        if self.sim._closed is not True or not self.confirm('release',self.sim._state):
            raise BackendError('Known holding and current supported placement required')
        self.sim.command_gripper(False,confirmed=True)
        self.last_support=None
        return super().release(steps)

    def _wrist_turn_range(self):
        j7=getattr(self.sim,'wrist_joint_rad',None)
        return None if j7 is None else wrist_turn_range(float(j7))

    def rotate_z(self,angle_deg,torque_cap=None,stop_on_resistance_drop=False,rate_deg_s=8.):
        if abs(float(angle_deg))>math.degrees(.55):   # the robot side allows 0.60 rad per motion
            raise ValueError('Initial real rotation envelope is at most 0.09 rad per primitive')
        if not 0<rate_deg_s<=math.degrees(ROT_RATE_MAX_RAD_S):
            raise ValueError('Rotation rate must be positive and <= %.2f rad/s' % ROT_RATE_MAX_RAD_S)
        return super().rotate_z(angle_deg,torque_cap,stop_on_resistance_drop,rate_deg_s)

    def _raw_hold(self,s,st,cap):
        # Watch every reading a stop watches (untared, tared and the robot side's raw; they can differ by
        # several N), stop HOLD_AHEAD_S ahead of the rise, and once held push again only after the reading
        # fell HOLD_RELEASE_N (a flat reading would let the push run blind for another 0.25 s)
        now=max(float(np.linalg.norm(s['reaction_raw'])),float(np.linalg.norm(s['reaction'])),
                float(np.linalg.norm(self.sim._state['wrench'][:3])))
        hist=st.setdefault('hold_hist',[])
        hist.append((s['t'],now))
        while hist[0][0]<s['t']-.2:
            hist.pop(0)
        t0,f0=hist[0]
        rise=(now-f0)/(s['t']-t0) if s['t']-t0>1e-6 else 0.
        if now+max(0.,rise)*HOLD_AHEAD_S>=cap:
            st['hold_latched']=True
        elif now<cap-HOLD_RELEASE_N:
            st['hold_latched']=False
        return st.get('hold_latched',False),now

    def screw(self,angle_deg,push=SCREW_PUSH_N,rate_deg_s=5.,quick=False):
        """Press about `push` N down the vertical axis first, then turn about it like rotate_z
        (negative = clockwise from above = tightens a right-hand thread) while still pressing: the target
        stays the press lead below the hand, down as the thread pulls, up if the screw rises. Sideways it
        holds while pressing and is soft while turning. While turning the push ramps from SCREW_TURN_START_N
        to push; quick=True (later segments of one call) presses only to the turn start, briefly. Stops after
        SCREW_SINK_MAX_M of sinking."""
        if not math.isfinite(angle_deg) or abs(float(angle_deg))>math.degrees(.55):   # as rotate_z
            raise ValueError('screw turns at most 0.55 rad per motion')
        if not 0<rate_deg_s<=math.degrees(ROT_RATE_MAX_RAD_S):
            raise ValueError('Rotation rate must be positive and <= %.2f rad/s' % ROT_RATE_MAX_RAD_S)
        if not 0<push<=SCREW_PUSH_MAX_N:
            raise ValueError(f'push must be in (0, {SCREW_PUSH_MAX_N:g}] N')
        requested=angle_deg=float(angle_deg)
        clipped=False
        turn_range=self._wrist_turn_range()
        if turn_range is not None:               # the wrist range, as rotate_z
            lo,hi=turn_range
            if not lo<=angle_deg<=hi:
                angle_deg,clipped=float(np.clip(angle_deg,lo,hi)),True
        lead=min(push/SCREW_KZ_N_PER_M,CONTACT_LEAD_M)
        # A full press turns at the full push; a quick (later) segment starts at SCREW_TURN_START_N
        start_n=min(push,SCREW_TURN_START_N) if quick else push
        turn_lead=min(lead,start_n/SCREW_KZ_N_PER_M)
        press_lead=turn_lead
        steps=max(1,int(math.ceil(abs(math.radians(angle_deg))/math.radians(rate_deg_s)/SCREW_RAMP_STEP_S)))
        free_lead=float(self.sim.port.limits.get('lead',.008))
        down=np.array([0.,0.,-1.])
        p0=self.sim.eef_pos.copy()
        yaw0=_yaw_deg_xyzw(self.sim.eef_quat)
        sgn=1. if angle_deg>=0 else -1.
        total,rate=abs(math.radians(angle_deg)),math.radians(rate_deg_s)
        acc=[0.]
        state=dict(stop=None,full_at=None,turn_t=None,turn_z=None,yaw_waits=0,zs=[],seated=False,capped=False,
                   turn_capped=False,turn_push=start_n)
        self.screw_phase='press'

        def ctrl(s):
            if float((s['pos']-p0)@down)>SCREW_SINK_MAX_M:
                state['stop']='sink_limit'
                return np.zeros(3),0.,True
            # the contact lead only while pressing on something (as slide/push)
            pressing=float(np.dot(s['reaction'],-down))>=self.f_contact
            turning=state['turn_t'] is not None
            if turning:                  # start_n -> push in steps over the expected turn time
                k=int((s['t']-state['turn_t'])/SCREW_RAMP_STEP_S)+1     # a step as it starts turning
                state['turn_push']=start_n+(push-start_n)*min(1.,k/steps)
                want=min(state['turn_push']/SCREW_KZ_N_PER_M,CONTACT_LEAD_M)
            else:
                want=press_lead if pressing else min(press_lead,free_lead)
            # the robot side grants its contact lead only once the force rose, else it shrinks the lead at its speed
            self.sim.step_lead=CONTACT_LEAD_M if want>free_lead else None
            # along the axis: the target `want` below the hand, ramped at SCREW_PRESS_SPEED while pressing
            # first, then following the hand at most speed_near per second
            # pressing and the ramp steps go at the press speed; following the thread down at speed_near
            v=SCREW_PRESS_SPEED if (state['turn_t'] is None or want>float((s['target']-s['pos'])@down)+1e-4) else self.speed_near
            # easing the 15 mm press lead back to the turn lead goes at the press speed too: at
            # speed_near it took 3 s, which the runner's 3.5 s tracking-stall guard nearly reads as stuck
            ax=float(np.clip(float(s['pos']@down)+want-float(s['target']@down),
                             -SCREW_PRESS_SPEED*self.sim.dt,v*self.sim.dt))
            capped,now=self._raw_hold(s,state,SCREW_PRESS_RAW_N) if SCREW_PRESS_RAW_N is not None else (False,0.)
            if capped:                   # stop pushing further; back off only while above the back-off
                ax=min(ax,0.)            # level, so the target does not move up and down (robot-side stall check)
                if now>=SCREW_BACKOFF_RAW_N:
                    ax=-self.speed_near*self.sim.dt
                state['turn_capped' if turning else 'capped']=True
            # pressing holds the start sideways; turning follows 30% of the hand's drift
            side=(p0+SCREW_TURN_SIDE_FOLLOW*(s['pos']-p0) if turning else p0)-s['target']
            side-=(side@down)*down
            dpos=side+down*ax
            if state['turn_t'] is None:
                at_full=capped or float((s['target']-s['pos'])@down)+ax>=press_lead-5e-4
                state['full_at']=(state['full_at'] if state['full_at'] is not None else s['t']) if at_full else None
                # after the hold, turn only once the screw stopped sinking (or the press time is up)
                state['zs']=[(t,z) for t,z in state['zs'] if t>=s['t']-SCREW_SEAT_WINDOW_S]+[(s['t'],float(s['pos'][2]))]
                still=(s['t']-state['zs'][0][0]>=SCREW_SEAT_WINDOW_S-1e-9
                       and state['zs'][0][1]-state['zs'][-1][1]<SCREW_SEAT_STILL_M)
                held=state['full_at'] is not None and s['t']-state['full_at']>=(SCREW_QUICK_HOLD_S if quick
                                                                                  else SCREW_PRESS_HOLD_S)
                if not (held and (still or quick)) and s['t']<SCREW_PRESS_MAX_S:
                    return dpos,0.,False
                state['seated']=held and still
                state['turn_t'],state['turn_z']=s['t'],float(s['pos'][2])
                self.screw_phase='turn'
            turned=((s['yaw']-yaw0+180.)%360.)-180.
            if acc[0]>=total-1e-9:
                return dpos,0.,(abs(math.radians(turned)-sgn*total)<math.radians(1.5)
                                or s['t']>state['turn_t']+total/rate+.5)
            if acc[0]-sgn*math.radians(turned)>SCREW_YAW_LEAD_RAD:   # wait for the wrist
                state['yaw_waits']+=1
                return dpos,0.,False
            d=min(rate*self.sim.dt,total-acc[0])
            acc[0]+=d
            return dpos,sgn*d,False

        def derive(rep):
            z=float(self.sim.eef_pos[2])      # rep.end_pos is rounded to 0.1 mm, too coarse per turn
            turn_z=z if state['turn_z'] is None else state['turn_z']
            out=dict(rotated_deg=rep.yaw_deg,push_N=push,push_lead_m=round(lead,4),start_z_m=round(float(p0[2]),5),
                     end_z_m=round(z,5),sink_m=round(float(p0[2])-z,5),
                     press_s=None if state['turn_t'] is None else round(state['turn_t'],2),
                     press_sink_m=round(float(p0[2])-turn_z,5),turn_sink_m=round(turn_z-z,5),
                     turn_push_N=[round(start_n,1),round(state['turn_push'],1)],turn_force_capped=state['turn_capped'],
                     quick_press=quick,
                     hand_followed_sideways_m=round(float(np.linalg.norm((self.sim.eef_pos-p0)[:2])),4),
                     yaw_wait_steps=state['yaw_waits'],press_seated=state['seated'],press_force_capped=state['capped'])
            if clipped:
                out.update(wrist_limit_clipped=True,requested_deg=requested,executed_deg=angle_deg,
                           note='rotation clipped by the wrist joint limit; release, turn the hand back and grip again')
            return out
        try:
            rep=self._run('screw',dict(angle_deg=requested,push=push),ctrl,
                          SCREW_PRESS_MAX_S+SCREW_PRESS_HOLD_S+2*total/rate+1.5,stop_on_contact=False,
                          grip=1. if self.grip_closed else None,derive=derive,max_lead=CONTACT_LEAD_M)
        finally:
            self.screw_phase=None
        if state['stop']:
            rep.termination=state['stop']
        rep.quantities=[q for q in rep.quantities if q.get('name')!='insertion_depth']+screw_quantities([rep])
        return rep

    def wiggle(self,depth=WIGGLE_DEPTH_M,force=WIGGLE_FORCE_N,yaw_deg=WIGGLE_YAW_DEG,shift=WIGGLE_SHIFT_M,
               duration=WIGGLE_TIME_S):
        """Press straight down (WIGGLE_FORCE_START_N ramping to `force` N) and rock the held part
        +-yaw_deg about the vertical and round a `shift` circle sideways; once it is held sideways stop rocking
        and press until it stops sinking. seated = `depth` below the nearby support surface (else below the start);
        held but shallower = jammed; out of time = not_seated."""
        for name,value,hi in (('depth',depth,WIGGLE_DEPTH_MAX_M),('force',force,WIGGLE_FORCE_MAX_N),
                              ('duration',duration,WIGGLE_TIME_MAX_S)):
            if not math.isfinite(value) or not 0<value<=hi:
                raise ValueError(f'{name} must be in (0, {hi:g}]')
        for name,value,hi in (('yaw_deg',yaw_deg,WIGGLE_YAW_MAX_DEG),('shift',shift,WIGGLE_SHIFT_MAX_M)):
            if not math.isfinite(value) or not 0<=value<=hi:
                raise ValueError(f'{name} must be in [0, {hi:g}]')
        amp=float(yaw_deg)
        turn_range=self._wrist_turn_range()
        if turn_range is not None:               # stay inside the wrist range
            amp=max(0.,min(amp,turn_range[1],-turn_range[0]))
        amp=math.radians(amp)
        rate=.9*ROT_RATE_MAX_RAD_S
        period=max(WIGGLE_YAW_PERIOD_S,2*math.pi*amp/rate)     # a sine's peak rate is 2 pi A / T
        start_n=min(WIGGLE_FORCE_START_N,force)
        free_lead=float(self.sim.port.limits.get('lead',.008))
        down=np.array([0.,0.,-1.])
        p0=self.sim.eef_pos.copy()
        yaw0=_yaw_deg_xyzw(self.sim.eef_quat)
        surface=None                             # depth along `down` of the support surface from p0 (<= 0: above)
        sup=self.last_support
        if sup is not None and float(np.dot(sup['normal'],down))>.95:
            off=sup['pos']-p0
            if np.linalg.norm(off-np.dot(off,down)*down)<=HOP_PRIOR_NEAR_M:
                surface=float(np.dot(off,down))
        state=dict(stop=None,phase='rock',cmd=0.,adv=0.,capped=False,yaw_waits=0,held_at=None,hist=[],zs=[],
                   frozen=None,seated_at=None,yaw_at=None,side_at=None)

        def below(adv):                          # insertion depth below the surface (or the start)
            return adv-(surface if surface is not None else 0.)

        def ctrl(s):
            adv=float((s['pos']-p0)@down)
            state['adv']=max(state['adv'],adv)
            turned=math.radians(((s['yaw']-yaw0+180.)%360.)-180.)
            if s['t']>=duration:
                state['stop']='seated' if below(adv)>=depth else 'not_seated'
                return np.zeros(3),0.,True
            # along the axis as screw: the target `want` below the hand, the contact lead only once pressing
            push=start_n+(force-start_n)*min(1.,s['t']/WIGGLE_FORCE_RAMP_S)
            lead=min(push/SCREW_KZ_N_PER_M,CONTACT_LEAD_M)
            pressing=float(np.dot(s['reaction'],-down))>=self.f_contact
            want=lead if pressing else min(lead,free_lead)
            self.sim.step_lead=CONTACT_LEAD_M if want>free_lead else None
            ax=float(np.clip(float(s['pos']@down)+want-float(s['target']@down),
                             -SCREW_PRESS_SPEED*self.sim.dt,SCREW_PRESS_SPEED*self.sim.dt))
            capped,now=self._raw_hold(s,state,SCREW_PRESS_RAW_N)
            if capped:
                ax=min(ax,0.)
                if now>=SCREW_BACKOFF_RAW_N:
                    ax=-self.speed_near*self.sim.dt
                state['capped']=True
            ramp=min(1.,s['t']/WIGGLE_RAMP_S)
            w=2*math.pi*s['t']/WIGGLE_SHIFT_PERIOD_S
            circle=p0+shift*ramp*np.array([math.cos(w),math.sin(w),0.])
            if state['phase']=='rock':
                # held sideways: the hand stopped following the circle (in the port, or jammed on its edge)
                hist=state['hist']
                hist.append((s['t'],np.asarray(s['pos'],dtype=float)[:2].copy(),circle[:2].copy()))
                while len(hist)>1 and hist[1][0]<=s['t']-1.:
                    hist.pop(0)
                path=sum(float(np.linalg.norm(b[2]-a[2])) for a,b in zip(hist,hist[1:]))
                moved=float(np.linalg.norm(hist[-1][1]-hist[0][1]))
                if (s['t']-hist[0][0]>=1.-1e-9 and path>=WIGGLE_HELD_PATH_M and moved<WIGGLE_HELD_M) or below(adv)>=depth:
                    # freeze the rocking (a moving target on a held hand trips the stall guard), keep pressing
                    state.update(phase='press',held_at=round(s['t'],2),frozen=np.asarray(s['target'],dtype=float).copy())
            if state['phase']=='press':
                side=state['frozen']-s['target']
                side[2]=0.
                zs=state['zs']
                zs.append((s['t'],adv))
                while len(zs)>1 and zs[1][0]<=s['t']-WIGGLE_BOTTOM_S:
                    zs.pop(0)
                still=s['t']-zs[0][0]>=WIGGLE_BOTTOM_S-1e-9 and zs[-1][1]-zs[0][1]<WIGGLE_BOTTOM_M
                full=s['t']>=WIGGLE_FORCE_RAMP_S
                if (still and full) or s['t']-state['held_at']>=WIGGLE_PRESS_MAX_S:
                    seated=below(adv)>=depth
                    state['stop']='seated' if seated else 'jammed'
                    if seated:
                        state.update(seated_at=round(s['t'],2),yaw_at=round(math.degrees(turned),2),
                                     side_at=np.round((s['pos']-p0)[:2],4).tolist())
                    return np.zeros(3),0.,True
                return side+down*ax,0.,False
            side=circle-s['target']
            side[2]=0.
            want_yaw=amp*ramp*math.sin(2*math.pi*s['t']/period)
            d=float(np.clip(want_yaw-state['cmd'],-rate*self.sim.dt,rate*self.sim.dt))
            if d and (state['cmd']-turned)*math.copysign(1.,d)>SCREW_YAW_LEAD_RAD:   # wait for the wrist
                state['yaw_waits']+=1
                d=0.
            state['cmd']+=d
            return side+down*ax,d,False

        def derive(rep):
            adv=float((self.sim.eef_pos-p0)@down)
            out=dict(seated=state['stop']=='seated',advance_m=round(adv,4),max_advance_m=round(state['adv'],4),
                     depth_below_surface_m=None if surface is None else round(below(adv),4),
                     surface_known=surface is not None,depth_m=depth,force_N=[round(start_n,1),round(force,1)],
                     held_sideways_after_s=state['held_at'],yaw_amplitude_deg=round(math.degrees(amp),2),
                     yaw_period_s=round(period,2),shift_m=shift,force_capped=state['capped'],yaw_wait_steps=state['yaw_waits'])
            where=('below the support surface' if surface is not None else 'below where wiggle started')
            if state['stop']=='seated':
                out.update(seated_after_s=state['seated_at'],yaw_at_seat_deg=state['yaw_at'],side_at_seat_m=state['side_at'],
                           note=f'{below(adv)*1000:.1f} mm {where}: seated')
            elif state['stop']=='jammed':
                out['note']=(f'held sideways but only {below(adv)*1000:.1f} mm {where} after pressing: jammed on the '
                             'mouth; lift, check the photos (turned 180 deg?) and align again')
            elif state['stop']=='not_seated':
                out['note']=(f'only {below(adv)*1000:.1f} mm {where}: if it barely moved, the part may be turned the '
                             'wrong way (180 deg) or off by more than the rocking covers; lift, check the photos and align again')
            return out
        rep=self._run('wiggle',dict(depth=depth,force=force,yaw_deg=yaw_deg,shift=shift,duration=duration),ctrl,
                      duration+2.,stop_on_contact=False,grip=1. if self.grip_closed else None,derive=derive,
                      max_lead=CONTACT_LEAD_M)
        if state['stop']:            # a guard or a timeout keeps its own termination
            rep.termination=state['stop']
        return rep

    def point_tool(self,direction,rate_deg_s=2.):
        a = np.asarray(direction,dtype=float)
        if a.shape!=(3,) or not np.isfinite(a).all() or np.linalg.norm(a)<1e-9:
            raise ValueError('direction must be a nonzero finite 3-vector')
        if not 0<rate_deg_s<=math.degrees(ROT_RATE_MAX_RAD_S):
            raise ValueError('Rotation rate must be positive and <= %.2f rad/s' % ROT_RATE_MAX_RAD_S)
        a /= np.linalg.norm(a)
        r0 = R.from_quat(self.sim.eef_quat)
        # Minimal rotation aligns the tool axis while preserving finger roll.
        z = r0.apply([0,0,1])
        ang = math.acos(float(np.clip(z@a,-1,1)))
        if ang>.09:
            raise ValueError('point_tool exceeds 0.09 rad adaptation envelope')
        axis = np.cross(z,a)
        rg = r0 if ang<1e-8 else R.from_rotvec(axis/np.linalg.norm(axis)*ang)*r0
        duration = max(.1,ang/math.radians(rate_deg_s))
        interp = Slerp([0,1],R.concatenate([r0,rg]))
        def ctrl(s):
            self.target_quat = interp(min(1.,s['t']/duration)).as_quat()
            return np.zeros(3),0.,s['t']>=duration+.1
        rep = self._run('point_tool',dict(direction=a.tolist()),ctrl,duration+2.,
                        stop_on_contact=self._stop_on_contact(None),
                        grip=1. if self.grip_closed else None)
        self.target_quat = self.sim.eef_quat.copy()
        rep.derived.update(force_rezeroed=False,rotation_requested_rad=ang)
        return rep

    def turn(self,axis,pivot,angle_deg,rate_deg_s=2.,torque_cap=None,stop_on_resistance_drop=False):
        if not np.isfinite(angle_deg) or abs(math.radians(angle_deg))>.55:   # the robot side allows 0.60 rad per motion
            raise ValueError('turn exceeds 0.09 rad adaptation envelope')
        if not 0<rate_deg_s<=math.degrees(ROT_RATE_MAX_RAD_S):
            raise ValueError('Invalid turn rate')
        return super().turn(axis,pivot,angle_deg,rate_deg_s,torque_cap,stop_on_resistance_drop)

    def call(self,name,**kwargs):
        # Real actions must not silently discard misspelled/unknown parameters.
        fn = getattr(self,name,None)
        allowed = set(inspect.signature(fn).parameters) if callable(fn) else set()
        clean = {}
        for key,value in kwargs.items():
            dest = key if key in allowed else self.ARG_ALIASES.get(name,{}).get(key)
            if dest not in allowed or dest in clean:
                raise ValueError('Unknown/duplicate real primitive argument: '+key)
            clean[dest] = value
        # Bound defaults as well as supplied values before any controller can
        # start (search with a zero pitch would otherwise never finish planning).
        bound = inspect.signature(fn).bind(**clean)
        bound.apply_defaults()
        values = bound.arguments
        positive = {'speed','max_dist','distance','depth','amplitude','radius_max','pitch',
                    'drop','hop','guide_step','f_catch','normal_force','target_force','force',
                    'max_force','f_threshold','rate_deg_s','max_time','hold_time','duration',
                    'seconds','height','lead','torque_cap'}
        for key,value in values.items():
            if key in positive and value is not None:
                if isinstance(value,bool) or not np.isfinite(value) or value<=0:
                    raise ValueError(key+' must be finite and positive')
            if key in ('direction','normal','axis','pivot'):
                a = np.asarray(value,dtype=float)
                if a.shape!=(3,) or not np.isfinite(a).all() or (key!='pivot' and np.linalg.norm(a)<1e-9):
                    raise ValueError('Invalid '+key)
        if 'wiggle' in values and not 0<=values['wiggle']<=.005:
            raise ValueError('wiggle must be between 0 and 5 mm')
        if values.get('radius_max') is not None and values['radius_max']>SEARCH_RADIUS_MAX_M:
            raise ValueError('radius_max must be <= %g mm' % (SEARCH_RADIUS_MAX_M*1000))
        if values.get('hop') is not None and values['hop']>.01:
            raise ValueError('hop must be <= 10 mm for initial adaptation')
        # Fail before controller activation for common infeasible requests.
        for key in ('speed',):
            if key in clean and clean[key] is not None and not 0<float(clean[key])<=.008:
                raise ValueError('Real speed must be 0 < speed <= .008 m/s')
        for key in ('target_force','normal_force','force','max_force','f_threshold'):
            if key in clean and clean[key] is not None and not 0<float(clean[key])<self.f_cap:
                raise ValueError('Force target must be below raw force ceiling')
        if clean.get('torque_cap') is not None and clean['torque_cap']>self.t_cap:
            raise ValueError('Torque cap cannot exceed raw torque ceiling')
        if name in ('turn','rotate_z') and clean.get('stop_on_resistance_drop'):
            if not self.sim.calibration.get('torque_verified'):
                raise BackendError('Resistance-drop rotation requires verified torque sign/calibration')
        return super().call(name,**clean)


def screw_quantities(reps):
    """Tool-result quantities for one or more consecutive screw motions: how far the screw went down its axis
    while turning (and per turn), and how the sideways force changed (a side load = turning off-axis)."""
    first,last=reps[0],reps[-1]
    sink=float(first.derived['start_z_m'])-float(last.derived['end_z_m'])
    turned=sum(((float(r.yaw_deg or 0)+180.)%360.)-180. for r in reps)
    # each motion presses before it turns; only the sink while turning tells the thread
    pressed=sum(float(r.derived.get('press_sink_m',0.)) for r in reps)
    turning=sum(float(r.derived.get('turn_sink_m',r.derived.get('sink_m',0.))) for r in reps)
    if abs(turning)<SCREW_STILL_M:
        flag='not_advancing'
    else:
        flag='advancing' if turning>0 else 'rising'
    per_turn=turning/abs(turned)*360. if abs(turned)>=10. else None
    unit=('mm down the screw axis in all: %.2f mm while pressing before turning, %.2f mm while turning %.0f deg'
          % (pressed*1000.,turning*1000.,turned))
    if per_turn is not None:
        unit+=' (%.1f mm per turn)' % (per_turn*1000.)
    out=[Quantity('screw_advance',round(sink*1000.,2),unit,None,sum(len(r.pos_rows) for r in reps),
                  'whole call','TCP travel along the vertical axis while the hand pressed and followed it',[flag],
                  dict(turned_deg=round(turned,1),per_turn_mm=None if per_turn is None else round(per_turn*1000.,2),
                       press_sink_mm=round(pressed*1000.,2),turn_sink_mm=round(turning*1000.,2),
                       segment_sink_mm=[round(float(r.derived.get('sink_m',0.))*1000.,2) for r in reps])).to_dict()]
    a,b=np.asarray(first.wrench_rows,dtype=float),np.asarray(last.wrench_rows,dtype=float)
    if len(a)>=3 and len(b)>=3:
        side=(b[-5:,1:3].mean(0)-a[:3,1:3].mean(0))
        out.append(Quantity('side_load_change',round(float(np.linalg.norm(side)),2),'N (start to end)',None,
                            len(a)+len(b),'whole call','change of the sideways reaction; FR3 estimate drifts 2-4 N',
                            [],dict(direction_deg=round(math.degrees(math.atan2(side[1],side[0])),0))).to_dict())
    return out

