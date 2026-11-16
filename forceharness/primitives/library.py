"""Self-terminating force primitives (layer 2 of ForceHarness).

The model decides only between primitives.  Inside a primitive the harness closes the loop at the control rate
(20 Hz in simulation), and a fixed guard (force / torque caps, lead and speed limits) is always active; the model can
choose targets and parameters but cannot change the guard.

Force through lead
    Every primitive keeps a target pose p* and advances it a little on every tick; the backend receives
    dpos = p* - p, and its impedance controller (OSC in robosuite, the stock joint impedance controller on the real
    FR3) acts as a spring between the hand and the target, so the contact force is F ~ K (p* - p).  The lead
    |p* - p| is clamped (``max_lead_contact`` for compliant contact-seeking motion, ``max_lead_guard`` for guarded
    moves, ``max_lead_free`` otherwise, or a per-primitive ``max_lead``), which bounds the force any primitive can
    build.  When the hand is blocked the target keeps advancing only up to the clamp, the force rises and saturates,
    the event detector reports contact / stall, and the primitive stops.  The position target is re-synchronised
    with the measured pose at the start of every primitive; the orientation target persists between primitives
    (initially tool straight down) and only the rotating primitives change it.

One shared loop
    Every primitive runs in ``_run`` and differs only in a small control law,
    ``controller(state) -> (dtarget, dyaw, done)``, that reads the tared reaction force and the pose.  To hold a force
    F_d along a direction u the law uses admittance control, v = clip(k_f (F_d - F_u), +-v_max) with F_u = -F . u,
    and moves the target by u v dt; slide, search and insert keep this law along the contact normal and add their own
    motion along the surface.  ``_run`` clamps the lead, steps the backend, feeds the event detector, logs the wrench
    and the pose, builds the ContactReport and runs the physics translation layer (``translation.estimators``) on it.

Self-termination
    A primitive ends on an event (contact, stall, a resistance drop, or its own completion condition), with the force
    / torque caps and a time limit as fallbacks.  After an early stop the target is pulled back next to the measured
    pose (10% of the lead is kept), which unloads the spring, and the hand settles before the report is built.

Action interfaces (the columns of the evaluation matrix, see ``INTERFACES``)
    pose   scene / move_to / move_by / rotate_z / grasp / release / hold / done.  Moves do NOT stop on contact; only
           the safety caps abort them.
    guard  the same tools, but every move stops on contact (contact guard) and reports termination=contact.
    force  guard + move_until_contact / push / slide / search / insert / lift / probe, each with a force target or a
           force-based stop condition.
    A task can open further primitives through ``extra_tools`` (e.g. pull, turn, point_tool); the ones in
    ``FORCE_ONLY_EXTRA`` close their loop on force readings and exist only under the force interface.

Load taring
    Holding an object adds about -m g to the wrist force.  The load is recorded as ``carry_bias`` by the still phase
    of ``lift``, at the end of any other completed primitive that ends at rest with a downward, load-like net force,
    and at the start of a primitive when the held load has clearly shifted; event detection and force regulation then
    use reaction - carry_bias.  ``grasp`` and ``release`` reset it.  While the gripper is closed, a downward load below
    ``payload_max`` does not raise contact_start when the hand moves up or before the load is tared (lift, hold, pull
    and turn suppress contact_start entirely while holding); the force caps still apply.

Real-robot hooks
    The class attributes ``search_*`` and ``lead_override`` and the methods ``_raw_hold`` and ``_wrist_turn_range``
    let a real-robot subclass change how ``search`` arms, pauses, presses and decides, and how far the wrist may turn;
    the defaults reproduce the simulation behaviour.

All coordinates are world-frame metres and direction vectors are normalised.  Speeds: ``speed_free`` in free space,
``speed_near`` near or in contact.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import inspect

from forceharness.primitives.events import EventDetector, main_axis
from forceharness.translation.estimators import EstimatorContext, estimate
from forceharness.primitives.report import ContactEvent, ContactReport

# The tools of each action interface, in tool-menu order (see the module docstring).
INTERFACES = {
    "pose": ["scene", "move_to", "move_by", "rotate_z", "grasp", "release", "hold", "done"],
    "guard": ["scene", "move_to", "move_by", "rotate_z", "grasp", "release", "hold", "done"],
    "force": ["scene", "move_to", "move_by", "rotate_z", "move_until_contact", "push", "slide", "search", "insert", "lift", "probe",
              "grasp", "release", "hold", "done"],
}

G = 9.81
# extra primitives a task may open only under the force interface (they close their loop on force readings, so the
# no-force control cannot use them)
FORCE_ONLY_EXTRA = {"pull"}


def _unit(v: Sequence[float]) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-9:
        raise ValueError("direction vector is zero")
    return v / n


def _yaw_deg_xyzw(q: np.ndarray) -> float:
    x, y, z, w = q
    return math.degrees(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _load_like(f: np.ndarray, payload_max: float) -> bool:
    """Whether a net force looks like a load hanging in the hand (and may be tared): pointing down, below payload_max,
    and roughly vertical (horizontal part <= max(1 N, half the vertical part)), since gravity is vertical.

    The verticality test matters: while holding a locked door handle (lock_or_stiff), the residual force of a
    previous pull (about 10 N with a -4 N vertical part) would otherwise be tared as a load, and the next pull would
    under-read by 10 N and overshoot its target."""
    f = np.asarray(f, dtype=float)
    return bool(f[2] < -1.0 and float(np.linalg.norm(f)) < payload_max
                and float(np.linalg.norm(f[:2])) <= max(1.0, 0.5 * abs(float(f[2]))))


def _arm_limit_margin(sim) -> Optional[float]:
    """Smallest margin (rad) of the 7 arm joints to their limits; None when it cannot be read."""
    obs = getattr(sim, "obs", None)
    q = obs.get("robot0_joint_pos") if isinstance(obs, dict) else None
    rng = getattr(sim, "_arm_ranges", None)
    if rng is None:
        try:
            mj = sim.mujoco
            ids = [mj.mj_name2id(sim.m, mj.mjtObj.mjOBJ_JOINT, f"robot0_joint{i}") for i in range(1, 8)]
            rng = np.array([sim.m.jnt_range[j] for j in ids], dtype=float) if all(j >= 0 for j in ids) else False
        except Exception:
            rng = False
        try:
            sim._arm_ranges = rng
        except Exception:
            pass
    if q is None or rng is False or len(q) < 7:
        return None
    q = np.asarray(q[:7], dtype=float)
    return float(np.min(np.minimum(q - rng[:, 0], rng[:, 1] - q)))


class Primitives:
    """The primitive library bound to one backend (``sim``) and one action interface.

    The model reaches it through ``call(name, **kwargs)``, which checks that the tool exists under the interface,
    maps common argument aliases and returns a ContactReport.  See the module docstring for the control structure.
    """
    # Real-robot hooks for search(mode="slide").  The defaults reproduce the simulation behaviour; a real-robot
    # subclass overrides them (on a real arm a threaded hole never lets the tip fall free, and the estimated force
    # drifts while the hand moves in the air, so sinking, arming and pressing are judged differently).
    # slide search: sinking deeper than drop below the touched surface ends the search as "dropped" on its own
    # (False = sim behaviour, which also asks for the hand to be blocked sideways)
    search_sink_is_drop = False
    # slide search: (pause_m, hole_m, hold_s, press_n) = once the tip sinks pause_m below the surface it slid on
    # in the last 2 s, stop the spiral and press at max(normal_force, press_n) for hold_s, holding the spot
    # sideways; still hole_m (or drop) deeper than that surface = in a hole (None = sim: the spiral never pauses)
    search_settle = None
    search_guided_hops = 8      # slide: lift-and-move hops along a catching side force (sim: a peg corner)
    search_arm_rise = None      # slide: arm on a force rise of this many N over the start reading (None = sim)
    search_arm_blocked = None   # slide: (lead_m, still_m, window_s): arm once the descending target leads the hand
                                # by lead_m while the hand moved < still_m in window_s (None = force arming)
    search_fall_rule = True     # slide: "fell drop deeper within 1 s while unsupported" counts as dropped
    search_press = None         # slide pause: (lead_m, raw_cap_n, hole_m): drive the press down at speed_free with
                                # this much lead allowed until press_n or the raw cap, hold it, finish the hold, then
                                # a hole is >= hole_m below the surface whatever drop says (None = sim: admittance
                                # press at speed_near within the 8 mm lead, judged against min(drop, hole_m))
    lead_override = None        # set by a primitive for a phase: a larger lead cap than its max_lead

    def __init__(
        self,
        sim,
        interface: str = "force",
        speed_free: float = 0.10,
        speed_near: float = 0.02,
        kp_contact: float = 40.0,     # OSC stiffness while seeking contact / pushing / sliding (at the default 150 a 6 mm lead already gives a ~45 N impact)
        f_contact: float = 2.0,
        f_cap: float = 25.0,
        t_cap: float = 3.0,
        kf: float = 0.0015,          # admittance gain, m/s per N
        pos_tol: float = 0.003,
        # slow_in_contact=True: in contact, move at min(requested speed, speed_near).  False (default): always speed_near
        # in contact, even when the caller asked for a slower speed.
        slow_in_contact: bool = False,
        max_rows: int = 160,
        payload_max: float = 15.0,   # while lifting, a load |F| below this is not treated as a collision
        detector_kwargs: Optional[Dict[str, Any]] = None,
        extra_tools: Sequence[str] = (),   # primitives a task opens in addition (e.g. point_tool for doors); empty = the standard tool list
    ):
        assert interface in INTERFACES, interface
        self.sim = sim
        self.interface = interface
        self.extra_tools = tuple(e for e in extra_tools if e != "push_click")     # push_click is a description switch, not a tool
        # pull_tare_lateral: tare the static sideways force present when a pull starts.  Off by default: on an ordinary
        # door the residual sideways force left by the grasp vanishes as soon as the pull starts, so taring it would create
        # a spurious opposite force.
        self.pull_tare_lateral = False
        self.pull_adapt_after_m = 0.005    # pull: travel (m) before the pulling direction adapts to the reaction force
        self.pull_r_min = 0.15             # pull: per tick the direction turns at most (hand travel / pull_r_min) rad (smallest radius of a mechanism path)
        self.speed_free, self.speed_near = speed_free, speed_near
        self.f_contact, self.f_cap, self.t_cap = f_contact, f_cap, t_cap
        self.kf = kf
        self.pos_tol = pos_tol
        self.slow_in_contact = slow_in_contact
        self.max_rows = max_rows
        self.payload_max = payload_max
        self.kp_contact = kp_contact
        self.kp_free = float(getattr(sim, "kp", 150.0))
        self.max_lead_contact = 0.006    # contact-seeking moves (low stiffness kp_contact): lead <= 6 mm keeps an impact within about 10 N
        self.max_lead_guard = 0.012      # ordinary moves with the contact guard (full stiffness): <= 12 mm
        self.max_lead_free = 0.03        # free moves: <= 3 cm
        self.grip_closed = False
        self.carry_bias = np.zeros(3)
        self.n_reports = 0
        dk = dict(f_contact=f_contact, f_cap=f_cap, t_cap=t_cap, min_samples=1)
        dk.update(detector_kwargs or {})
        self.detector = EventDetector(**dk)
        self.target_pos = sim.eef_pos
        # The orientation target persists: tool straight down (Rz(yaw) Rx(pi)), yaw changed only by the rotating primitives.
        # robosuite's Panda starts pitched by about 9 deg; uncorrected, the wiping pad and the fingertips would be tilted
        # (the Wipe task counts a stain as wiped only with the pad parallel to the table).
        self.target_quat = self._level_quat(math.radians(_yaw_deg_xyzw(sim.eef_quat)))
        self._level_and_rezero()

    def _level_and_rezero(self, max_steps: int = 40) -> None:
        """At the start, turn the tool straight down (every primitive keeps this orientation), then re-zero the sensor in
        this pose (see ForceSim.rezero_bias)."""
        sim = self.sim
        if not hasattr(sim, "rezero_bias"):
            return
        try:
            for _ in range(max_steps):
                drot = self._drot_world()
                if float(np.linalg.norm(drot)) < math.radians(0.5):
                    break
                sim.step(np.zeros(3), drot, None)
            sim.hold(4)
            sim.rezero_bias(10)
        except Exception:
            pass
        self.target_pos = sim.eef_pos

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _level_quat(yaw_rad: float) -> np.ndarray:
        return (R.from_euler("z", yaw_rad) * R.from_euler("x", math.pi)).as_quat()

    def _sync_target(self) -> None:
        self.target_pos = self.sim.eef_pos          # position re-synchronised for every primitive; the orientation target persists

    def _drot_world(self) -> np.ndarray:
        """World-frame rotation vector from the measured to the target orientation."""
        r_err = R.from_quat(self.target_quat) * R.from_quat(self.sim.eef_quat).inv()
        return r_err.as_rotvec()

    def _run(self, name: str, params: Dict[str, Any], controller, max_time: float,
             stop_on_contact: bool, stop_on_stall: bool = False, stop_on_drop: bool = False,
             grip: Optional[float] = None, derive=None, max_lead: Optional[float] = None,
             compliant: Optional[bool] = None, soft_dir: Optional[np.ndarray] = None,
             kp_vec: Optional[np.ndarray] = None, suppress_contact: bool = False) -> ContactReport:
        """The shared loop of every primitive.

        controller(state) -> (dtarget[3], dyaw_rad, done) is called once per tick; state holds pos, reaction
        (carry_bias removed), reaction_raw, torque, t, step, yaw, in_contact and target.  The loop adds dtarget to the
        target, clamps the lead, steps the backend, feeds the event detector and stops on a force / torque cap
        (always), contact_start (stop_on_contact), stall (stop_on_stall), resistance_drop (stop_on_drop), done, or
        max_time (timeout).
        compliant=True lowers the OSC translational stiffness to kp_contact only along soft_dir (on all three axes when
        soft_dir is None), the other axes keep full stiffness; kp_vec sets the per-axis stiffness explicitly.  The
        stiffness is restored at the end.  suppress_contact keeps contact_start from being reported.
        """
        sim = self.sim
        self.detector.reset()
        self._sync_target()
        use_soft = bool(compliant) or kp_vec is not None
        if hasattr(sim, "set_kp"):
            if kp_vec is not None:                 # explicit per-axis stiffness (insert: soft tangentially, stiff along the normal)
                sim.set_kp(np.asarray(kp_vec, dtype=float))
            elif bool(compliant) and soft_dir is not None:
                u = np.abs(_unit(soft_dir))
                kp6 = np.concatenate([self.kp_free - (self.kp_free - self.kp_contact) * u, np.ones(3) * self.kp_free])
                sim.set_kp(kp6)
            else:
                sim.set_kp(self.kp_contact if bool(compliant) else self.kp_free)
        t0 = sim.sim_time
        p0 = sim.eef_pos
        yaw0 = _yaw_deg_xyzw(sim.eef_quat)
        rows: List[List[float]] = []
        prows: List[List[float]] = []
        events: List[ContactEvent] = []
        cmd_acc = np.zeros(3)
        peak_f = peak_t = 0.0
        contact_f: List[float] = []
        termination = "timeout"
        n = 0
        max_steps = max(1, int(max_time / sim.dt))
        w = sim.wrench()
        reaction_raw = sim.reaction_force(w)
        # tare at the start of a primitive: holding something at rest with a net force that looks like a downward load and
        # differs from the current carry_bias (the load shifted in the fingers, or it was never lifted) -> re-estimate
        if self.grip_closed and name not in ("grasp", "release"):
            f0 = reaction_raw
            if _load_like(f0, self.payload_max) and float(np.linalg.norm(f0 - self.carry_bias)) > 1.0:
                samples = [f0]
                for _ in range(2):
                    w = sim.step(np.zeros(3), None, grip)
                    samples.append(sim.reaction_force(w))
                f0 = np.mean(samples, axis=0)
                if _load_like(f0, self.payload_max):
                    self.carry_bias = f0.copy()
                reaction_raw = sim.reaction_force(w)
        reaction = reaction_raw - self.carry_bias
        last_pos = p0
        truth_contact_steps = 0
        for n in range(1, max_steps + 1):
            state = dict(pos=last_pos, reaction=reaction, reaction_raw=reaction_raw, torque=w.net_torque_world, step=n,
                         t=sim.sim_time - t0, yaw=_yaw_deg_xyzw(sim.eef_quat), in_contact=self.detector.in_contact,
                         target=self.target_pos.copy())
            dtarget, dyaw, done = controller(state)
            dtarget = np.asarray(dtarget, dtype=float)
            self.target_pos = self.target_pos + dtarget
            # clamp the lead: the target may lead the measured pose by at most lead_cap; otherwise the whole displacement stored
            # in the spring turns into impact force the moment the hand hits something
            lead_cap = max_lead if max_lead is not None else (self.max_lead_contact if use_soft else self.max_lead_free)
            if stop_on_contact and not use_soft:
                lead_cap = min(lead_cap, self.max_lead_guard)      # guarded move: full stiffness but a smaller lead, so a hit stays around 10 N
            if self.lead_override:
                lead_cap = max(lead_cap, self.lead_override)
            lead = self.target_pos - last_pos
            ln = float(np.linalg.norm(lead))
            if ln > lead_cap:
                self.target_pos = last_pos + lead / ln * lead_cap
            if dyaw:
                self.target_quat = (R.from_rotvec([0, 0, dyaw]) * R.from_quat(self.target_quat)).as_quat()
            dpos = self.target_pos - last_pos
            drot = self._drot_world()
            if not dyaw and np.linalg.norm(drot) < 1e-4:
                drot = None
            w = sim.step(dpos, drot, grip)
            reaction_raw = sim.reaction_force(w)
            reaction = reaction_raw - self.carry_bias
            pos = sim.eef_pos
            actual = pos - last_pos
            last_pos = pos
            cmd_acc += dtarget
            # holding something with a downward net load that is not tared yet (carry_bias ~ 0): do not report contact whichever
            # way the hand moves (suppressing only upward moves flags the load as contact on the tick where a move ends and
            # dtarget returns to zero); once tared, reaction excludes the load and collisions are detected as usual
            untared = float(np.linalg.norm(self.carry_bias)) < 0.5
            lifting_load = (self.grip_closed and reaction[2] < 0 and float(np.linalg.norm(reaction)) < self.payload_max
                            and (dtarget[2] > 1e-6 or untared))
            ev = self.detector.update(sim.sim_time, reaction, w.net_torque_world, pos, dtarget, actual, self.grip_closed,
                                      suppress_contact=lifting_load or suppress_contact)
            events.extend(ev)
            f = float(np.linalg.norm(reaction))
            tq = float(np.linalg.norm(w.net_torque_world))
            peak_f, peak_t = max(peak_f, f), max(peak_t, tq)
            if self.detector.in_contact:
                contact_f.append(f)
            if sim.gripper_contact_state()["in_contact"]:
                truth_contact_steps += 1
            rows.append([round(sim.sim_time - t0, 3), *np.round(reaction_raw, 3).tolist(), *np.round(w.net_torque_world, 3).tolist()])
            prows.append(np.round(pos, 4).tolist())
            kinds = {e.kind for e in ev}
            if "force_cap" in kinds or "torque_cap" in kinds:
                termination = "force_cap" if "force_cap" in kinds else "torque_cap"
                break
            if stop_on_contact and "contact_start" in kinds:
                termination = "contact"
                break
            if stop_on_stall and "stall" in kinds:
                termination = "stall"
                break
            if stop_on_drop and "resistance_drop" in kinds:
                termination = "resistance_drop"
                break
            if done:
                termination = "completed"
                break
        # stopped by contact / stall / a cap: pull the target back next to the measured pose (keep 10% of the lead) to unload
        # the force stored in the spring; in every case the hand then settles for 2 ticks
        if termination in ("contact", "force_cap", "torque_cap", "stall", "resistance_drop"):
            self.target_pos = sim.eef_pos + 0.1 * (self.target_pos - sim.eef_pos)
        for _ in range(2):
            w = sim.step(self.target_pos - sim.eef_pos, None, grip)
        if hasattr(sim, "set_kp") and use_soft:
            sim.set_kp(self.kp_free)
        reaction_raw = sim.reaction_force(w)
        # carry tare: holding something, primitive completed, hand at rest, net force like a downward load -> record it as
        # carry_bias.  Otherwise only lift would tare, and after a lift with move_to the load would be taken for contact at
        # the start of every later move.
        tared = False
        if self.grip_closed and termination == "completed" and name not in ("grasp", "release", "lift") and len(rows) >= 5:
            tail = np.asarray(rows[-5:], dtype=float)[:, 1:4]
            f_tail = tail.mean(axis=0)
            steady = float(np.linalg.norm(tail, axis=1).std()) < 1.0
            load_like = _load_like(f_tail, self.payload_max)
            if steady and load_like:
                self.carry_bias = f_tail.copy()
                tared = True
        if len(rows) > self.max_rows:  # uniform downsampling, first and last rows kept
            idx = np.linspace(0, len(rows) - 1, self.max_rows).round().astype(int)
            rows = [rows[i] for i in idx]
            prows = [prows[i] for i in idx]
        rep = ContactReport(
            primitive=name, params=params, termination=termination, t_start=t0, t_end=sim.sim_time,
            start_pos=np.round(p0, 4).tolist(), end_pos=np.round(sim.eef_pos, 4).tolist(),
            commanded_disp=np.round(cmd_acc, 4).tolist(), actual_disp=np.round(sim.eef_pos - p0, 4).tolist(),
            events=events, wrench_rows=rows, pos_rows=prows,
            final_wrench=[*np.round(reaction_raw, 3).tolist(), *np.round(w.net_torque_world, 3).tolist()],
            final_wrench_net=[*np.round(reaction_raw - self.carry_bias, 3).tolist(), *np.round(w.net_torque_world, 3).tolist()],
            peak_force=round(peak_f, 3), peak_torque=round(peak_t, 3),
            mean_force_in_contact=round(float(np.mean(contact_f)), 3) if contact_f else 0.0,
            n_steps=n, gripper_opening=sim.gripper_opening, yaw_deg=round(_yaw_deg_xyzw(sim.eef_quat) - yaw0, 2),
            truth=dict(contact_fraction=round(truth_contact_steps / max(n, 1), 3),
                       touching_end=sim.gripper_contact_state()["touching"],
                       peak_object_contact_force=round(float(sim.peak_contact_force), 3),
                       grip_closed=self.grip_closed, tared=tared, task_state=dict(getattr(sim, "task_state", {}))),
        )
        tf = getattr(getattr(sim, "task", None), "truth_fn", None)
        if tf is not None:
            try:
                rep.truth["task"] = tf(sim)
            except Exception as e:
                rep.truth["task"] = dict(error=str(e))
        if derive is not None:
            try:
                rep.derived.update(derive(rep))
            except Exception as e:  # a failing derive does not fail the primitive
                rep.derived["derive_error"] = str(e)
        self._estimate(rep)
        self.n_reports += 1
        return rep

    def _estimate(self, rep: ContactReport) -> None:
        """Run the physics translation layer: turn the force trace in rep into physical quantities (rep.quantities).

        It runs once at the end of _run; primitives that rewrite termination / primitive afterwards (push, search,
        insert, pull, probe) run it again, so that e.g. the insertion outcome flag sees "dropped" or "seated" instead of
        "completed"."""
        sim = self.sim
        try:
            ctx = EstimatorContext(carry_bias=self.carry_bias.copy(), grip_closed=self.grip_closed,
                                   noise_std=float(getattr(sim, "force_noise_std", 0.5) or 0.5),
                                   bias_std=float(getattr(sim, "force_bias_std", 0.3) or 0.3), dt=float(sim.dt),
                                   f_contact=self.f_contact, speed_near=self.speed_near,
                                   filter_n=int(getattr(self.detector, "filter_n", 4)))
            rep.quantities = [q.to_dict() for q in estimate(rep, ctx)]
            rep.derived.pop("physics_error", None)
        except Exception as e:  # a failing translation layer does not fail the primitive
            rep.derived["physics_error"] = str(e)

    def _stop_on_contact(self, explicit: Optional[bool]) -> bool:
        if self.interface == "pose":
            return False
        return True if explicit is None else bool(explicit)

    # ------------------------------------------------------------------ public primitives
    def move_to(self, x: float, y: float, z: float, stop_on_contact: Optional[bool] = None,
                speed: Optional[float] = None, max_time: float = 8.0) -> ContactReport:
        """Move the TCP to (x, y, z), orientation unchanged.  Stops on contact under the guard / force interfaces
        (stop_on_contact=None) and never under pose; in contact the speed drops to speed_near."""
        goal = np.array([x, y, z], dtype=float)
        v = speed or self.speed_free
        max_time = max(max_time, 2.0 * float(np.linalg.norm(goal - self.sim.eef_pos)) / v + 3.0)
        last = [None, 0]        # previous position, number of consecutive still ticks

        def ctrl(s):
            err = goal - s["target"]
            d = np.linalg.norm(err)
            reached = np.linalg.norm(goal - s["pos"]) < self.pos_tol
            moved = np.linalg.norm(s["pos"] - last[0]) if last[0] is not None else 1.0
            last[0] = s["pos"].copy()
            last[1] = last[1] + 1 if moved < 3e-4 else 0
            if reached:
                return np.zeros(3), 0.0, True
            if d < 1e-6:                      # target reached: wait for the robot to catch up or settle
                return np.zeros(3), 0.0, last[1] >= 6
            vv = (min(v, self.speed_near) if self.slow_in_contact else self.speed_near) if s["in_contact"] else v
            step = err / d * min(d, vv * self.sim.dt)
            return step, 0.0, False

        return self._run("move_to", dict(x=x, y=y, z=z), ctrl, max_time, self._stop_on_contact(stop_on_contact))

    def move_by(self, dx: float, dy: float, dz: float, stop_on_contact: Optional[bool] = None,
                speed: Optional[float] = None, max_time: float = 8.0) -> ContactReport:
        """move_to relative to the current position."""
        p = self.sim.eef_pos
        rep = self.move_to(p[0] + dx, p[1] + dy, p[2] + dz, stop_on_contact, speed, max_time)
        rep.primitive, rep.params = "move_by", dict(dx=dx, dy=dy, dz=dz)
        return rep

    def move_until_contact(self, direction: Sequence[float], max_dist: float = 0.1,
                           f_threshold: Optional[float] = None, speed: Optional[float] = None) -> ContactReport:
        """Move along direction at speed_near, compliant along it, until the detector reports contact (threshold
        f_threshold) or max_dist has been travelled (termination no_contact_within_max_dist)."""
        u = _unit(direction)
        v = speed or self.speed_near
        p0 = self.sim.eef_pos
        thr = float(f_threshold or self.f_contact)
        self.detector.f_contact = thr

        def ctrl(s):
            if np.dot(s["pos"] - p0, u) >= max_dist:
                return np.zeros(3), 0.0, True
            return u * v * self.sim.dt, 0.0, False

        try:
            rep = self._run("move_until_contact", dict(direction=u.round(3).tolist(), max_dist=max_dist, f_threshold=thr),
                            ctrl, max_dist / v + 3.0, stop_on_contact=True, compliant=True, soft_dir=u)
        finally:
            self.detector.f_contact = self.f_contact
        if rep.termination == "completed":
            rep.termination = "no_contact_within_max_dist"
        return rep

    def push(self, direction: Sequence[float], target_force: float, duration: float = 2.0,
             max_dist: float = 0.05, lead: float = 0.03, stop_on_resistance_drop: bool = False) -> ContactReport:
        """Press along direction and regulate the reaction force to target_force (admittance: force error -> target
        velocity), for duration seconds.

        stop_on_resistance_drop: stop as soon as the resistance peaks and then clearly falls (a "click", e.g. a handle
        unlatching).  While the target force exceeds the resistance the push advances at a constant speed_near
        (2 cm/s by default, at most max_dist), so the force trace is the resistance profile."""
        u = _unit(direction)
        p0 = self.sim.eef_pos
        target_force = float(min(target_force, self.f_cap * 0.8))

        # push uses 3x the admittance gain: with 0.0015 it does not converge within 2 s (a 5 N target reached only 3.8 N);
        # 3x with a 3 cm lead gives 5 N -> 5.0 N and 12 N -> 11.9 N with <= 25% peak overshoot
        kf = self.kf * 3.0
        # push detects a "click" itself: the generic event detector smooths over 4 ticks and confirms over 2, but the dip after a
        # snap-fit click lasts a single tick and is missed.  Here: the raw force along the push (2-tick mean), from 0.6 s after
        # the start; stop once it has fallen from its peak by more than half and by at least 2 N
        clk = dict(buf=[], peak=0.0, hit=False)

        def ctrl(s):
            f_u = float(np.dot(s["reaction"], -u))         # pushing along u, the reaction points along -u
            if stop_on_resistance_drop:
                clk["buf"] = (clk["buf"] + [f_u])[-2:]
                fa = float(np.mean(clk["buf"]))
                if s["t"] >= 0.6:
                    if fa > clk["peak"]:
                        clk["peak"] = fa
                    elif clk["peak"] >= 3.0 and (clk["peak"] - fa) >= max(2.0, 0.5 * clk["peak"]):
                        clk["hit"] = True
                        self.target_pos = np.asarray(s["pos"], dtype=float).copy()   # target back to the measured pose: in compliant mode it leads by a few cm and the hand would lunge forward
                        return np.zeros(3), 0.0, True
            e = target_force - f_u
            v = float(np.clip(kf * e, -self.speed_near, self.speed_near))
            if np.dot(s["target"] - p0, u) >= max_dist and v > 0:
                v = 0.0
            return u * v * self.sim.dt, 0.0, s["t"] >= duration

        def derive(rep):
            arr = rep.wrench_array()
            if len(arr) < 4:
                return {}
            fu = -((arr[:, 1:4] - self.carry_bias) @ u)
            tail = fu[-max(4, len(fu) // 4):]
            return dict(force_along_push_mean_N=round(float(tail.mean()), 2), force_along_push_std_N=round(float(tail.std()), 2),
                        regulation_error_N=round(float(target_force - tail.mean()), 2))

        # 3 cm lead: with the default 6 mm the compliant spring cannot build the force (a 5 N target settled at 3.65 N and
        # 8 N reached only 3.5 N); slide uses 2 cm and reaches more than 8 N
        params = dict(direction=u.round(3).tolist(), target_force=target_force, duration=duration)
        if stop_on_resistance_drop:
            params["stop_on_resistance_drop"] = True
        rep = self._run("push", params, ctrl, duration + 1.0, stop_on_contact=False,
                        derive=derive, compliant=True, soft_dir=u, max_lead=lead,
                        grip=1.0 if self.grip_closed else None)
        if clk["hit"]:
            rep.termination = "resistance_drop"
            rep.derived["click_peak_N"] = round(clk["peak"], 2)
            self._estimate(rep)
        return rep

    def slide(self, direction: Sequence[float], distance: float, normal_force: float = 5.0,
              normal: Sequence[float] = (0, 0, -1), speed: Optional[float] = None) -> ContactReport:
        """Slide on a surface: hold normal_force against it along normal while moving distance along direction
        (tangential)."""
        u = _unit(direction)
        nrm = _unit(normal)
        u = _unit(u - np.dot(u, nrm) * nrm)
        v = speed or self.speed_near
        p0 = self.sim.eef_pos
        normal_force = float(min(normal_force, self.f_cap * 0.8))

        def ctrl(s):
            if float(np.dot(s["pos"] - p0, nrm)) < -0.02:      # 2 cm off the surface: contact lost
                return np.zeros(3), 0.0, True
            f_n = float(np.dot(s["reaction"], -nrm))
            e = normal_force - f_n
            vn = float(np.clip(self.kf * e, -self.speed_near, self.speed_near))
            travelled = float(np.dot(s["pos"] - p0, u))
            done = travelled >= distance
            vt = 0.0 if done else v
            return (u * vt + nrm * vn) * self.sim.dt, 0.0, done

        def derive(rep):
            arr = rep.wrench_array()
            if len(arr) < 4:
                return {}
            F = arr[:, 1:4] - self.carry_bias
            fn = -(F @ nrm)
            ft = F @ u
            return dict(normal_force_mean_N=round(float(fn.mean()), 2), normal_force_std_N=round(float(fn.std()), 2),
                        tangential_force_mean_N=round(float(ft.mean()), 2),
                        friction_ratio=round(float(abs(ft.mean()) / max(fn.mean(), 1e-3)), 2),
                        lost_contact_fraction=round(float((fn < self.f_contact * 0.5).mean()), 2))

        # compliant only along the normal, full stiffness tangentially, 2 cm lead: at low stiffness the tangential force cannot
        # overcome friction (kp=40 with a 6 mm lead gives only ~0.5 N) and the tool would stay stuck in place
        return self._run("slide", dict(direction=u.round(3).tolist(), distance=distance, normal_force=normal_force),
                         ctrl, distance / v + 3.0, stop_on_contact=False, derive=derive, compliant=True, soft_dir=nrm, max_lead=0.02)

    def _raw_hold(self, s, st, cap) -> Tuple[bool, float]:
        """Hook for the search press: whether a press capped at ``cap`` (raw |F|, N) must stop growing now, and the
        reading it was judged by.  A real-robot subclass may also watch the tared and the controller-side readings
        and stop ahead of a fast rise."""
        now = float(np.linalg.norm(s["reaction_raw"]))
        return now >= cap, now

    def search(self, normal: Sequence[float] = (0, 0, -1), radius_max: float = 0.02, pitch: float = 0.004,
               normal_force: float = 1.5, drop: float = 0.02, speed: Optional[float] = None,
               mode: str = "hop", hop: float = 0.012, f_catch: float = 6.0, guide_step: float = 0.006) -> ContactReport:
        """Hole / insertion search around the current position.

        mode="hop": at each spiral sample point (spacing pitch, out to radius_max) lift by hop, move sideways and probe
        gently along normal until contact.  Ends dropped as soon as the tool sinks more than drop below the depth of
        the last probe, completed after the last point.
        mode="slide" (force-guided): press on the surface with normal_force and slide along a spiral.  When a corner of
        a square peg dips into the hole mouth and catches on the rim (side force > f_catch), the rim's reaction points
        toward the hole centre: lift 4 mm, move guide_step along it, come down again and restart the spiral around the
        new spot.  The search first descends (at most 3 cm) to a supporting surface (no_surface otherwise).
        Results: dropped = support lost, little side force and drop deeper along normal within 1 s, or deeper than
        drop below the start surface while blocked sideways; completed = radius_max searched without a drop; stuck =
        sideways motion blocked for 1.5 s beyond the 1 s window; out_of_range = more than radius_max + 1 cm from the
        start; lost_contact = 1.5 cm off the surface.  The real-robot hooks (class attributes) change how the slide
        search arms, pauses, presses and decides.
        """
        nrm = _unit(normal)
        if mode == "hop":
            rep = self._search_hop(nrm, radius_max, pitch, drop, speed, hop)
            self._estimate(rep)
            return rep
        a = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = _unit(a - np.dot(a, nrm) * nrm)
        e2 = np.cross(nrm, e1)
        v = speed or self.speed_near
        normal_force = float(min(normal_force, self.f_cap * 0.8))
        p0 = np.array(self.sim.eef_pos, dtype=float).copy()
        dt = self.sim.dt
        state = dict(theta=0.0, dropped=False, lost=False, stuck=False, phase="spiral", t_phase=0.0, guide=None, hops=0,
                     center=p0.copy(), blocked=0, armed=False, no_surface=False, settles=0, settle_from=None,
                     settled_in=None, f_start=None, advhist=[], settle_ref=0.0, armhist=[], reach=0.0)
        window: List[float] = []           # advance over the last 1 s
        flwin: List[float] = []            # side force over the last 0.3 s
        poshist: List[np.ndarray] = []     # measured lateral position over the last 1 s
        tgthist: List[np.ndarray] = []     # target lateral position over the last 1 s

        def ctrl(s):
            adv = float(np.dot(s["pos"] - p0, nrm))
            window.append(adv)
            if len(window) > max(2, int(1.0 / dt)):
                window.pop(0)
            f_n = float(np.dot(s["reaction"], -nrm))
            supported = f_n >= 0.3 * normal_force
            F_lat = s["reaction"] - np.dot(s["reaction"], nrm) * nrm
            fl = float(np.linalg.norm(F_lat))
            flwin.append(fl)
            if len(flwin) > max(2, int(0.3 / dt)):
                flwin.pop(0)
            # models often call search in mid-air: first descend to a supporting surface (at most 3 cm), and do not judge
            # "dropped" before it is found
            if not state["armed"]:
                if state["f_start"] is None:
                    state["f_start"] = f_n
                if self.search_arm_blocked is not None:      # arm when the surface stops the hand (position, not force)
                    lead_m, still_m, win_s = self.search_arm_blocked
                    hist = state["armhist"]
                    hist.append((s["t"], adv))
                    while len(hist) > 1 and hist[1][0] <= s["t"] - win_s:
                        hist.pop(0)
                    arm = (s["t"] - hist[0][0] >= win_s - 1e-9 and adv - hist[0][1] < still_m
                           and float(np.dot(s["target"] - s["pos"], nrm)) >= lead_m)
                elif self.search_arm_rise is not None:
                    arm = f_n - state["f_start"] >= self.search_arm_rise
                else:
                    arm = supported
                if arm:
                    state["armed"] = True
                    p0[:] = s["pos"]; state["center"] = p0.copy(); window.clear()
                    return np.zeros(3), 0.0, False
                if adv > 0.03:
                    state["no_surface"] = True
                    return np.zeros(3), 0.0, True
                return nrm * self.speed_near * dt, 0.0, False
            if (self.search_fall_rule and adv - min(window) > drop and not supported
                    and float(np.mean(flwin)) < 4.0):   # really fell in
                state["dropped"] = True
                return np.zeros(3), 0.0, True
            # "blocked" = the target moves but the hand does not: over the last 1 s the hand moved < 2 mm sideways while the
            # target moved > 8 mm
            lat_pos = s["pos"] - adv * nrm
            state["reach"] = max(state["reach"], float(np.linalg.norm(lat_pos - p0)))
            tgt_lat = s["target"] - np.dot(s["target"] - p0, nrm) * nrm
            poshist.append(lat_pos.copy()); tgthist.append(tgt_lat.copy())
            nwin = max(2, int(1.0 / dt))
            if len(poshist) > nwin:
                poshist.pop(0); tgthist.pop(0)
            moved = float(np.linalg.norm(poshist[-1] - poshist[0])); tmoved = float(np.linalg.norm(tgthist[-1] - tgthist[0]))
            blocked_now = len(poshist) >= nwin and moved < 0.002 and tmoved > 0.008
            state["blocked"] = state["blocked"] + 1 if blocked_now else 0
            if (adv > drop and (blocked_now or self.search_sink_is_drop)
                    and not (self.search_press is not None and state["phase"] in ("settle", "unpress"))):   # already in the hole (held sideways by the hole wall)
                state["dropped"] = True
                return np.zeros(3), 0.0, True
            if adv < -0.015:
                state["lost"] = True
                return np.zeros(3), 0.0, True
            ph = state["phase"]
            # more than radius_max + 1 cm from the start: out of the search range (the force guidance led it astray) -> stop
            # before sliding off the edge of the part
            if float(np.linalg.norm(lat_pos - p0)) > radius_max + 0.01:      # lat_pos already has p0's normal component
                state["out"] = True
                return np.zeros(3), 0.0, True
            if ph == "spiral" and self.search_settle is not None:
                # the reference is the surface it slid on 0.5-2 s ago, not the start height: the surface need not be level and
                # the tool tilts a little, so the start height drifts by 0.1-0.15 mm, enough to miss a shallow dip
                pause_m = self.search_settle[0]
                hist = state["advhist"]
                hist.append((s["t"], adv))
                while hist and hist[0][0] < s["t"] - 2.0:
                    hist.pop(0)
                past = [a for tt, a in hist if tt <= s["t"] - 0.5]
                ref = float(np.median(past)) if past else 0.0
                if adv - ref > pause_m and (state["settle_from"] is None
                                            or float(np.linalg.norm(lat_pos - state["settle_from"])) > 0.002):
                    state.update(phase="settle", t_phase=s["t"], settle_from=lat_pos.copy(), settles=state["settles"] + 1,
                                 settle_ref=ref)
                    ph = "settle"
            if ph == "unpress":
                # ease the press lead back under the normal lead before sliding on: dropping the robot-side contact lead at once,
                # with the target ~20 mm ahead, would break either the rate limit or the lead limit
                if float(np.dot(s["target"] - s["pos"], nrm)) <= 0.006:
                    self.lead_override = None
                    if hasattr(self.sim, "step_lead"):
                        self.sim.step_lead = None
                    state["phase"] = "spiral"
                    return state["settle_from"] - tgt_lat, 0.0, False
                return state["settle_from"] - tgt_lat - nrm * self.speed_free * dt, 0.0, False
            if ph == "settle":
                # the tip sank: stop sliding (the spiral would drag it back out) and keep pressing, holding the spot sideways
                # (a pressed part that is free to move sideways tips over in the grip)
                _, hole_m, hold_s, press_n = self.search_settle
                press = self.search_press
                if s["t"] - state["t_phase"] >= hold_s:
                    # a hard press sinks into a flat spot too (fingers, grip and surface give ~0.9 mm at 15 N), so after a press
                    # only its own, deeper threshold counts
                    need = min(drop, hole_m) if press is None else press[2]
                    if adv - state["settle_ref"] >= need:  # in: the finish ramps the press back
                        state.update(dropped=True, settled_in=round(adv - state["settle_ref"], 4))
                        return np.zeros(3), 0.0, True
                    state["phase"] = "spiral" if press is None else "unpress"   # a dent, not a hole: slide on
                    return state["settle_from"] - tgt_lat, 0.0, False
                if press is None:
                    vn = float(np.clip(self.kf * (max(normal_force, press_n) - f_n), -self.speed_near, self.speed_near))
                else:
                    # press it in at once: the lead may grow to lead_m (the robot-side contact lead) along the axis, the target
                    # moving down at speed_free until press_n or the raw cap, then held
                    lead_m, raw_cap, _ = press
                    self.lead_override = lead_m
                    if hasattr(self.sim, "step_lead"):
                        self.sim.step_lead = lead_m
                    held, now = self._raw_hold(s, state, raw_cap) if raw_cap is not None else (False, 0.0)
                    vn = self.speed_free if (f_n < max(normal_force, press_n) and not held) else 0.0
                    if raw_cap is not None and now >= raw_cap + 1.0:      # back off once 1 N above the cap
                        vn = -self.speed_near
                d_lat = state["settle_from"] - tgt_lat
                return d_lat + nrm * vn * dt, 0.0, False
            if ph == "spiral":
                # a corner caught on the rim = side force above f_catch for 0.3 s AND almost no lateral motion in those 0.3 s
                # (a friction spike does not block the hand, the rim does)
                n3 = max(2, int(0.3 / dt))
                caught = (supported and len(flwin) >= n3 and min(flwin[-n3:]) > f_catch and len(poshist) >= n3
                          and float(np.linalg.norm(poshist[-1] - poshist[-n3])) < 0.001
                          and state["hops"] < self.search_guided_hops)
                if caught:
                    state.update(phase="lift", t_phase=s["t"], guide=F_lat / fl)
                    state["hops"] += 1
                    return np.zeros(3), 0.0, False
                if state["blocked"] >= int(1.5 / dt):                  # blocked for 1.5 s (beyond the 1 s window)
                    state["stuck"] = True
                    return np.zeros(3), 0.0, True
                if not supported:                                   # slid over the hole: freeze laterally, only descend
                    return nrm * self.speed_near * 1.5 * dt, 0.0, False
                vn = float(np.clip(self.kf * (normal_force - f_n), -self.speed_near, self.speed_near))
                r_now = pitch * state["theta"] / (2 * math.pi)
                state["theta"] += v * dt / max(r_now, pitch / 4)
                r = pitch * state["theta"] / (2 * math.pi)
                if r > radius_max:
                    return np.zeros(3), 0.0, True
                off = r * (math.cos(state["theta"]) * e1 + math.sin(state["theta"]) * e2)
                d_lat = state["center"] + off - s["target"]
                d_lat -= np.dot(d_lat, nrm) * nrm
                return d_lat + nrm * vn * dt, 0.0, False
            if ph == "lift":                                         # lift 4 mm (0.4 s)
                if s["t"] - state["t_phase"] >= 0.4:
                    state.update(phase="move", t_phase=s["t"])
                return -nrm * 0.01 * dt, 0.0, False
            if ph == "move":                                         # move guide_step along the rim's reaction
                if s["t"] - state["t_phase"] >= guide_step / v:
                    state.update(phase="descend", t_phase=s["t"])
                return state["guide"] * v * dt, 0.0, False
            # descend: back down to the surface; once supported again, restart the spiral at a small radius around this spot
            if supported and s["t"] - state["t_phase"] > 0.2:
                state.update(phase="spiral", center=s["pos"] - adv * nrm, theta=2 * math.pi * 0.5)
                return np.zeros(3), 0.0, False
            return nrm * self.speed_near * dt, 0.0, False

        def search_note(rep, adv):
            note = ("the hand itself reached %.1f mm from the start (spiral target %.1f mm); friction keeps it a few "
                    "mm behind the target" % (state["reach"] * 1000, pitch * state["theta"] / (2 * math.pi) * 1000))
            off = float(np.linalg.norm((np.array(rep.end_pos) - p0) - adv * nrm))
            if state["dropped"]:
                note += "; the hole was found %.1f mm from where the search started" % (off * 1000)
                if off > 0.003:
                    note += (" - if the start was aimed at the measured hole centre this is probably not that hole: "
                             "check the photos before turning")
            return note

        def derive(rep):
            adv = float(np.dot(np.array(rep.end_pos) - p0, nrm))
            return dict(dropped=state["dropped"], advance_along_normal_m=round(adv, 4),
                        lateral_offset_at_end_m=np.round((np.array(rep.end_pos) - p0) - adv * nrm, 4).tolist(),
                        radius_searched_m=round(pitch * state["theta"] / (2 * math.pi), 4), force_guided_moves=state["hops"],
                        **({} if self.search_settle is None else
                           dict(sink_pauses=state["settles"], found_by_pause_at_m=state["settled_in"],
                                radius_reached_m=round(state["reach"], 4),
                                found_from_start_m=(round(float(np.linalg.norm((np.array(rep.end_pos) - p0) - adv * nrm)), 4)
                                                    if state["dropped"] else None),
                                note=search_note(rep, adv))))

        try:
            # 8 mm lead: sideways the hand is only about 1 N/mm stiff, so 2 mm cannot overcome friction and the spiral spins in
            # place; caught at 8 mm the side force is ~8-10 N
            rep = self._run("search", dict(normal=nrm.round(3).tolist(), radius_max=radius_max, pitch=pitch, normal_force=normal_force, drop=drop, mode="slide"),
                            ctrl, (radius_max ** 2 * math.pi / pitch) / v + 20.0, stop_on_contact=False, derive=derive,
                            compliant=True, soft_dir=nrm, max_lead=0.008)
        finally:
            self.lead_override = None
        if state["dropped"]:
            rep.termination = "dropped"
        elif state["no_surface"]:
            rep.termination = "no_surface"
        elif state["stuck"]:
            rep.termination = "stuck"
        elif state.get("out"):
            rep.termination = "out_of_range"
        elif state["lost"]:
            rep.termination = "lost_contact"
        self._estimate(rep)
        return rep

    def _search_hop(self, nrm: np.ndarray, radius_max: float, pitch: float, drop: float, speed: Optional[float], hop: float) -> ContactReport:
        a = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = _unit(a - np.dot(a, nrm) * nrm)
        e2 = np.cross(nrm, e1)
        v_lat = speed or self.speed_free
        v_n = self.speed_near * 0.5
        p0 = self.sim.eef_pos
        # spiral sample points, spaced about pitch along the arc
        pts = [np.zeros(3)]
        theta, r = 0.0, 0.0
        while r < radius_max:
            theta += pitch / max(pitch * theta / (2 * math.pi), pitch / 2)
            r = pitch * theta / (2 * math.pi)
            pts.append(r * (math.cos(theta) * e1 + math.sin(theta) * e2))
        st = dict(i=0, phase="up", dropped=False, probes=0, depth_max=0.0, last_contact_adv=0.0)
        kc, kf = self.kp_contact, self.kp_free
        kp_up = np.concatenate([kf - (kf - kc) * np.abs(nrm), np.ones(3) * kf])     # moving sideways: soft along the normal, stiff tangentially (accurate positioning)
        # probing down stays stiff tangentially too: with all three axes soft the null-space term dragged the tool 1-2 cm
        # sideways and the sample points went astray
        kp_down = kp_up
        has_kp = hasattr(self.sim, "set_kp")

        def ctrl(s):
            adv = float(np.dot(s["pos"] - p0, nrm))
            st["depth_max"] = max(st["depth_max"], adv)
            # "dropped" = suddenly drop deeper than the last probed depth (not the start: a held part pushed again and again
            # slowly slides up in the grip, so the start drifts)
            if adv - st["last_contact_adv"] > drop:
                st["dropped"] = True
                return np.zeros(3), 0.0, True
            if st["i"] >= len(pts):
                return np.zeros(3), 0.0, True
            tgt_lat = p0 + pts[st["i"]]
            if st["phase"] == "up":            # rise to hop above the sample point
                goal = tgt_lat - nrm * hop
                err = goal - s["target"]
                lat_err = goal - s["pos"]
                lat_err -= np.dot(lat_err, nrm) * nrm
                if np.linalg.norm(lat_err) < 1.5e-3 and np.linalg.norm(err) < 2e-3:   # measured lateral position arrived (the target leads the hand, so the target alone is not enough)
                    st["phase"] = "down"
                    if has_kp:
                        self.sim.set_kp(kp_down)
                    return np.zeros(3), 0.0, False
                if np.linalg.norm(err) < 1e-6:
                    return np.zeros(3), 0.0, False
                return err / np.linalg.norm(err) * min(np.linalg.norm(err), v_lat * self.sim.dt), 0.0, False
            # down: probe gently along normal; on contact (reaction above threshold) record the depth and go to the next point
            f_n = float(np.dot(s["reaction"], -nrm))
            if f_n >= self.f_contact:
                st["probes"] += 1
                st["i"] += 1
                st["phase"] = "up"
                st["last_contact_adv"] = adv
                if has_kp:
                    self.sim.set_kp(kp_up)
                return np.zeros(3), 0.0, False
            return nrm * v_n * self.sim.dt, 0.0, False

        def derive(rep):
            adv = float(np.dot(np.array(rep.end_pos) - p0, nrm))
            return dict(dropped=st["dropped"], probes=st["probes"], advance_along_normal_m=round(adv, 4),
                        max_depth_m=round(st["depth_max"], 4), radius_searched_m=round(float(np.linalg.norm(pts[min(st["i"], len(pts) - 1)])), 4),
                        lateral_offset_at_end_m=np.round((np.array(rep.end_pos) - p0) - adv * nrm, 4).tolist())

        n_pts = len(pts)
        max_time = n_pts * (2 * hop / v_n + 0.5) + 5.0
        rep = self._run("search", dict(normal=nrm.round(3).tolist(), radius_max=radius_max, pitch=pitch, drop=drop, mode="hop", hop=hop),
                        ctrl, max_time, stop_on_contact=False, derive=derive, compliant=True, soft_dir=nrm, max_lead=0.02)
        if st["dropped"]:
            rep.termination = "dropped"
        return rep

    def insert(self, depth: float = 0.02, normal: Sequence[float] = (0, 0, -1), max_force: float = 12.0,
               speed: Optional[float] = None, max_time: Optional[float] = None, wiggle: float = 0.002) -> ContactReport:
        """Compliant insertion: soft tangentially (kp_contact, so the part centres itself), stiff along the normal.

        Advances along normal at a speed that falls linearly with the normal reaction force (zero at max_force, backing
        off above it), with a small circular wiggle.  Advance >= depth -> seated; less than 1 mm of progress in 1 s ->
        jammed if the normal force is high, otherwise stalled (bottomed out, or the part slipping in the fingers)."""
        nrm = _unit(normal)
        v = speed or self.speed_near
        max_force = float(min(max_force, self.f_cap * 0.8))
        p0 = self.sim.eef_pos
        hist: List[tuple] = []
        state = dict(seated=False, jammed=False, stalled=False)
        a_ = np.array([1.0, 0, 0]) if abs(nrm[0]) < 0.9 else np.array([0, 1.0, 0])
        e1 = _unit(a_ - np.dot(a_, nrm) * nrm)
        e2 = np.cross(nrm, e1)
        if max_time is None:
            max_time = depth / v * 3.0 + 3.0

        def ctrl(s):
            adv = float(np.dot(s["pos"] - p0, nrm))
            hist.append((s["t"], adv))
            if adv >= depth:
                state["seated"] = True
                return np.zeros(3), 0.0, True
            f_n = float(np.dot(s["reaction"], -nrm))
            recent = [a for (t, a) in hist if t >= s["t"] - 1.0]
            if s["t"] > 1.0 and len(recent) >= 10 and (adv - min(recent)) < 0.001:
                state["jammed" if f_n >= 0.5 * max_force else "stalled"] = True
                return np.zeros(3), 0.0, True
            vn = float(np.clip(v * (1.0 - f_n / max_force), -v, v))
            # small circles (1 Hz, radius wiggle): the classic insertion trick that lets a caught corner find the entry; the
            # tangential axes are soft, so this builds no large force
            w = 2 * math.pi * s["t"]
            lat_goal = p0 + wiggle * (math.cos(w) * e1 + math.sin(w) * e2) if wiggle > 0 else p0
            d_lat = lat_goal - s["target"]
            d_lat -= np.dot(d_lat, nrm) * nrm
            return d_lat * 0.5 + nrm * vn * self.sim.dt, 0.0, False

        kc, kf_ = self.kp_contact, self.kp_free
        kp6 = np.concatenate([kc + (kf_ - kc) * np.abs(nrm), np.ones(3) * kf_])     # stiff along the normal, soft tangentially

        def derive(rep):
            arr = rep.wrench_array()
            adv = float(np.dot(np.array(rep.end_pos) - p0, nrm))
            lat_mag = 0.0
            if len(arr) >= 3:
                F = arr[-5:, 1:4] - self.carry_bias
                lat = F - np.outer(F @ nrm, nrm)
                lat_mag = float(np.linalg.norm(lat.mean(axis=0)))
            return dict(seated=state["seated"], jammed=state["jammed"], stalled=state["stalled"], advance_along_normal_m=round(adv, 4),
                        lateral_force_end_N=round(lat_mag, 2))

        rep = self._run("insert", dict(depth=depth, normal=nrm.round(3).tolist(), max_force=max_force), ctrl, max_time,
                        stop_on_contact=False, derive=derive, kp_vec=kp6, max_lead=0.002)
        if state["jammed"]:
            rep.termination = "jammed"
        elif state["stalled"]:
            rep.termination = "stalled"
        elif state["seated"]:
            rep.termination = "seated"
        self._estimate(rep)
        return rep

    def pull(self, direction: Sequence[float], force: float = 8.0, distance: float = 0.10,
             speed: Optional[float] = None, max_time: Optional[float] = None) -> ContactReport:
        """Pull along a constraint: for something held that is attached to a mechanism (door handle, drawer).

        Admittance along the pulling direction regulates the pull to force.  In the horizontal plane the component
        perpendicular to the pull follows the force (when the mechanism pushes the hand sideways the hand gives way and
        the target stores no spring force), the pulling direction is corrected toward the side the constraint force
        leans to, and the wrist yaw stores no spring either and turns with the handle.  The mechanism's constraint thus
        leads the hand along the path it allows (an arc for a door, a line for a drawer), and the direction from the
        model only has to be roughly right.  Height is held (full stiffness).

        Results: completed after distance along the path; stalled when the hand barely moves for 1 s while the pull is
        at its target (end of travel, locked, or slipping); joint_limit when the arm cannot follow any further; the
        safety caps apply as usual.  The report gives the turn of the path, the fitted radius and pivot, the final
        free direction and the forces.

        Rationale: at a door handle the door normal and the direction of motion differ by 16-17 deg; with push or
        move_by, which are stiff sideways, a 16 deg error halves the progress and 46 deg blocks the door entirely.
        """
        sim = self.sim
        d0 = np.asarray(direction, dtype=float).reshape(3).copy()
        d0[2] = 0.0
        d = _unit(d0)
        force = float(min(force, self.f_cap * 0.8))
        v_max = float(speed or 0.03)
        if max_time is None:
            max_time = distance / v_max * 2.5 + 3.0
        kf = self.kf * 3.0
        zax = np.array([0.0, 0.0, 1.0])
        p0 = sim.eef_pos
        st = dict(d=d.copy(), path=0.0, last=p0.copy(), mark=p0.copy(), hist=[], ds=[], fp=[], stalled=False, turned=0.0)

        def ctrl(s):
            pos = s["pos"]
            step = pos - st["last"]
            step[2] = 0.0
            st["path"] += float(np.linalg.norm(step))
            st["last"] = pos.copy()
            st["hist"].append((s["t"], pos.copy()))
            dcur = st["d"]
            lat = np.cross(zax, dcur)
            # optional tare of the static sideways force already on the hand when the pull starts (e.g. a pressed spring lever
            # pushing the hand at an angle); never along the pull: on a retry the pull left over from the previous attempt must
            # not be subtracted as "static", or the target force would stack on top of it (untared, a 2-3 N sideways spring
            # force turned the pull ~88 deg off; tared as a whole vector, a retry overshot to 16 N)
            if "f0" not in st:
                st["f0"] = np.asarray(s["reaction"], dtype=float).copy() if self.pull_tare_lateral else np.zeros(3)
            react = np.asarray(s["reaction"], dtype=float)
            f_pull = float(np.dot(react, -dcur))
            f_lat = float(np.dot(react - st["f0"], lat))
            st["fp"].append((s["t"], f_pull))
            # The pulling direction adapts by force, not by displacement (correcting by displacement fails: once the direction is
            # off the hand slips along the handle bar, the slip looks like the door moving, and the correction runs away).  The
            # free direction lies on the side the constraint's reaction leans to, so each tick the direction turns toward it by
            # 10% of the reaction's angle (at most 3 deg per tick).
            # Corrections start only once the mechanism moves (pull_adapt_after_m of travel): before static friction breaks,
            # stray sideways forces say nothing about the constraint (a threshold on instantaneous speed would never correct a
            # slow door).  They also need a real pull, f_pull >= max(1 N, a quarter of the target): once a door gives way the
            # pull is small or even negative, a little sideways noise would read as a +-90 deg angle and the direction would turn
            # past the path until the hand pushes into the hinge.  Per tick the direction turns at most (horizontal hand travel
            # this tick) / pull_r_min: a mechanism path bends no more sharply than a circle of that radius, so its tangent
            # cannot turn faster.
            step_len = float(np.linalg.norm(step[:2]))
            if (st["path"] >= self.pull_adapt_after_m and float(np.hypot(f_pull, f_lat)) >= 1.5
                    and f_pull >= max(1.0, 0.25 * force)):
                psi = math.atan2(f_lat, f_pull)
                lim = min(math.radians(3), step_len / self.pull_r_min)
                dphi = float(np.clip(0.10 * psi, -lim, lim))
                c, s_ = math.cos(dphi), math.sin(dphi)
                dcur = _unit(np.array([c * dcur[0] - s_ * dcur[1], s_ * dcur[0] + c * dcur[1], 0.0]))
                st["d"] = dcur
                st["turned"] += dphi
                lat = np.cross(zax, dcur)
            st["ds"].append(dcur.copy())
            # no spring in wrist yaw: the target yaw follows the measured yaw every tick, so the wrist turns as much as the
            # handle does (it is not twisted to follow changes of the pulling direction)
            dyaw = math.radians(_yaw_deg_xyzw(sim.eef_quat) - _yaw_deg_xyzw(self.target_quat))
            v = float(np.clip(kf * (force - f_pull), -v_max, v_max))
            vl = float(np.clip(kf * f_lat, -v_max, v_max))
            follow = lat * float(np.dot(pos - s["target"], lat))      # sideways target follows the hand: no spring force stored sideways
            done = st["path"] >= distance
            # stop when the arm runs out of reach: a door opening toward the robot drives the elbow (joint 4) into its limit past
            # about 35 deg, the arm can no longer follow and the gripper wrenches the handle (19-28 N); a real Franka also stops
            # before a joint limit.  When the pull starts right at a limit (e.g. after a turn pressed the handle all the way down)
            # it stops only if the margin keeps shrinking, otherwise the pull that cracks the door open would be stopped at once
            margin = _arm_limit_margin(sim)
            if margin is not None and "m0" not in st:
                st["m0"] = margin
            if not done and margin is not None and margin < min(math.radians(1.0), st.get("m0", margin) - 0.003):
                st["joint_limit"] = True
                done = True
            if not done and s["t"] > 1.5:
                recent = [p for (t, p) in st["hist"] if t >= s["t"] - 1.0]
                # "cannot pull further" needs the pull (0.5 s mean) at 95% of the target: a lower bar (a single tick at 70%, or 90%)
                # reported stalled while the force was still rising toward the target
                f_recent = [f for (t, f) in st["fp"] if t >= s["t"] - 0.5]
                if (len(recent) >= 10 and float(np.linalg.norm((recent[-1] - recent[0])[:2])) < 0.003
                        and float(np.mean(f_recent)) >= 0.95 * force):
                    st["stalled"] = True
                    done = True
            return dcur * v * sim.dt + lat * vl * sim.dt + follow, dyaw, done

        kc, kfree = self.kp_contact, self.kp_free
        kp6 = np.array([kc, kc, kfree, kfree, kfree, 10.0])     # soft horizontally, stiff vertically; wrist yaw very soft (and its target follows the measured yaw)

        def derive(rep):
            arr = rep.wrench_array()
            pos = np.asarray(rep.pos_rows, dtype=float).reshape(-1, 3) if rep.pos_rows else np.zeros((0, 3))
            # the turn of the path and the final free direction come from the path the hand actually travelled (direction of the
            # first / last 1.5 cm), not from the corrections of the pulling direction (which include the initial alignment)
            turned, final_dir = 0.0, st["d"]
            if len(pos) >= 4 and st["path"] >= 0.03:
                xy = pos[:, :2]
                seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
                i1 = int(np.searchsorted(seg, 0.015)); i2 = int(np.searchsorted(seg, seg[-1] - 0.015))
                a_ = xy[min(i1, len(xy) - 1)] - xy[0]; b_ = xy[-1] - xy[max(i2, 0)]
                if np.linalg.norm(a_) > 1e-4 and np.linalg.norm(b_) > 1e-4:
                    turned = math.atan2(a_[0] * b_[1] - a_[1] * b_[0], float(a_ @ b_))
                    final_dir = np.array([*(b_ / np.linalg.norm(b_)), 0.0])
            out = dict(path_length_m=round(st["path"], 4), turned_deg=round(math.degrees(turned), 1),
                       final_direction=np.round(final_dir, 3).tolist(), pull_direction_corrected_deg=round(math.degrees(st["turned"]), 1),
                       stalled=st["stalled"])
            ds = np.asarray(st["ds"], dtype=float)
            if len(arr) >= 4 and len(ds) >= 4:
                n = min(len(arr), len(ds))
                F = arr[-n:, 1:4] - self.carry_bias
                D = ds[-n:]
                fp = -np.einsum("ij,ij->i", F, D)
                lat = np.cross(np.tile(zax, (n, 1)), D)
                fl = np.einsum("ij,ij->i", F, lat)
                tail = slice(n // 5, n)
                out.update(pull_force_mean_N=round(float(fp[tail].mean()), 2), pull_force_std_N=round(float(fp[tail].std()), 2),
                           lateral_force_mean_N=round(float(np.abs(fl[tail]).mean()), 2))
            if len(pos) >= 8 and abs(turned) >= math.radians(4) and st["path"] >= 0.03:
                x, y = pos[:, 0], pos[:, 1]
                A = np.column_stack([x, y, np.ones_like(x)])
                b = -(x ** 2 + y ** 2)
                (D_, E_, F_), *_ = np.linalg.lstsq(A, b, rcond=None)
                cx, cy = -D_ / 2, -E_ / 2
                r = math.sqrt(max(cx * cx + cy * cy - F_, 0.0))
                resid = float(np.sqrt(np.mean((np.hypot(x - cx, y - cy) - r) ** 2)))
                out.update(pivot_xy=[round(float(cx), 3), round(float(cy), 3)], radius_m=round(float(r), 3), radius_fit_rms_m=round(resid, 4))
            return out

        # the hand already holds the handle: a rising pull is not a new contact, so contact is not reported (the safety caps
        # still apply)
        rep = self._run("pull", dict(direction=np.round(d, 3).tolist(), force=force, distance=distance), ctrl, max_time,
                        stop_on_contact=False, derive=derive, kp_vec=kp6, max_lead=0.03, grip=1.0 if self.grip_closed else None,
                        suppress_contact=self.grip_closed)
        if st.get("joint_limit"):
            rep.termination = "joint_limit"
            rep.derived["note"] = ("stopped: the arm reached a joint limit and cannot follow the mechanism any further "
                                   "(out of reach), not a blockage of the mechanism")
        elif st["stalled"]:
            rep.termination = "stalled"
        self._estimate(rep)
        return rep

    def turn(self, axis: Sequence[float], pivot: Sequence[float], angle_deg: float, rate_deg_s: float = 10.0,
             torque_cap: Optional[float] = None, stop_on_resistance_drop: bool = False) -> ContactReport:
        """Turn the hand (and whatever it holds) rigidly by angle_deg about the axis through pivot with direction axis
        (right-hand rule): the TCP moves on the circle around the axis and the tool orientation turns with it.

        For lever handles and knobs: pushing a firmly held lever straight down instead makes it tilt in the fingers
        while the wrist orientation is locked (12 N of push turned it by only 7 deg).  Under the force interface
        torque_cap and stop_on_resistance_drop (stop when the resistance passes a peak and falls = a "click") are
        available; the other interfaces turn to the end (the safety caps apply).  Full stiffness."""
        if self.interface != "force":
            torque_cap, stop_on_resistance_drop = None, False
        ax = _unit(axis)
        pv = np.asarray(pivot, dtype=float).reshape(3)
        total = math.radians(float(angle_deg))
        sgn = 1.0 if total >= 0 else -1.0
        rate = math.radians(float(rate_deg_s))
        p0 = self.sim.eef_pos
        q0 = R.from_quat(self.target_quat)
        acc = [0.0]
        old_tcap = self.detector.t_cap
        if torque_cap:
            self.detector.t_cap = float(torque_cap)
        # the "click" is judged on the total resisting moment about the turn axis, not on |F|: the hand holds the lever
        # rigidly, so part of the lever's resistance arrives as wrist torque (judged on |F| alone the click was diluted by
        # more than half).  Resisting moment = ax.T_sensor - ax.(r x F_reaction), positive = opposing the turn; the sign was
        # calibrated on the latch spring of the door_latch task.
        mom = dict(buf=[], peak=0.0, peak_deg=0.0, below=0, click=False, click_deg=None, hist=[])

        def ctrl(s):
            if abs(acc[0]) < abs(total) - 1e-9:
                acc[0] += sgn * min(rate * self.sim.dt, abs(total) - abs(acc[0]))
            rot = R.from_rotvec(ax * acc[0])
            goal = pv + rot.apply(p0 - pv)
            self.target_quat = (rot * q0).as_quat()
            m_res = sgn * (float(ax @ np.asarray(s["torque"], dtype=float)) - float(ax @ np.cross(s["pos"] - pv, s["reaction"])))
            mom["buf"] = (mom["buf"] + [m_res])[-4:]
            m_s = float(np.mean(mom["buf"]))
            ang = math.degrees(abs(acc[0]))
            mom["hist"].append((round(ang, 1), round(m_s, 3)))
            # no peak in the first 0.6 s: the start-up transient overshoots and falls back (it was taken for a "click" at 5-7 deg)
            if s["t"] < 0.6:
                pass
            elif m_s > mom["peak"]:
                mom["peak"], mom["peak_deg"], mom["below"] = m_s, ang, 0
            elif mom["peak"] > 0 and (mom["peak"] - m_s) >= max(0.15, 0.2 * mom["peak"]):
                mom["below"] += 1
                if mom["below"] >= 2 and not mom["click"]:
                    mom["click"], mom["click_deg"] = True, ang
            else:
                mom["below"] = 0
            if stop_on_resistance_drop and mom["click"]:
                return np.zeros(3), 0.0, True
            settled = float(np.linalg.norm(goal - s["pos"])) < 0.003 or s["t"] > abs(total) / rate + 1.0
            return goal - s["target"], 0.0, abs(acc[0]) >= abs(total) - 1e-9 and settled

        def derive(rep):
            v0 = np.asarray(rep.start_pos, dtype=float) - pv
            v1 = np.asarray(rep.end_pos, dtype=float) - pv
            v0 -= (v0 @ ax) * ax
            v1 -= (v1 @ ax) * ax
            turned = math.degrees(math.atan2(float(ax @ np.cross(v0, v1)), float(v0 @ v1))) if np.linalg.norm(v0) > 1e-4 else 0.0
            out = dict(turned_deg=round(turned, 1), commanded_deg=round(math.degrees(acc[0]), 1), radius_m=round(float(np.linalg.norm(v0)), 3))
            if self.interface == "force":                       # force information only under the force interface (it must not appear in pose / guard results)
                h = mom["hist"]
                out.update(resisting_moment_peak_Nm=round(mom["peak"], 3), resisting_moment_peak_at_deg=round(mom["peak_deg"], 1),
                           resisting_moment_end_Nm=(h[-1][1] if h else None), click_detected=mom["click"], click_at_deg=mom["click_deg"],
                           note="resisting moment about the turn axis; a peak followed by a clear drop is a 'click' (e.g. a latch giving way)")
            return out

        try:
            rep = self._run("turn", dict(axis=np.round(ax, 3).tolist(), pivot=np.round(pv, 4).tolist(), angle_deg=float(angle_deg),
                                         torque_cap=torque_cap, stop_on_resistance_drop=stop_on_resistance_drop),
                            ctrl, abs(total) / rate + 2.0, stop_on_contact=False,
                            grip=1.0 if self.grip_closed else None, derive=derive, suppress_contact=self.grip_closed)
        finally:
            self.detector.t_cap = old_tcap
        if stop_on_resistance_drop and mom["click"]:
            rep.termination = "resistance_drop"
        if rep.termination != "completed":                  # stopped midway (click / cap): leave the orientation target at the measured one, do not keep twisting
            self.target_quat = np.asarray(self.sim.eef_quat, dtype=float).copy()
        return rep

    def _wrist_turn_range(self):
        """Range of angles (deg, positive = counter-clockwise seen from above) rotate_z may still turn; None without a
        joint-7 reading.  A real-robot subclass may narrow it (e.g. for the wrist cable)."""
        j7 = getattr(self.sim, "wrist_joint_rad", None)
        if j7 is None:
            return None
        # predicted joint 7: d(j7) ~ -d(yaw); keep it within +-2.6 rad
        lo, hi = math.degrees(j7 - 2.6), math.degrees(j7 + 2.6)   # allowed range of -d(yaw) -> d(yaw) in [-hi, -lo]
        return -hi, -lo

    def rotate_z(self, angle_deg: float, torque_cap: Optional[float] = None, stop_on_resistance_drop: bool = False,
                 rate_deg_s: float = 30.0) -> ContactReport:
        """Rotate about the vertical axis by angle_deg (positive = counter-clockwise seen from above), clipped to the
        wrist joint range.  Under the force interface it can stop on torque_cap or when the resistance passes a peak
        and falls."""
        if self.interface != "force":
            torque_cap, stop_on_resistance_drop = None, False
        requested = float(angle_deg)
        clipped = False
        turn_range = self._wrist_turn_range()
        if turn_range is not None:
            a_min, a_max = turn_range
            if angle_deg < a_min or angle_deg > a_max:
                angle_deg = float(np.clip(angle_deg, a_min, a_max))
                clipped = True
        sgn = 1.0 if angle_deg >= 0 else -1.0
        total = abs(math.radians(angle_deg))
        rate = math.radians(rate_deg_s)
        acc = [0.0]
        old_tcap = self.detector.t_cap
        if torque_cap:
            self.detector.t_cap = float(torque_cap)

        def ctrl(s):
            if acc[0] >= total - 1e-9:
                # wait until the measured orientation catches up (or 0.5 s pass)
                return np.zeros(3), 0.0, abs(math.radians(s["yaw"] - yaw_start) - sgn * total) < math.radians(1.5) or s["t"] > total / rate + 0.5
            d = min(rate * self.sim.dt, total - acc[0])
            acc[0] += d
            return np.zeros(3), sgn * d, False

        yaw_start = _yaw_deg_xyzw(self.sim.eef_quat)

        def derive(rep):
            out = dict(rotated_deg=rep.yaw_deg)
            if clipped:
                out.update(wrist_limit_clipped=True, requested_deg=requested, executed_deg=angle_deg,
                           note="rotation clipped by the wrist joint limit; rotate the other way (angle ± 90/180) if you need more")
            arr = rep.wrench_array()
            if len(arr) < 4:
                return out
            tz = arr[:, 6]
            i = int(np.argmax(np.abs(tz)))
            out.update(torque_z_peak_Nm=round(float(tz[i]), 3), torque_z_peak_at_s=round(float(arr[i, 0]), 2),
                       torque_z_final_Nm=round(float(tz[-1]), 3))
            return out

        try:
            return self._run("rotate_z", dict(angle_deg=requested, torque_cap=torque_cap, stop_on_resistance_drop=stop_on_resistance_drop),
                             ctrl, total / rate + 1.5, stop_on_contact=False, stop_on_drop=stop_on_resistance_drop,
                             grip=1.0 if self.grip_closed else None, derive=derive)
        finally:
            self.detector.t_cap = old_tcap

    def point_tool(self, direction: Sequence[float], rate_deg_s: float = 30.0) -> ContactReport:
        """Re-orient the tool so the fingers point along direction; the TCP stays where it is.

        [0, 0, -1] = straight down (the default orientation); a horizontal direction lays the tool flat with the
        fingers pointing that way and opening up / down (to grasp a horizontal bar such as a door handle from the side;
        grasping it from above would make the 21 cm wide hand hit the door).  The finger opening axis is the direction
        perpendicular to direction that is closest to vertical (for a downward direction, the horizontal projection of
        the current opening axis).  Stops on contact (guard / force interfaces).  After a completed turn with nothing
        held and no contact the sensor is re-zeroed in the new orientation: the component of the gripper's own weight
        in the sensor frame changes with orientation and would otherwise add about 7 N once the tool is flat."""
        a = _unit(direction)
        r0 = R.from_quat(self.target_quat)
        y_cur = r0.apply([0.0, 1.0, 0.0])
        if abs(a[2]) > 0.95:                          # roughly down / up: keep the horizontal projection of the current opening axis
            pref = np.array([y_cur[0], y_cur[1], 0.0])
            if np.linalg.norm(pref) < 1e-3:
                pref = np.cross(a, [1.0, 0.0, 0.0])
        else:                                         # with a horizontal component: the opening axis closest to vertical, signed like the current one (less turning)
            pref = np.array([0.0, 0.0, 1.0]) * (1.0 if y_cur[2] >= 0 else -1.0)
        y = pref - float(pref @ a) * a
        y /= np.linalg.norm(y)
        x = np.cross(y, a)
        rg = R.from_matrix(np.column_stack([x, y, a]))
        ang = float((rg * r0.inv()).magnitude())
        T = max(ang / math.radians(rate_deg_s), 0.1)
        slerp = Slerp([0.0, 1.0], R.concatenate([r0, rg]))
        sim = self.sim
        # while turning, the component of the gripper's weight in the sensor frame changes: re-express the zero on every tick
        # from that weight, which is constant in the world frame; otherwise a few newtons (3.2 N measured) appear halfway and
        # trigger the contact stop
        grav = hasattr(sim, "raw_wrench") and hasattr(sim, "bias_force")
        if grav:
            fw0 = np.array(sim.raw_wrench()[2]) @ np.asarray(sim.bias_force, dtype=float)

        def regravity():
            if grav:
                sim.bias_force = np.array(sim.raw_wrench()[2]).T @ fw0

        def ctrl(s):
            regravity()
            u = min(1.0, s["t"] / T)
            self.target_quat = slerp([u])[0].as_quat()
            err = float((rg * R.from_quat(self.sim.eef_quat).inv()).magnitude())
            return np.zeros(3), 0.0, (u >= 1.0 and err < math.radians(2.0)) or s["t"] > T + 2.0

        def derive(rep):
            rm = R.from_quat(self.sim.eef_quat)
            z_now, y_now = rm.apply([0, 0, 1.0]), rm.apply([0, 1.0, 0])
            return dict(tool_direction=np.round(z_now, 3).tolist(), fingers_open_along=np.round(y_now, 3).tolist(),
                        direction_error_deg=round(math.degrees(math.acos(float(np.clip(z_now @ a, -1, 1)))), 1),
                        rotated_deg=round(math.degrees(ang), 1))

        rep = self._run("point_tool", dict(direction=np.round(a, 3).tolist()), ctrl, T + 3.0,
                        stop_on_contact=self._stop_on_contact(None), grip=1.0 if self.grip_closed else None, derive=derive)
        regravity()
        if rep.termination != "completed":            # stopped midway (contact / cap): leave the orientation target at the measured one so the wrist does not keep twisting
            self.target_quat = np.asarray(self.sim.eef_quat, dtype=float).copy()
        elif not self.grip_closed and not self.detector.in_contact and hasattr(self.sim, "rezero_bias"):
            self.sim.rezero_bias(10)
            self.target_pos = self.sim.eef_pos
            rep.derived["force_rezeroed"] = True
            rep.derived["note"] = "wrist force re-zeroed in the new orientation"
        return rep

    def lift(self, height: float = 0.08, hold_time: float = 1.0) -> ContactReport:
        """Rise vertically by height, then hold still for hold_time; derive a payload estimate (net z force over the
        still phase / g) and record the load as carry_bias."""
        p0 = self.sim.eef_pos
        v = self.speed_near
        phase_t = [None]

        def ctrl(s):
            if s["pos"][2] - p0[2] < height and phase_t[0] is None:
                return np.array([0, 0, v * self.sim.dt]), 0.0, False
            if phase_t[0] is None:
                phase_t[0] = s["t"]
            return np.zeros(3), 0.0, (s["t"] - phase_t[0]) >= hold_time

        def derive(rep):
            arr = rep.wrench_array()
            if len(arr) < 6:
                return {}
            tail = arr[-max(5, len(arr) // 4):]
            fz = float(tail[:, 3].mean())
            if self.grip_closed:
                self.carry_bias = tail[:, 1:4].mean(axis=0)
            txy = [float(tail[:, 4].mean()), float(tail[:, 5].mean())]
            return dict(hold_force_z_N=round(fz, 2), hold_force_z_std_N=round(float(tail[:, 3].std()), 2),
                        payload_estimate_kg=round(abs(fz) / G, 3),
                        hold_torque_xy_Nm=[round(txy[0], 3), round(txy[1], 3)], hold_torque_xy_mag_Nm=round(float(np.hypot(*txy)), 3),
                        note="a torque about a horizontal axis while holding still means the load's centre of mass is offset from the grasp")

        # while holding something, the load during the lift and the hover is not contact: contact is not reported for the
        # whole primitive (otherwise a 0.9 kg load was reported as contact_start when the hover began)
        return self._run("lift", dict(height=height, hold_time=hold_time), ctrl, height / v + hold_time + 2.0,
                         stop_on_contact=False, grip=1.0 if self.grip_closed else None, derive=derive,
                         suppress_contact=self.grip_closed)

    def probe(self, direction: Sequence[float] = (0, 0, -1), amplitude: float = 0.02,
              f_threshold: Optional[float] = None) -> ContactReport:
        """Tap: move along direction until contact (at most amplitude), then return to the start; reports whether and
        where contact was found."""
        u = _unit(direction)
        p0 = self.sim.eef_pos
        rep = self.move_until_contact(u, amplitude, f_threshold)
        contact_pos = self.sim.eef_pos
        back = self.move_to(*p0, stop_on_contact=False, max_time=4.0)
        rep.primitive = "probe"
        rep.params = dict(direction=u.round(3).tolist(), amplitude=amplitude)
        rep.end_pos = back.end_pos
        rep.derived.update(contact_found=(rep.termination == "contact"),
                           contact_distance_m=round(float(np.dot(contact_pos - p0, u)), 4),
                           contact_pos=np.round(contact_pos, 4).tolist())
        self._estimate(rep)
        return rep

    def grasp(self, steps: int = 16) -> ContactReport:
        """Close the gripper (resets carry_bias)."""
        self.carry_bias = np.zeros(3)
        return self._gripper("grasp", 1.0, steps)

    def release(self, steps: int = 16) -> ContactReport:
        """Open the gripper (resets carry_bias and drops a simulated weld, if any)."""
        self.carry_bias = np.zeros(3)
        if hasattr(self.sim, "set_held_weld"):
            try:
                self.sim.set_held_weld(False)
            except Exception:
                pass
        return self._gripper("release", -1.0, steps)

    def _gripper(self, name: str, cmd: float, steps: int) -> ContactReport:
        if not self.sim.task.has_gripper or self.sim.action_dim <= 6:
            rep = self.hold(0.1)
            rep.primitive, rep.termination, rep.error = name, "error", "this tool has no fingers"
            return rep
        self.grip_closed = cmd > 0

        def ctrl(s):
            return np.zeros(3), 0.0, s["step"] >= steps

        def derive(rep):
            op = self.sim.gripper_opening
            return dict(opening_m=round(op, 4) if op is not None else None,
                        closed_on_something=bool(cmd > 0 and op is not None and op > 0.006),
                        fully_closed=bool(cmd > 0 and op is not None and op <= 0.006))

        return self._run(name, dict(), ctrl, steps * self.sim.dt + 0.5, stop_on_contact=False, grip=cmd, derive=derive)

    def hold(self, seconds: float = 0.5) -> ContactReport:
        """Stay still for seconds while recording the wrench."""
        def ctrl(s):
            return np.zeros(3), 0.0, s["t"] >= seconds

        return self._run("hold", dict(seconds=seconds), ctrl, seconds + 0.5, stop_on_contact=False,
                         grip=1.0 if self.grip_closed else None, suppress_contact=self.grip_closed)

    # ------------------------------------------------------------------ dispatch
    # argument aliases models commonly use (e.g. dz for lift's height; a tool error there can derail an episode); unknown
    # arguments are ignored and listed in the report instead of raising an error
    ARG_ALIASES = {
        "lift": {"dz": "height", "z": "height", "height_m": "height", "distance": "height", "seconds": "hold_time", "hold": "hold_time", "duration": "hold_time"},
        "hold": {"duration": "seconds", "time": "seconds", "hold_time": "seconds"},
        "move_by": {"x": "dx", "y": "dy", "z": "dz"},
        "rotate_z": {"degrees": "angle_deg", "angle": "angle_deg", "deg": "angle_deg", "yaw_deg": "angle_deg", "dyaw_deg": "angle_deg"},
        "move_until_contact": {"dir": "direction", "distance": "max_dist", "max_distance": "max_dist", "threshold": "f_threshold", "force_threshold": "f_threshold"},
        "probe": {"dir": "direction", "distance": "amplitude", "max_dist": "amplitude", "threshold": "f_threshold"},
        "slide": {"dir": "direction", "force": "normal_force", "normal_force_N": "normal_force"},
        "push": {"dir": "direction", "force": "target_force", "target_force_N": "target_force", "seconds": "duration",
                 "stop_on_drop": "stop_on_resistance_drop", "stop_on_click": "stop_on_resistance_drop", "distance": "max_dist"},
        "search": {"radius": "radius_max", "max_radius": "radius_max"},
        "point_tool": {"dir": "direction", "toward": "direction", "approach": "direction", "vector": "direction"},
        "turn": {"angle": "angle_deg", "degrees": "angle_deg", "center": "pivot", "point": "pivot", "rate": "rate_deg_s",
                 "stop_on_drop": "stop_on_resistance_drop", "stop_on_click": "stop_on_resistance_drop"},
        "pull": {"dir": "direction", "target_force": "force", "force_N": "force", "distance_m": "distance", "dist": "distance", "max_dist": "distance"},
    }

    def call(self, name: str, **kwargs) -> ContactReport:
        """Run a model tool call: check that the tool exists under this interface (or is an allowed extra tool), map
        argument aliases, drop unknown arguments, and note both in report.derived["arg_notes"]."""
        extra_ok = name in self.extra_tools and (name not in FORCE_ONLY_EXTRA or self.interface == "force")
        if name not in INTERFACES[self.interface] and not extra_ok:
            raise ValueError(f"tool '{name}' is not available under interface '{self.interface}'")
        fn = getattr(self, name)
        sig = inspect.signature(fn).parameters
        aliases = self.ARG_ALIASES.get(name, {})
        clean: Dict[str, Any] = {}
        renamed, ignored = [], []
        for k, v in kwargs.items():
            k2 = k if k in sig else aliases.get(k)
            if k2 is not None and k2 in sig and k2 not in clean:
                clean[k2] = v
                if k2 != k:
                    renamed.append(f"{k}->{k2}")
            else:
                ignored.append(k)
        rep = fn(**clean)
        notes = []
        if renamed:
            notes.append("renamed argument(s) " + ", ".join(renamed))
        if ignored:
            notes.append("ignored unknown argument(s) " + ", ".join(ignored) + f" (valid: {', '.join(sig)})")
        if notes:
            rep.derived["arg_notes"] = "; ".join(notes)
        return rep
