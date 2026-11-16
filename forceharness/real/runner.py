"""The real-robot episode loop: a frozen model drives the FR3 through the primitive menu, one call per turn.

Every turn the runner photographs the scene (wrist RGB-D, exterior RGB), detects the objects on the table,
writes the observation to the relay folder and waits for the model's call (relay.py; the web monitor or a
person answers). It validates the call before anything moves (validation.py), runs one self-terminating
primitive, relaxes the controller and takes a still reading, and renders the tool result: condition A
(ForceHarness) adds the phys2 translation of the force trace and the still readings (force on the hand relative
to the hand at the start, the change since the grasp and the weight it implies); condition B (the no-force
control) has the same robot, the same guards and the same images, with tool results that carry no force and a
menu without the force-only tools. The task is either the operator's text (free task) or a fixed instruction
(tasks.py).

Guards that protect the robot stay as they are: the robot-side limits (8 mm/s, lead, 15 mm tracking, workspace
box, joint margins, per-motion envelopes), the raw 25 N cap (the episode ends), the 3 N*m torque cap (the motion
stops and the episode goes on with a tool result), the 8 N collision stop (12 N while holding), the 15/20 mm path
stop and the 30 mm fatal deviation, no gripper actuation in an unknown state, the hold renewal while the model
thinks with something gripped, and relax-and-read after every primitive. Motions the arm cannot follow are
refused before they start, with the largest feasible amount in the tool result.

A closed gripper at the start is an unknown held state and the episode does not start, unless the operator
names what it holds (--resume-held): the episode then starts holding it, as if it had just been grasped here.

    python scripts/real/free_task.py --instruction "..." --out runs/real/<name>/episode --relay-dir runs/real/<name>/relay
    (demo by default: synthetic plant with two synthetic bottles, scripted calls or --demo-model)
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
from functools import partial
import json
import math
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from forceharness.primitives.report import ContactEvent
from forceharness.translation.base import make_injector
from forceharness.real import posture_bias
from forceharness.real import posture_calibration as posture_cal
from forceharness.real import receipt_text as T
from forceharness.real import scene_objects, pixel_target
from forceharness.real.cameras import GripperCameras, camera_specs, live_photographer
from forceharness.real.checks import (HELD_TURN_COLLISION_N, TCP_TRANSLATION, TURN_FLANGE_REACH_M,  # noqa: F401
                                      TURN_JOINT_DELTA, TURN_JOINT_MARGIN, arc_angle_deg, hand_tilt_deg,
                                      rebased_quat, rotation_reach, translation_reach, turn_flange_chord, vector)
from forceharness.real.config import BackendConfig, load_robot_config
from forceharness.real.menu import (ACTION_RANGE, CONDITIONS, CONTACT_SEARCH_MAX_M, FORCE_LIMITS,  # noqa: F401
                                    FORCE_MAX_N, FORCE_ONLY_ARGS, GRIP_RULES, INSERT_FORCE_MAX_N, MAX_ACTIONS,
                                    MAX_MINUTES, MINUTE_RANGE, MOVE_MAX_M, PRIMITIVES, PULL_FORCE_DEFAULT_N,
                                    PULL_FORCE_MAX_N, PUSH_FORCE_MAX_N, RAW_FORCE_CAP_N, ROT_MAX_DEG,
                                    ROTATE_Z_MAX_DEG, SLIDE_FORCE_DEFAULT_N, TARGET_FORCE_CEILING_N, TASK_ONLY,
                                    action_prompt, menu_text, signature)
from forceharness.real.primitives import (SCREW_PUSH_N, WRIST_Q7_RAD, ContinuousPrimitives,  # noqa: F401
                                          screw_quantities)
from forceharness.real.relax import STEP_S, relax_and_read
from forceharness.real.relay import OperatorConfirm, RelayClient, atomic_json
from forceharness.real.rotation_progress import RotationProgress, TrackingProgress
from forceharness.real.types import BackendError
from forceharness.real.validation import grasp_reach, unit, validate  # noqa: F401
from forceharness.real.wrist_overlay import use_hand_eye

ROOT = Path(__file__).resolve().parents[2]
PROMPT_DIR = ROOT/'forceharness/prompts/real'
VERSION = 'free_task/1'
# Upper bound on a task instruction (the longest shipped one, screw insertion, is about 8400 characters).
MAX_INSTRUCTION_CHARS = 16000
FREE_COLLISION_N = 8.
EMPTY_CLOSE_M = .008      # closed width at or below this: the fingers met, nothing is held
RESUME_MIN_M = .001       # --resume-held floor (a tool clamped on its flat faces can read under 2 mm)
G = 9.81
# The FR3 force is estimated from joint torques and moves by 2-4 N with the hand's pose and approach direction
# (the same pose reached differently differed by up to 2.9 N; a 6.2 g eraser read hundreds of grams after a
# grasp and lift), so every absolute weight carries this warning.
WEIGHT_ERROR_G = 400
REBASE_MIN_RADIUS_M = .02  # closer to the turn axis, the TCP angle around the pivot is noise
REBASE_MAX_DEG = 4.        # a larger start correction adds to the requested turn (0.60 rad robot-side envelope)
# The 3 N*m torque cap stops the motion as before; the episode then goes on with this tool result. A held lift
# after pressing on a surface can see the torque estimate climb by 2-3 N*m with nothing stuck.
TORQUE_STOP_NOTE = T.TORQUE_STOP
TORQUE_STOP_NOTE_B = T.TORQUE_STOP_B


# ---- backend and primitives -------------------------------------------------------------------------
class FreeGripper:
    """Gripper and force-scope rules of the runner, mixed over ContinuousBackend (the shared class is unchanged).

    - More than one grasp per episode: every close after the first opens a new gripper trial of the port
      (RobotPort.next_weigh_trial: each only after a confirmed release; the runner's port has no trial
      budget). The port keeps its one-close-per-trial ledger.
    - Fingers that met (<= 8 mm, read back after the close) are a known empty state, not a fault,
      even though the port reports the close as a failed grasp: the model may open again.
    - Any closed width keeps the force readings usable with the calibrated closure correction
      (measured on a 29 mm bottle cap); the width is recorded. Raw caps never depend on this.
    Unknown states (lost reply, closed but >= 75 mm, open below 78 mm) still latch a fault and
    nothing is actuated again."""
    closed_empty = False
    width_outside_calibration = None

    def grasps_left(self):
        """Closes still possible in this episode (None: the port has no trial budget, as the runner's own
        port, or a test fixture)."""
        limit = getattr(self.port, 'max_trials', None)
        if limit is None or not hasattr(self.port, 'next_weigh_trial') or not hasattr(self.port, 'trial'):
            return None
        return (limit-self.port.trial) + (0 if self._grasp_attempted else 1)

    def command_gripper(self, closed, *, confirmed=False):
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
        if self.clock()-self._last_grip < 3:
            raise BackendError('Gripper interval (3 s) not elapsed')
        if closed and self._grasp_attempted and hasattr(self.port, 'next_weigh_trial'):
            # before any actuation; refuses unless the previous release was confirmed
            self.port.next_weigh_trial(getattr(self.port, 'trial', 0)+1)
        self._last_grip = self.clock()
        self._grasp_attempted |= closed
        self._closed = None      # lost reply => held state unknown, never retry/open
        try:
            try:
                self.port.grip(closed)
            except Exception as exc:
                if not closed:
                    raise
                self._request('state')       # the port rejects a close below its minimum width
                if self.gripper_opening > EMPTY_CLOSE_M:
                    raise
                self.empty_close_note = str(exc)
            self._request('state')
            width = self.gripper_opening
            if closed and width <= EMPTY_CLOSE_M:
                self.closed_empty = True
            elif (closed and not width < .075) or (not closed and width < .078):
                raise BackendError('Gripper width acceptance failed')
            else:
                self.closed_empty = False
            self._closed = closed
        except BaseException as exc:
            self.fault = 'Gripper outcome unknown: '+str(exc)
            raise

    def resume_holding(self):
        """The operator confirms that the closed gripper still holds the previous episode's object.

        Only from the unknown start state (closed below 78 mm) and only at a width that is a real grasp;
        returns the width. The empty-hand bias cannot be sampled while holding: readings use the calibrated
        bias with the closure correction, and the runner reads the held reaction still right away."""
        width = float(self.gripper_opening)
        if self.fault or self.active or self._closed is not None:
            raise BackendError('Resume holding only from a closed gripper at the start of a session')
        # with the width checks off, any closed reading counts (a flat tool can read about 0 mm)
        if not (RESUME_MIN_M if self.config.gripper_width_check else -math.inf) < width < .075:
            raise BackendError(f'A gripper width of {width*1000:.1f} mm does not look like holding something (it must '
                               f'be between {RESUME_MIN_M*1000:.0f} and 75 mm); the episode did not start')
        self._closed = True
        return width

    def wrench(self):
        w = super().wrench()
        if (self._closed and not w.bias_valid and self.calibration_valid()
                and self.calibration.get('gripper_verified') is True):
            self.width_outside_calibration = round(float(self.gripper_opening), 4)
            w = replace(w, bias_valid=True)
        return w


def free_backend(base):
    return type('Free'+base.__name__, (FreeGripper, base), {})


def axis_radius(axis, pivot, point):
    """Distance of point from the line through pivot along axis."""
    a = unit(axis)
    v = np.asarray(point, dtype=float)-vector(pivot)
    return float(np.linalg.norm(v-(v@a)*a))


def held_turn_start(grasp_ref, axis, pivot, eef_pos, eef_quat):
    """(start orientation, record) for a held turn.

    A held turn starts from the grasp orientation turned by the held object's angle around the pivot, so the
    lag of one segment cannot accumulate into a twist of the held object. That angle is only measurable when
    the grasp TCP and the current TCP both sit off the axis, and the correction must stay small; otherwise the
    turn starts from the hand's current orientation."""
    current = np.asarray(eef_quat, dtype=float).copy()
    radius = min(axis_radius(axis, pivot, grasp_ref[0]), axis_radius(axis, pivot, eef_pos))
    record = dict(radius_m=round(radius, 4), applied=False)
    if radius < REBASE_MIN_RADIUS_M:
        return current, dict(record, reason=f'TCP within {REBASE_MIN_RADIUS_M*100:.0f} cm of the turn axis')
    rebased = rebased_quat(grasp_ref, axis, pivot, eef_pos)
    offset = math.degrees((R.from_quat(rebased)*R.from_quat(current).inv()).magnitude())
    record['offset_deg'] = round(offset, 1)
    if offset > REBASE_MAX_DEG:
        return current, dict(record, reason=f'start correction above {REBASE_MAX_DEG:g} deg')
    return rebased, dict(record, applied=True)


def held_reading(backend, settle_s=.3, read_s=.5):
    """The held hand's reaction here and now, read still, from one short motion of zero steps (the tare
    recorded at the lift drifts by 2-4 N once the hand has moved)."""
    rows = []
    n_settle, n_read = int(round(settle_s/STEP_S)), max(1, int(round(read_s/STEP_S)))
    with backend.motion(settle_s+read_s+2.):
        for i in range(n_settle+n_read):
            backend.step(np.zeros(3), np.zeros(3))
            if i >= n_settle:
                rows.append(backend.reaction_force(backend.wrench()))
        backend.motion_result('completed')
    return np.mean(rows, axis=0)


COLLISION_GUARDED = ('move_to', 'turn', 'hold', 'rotate_z', 'point_tool', 'lift')


def merge_segments(reps, requested):
    """One tool result for a segmented move_until_contact: the last segment's contact, the whole call's motion."""
    first, last = reps[0], reps[-1]
    rows, events, pos_rows = [], [], []
    for r in reps:
        off = r.t_start-first.t_start
        rows += [[row[0]+off, *row[1:]] for row in r.wrench_rows]
        events += [replace(e, t=e.t+off) for e in r.events]
        pos_rows += r.pos_rows
    merged = replace(last, params=dict(last.params, max_dist=requested), t_start=first.t_start,
                     start_pos=first.start_pos, commanded_disp=np.sum([r.commanded_disp for r in reps], 0).tolist(),
                     actual_disp=(np.asarray(last.end_pos)-np.asarray(first.start_pos)).tolist(),
                     events=events, wrench_rows=rows, pos_rows=pos_rows, n_steps=sum(r.n_steps for r in reps),
                     peak_force=max(r.peak_force for r in reps), peak_torque=max(r.peak_torque for r in reps),
                     derived=dict(last.derived))
    merged.derived['segments'] = [dict(max_dist=r.params.get('max_dist'), termination=r.termination) for r in reps]
    return merged


def merge_rotations(reps, requested):
    """One tool result for a segmented rotate_z: the last segment's end state, the whole call's motion."""
    first, last = reps[0], reps[-1]
    rows, events, pos_rows = [], [], []
    for r in reps:
        off = r.t_start-first.t_start
        rows += [[row[0]+off, *row[1:]] for row in r.wrench_rows]
        events += [replace(e, t=e.t+off) for e in r.events]
        pos_rows += r.pos_rows
    steps = sum(r.n_steps for r in reps)
    yaw = sum(((float(r.yaw_deg or 0)+180.) % 360.)-180. for r in reps)
    merged = replace(last, params=dict(last.params, angle_deg=requested), t_start=first.t_start,
                     start_pos=first.start_pos,
                     commanded_disp=np.sum([r.commanded_disp for r in reps], 0).tolist(),
                     actual_disp=(np.asarray(last.end_pos)-np.asarray(first.start_pos)).tolist(),
                     events=events, wrench_rows=rows, pos_rows=pos_rows, n_steps=steps,
                     peak_force=max(r.peak_force for r in reps), peak_torque=max(r.peak_torque for r in reps),
                     mean_force_in_contact=sum(r.mean_force_in_contact*r.n_steps for r in reps)/max(steps, 1),
                     yaw_deg=round(yaw, 2), derived=dict(last.derived))
    merged.derived.update(rotated_deg=round(yaw, 2), requested_deg=requested,
                          segments=[dict(requested_deg=r.params.get('angle_deg'), rotated_deg=r.yaw_deg,
                                         termination=r.termination) for r in reps])
    clipped = next((r.derived for r in reps if r.derived.get('wrist_limit_clipped')), None)
    if clipped:
        merged.derived.update(wrist_limit_clipped=True, note=clipped.get('note'))
    return merged


class FreePrimitives(ContinuousPrimitives):
    """ContinuousPrimitives with the runner's guards: relax-and-read after every primitive, the 8 N (12 N when
    holding) collision stop, a path-deviation stop with a path model for every primitive, the held-turn rebase,
    a resistance drop from the relative tangential force, segmented long motions, and the stall watchdogs."""
    def __init__(self, *args, tick, **kwargs):
        self.budget_tick = tick
        self.relative_drop = None
        self.guard_stop = None
        self.grasp_ref = None
        self._call_args = {}
        super().__init__(*args, **kwargs)

    def pull(self, direction, force=PULL_FORCE_DEFAULT_N, distance=.02, speed=.004, max_time=None):
        # Reject instead of silently clamping a pull force (the message reaches the model).
        if not math.isfinite(force) or not 0 < force <= min(PULL_FORCE_MAX_N, self.f_cap*.8):
            raise ValueError('pull force must be positive, <= 20 N and <= 0.8 * raw force cap')
        return super().pull(direction, force, distance, speed, max_time)

    def _drot_world(self):
        # The robot side refuses |drot| > 0.10 rad per step; a rebased held turn can start up to about 9 deg
        # behind the hand, so ask for at most 0.08 rad per step and let the robot-side rate limit carry the rest.
        v = super()._drot_world()
        n = float(np.linalg.norm(v))
        return v*(.08/n) if n > .08 else v

    def held_angle_deg(self, axis, pivot):
        if self.grasp_ref is None or min(axis_radius(axis, pivot, self.grasp_ref[0]),
                                         axis_radius(axis, pivot, self.sim.eef_pos)) < REBASE_MIN_RADIUS_M:
            return None
        return arc_angle_deg(axis, pivot, self.grasp_ref[0], self.sim.eef_pos)

    def point_tool(self, direction, rate_deg_s=2.):
        """ContinuousPrimitives.point_tool with the full per-motion rotation envelope (0.55 rad) instead of 0.09 rad."""
        a = unit(direction)
        if not 0 < rate_deg_s <= math.degrees(.14):
            raise ValueError('Rotation rate must be positive and <= 0.04 rad/s')
        r0 = R.from_quat(self.sim.eef_quat)
        z = r0.apply([0, 0, 1])
        ang = math.acos(float(np.clip(z@a, -1, 1)))
        if ang > .55:
            raise ValueError('point_tool exceeds the 0.55 rad envelope')
        axis = np.cross(z, a)
        rg = r0 if ang < 1e-8 else R.from_rotvec(axis/np.linalg.norm(axis)*ang)*r0
        duration = max(.1, ang/math.radians(rate_deg_s))
        interp = Slerp([0, 1], R.concatenate([r0, rg]))

        def ctrl(s):
            self.target_quat = interp(min(1., s['t']/duration)).as_quat()
            return np.zeros(3), 0., s['t'] >= duration+.1
        rep = self._run('point_tool', dict(direction=a.tolist()), ctrl, duration+2.,
                        stop_on_contact=self._stop_on_contact(None), grip=1. if self.grip_closed else None)
        self.target_quat = self.sim.eef_quat.copy()
        rep.derived.update(force_rezeroed=False, rotation_requested_rad=ang)
        return rep

    def move_until_contact(self, direction, max_dist=.02, f_threshold=None, speed=None):
        """Up to CONTACT_SEARCH_MAX_M per call, run as segments of at most MOVE_MAX_M, each its own robot-side
        motion with every guard. Before each later segment a held hand is read still again (the held tare
        drifts 2-4 N once the hand has moved); an empty hand is re-zeroed by the backend anyway. A segment
        that ends any other way than without contact ends the call."""
        total = float(max_dist)
        if total <= MOVE_MAX_M+1e-9:
            return super().move_until_contact(direction, total, f_threshold, speed)
        reps, left, zeros = [], total, []
        while left > 1e-6:
            if reps and self.sim._closed:
                previous = np.asarray(self.carry_bias, dtype=float).copy()
                self.carry_bias = held_reading(self.sim)
                zeros.append((self.carry_bias-previous).round(3).tolist())
            step = min(MOVE_MAX_M, left)
            reps.append(super().move_until_contact(direction, step, f_threshold, speed))
            left -= step
            if reps[-1].termination != 'no_contact_within_max_dist' or self.guard_stop or self.sim.fault:
                break
        merged = merge_segments(reps, total)
        if zeros:
            merged.derived['held_zero_shifts_N'] = zeros
        return merged

    def rotate_z(self, angle_deg, torque_cap=None, stop_on_resistance_drop=False, rate_deg_s=8.):
        """Up to ROTATE_Z_MAX_DEG per call, run as equal segments of at most ROT_MAX_DEG, each its own robot-side
        motion (0.60 rad per motion unchanged) with every guard. A segment that ends early (contact, stall,
        wrist limit, fault) ends the call."""
        total = float(angle_deg)
        segments = max(1, int(math.ceil(abs(total)/ROT_MAX_DEG-1e-9)))
        if segments == 1:
            return super().rotate_z(total, torque_cap, stop_on_resistance_drop, rate_deg_s)
        reps = []
        for _ in range(segments):
            rep = super().rotate_z(total/segments, torque_cap, stop_on_resistance_drop, rate_deg_s)
            reps.append(rep)
            if (rep.termination != 'completed' or self.guard_stop or self.sim.fault
                    or rep.derived.get('wrist_limit_clipped')):
                break
            self.target_quat = np.asarray(self.sim.eef_quat, dtype=float).copy()
        return merge_rotations(reps, total)

    def screw(self, angle_deg, push=SCREW_PUSH_N, rate_deg_s=5.):
        """Up to ROTATE_Z_MAX_DEG per call as segments of at most ROT_MAX_DEG, like rotate_z;
        the tool result sums how far the screw went down over the whole call."""
        total = float(angle_deg)
        segments = max(1, int(math.ceil(abs(total)/ROT_MAX_DEG-1e-9)))
        reps = []
        for _ in range(segments):
            rep = super().screw(total/segments, push, rate_deg_s, quick=bool(reps))   # later segments press quickly
            reps.append(rep)
            if (rep.termination != 'completed' or self.guard_stop or self.sim.fault
                    or rep.derived.get('wrist_limit_clipped')):
                break
            self.target_quat = np.asarray(self.sim.eef_quat, dtype=float).copy()
        if len(reps) == 1:
            return reps[0]
        merged = merge_rotations(reps, total)
        merged.derived.update(start_z_m=reps[0].derived['start_z_m'],
                              sink_m=round(reps[0].derived['start_z_m']-reps[-1].derived['end_z_m'], 5))
        merged.quantities = [q for q in merged.quantities if q.get('name') not in ('screw_advance', 'side_load_change')]
        merged.quantities += screw_quantities(reps)
        return merged

    def call(self, name, **kwargs):
        rebase = held_zero = None
        if name == 'turn' and self.sim._closed and self.grasp_ref is not None:
            # start a held turn from the orientation matching the held object's real angle
            self.target_quat, rebase = held_turn_start(self.grasp_ref, kwargs['axis'], kwargs['pivot'],
                                                       self.sim.eef_pos, self.sim.eef_quat)
        if name == 'move_until_contact' and self.sim._closed:
            # A held search compares with the reading here, in free air, not with the tare from the lift:
            # after the hand moves, that tare can be off by 2-4 N and read as a contact at once.
            previous = np.asarray(self.carry_bias, dtype=float).copy()
            self.carry_bias = held_reading(self.sim)
            held_zero = dict(carry_bias_N=self.carry_bias.round(3).tolist(),
                             shift_N=(self.carry_bias-previous).round(3).tolist())
        torque = bool(self.sim.calibration.get('torque_verified'))
        force_drop = name == 'turn' and bool(kwargs.get('stop_on_resistance_drop')) and not torque
        if force_drop:
            # the uncertified sensor torque is not used: tangential force along the arc instead
            self.relative_drop = dict(buf=[], peak=0., below=0, detected=False)
            kwargs = dict(kwargs, stop_on_resistance_drop=False)
        self.guard_stop = None
        self._call_args = dict(kwargs)
        try:
            rep = super().call(name, **kwargs)
            if not self.sim.fault:
                rep.derived['static_after_relax'] = relax_and_read(self.sim)     # relax, then read still
                self.target_quat = np.asarray(self.sim.eef_quat, dtype=float).copy()
            if name == 'grasp' and self.sim._closed:
                self.grasp_ref = (self.sim.eef_pos.copy(), np.asarray(self.target_quat, dtype=float).copy())
                rep.derived.update(closed_width_m=round(float(self.sim.gripper_opening), 4),
                                   closed_empty=bool(getattr(self.sim, 'closed_empty', False)))
            if name == 'release' and self.sim._closed is False:
                self.grasp_ref = None
            if rebase is not None:
                rep.derived['held_turn_rebase'] = rebase
            if held_zero is not None:
                rep.derived['held_contact_zero'] = held_zero
            if self.guard_stop:
                kind, value = self.guard_stop
                if rep.termination == 'completed':
                    rep.termination = kind
                rep.derived['relay_guard'] = dict(kind=kind, value=value,
                                                  note='primitive stopped by the relay guard; the episode continues')
            if force_drop:
                rep.params['stop_on_resistance_drop'] = True
                rep.derived.update(resistance_signal='relative tangential force, not certified torque',
                                   resistance_peak_N=self.relative_drop['peak'])
                if self.relative_drop['detected'] and rep.termination == 'completed':
                    rep.termination = 'resistance_drop'
                    rep.events.append(ContactEvent(rep.duration, 'resistance_drop', self.relative_drop['end'],
                                                   'Relative tangential force dropped after a peak', rep.end_pos))
            # weight comes from the still readings (posture corrected, relative to the grasp); the
            # lift estimate uses one bias for every pose. Torque-based quantities need certified torque.
            drop = {'payload_mass'} | (set() if torque else {'com_offset'})
            rep.quantities = [q for q in rep.quantities if q.get('name') not in drop]
            if not torque:
                for key in list(rep.derived):
                    if key.startswith('resisting_moment') or key in ('click_detected', 'click_at_deg'):
                        rep.derived.pop(key)
            return rep
        finally:
            self.relative_drop = None

    def _path_error(self, name, params, start, p):
        """(sideways deviation from what this primitive is meant to do, stop tolerance), metres."""
        d = p-start

        def off(u):
            u = np.asarray(u, dtype=float)
            n = float(np.linalg.norm(u))
            if n < 1e-9:
                return float(np.linalg.norm(d))
            u = u/n
            return float(np.linalg.norm(d-(d@u)*u))
        if name == 'turn':
            ax = vector(params['axis'])
            ax = ax/np.linalg.norm(ax)
            pv = vector(params['pivot'])
            v, v0 = p-pv, start-pv
            return math.hypot(float((v-v0)@ax), float(np.linalg.norm(v-(v@ax)*ax)-np.linalg.norm(v0-(v0@ax)*ax))), .02
        if name == 'move_to':
            return off(np.array([params[k] for k in ('x', 'y', 'z')])-start), .015
        if name in ('move_until_contact', 'push'):
            return off(params['direction']), .015
        if name == 'lift':
            return off([0, 0, 1]), .015
        if name == 'insert':
            return off(params['normal']), .015
        if name == 'slide':
            u = unit(params['direction'])
            n = np.asarray(self._call_args.get('normal', (0, 0, -1)), dtype=float)
            n = n-(n@u)*u
            rest = d-(d@u)*u
            if np.linalg.norm(n) > 1e-9:
                n = n/np.linalg.norm(n)
                rest = rest-(rest@n)*n
            return float(np.linalg.norm(rest)), .015
        if name == 'search':
            return max(0., off(params['normal'])-float(params['radius_max'])), .015
        if name == 'screw':
            return off([0, 0, 1]), .01          # follows the thread down; sideways firm while pressing, soft while turning
        if name == 'pull':
            # compliant sideways by design (follows the mechanism): only travel past the distance counts
            return max(0., float(np.linalg.norm(d))-float(params['distance'])), .015
        return float(np.linalg.norm(d)), .015          # hold, grasp, release, rotate_z, point_tool

    def _rotation_deg(self, name, params):
        """Degrees the hand is meant to turn in this motion (None for translations)."""
        if name in ('turn', 'rotate_z', 'screw'):
            return abs(float(params.get('angle_deg', 0)))
        if name == 'point_tool':
            z = R.from_quat(self.sim.eef_quat).apply([0, 0, 1])
            return math.degrees(math.acos(float(np.clip(z@unit(params['direction']), -1, 1))))
        return None

    def _run(self, name, params, controller, *args, **kwargs):
        start = self.sim.eef_pos.copy()
        held = bool(self.sim._closed)
        above_collision = 0
        rotation_deg = self._rotation_deg(name, params)
        progress = (RotationProgress(axis=params.get('axis') if name == 'turn' else None,
                                     pivot=params.get('pivot') if name == 'turn' else None)
                    if rotation_deg is not None and rotation_deg > 1 else None)
        tracking, stuck_stop = TrackingProgress(), []

        def guarded(state):
            nonlocal above_collision
            self.budget_tick()
            stuck = tracking.observe(state['t'], state['pos'], state['target'])
            # a screw presses before it turns; the wrist is only expected to move once it turns
            watch = progress and (name != 'screw' or self.screw_phase == 'turn')
            stalled = progress.observe(state['t'], state['pos'], self.sim.eef_quat) if watch else False
            if name in COLLISION_GUARDED:
                # The raw reaction, so a held load cannot tare an 8 N collision away; holding, free-moving
                # joint friction and the load itself read up to about 8 N, hence 12 N then.
                limit = HELD_TURN_COLLISION_N if held else FREE_COLLISION_N
                above_collision = (above_collision+1 if state['t'] >= .3 and
                                   np.linalg.norm(state['reaction_raw']) >= limit else 0)
                if above_collision >= 3:
                    self.guard_stop = ('contact', float(np.linalg.norm(state['reaction_raw'])))
                    return np.asarray(state['pos'])-np.asarray(state['target']), 0., True
            p = np.asarray(state['pos'])
            err, tolerance = self._path_error(name, params, start, p)
            if err > .03:
                raise BackendError('path_deviation > 30 mm')
            if err > tolerance:
                self.guard_stop = ('path_deviation', float(err))
                return p-np.asarray(state['target']), 0., True
            if name == 'turn' and self.relative_drop is not None:
                ax = vector(params['axis'])
                ax = ax/np.linalg.norm(ax)
                pv = vector(params['pivot'])
                tangent = np.cross(ax, p-pv)*np.sign(params['angle_deg'])
                tangent /= max(float(np.linalg.norm(tangent)), 1e-9)
                resistance = -float(np.asarray(state['reaction_raw'])@tangent)
                drop = self.relative_drop
                drop['buf'] = (drop['buf']+[resistance])[-4:]
                signal = float(np.mean(drop['buf']))
                drop['end'] = signal
                if state['t'] >= .6:
                    drop['peak'] = max(drop['peak'], signal)
                    drop['below'] = (drop['below']+1 if drop['peak'] >= 1. and
                                     drop['peak']-signal >= max(.5, .2*drop['peak']) else 0)
                    if drop['below'] >= 2:
                        drop['detected'] = True
                        self.target_quat = self.sim.eef_quat.copy()
                        return p-np.asarray(state['target']), 0., True
            command = controller(state)
            if (stalled or stuck) and not command[2]:
                if not stalled:
                    stuck_stop.append(dict(version='stall1', **tracking.metrics))
                self.guard_stop = (('stalled', progress.metrics['wrist_span_deg']) if stalled else
                                   ('stalled', tracking.metrics['target_lead_mm']))
                self.target_quat = self.sim.eef_quat.copy()
                return p-np.asarray(state['target']), 0., True
            return command
        rep = super()._run(name, params, guarded, *args, **kwargs)
        if progress:
            rep.derived['rotation_progress'] = dict(version='r33c', **progress.metrics)
        if stuck_stop:
            m = rep.derived['tracking_progress'] = stuck_stop[0]
            note = ('the hand stayed put for %.1f s while its target kept moving (target %.1f mm ahead): pressed '
                    'too hard to slide, or blocked; stopped here before the NUC tracking-stall fault would end '
                    'the session' % (m['stuck_s'], m['target_lead_mm']))
            rep.derived['note'] = (rep.derived['note']+'; '+note) if rep.derived.get('note') else note
        return rep


# ---- demo: synthetic plant with two synthetic bottles (never used live) --------------------------
def demo():
    from forceharness.real.backend import ContinuousBackend
    from forceharness.real.plant import DemoPort, DemoWorld
    cfg = BackendConfig(motion_enabled=True, f_cap=RAW_FORCE_CAP_N, tcp_translation=TCP_TRANSLATION)
    # flange-z plane: the TCP stops at the synthetic table (hand pointing down, TCP 0.1185 m below the flange)
    port = DemoPort(cfg, payload=0, plane=DemoWorld.TABLE_Z+TCP_TRANSLATION[2])
    cal = port.calibration()
    cal.update(force_sign=1, torque_verified=False)

    class SyntheticBackend(ContinuousBackend):
        def calibration_valid(self):
            return self._closed is not None    # exact synthetic sensors, every pose
    b = free_backend(SyntheticBackend)(port, config=cfg, calibration=cal, clock=port.clock, sleep=port.sleep,
                                       keep_controller=True)
    q0 = np.array([1., 0, 0, 0])
    port.driver.p = np.array([.40, 0, .25])-R.from_quat(q0).apply(cfg.tcp_translation)
    world = DemoWorld(port, cfg.tcp_translation)
    world.install()
    a = world.objects[0]['xy']
    calls = [dict(name='scene', args={}),
             dict(name='move_to', args=dict(x=(.40+a[0])/2, y=a[1]/2, z=.25)),
             dict(name='move_to', args=dict(x=a[0], y=a[1], z=.25)),
             dict(name='move_to', args=dict(x=a[0], y=a[1], z=.16)),
             dict(name='grasp', args={}),
             dict(name='move_by', args=dict(dx=0, dy=0, dz=.03)),        # A and B share the script: no lift (force menu)
             dict(name='move_by', args=dict(dx=0, dy=0, dz=-.03)),
             dict(name='release', args={}),
             dict(name='move_by', args=dict(dx=0, dy=0, dz=.08)),
             dict(name='done', args=dict(success=True, note='synthetic scene only'))]
    return b, world, calls


# ---- episode -----------------------------------------------------------------------------------------
def score(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--outcome', choices=['success', 'partial', 'failed', 'not_attempted'], required=True)
    ap.add_argument('--note', default='')
    a = ap.parse_args(argv)
    path = a.out/'episode.json'
    r = json.loads(path.read_text())
    if r.get('version') != VERSION or not r.get('ended'):
        ap.error('Expected a completed free-task episode')
    if r.get('scored'):
        ap.error('Already scored; preserve the original operator outcome')
    r.update(scored=True, outcome=a.outcome, success=a.outcome == 'success', score_source='operator_visual',
             score_note=a.note)
    atomic_json(path, r)
    print(json.dumps(dict(outcome=r['outcome'], peak_force_N=r['peak_force_N'], action_count=len(r['actions']))))
    return 0


def tool_frame(backend):
    rot = R.from_quat(backend.eef_quat)
    return dict(pointing=rot.apply([0, 0, 1]).round(3).tolist(), closing_axis=rot.apply([0, 1, 0]).round(3).tolist())


def build_parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True, help='episode folder (private record; must not exist)')
    ap.add_argument('--relay-dir', type=Path, required=True, help='relay folder the model side reads and writes')
    ap.add_argument('--instruction', help='the task, as the operator wrote it')
    ap.add_argument('--instruction-file', type=Path)
    ap.add_argument('--condition', choices=sorted(CONDITIONS), default='A',
                    help='A = ForceHarness (phys2 force tool results), B = no-force control')
    ap.add_argument('--live', action='store_true', help='drive the real robot (default: synthetic demo)')
    ap.add_argument('--execute', action='store_true', help='required with --live')
    ap.add_argument('--calibration', type=Path, help='force calibration file (configs/real/force_calibration.json)')
    ap.add_argument('--robot-config', type=Path, default=None,
                    help='site file (default: $FH_ROBOT_CONFIG, else configs/real/robot.json)')
    ap.add_argument('--demo-model', action='store_true', help='synthetic plant driven by the relay model (no robot)')
    ap.add_argument('--max-actions', type=int, default=MAX_ACTIONS)
    ap.add_argument('--max-minutes', type=float, default=MAX_MINUTES)
    ap.add_argument('--auto-confirm', action='store_true',
                    help='grasp/release without the operator confirmation (the model runs the whole episode)')
    ap.add_argument('--posture-calibration', type=Path, default=posture_bias.DEFAULT_FILE)
    ap.add_argument('--no-auto-calibration', action='store_true')
    ap.add_argument('--recalibrate', action='store_true')
    ap.add_argument('--force-max', type=float, default=FORCE_MAX_N,
                    help=f'contact force targets (move_until_contact, push, slide, search), up to {TARGET_FORCE_CEILING_N:g} N')
    ap.add_argument('--insert-force-max', type=float, default=INSERT_FORCE_MAX_N,
                    help=f'insert max_force, up to {TARGET_FORCE_CEILING_N:g} N')
    ap.add_argument('--slide-force-default', type=float, default=SLIDE_FORCE_DEFAULT_N,
                    help='slide normal_force when the model gives none (at most --force-max)')
    ap.add_argument('--resume-held', metavar='OBJECT',
                    help='operator: the closed gripper holds OBJECT (left from the previous episode or closed on by '
                         'hand with scripts/real/gripper_check.py); start holding it')
    return ap


def main(argv=None, *, task=None):
    """Run one episode; task is a tasks.TaskPreset (None: the free task). Exit code 0 when finished cleanly."""
    from forceharness.real.tasks import FREE
    task = task or FREE
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'score':
        return score(argv[1:])
    ap = build_parser(__doc__)
    a = ap.parse_args(argv)
    if not FORCE_MAX_N <= a.force_max <= TARGET_FORCE_CEILING_N:
        ap.error(f'--force-max {FORCE_MAX_N:g}..{TARGET_FORCE_CEILING_N:g} N (0.8 * the {RAW_FORCE_CAP_N:g} N raw cap)')
    if not INSERT_FORCE_MAX_N <= a.insert_force_max <= TARGET_FORCE_CEILING_N:
        ap.error(f'--insert-force-max {INSERT_FORCE_MAX_N:g}..{TARGET_FORCE_CEILING_N:g} N (0.8 * the {RAW_FORCE_CAP_N:g} N raw cap)')
    if not 0 < a.slide_force_default <= a.force_max:
        ap.error('--slide-force-default must be positive and at most --force-max')
    limits = dict(contact=a.force_max, insert=a.insert_force_max, slide_default=a.slide_force_default)
    if a.resume_held is not None:
        a.resume_held = a.resume_held.strip()
        if not a.resume_held or len(a.resume_held) > 40 or '\n' in a.resume_held:
            ap.error('--resume-held names the held object in one short line (at most 40 characters)')
    fixed_instruction = task.instruction()
    if fixed_instruction is not None and (a.instruction is not None or a.instruction_file is not None):
        ap.error('a fixed task does not accept --instruction; to change the task use scripts/real/free_task.py')
    instruction = (fixed_instruction if fixed_instruction is not None else
                   a.instruction_file.read_text() if a.instruction_file else a.instruction or '').strip()
    if not instruction:
        ap.error('--instruction (or --instruction-file) is required: it is the task')
    if len(instruction) > MAX_INSTRUCTION_CHARS:
        ap.error(f'instruction longer than {MAX_INSTRUCTION_CHARS} characters')
    if not ACTION_RANGE[0] <= a.max_actions <= ACTION_RANGE[1] or not MINUTE_RANGE[0] <= a.max_minutes <= MINUTE_RANGE[1]:
        ap.error(f'--max-actions {ACTION_RANGE}, --max-minutes {MINUTE_RANGE}')
    conditions, primitives = task.menus()
    cond = conditions[a.condition]
    if a.out.resolve() == a.relay_dir.resolve() or a.out.resolve() in a.relay_dir.resolve().parents or a.relay_dir.resolve() in a.out.resolve().parents:
        ap.error('Private episode and public relay must be separate siblings')
    world = None
    robot = None
    if a.live:
        if not a.execute or not a.calibration:
            ap.error('Live requires --execute --calibration')
        cal = json.loads(a.calibration.read_text())
        if cal.get('verified') is not True or cal.get('gripper_verified') is not True or cal.get('force_sign') not in (-1, 1):
            ap.error('Live needs a verified force calibration with the gripper closure correction')
        try:
            robot = load_robot_config(a.robot_config).require_live()
        except (OSError, ValueError) as exc:
            ap.error(str(exc))
        use_hand_eye(robot.cameras.hand_eye_path)
        from forceharness.real.backend import ContinuousBackend
        from forceharness.real.transport import RobotPort
        cfg = BackendConfig(motion_enabled=True, f_cap=RAW_FORCE_CAP_N, tcp_translation=TCP_TRANSLATION,
                            gripper_width_check=task.gripper_width_check)
        # thin parts: a closed width down to 8 mm is a grasp; objects of any width up to 75 mm
        port = RobotPort(cfg, a.out/'nuc', robot=robot, grasp_min_width_m=.008,
                         camera_factory=partial(GripperCameras, camera_specs(robot.cameras)), max_trials=None)
        port.next_weigh_trial(1)
        b = free_backend(ContinuousBackend)(port, config=cfg, calibration=cal, keep_controller=True)
        calls = None
    else:
        if a.execute or a.calibration:
            ap.error('Demo uses only synthetic inputs')
        b, world, calls = demo()
        if task.demo_note is not None:
            world.objects = []
            calls = [dict(name='done', args=dict(success=False, note=task.demo_note))]
        if a.demo_model:
            calls = None
    a.out.mkdir(parents=True, exist_ok=False)
    client = RelayClient(a.relay_dir, timeout_s=600, action_prompt=action_prompt(a.condition, conditions))
    start = time.monotonic()
    budget_s = a.max_minutes*60
    r = dict(version=VERSION, mode='live' if a.live else 'demo', condition=cond['label'], interface=cond['interface'],
             task_id=task.task_id, task_version=task.version, force_limits_N=limits,
             camera_version='free_camera/r41',
             gripper_camera_version='free_camera/r42',
             instruction=instruction, menu=cond['menu'],
             operator_confirmation='auto (no CLOSE/OPEN approval)' if a.auto_confirm or not a.live else 'chat',
             model='relay' if a.live or a.demo_model else 'synthetic', max_actions=a.max_actions, max_minutes=a.max_minutes,
             started=datetime.now().astimezone().isoformat(), actions=[], scene_updates=[],
             config=asdict(b.config), peak_force_N=0., peak_raw_force_N=0., ended=False, finished=False, fault=None,
             outcome=None, scored=False, physical_validation=False)
    if robot is not None:
        r['robot_config'] = robot.source
    injector = make_injector(cond['injector'])

    def budget():
        if time.monotonic()-start >= budget_s:
            raise TimeoutError(f'{a.max_minutes:g}-minute episode budget reached')

    def tick():
        budget()
        b.refresh()
        if b.holding and b._closed and b._session_stats['holds'][-1]['elapsed_s'] > 45:
            # renew the hold while the model thinks with something gripped (robot-side 60 s idle rule)
            with b.motion(2.):
                b.step([0, 0, 0])
                b.motion_result('completed')
        b.sleep(.05)
    confirm = OperatorConfirm(a.out/'confirmations') if a.live and not a.auto_confirm else None
    last_confirm = {}

    def authorize(kind, state):
        if not confirm:
            return True
        ok = confirm('close' if kind == 'grasp' else 'open', dict(action=kind, robot_state=state), tick)
        try:
            results = sorted((a.out/'confirmations').glob('*.result.json'))
            last_confirm['text'] = json.loads(results[-1].read_text()).get('user_text', '') if results else ''
        except Exception:
            last_confirm['text'] = ''
        return ok
    grip = GRIP_RULES['auto' if a.auto_confirm or not a.live else 'operator']

    def build_prompt(table_z):
        values = dict(instruction=instruction, menu=menu_text(a.condition, grip, limits, conditions=conditions,
                                                              primitives=primitives),
                      table_z=f'{table_z:.3f}' if table_z is not None else '≈ 0.005 (go by the object list)',
                      max_actions=str(a.max_actions), max_minutes=f'{a.max_minutes:g}')
        # one pass, so braces inside the operator's instruction are never filled in
        return re.sub(r'\{(' + '|'.join(values) + r')\}', lambda m: values[m.group(1)], (PROMPT_DIR/cond['prompt']).read_text())
    messages = []
    atomic_json(a.out/'episode.json', r)
    try:
        with b:
            if a.resume_held and b._closed is None:
                # the operator names what the gripper still holds, so the held state is known
                r['resumed_holding'] = dict(object=a.resume_held, width_m=round(b.resume_holding(), 4))
            elif a.resume_held:
                r['resume_held_ignored'] = f'gripper open at the start ({b.gripper_opening*1000:.0f} mm): nothing held'
            if b._closed is None:
                # A gripper left closed by a stopped episode makes the held state unknown; the first force
                # reading would then fail anyway. Say what to do instead.
                raise BackendError(f'The gripper is not open at the start (width {b.gripper_opening*1000:.0f} mm, needs '
                                   '>= 78 mm): the hand may still hold something. With the held state unknown nothing '
                                   'moves and the gripper does not open; the episode did not start. If the hand is empty '
                                   'and only fails to open to 78 mm: clear the space around the fingers and restart the '
                                   'Polymetis gripper server on the robot PC (it homes on start), and do not use '
                                   '--resume-held. If the hand really holds something and the task should continue: '
                                   'name what it holds at the start (--resume-held)')
            p = FreePrimitives(b, confirm=authorize, tick=budget, stop_protocol='r9', interface=cond['interface'])
            p.extra_tools = (*p.extra_tools, *cond['menu'])      # the menu decides; B's tool results stay force-free
            client.wait_tick = tick
            posture = None
            if a.live:
                posture = None if a.recalibrate else posture_cal.load(a.posture_calibration)
                if posture is None:
                    if a.no_auto_calibration:
                        raise BackendError('posture calibration missing: '+str(a.posture_calibration))
                    if b._closed:
                        raise BackendError('The posture calibration needs an empty hand and is not run while holding '
                                           'something; the episode did not start: first start an episode with an '
                                           'empty hand to complete the calibration')
                    print('First use (or --recalibrate): running a one-time empty-hand posture calibration first '
                          '(about 20 minutes), reused afterwards:', a.posture_calibration, flush=True)
                    r['posture_calibration_run'] = posture_cal.run_calibration(
                        p, a.posture_calibration, out_dir=a.out/'posture_calibration', log=lambda m: print(m, flush=True))['stats']
                    atomic_json(a.out/'episode.json', r)
                    posture = posture_cal.load(a.posture_calibration)
            zero = relax_and_read(b)
            p.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
            if r.get('resumed_holding'):
                # what a grasp here would have left: held, the grasp reference here, and the held reaction
                # read still, here, as the carry tare
                p.grip_closed = True
                p.grasp_ref = (np.asarray(b.eef_pos, dtype=float).copy(), p.target_quat.copy())
                p.carry_bias = held_reading(b)
                r['resumed_holding']['carry_bias_N'] = np.round(p.carry_bias, 3).tolist()
            if posture is not None and posture.meta.get('use_model') and zero.get('tau') is not None:
                posture.zero_check(zero['q'], zero['tau'])
            r['zero_check'] = zero
            r['posture_calibration'] = (None if posture is None else
                                        dict(file=str(a.posture_calibration), created=posture.meta.get('created'),
                                             use_model=bool(posture.meta.get('use_model')), stats=posture.meta.get('stats')))
            atomic_json(a.out/'episode.json', r)
            grasp_force, empty_force = {}, {}

            def static_force(st):
                """Force on the hand (reaction sign) relative to the hand at the start: empty, or holding
                the resumed object."""
                F = np.asarray(st['raw_force'], dtype=float)-np.asarray(zero['raw_force'], dtype=float)
                if posture is not None and posture.d_tau is not None:
                    F = F-(posture.wrench(st['q'])[:3]-posture.wrench(zero['q'])[:3])
                return (b.force_sign or -1)*F
            start_hand = (f'relative to the hand at the start, which already held the {r["resumed_holding"]["object"]}'
                          if r.get('resumed_holding') else 'relative to the empty hand at the start')
            photographer = live_photographer(a.out, b, robot.cameras) if a.live else None
            latest_scene = {}

            def snapshot(turn):
                """Photos (live) and the object list for this turn; text for the model, never fatal."""
                latest_scene.clear()
                record = dict(turn=turn)
                if photographer is None:
                    found = world.found()
                    record.update(objects=found['objects'], table_z=found['table_z'], synthetic=True)
                    r['scene_updates'].append(record)
                    return found, '\n'+T.SYNTHETIC_NO_IMAGES+'\n'+scene_objects.describe(found, synthetic=True)
                try:
                    folder = Path(photographer(f'turn{turn:02d}', tick)['wrist_images'][0]).parent
                except Exception as exc:
                    record.update(error=f'camera: {exc}')
                    r['scene_updates'].append(record)
                    return None, '\n'+T.CAMERA_FAILED+f'{exc}]'
                record['photo'] = str(folder)
                latest_scene.update(frame=turn, folder=folder)
                found, wrist = None, folder/'scene_wrist.png'
                try:
                    found = scene_objects.detect(folder, b.config.tcp_translation, b.config.tcp_quaternion)
                    try:
                        grasp_reach(b, found)
                    except Exception as exc:             # the hint without the joint check, never fatal
                        record['grasp_reach_error'] = str(exc)
                    marks = scene_objects.annotate(folder, folder/'scene_wrist_free.png', found,
                                                   b.config.tcp_translation, b.config.tcp_quaternion)
                    (folder/'objects_free.json').write_text(json.dumps(found, ensure_ascii=False, indent=1))
                    (folder/'overlay_free.json').write_text(json.dumps(marks, ensure_ascii=False))
                    wrist = folder/'scene_wrist_free.png'
                    record.update(objects=found['objects'], candidates=found.get('candidates', []),
                                  perception_version=found.get('version'), table_z=found['table_z'], view=found['view'],
                                  closing_deg=found.get('closing_deg'), image_axes=found.get('image_axes'))
                except Exception as exc:
                    record['error'] = f'objects: {exc}'
                r['scene_updates'].append(record)
                names = []
                for label, src in (('wrist', wrist), ('exterior', folder/'scene_exterior.png')):
                    if src.is_file():
                        dst = client.dir/f'obs_{turn}_{label}.png'
                        shutil.copyfile(src, dst)
                        names.append(dst.name)
                text = ('\n'+T.PHOTOS+', '.join(names)+'] The wrist camera is on the hand, about 5 cm ahead of the '
                        'fingers, looking down (the two finger pads are in the upper corners of the frame); the exterior '
                        'camera sees the whole scene from the front. Wrist image marks: red boxes = where the two finger '
                        'pads are now, yellow boxes and numbers = objects detected automatically, orange boxes with a '
                        'question mark = candidates to confirm; a cross marks the measured top-surface reference point. '
                        'The red boxes show only the current finger pads, not where they will land after descending.')
                text += ('\n'+scene_objects.describe(found)) if found else '\n'+T.DETECTION_FAILED+f'{record.get("error")}]'
                from PIL import Image
                try:
                    with Image.open(wrist) as im:
                        width, height = im.size
                    text += ('\n'+T.WRIST_FRAME+f'{turn}, original {width}×{height} pixels. Use locate(frame={turn}, '
                             'u=horizontal, v=vertical) to measure the surface coordinates of a part in the image; it '
                             'need not have its own object number.')
                except OSError:
                    latest_scene.clear()
                    text += '\n'+T.NO_WRIST_IMAGE+'; locate cannot be used.'
                return found, text

            def post_receipt(receipt):
                _, scene_text = snapshot(client.turn)
                receipt['text'] = receipt.get('text', '')+scene_text
                messages.append(dict(role='tool', parts=[dict(type='text', text=json.dumps(receipt, ensure_ascii=False))]))

            def states():
                return dict(position=np.round(b.eef_pos, 4).tolist(), gripper_width=round(float(b.gripper_opening), 4),
                            **tool_frame(b))
            found0, scene0 = snapshot(0)
            messages.append(dict(role='system', text=build_prompt(found0['table_z'] if found0 else None)))
            first = [dict(type='text', text=json.dumps(dict(states(), quaternion=np.round(b.eef_quat, 4).tolist())))]
            if r.get('resumed_holding'):
                held = r['resumed_holding']
                first.append(dict(type='text', text=(
                    T.RESUMED+f', holding "{held["object"]}" (confirmed by the operator, closed width '
                    f'{held["width_m"]*1000:.0f} mm). Go by this turn\'s photos for what is left of the task. No need to '
                    'grasp again; to change the grip, first set it on a support surface and release, then realign and '
                    'grip.')))
            messages.append(dict(role='user', parts=first+[dict(type='text', text=scene0)]))
            atomic_json(a.out/'episode.json', r)
            for index in range(a.max_actions):
                if calls is not None:
                    atomic_json(client.dir/f'cmd_{client.turn}.json', dict(calls=[calls[index]]))
                response = client.chat(messages, [])
                call = response.tool_calls[0]
                action = dict(name=call.name, args=call.args)
                if response.text:
                    action['note'] = response.text
                r['actions'].append(action)
                atomic_json(a.out/'episode.json', r)
                messages.append(dict(role='assistant', text=response.text))
                try:
                    args, ignored = validate(call.name, call.args, b, a.condition, limits,
                                             conditions=conditions, primitives=primitives)
                    if ignored:
                        action['ignored_args'] = ignored
                except (ValueError, TypeError) as exc:
                    receipt = dict(text=T.INVALID+str(exc), states=states())
                    action['invalid'] = str(exc)
                else:
                    if call.name == 'done':
                        r.update(finished=True, model_success=args['success'], model_note=args.get('note', ''))
                        break
                    if call.name == 'locate':
                        try:
                            if not latest_scene or args['frame'] != latest_scene['frame']:
                                raise ValueError('only the latest wrist frame can be located; use the frame of the '
                                                 'current observation, not an older or synthetic image')
                            target = pixel_target.locate(latest_scene['folder'], args['u'], args['v'],
                                                         b.config.tcp_translation, b.config.tcp_quaternion)
                            target['frame'] = args['frame']
                            action['target'] = target
                            receipt = dict(text=T.LOCATE_OK+json.dumps(target, ensure_ascii=False), states=states())
                        except (ValueError, OSError, KeyError) as exc:
                            action['invalid'] = str(exc)
                            receipt = dict(text=T.LOCATE_FAILED+str(exc), states=states())
                        action['receipt'] = receipt
                        post_receipt(receipt)
                        atomic_json(a.out/'episode.json', r)
                        continue
                    if call.name == 'scene':
                        receipt = dict(text=T.SCENE_OK, states=states())
                        action['receipt'] = receipt
                        post_receipt(receipt)
                        error = r['scene_updates'][-1].get('error')
                        if error:
                            action['invalid'] = 'scene capture failed: '+error
                        atomic_json(a.out/'episode.json', r)
                        continue
                    try:
                        rep = p.call(call.name, **args)
                    except ValueError as exc:
                        if b.fault:
                            raise
                        receipt = dict(text=T.INVALID+str(exc), states=states())
                        action.update(invalid=str(exc), receipt=receipt)
                        post_receipt(receipt)
                        atomic_json(a.out/'episode.json', r)
                        continue
                    except BackendError as exc:
                        # an operator decline (CLOSE/OPEN not given) is a tool result; the gripper is unchanged
                        if b.fault or not any(k in str(exc) for k in ('not confirmed', 'supported placement required')):
                            raise
                        hint = (last_confirm.get('text') or '').strip()
                        receipt = dict(text=T.OPERATOR_DECLINED+f'{call.name}. Operator\'s hint: {hint or "not aligned"}. '
                                       'The robot did not move and the gripper state is unchanged.',
                                       states=states())
                        action.update(declined=hint, receipt=receipt)
                        post_receipt(receipt)
                        atomic_json(a.out/'episode.json', r)
                        continue
                    atomic_json(a.out/f'action_{index:02d}.json', rep.to_dict())
                    action['termination'] = rep.termination
                    r['peak_force_N'] = max(r['peak_force_N'], rep.peak_force)
                    r['peak_raw_force_N'] = b.peak_wrench
                    parts = injector.render(rep, dict(proprio='full'))
                    extra = []
                    if call.name == 'grasp':
                        if rep.derived.get('closed_empty'):
                            extra.append(T.EMPTY_GRASP+f'the two fingers closed (width {rep.derived["closed_width_m"]*1000:.0f} mm); '
                                         'nothing is in the hand. After release opens the gripper you can realign and '
                                         'grasp again.')
                        else:
                            extra.append(T.HOLDING+f'{rep.derived.get("closed_width_m", 0)*1000:.0f} mm (≈ the width '
                                         'of the grasped part).')
                    if call.name == 'turn':
                        held_angle = p.held_angle_deg(vector(args['axis']), vector(args['pivot']))
                        extra.append(T.TURNED+f'{rep.derived.get("turned_deg")} deg of the requested {args["angle_deg"]} deg'
                                     + (f'; the held object is now rotated {held_angle:.1f} deg from where it was grasped '
                                        '(from the TCP position around the pivot)' if held_angle is not None else '')
                                     + f'; gripper tilt from pointing straight down is {hand_tilt_deg(b):.1f} deg')
                    st = rep.derived.get('static_after_relax')
                    if st is not None:
                        F = static_force(st)
                        action['static_force_N'] = F.round(3).tolist()
                        if call.name == 'grasp' and b._closed:
                            # Closing on a standing object can press it into the table or lift it a little, by as much
                            # as the weight differences of interest. The empty hand just before the close, at the same
                            # pose, is the clean reference.
                            before = (empty_force.get('F') if 'pos' in empty_force
                                      and np.linalg.norm(b.eef_pos-empty_force['pos']) < .003 else None)
                            grasp_force['F'] = F if before is None else before
                            action['weight_reference'] = 'after_grasp' if before is None else 'before_grasp'
                        if b._closed is False:
                            empty_force.update(F=F, pos=np.asarray(b.eef_pos, dtype=float).copy())
                        if call.name == 'release':
                            grasp_force.clear()
                        if a.condition == 'A':
                            line = (T.STILL_READING
                                    + f'[{F[0]:+.2f}, {F[1]:+.2f}, {F[2]:+.2f}] N (|F| {np.linalg.norm(F):.2f} N) '
                                    + start_hand)
                            if b._closed and 'F' in grasp_force and call.name != 'grasp':
                                dF = F-grasp_force['F']
                                grams = -dF[2]/G*1000
                                action['weight_since_grasp_g'] = round(float(grams), 1)
                                line += (T.CHANGE_SINCE_GRASP+f'[{dF[0]:+.2f}, {dF[1]:+.2f}, {dF[2]:+.2f}] N'
                                         + (T.WEIGHT_ESTIMATE+f'{grams:.0f} g. '
                                            + T.WEIGHT_CAVEAT.format(error=WEIGHT_ERROR_G) if grams > 20 else ''))
                            extra.append(line)
                    if a.condition == 'A' and rep.derived.get('held_contact_zero'):
                        shift = rep.derived['held_contact_zero']['shift_N']
                        extra.append(T.HELD_CONTACT_ZERO+'before starting, the hand settled in the air and re-zeroed '
                                     '(differs from the reading recorded at the lift by '
                                     f'[{shift[0]:+.2f}, {shift[1]:+.2f}, {shift[2]:+.2f}] N); contact is judged only from '
                                     'the force change during this downward segment.')
                    if (a.condition == 'A' and call.name == 'grasp' and b._closed
                            and getattr(b, 'width_outside_calibration', None) is not None):
                        extra.append(T.CLOSED_WIDTH_NOTE+f'{b.width_outside_calibration*1000:.0f} mm is not the '
                                     f'{b.calibration.get("closed_width_m", 0)*1000:.0f} mm of the force calibration: '
                                     'force readings while holding it use the calibrated gripper correction as an '
                                     'approximation, and may be off by about 1 N; the change from before to after the '
                                     'grasp is not affected)')
                    if a.condition == 'A' and rep.peak_force:
                        extra.append(T.NET_FORCE_PEAK+f'{rep.peak_force:.3f} N')
                    if rep.termination == 'torque_cap' and not b.fault:
                        action['torque_stop_Nm'] = round(float(rep.peak_torque), 3)
                        extra.append(TORQUE_STOP_NOTE if a.condition == 'A' else TORQUE_STOP_NOTE_B)
                    receipt = dict(text='\n'.join([part['text'] for part in parts]+extra), states=states())
                    # a guard that only STOPPED the motion is information, and so is the 3 N*m torque cap (the
                    # motion stops as before, the episode goes on); the force cap, unexpected motion or a
                    # backend fault end the episode
                    if bool(b.fault) or rep.termination in ('force_cap', 'unexpected_motion'):
                        raise BackendError('Primitive stopped: '+rep.termination+'; '+str(b.fault))
                action['receipt'] = receipt
                post_receipt(receipt)
                atomic_json(a.out/'episode.json', r)
            else:
                r['budget_stop'] = f'{a.max_actions} actions exhausted'
    except Exception as exc:
        r['fault'] = f'{type(exc).__name__}: {exc}'
    finally:
        r.update(ended=True, duration_s=time.monotonic()-start, controller_stopped=not b.active and not b.holding,
                 held_state=b._closed, relay_events=client.events, backend_fault=b.fault,
                 valid_episode=r['fault'] is None, excluded_from_success_rate=r['fault'] is not None)
        atomic_json(a.out/'episode.json', r)
        client.finish()
    print(json.dumps({k: r[k] for k in ('mode', 'condition', 'finished', 'fault', 'held_state', 'controller_stopped')},
                     ensure_ascii=False))
    return 0 if r['finished'] and not r['fault'] else 2
