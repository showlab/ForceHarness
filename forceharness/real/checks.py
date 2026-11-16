"""Kinematic checks of the real-robot runner: refuse before motion what the robot side would fault on (layer 2).

The robot-side guard faults (and ends the episode) on a flange leaving the workspace box, a joint close to its
limit, too much joint travel in one motion, or a turn whose flange chord exceeds its envelope. Before a motion
starts, the runner walks the planned flange path with an FR3 kinematic model from the logged joints and refuses
what would fault there, telling the model how far the arm can go. The walk is skipped when the logged joints do
not reproduce the logged flange pose (the synthetic plant).
"""
import math

import numpy as np
from scipy.spatial.transform import Rotation as R

from forceharness.real import kinematics as fr3
from forceharness.real.geometry import check_flange_workspace
from forceharness.real.primitives import WRIST_Q7_RAD
from forceharness.real.wrist_overlay import PAD_CENTRE_TCP_Z

# TCP = centre of the black finger pads (Desk TCP + 15 mm, see wrist_overlay): the Desk TCP put the pads
# 13 mm below a commanded point.
TCP_TRANSLATION = (0.0, 0.0, PAD_CENTRE_TCP_Z)
# The robot side rejects a desired flange pose > 120 mm from the motion start; leave the same 15 mm
# (lead + compensation) margin as the 105 mm move_to reach.
TURN_FLANGE_REACH_M = .105
# The walk refuses only what the robot side would fault on: a joint within 0.05 rad of its limit (robot side
# 0.03), the flange leaving the workspace box, or one joint moving > 0.85 rad in this motion (robot side 0.9).
TURN_JOINT_MARGIN = .05
TURN_JOINT_DELTA = .85
# Holding an object, free-moving joint friction and the load itself read up to about 8 N during small held
# turns, so the collision stop of held motions is 12 N; free motion keeps 8 N.
HELD_TURN_COLLISION_N = 12.


def vector(v):
    a = np.asarray(v, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError('Expected a finite three-vector')
    return a


def turn_flange_chord(backend, args):
    """Flange displacement of a rigid turn, and the largest angle that fits the reach."""
    axis = vector(args['axis'])/np.linalg.norm(vector(args['axis']))
    pivot = vector(args['pivot'])
    flange = np.asarray(backend._state['flange_pos'], dtype=float)
    lever = flange-pivot
    radius = float(np.linalg.norm(lever-(lever@axis)*axis))
    chord = 2*radius*abs(math.sin(math.radians(args['angle_deg'])/2))
    limit = 30. if radius <= TURN_FLANGE_REACH_M/2 else math.degrees(2*math.asin(TURN_FLANGE_REACH_M/(2*radius)))
    return chord, math.floor(limit)


def hand_tilt_deg(backend):
    """Angle between the gripper approach axis and straight down (0 = pointing down)."""
    approach = (R.from_quat(backend.eef_quat)).apply([0, 0, 1])
    return math.degrees(math.acos(float(np.clip(-approach[2], -1, 1))))


def arc_angle_deg(axis, pivot, p_ref, p_now):
    """Signed rotation (deg, right hand about axis) carrying p_ref to p_now around the pivot."""
    a = np.asarray(axis, dtype=float)/np.linalg.norm(axis)
    u, v = np.asarray(p_ref, dtype=float)-pivot, np.asarray(p_now, dtype=float)-pivot
    u, v = u-(u@a)*a, v-(v@a)*a
    return math.degrees(math.atan2(float(a@np.cross(u, v)), float(u@v)))


def rebased_quat(grasp_ref, axis, pivot, eef_pos):
    """Grasp orientation turned by the held object's real angle (TCP position around the pivot)."""
    a = vector(axis)/np.linalg.norm(vector(axis))
    phi = math.radians(arc_angle_deg(a, vector(pivot), grasp_ref[0], eef_pos))
    return (R.from_rotvec(a*phi)*R.from_quat(grasp_ref[1])).as_quat()


def _start_pose(backend):
    """(joints, flange pose) when the logged joints reproduce the logged flange (not the synthetic plant)."""
    s = backend._state
    if 'joints' not in s or 'flange_quat' not in s:
        return None
    q0 = np.asarray(s['joints'], dtype=float)
    T0 = np.eye(4)
    T0[:3, :3] = R.from_quat(s['flange_quat']).as_matrix()
    T0[:3, 3] = s['flange_pos']
    if np.linalg.norm(fr3.fk(q0)[:3, 3]-T0[:3, 3]) > .001:
        return None
    return q0, T0


def _walk(q0, poses, segment=None):
    """How many flange poses (in order) the arm follows from q0 before a robot-side limit, and which limit.
    segment: poses per robot-side motion (rotate_z segments); the joint travel limit restarts with each."""
    q = q0

    def wrist_out(q):          # how far joint 7 is outside the range the wrist cable allows
        return max(WRIST_Q7_RAD[0]-q[6], q[6]-WRIST_Q7_RAD[1], 0.)
    wrist_start = wrist_out(q0)
    for i, T in enumerate(poses):
        if segment and i and i % segment == 0:
            q0 = q
        try:
            check_flange_workspace(T[:3, 3])
        except ValueError:
            return i, "the hand's flange would leave the workspace box (x 0.20–0.65, |y| < 0.30, z 0.12–0.55 m)"
        q, err = fr3.ik(q, T)
        if err > 1e-4:
            return i, 'the arm cannot reach this pose'
        if fr3.margin(q)[0] < TURN_JOINT_MARGIN:
            return i, 'a joint would hit its limit'
        if wrist_out(q) > wrist_start+.01:
            return i, (f'the wrist (joint 7) would turn out of {WRIST_Q7_RAD[0]:g}–{WRIST_Q7_RAD[1]:g} rad '
                       '(turning further would pull the cable tight)')
        if np.abs(q-q0).max() > TURN_JOINT_DELTA:
            return i, 'a joint would turn by more than 0.85 rad in one action'
    return len(poses), None


def translation_reach(backend, delta, step=.005, segment_m=None):
    """Metres of a straight translation (orientation fixed) the arm can follow, and why it stops.
    segment_m: the translation runs as several robot-side motions of at most this length."""
    d = np.asarray(delta, dtype=float)
    dist = float(np.linalg.norm(d))
    start = _start_pose(backend)
    if start is None or dist < 1e-9:
        return dist, None
    q0, T0 = start
    n = max(1, int(math.ceil(dist/step)))
    poses = []
    for i in range(1, n+1):
        T = T0.copy()
        T[:3, 3] = T0[:3, 3]+d*i/n
        poses.append(T)
    k, why = _walk(q0, poses, max(1, int(segment_m/(dist/n))) if segment_m else None)
    return dist*k/n, why


def rotation_reach(backend, axis, pivot, angle_deg, segment_deg=None):
    """Degrees of a rigid hand rotation about axis through pivot the arm can follow, and why it stops.
    segment_deg: the rotation runs as several robot-side motions of at most this many degrees (rotate_z)."""
    requested = abs(float(angle_deg))
    start = _start_pose(backend)
    if start is None or requested < 1e-9:
        return requested, None
    q0, T0 = start
    a = vector(axis)/np.linalg.norm(vector(axis))
    pv, sign = vector(pivot), math.copysign(1., float(angle_deg))
    segments = int(math.ceil(requested/segment_deg)) if segment_deg else 1
    n = segments*max(1, int(math.ceil(requested/segments)))      # ~1 deg steps, whole steps per segment
    poses = []
    for deg in np.linspace(0, requested, n+1)[1:]:
        rot = R.from_rotvec(a*math.radians(sign*deg))
        T = np.eye(4)
        T[:3, :3] = (rot*R.from_matrix(T0[:3, :3])).as_matrix()
        T[:3, 3] = pv+rot.apply(T0[:3, 3]-pv)
        poses.append(T)
    k, why = _walk(q0, poses, n//segments)
    return requested*k/n, why
