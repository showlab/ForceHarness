"""Pad-centre TCP, the wrist overlay, and the kinematic checks of held turns (flange reach, joint walk, rebase)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from forceharness.real import kinematics as k
from forceharness.real import runner as free
from forceharness.real.checks import TCP_TRANSLATION, arc_angle_deg, hand_tilt_deg, rebased_quat, turn_flange_chord
from forceharness.real.config import BackendConfig
from forceharness.real.wrist_overlay import PAD_CENTRE_TCP_Z, overlay
from real_helpers import make_scene

DOWN = R.from_quat([1., 0, 0, 0])                             # gripper pointing straight down
# A logged cap grasp: joints, and the hand orientation of that grasp
GRASP_Q = np.array([.0314, .5222, .0203, -2.2148, .0250, 2.7004, .8518])
GRASP_HAND = R.from_quat([.9158370494842529, -.40100258588790894, -.02092854119837284, -.0012104988563805819])
# A logged still state (joints and the flange pose they produce)
LOGGED = dict(joints=[0.02726062759757042, 0.24772903323173523, 0.02015535533428192, -2.321366548538208,
                      0.025355558842420578, 2.535597324371338, 0.8477513790130615],
              flange_pos=[0.5186266303062439, 0.024504657834768295, 0.23118089139461517],
              flange_quat=[0.916561484336853, -0.3994295597076416, -0.019216977059841156, -0.0013357368297874928])


def test_pad_centre_tcp_is_desk_tcp_plus_15mm():
    assert TCP_TRANSLATION == (0., 0., PAD_CENTRE_TCP_Z)
    assert PAD_CENTRE_TCP_Z == pytest.approx(.1034+.015, abs=.001)


def test_overlay_reports_target_offset_and_ghost(tmp_path):
    cfg = BackendConfig(tcp_translation=TCP_TRANSLATION)
    scene = make_scene(tmp_path/'scene', tcp=(.45, .02, .20))
    target = np.array([.47, .03, .03])
    info = overlay(scene, tmp_path/'o.png', cfg.tcp_translation, cfg.tcp_quaternion, target=target.tolist(),
                   target_z=float(target[2]))
    assert (tmp_path/'o.png').is_file()
    tcp = np.array(info['tcp'])
    assert np.allclose(tcp, [.45, .02, .20], atol=1e-4)
    assert np.allclose(info['target_minus_tcp_mm'], (target-tcp)*1000, atol=.2)
    assert info['aim_point'][2] == pytest.approx(target[2])
    assert 0 < info['aim_px'][0] < 640 and 0 < info['aim_px'][1] < 480
    held = overlay(scene, tmp_path/'h.png', cfg.tcp_translation, cfg.tcp_quaternion)
    assert 'aim_point' not in held and 'target_px' not in held


def backend_at(flange, closed=False):
    tcp = np.asarray(flange)+DOWN.apply([0, 0, PAD_CENTRE_TCP_Z])
    return SimpleNamespace(_state=dict(flange_pos=list(flange)), eef_pos=tcp, eef_quat=DOWN.as_quat(), _closed=closed)


def test_held_turn_chord_limit_matches_the_robot_side_envelope():
    pivot = [.3592, .0305, .0061]
    b = backend_at([.5365, .0305, .0282+PAD_CENTRE_TCP_Z], closed=True)
    args = dict(axis=[0, -1, 0], pivot=pivot, angle_deg=30, stop_on_resistance_drop=True)
    chord, limit = turn_flange_chord(b, args)
    assert chord == pytest.approx(.117, abs=.002)     # 30 deg would sit 3 mm under the 120 mm robot-side limit
    assert 25 <= limit < 30
    with pytest.raises(ValueError, match=f'turn at most {limit}° per call'):
        free.validate('turn', args, b, 'A')
    free.validate('turn', dict(args, angle_deg=20), b, 'A')


def test_hand_tilt():
    b = backend_at([.5, 0, .2])
    assert hand_tilt_deg(b) == pytest.approx(0, abs=1e-6)
    b.eef_quat = (R.from_rotvec([0, -math.radians(30), 0])*DOWN).as_quat()
    assert hand_tilt_deg(b) == pytest.approx(30, abs=1e-6)


def held_backend(cap, tilt_deg=0., pivot=None, axis=(0, 1, 0)):
    """Backend state with joints consistent with a cap grasp, optionally already turned."""
    flange = np.asarray(cap)-GRASP_HAND.apply([0, 0, PAD_CENTRE_TCP_Z])
    turn = R.from_rotvec(np.radians(tilt_deg)*np.asarray(axis, dtype=float))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = (turn*GRASP_HAND).as_matrix(), np.asarray(pivot)+turn.apply(flange-pivot)
    q = GRASP_Q
    for step in np.linspace(0, 1, 30)[1:]:                 # follow the arc so the IK stays on one branch
        part = R.from_rotvec(np.radians(tilt_deg*step)*np.asarray(axis, dtype=float))
        Ts = np.eye(4)
        Ts[:3, :3], Ts[:3, 3] = (part*GRASP_HAND).as_matrix(), np.asarray(pivot)+part.apply(flange-pivot)
        q, _ = k.ik(q, Ts, iters=300)
    quat = R.from_matrix(T[:3, :3]).as_quat()
    tcp_quat = (R.from_quat(quat)*R.from_quat(BackendConfig().tcp_quaternion)).as_quat()
    return SimpleNamespace(_state=dict(flange_pos=T[:3, 3].tolist(), flange_quat=quat.tolist(), joints=q.tolist()),
                           eef_pos=T[:3, 3]+R.from_quat(quat).apply([0, 0, PAD_CENTRE_TCP_Z]), eef_quat=tcp_quat,
                           _closed=True)


def test_turn_feasibility_follows_the_arm_not_fixed_caps():
    # a bottle lying with its cap toward the robot at x 0.375, turned upright about its bottom edge
    cap, pivot = [.375, .03, .0308], [.574, .03, .006]
    turn = dict(axis=[0, 1, 0], pivot=pivot, stop_on_resistance_drop=True)
    start = held_backend(cap, 0, pivot)
    free.validate('turn', dict(turn, angle_deg=20), start, 'A')
    with pytest.raises(ValueError, match='turn at most 2[0-9]° per call'):    # the flange would move > 105 mm
        free.validate('turn', dict(turn, angle_deg=30), start, 'A')
    near_end = held_backend(cap, 62, pivot)
    assert hand_tilt_deg(near_end) == pytest.approx(62, abs=3)    # the real grasp pose starts ~2 deg off vertical
    with pytest.raises(ValueError, match='at most about [0-9]+° more') as info:
        free.validate('turn', dict(turn, angle_deg=20), near_end, 'A')
    allowed = int(str(info.value).split('angle_deg ≤ ')[1])
    assert 3 <= allowed <= 12                       # ~70 deg in total for this placement (offline IK)
    free.validate('turn', dict(turn, angle_deg=allowed), near_end, 'A')


def test_fr3_kinematics_matches_the_logged_pose_and_the_wrist_runs_out_for_a_cap_pointing_away():
    T = k.fk(LOGGED['joints'])
    assert np.allclose(T[:3, 3], LOGGED['flange_pos'], atol=1e-6)
    assert (R.from_matrix(T[:3, :3]).inv()*R.from_quat(LOGGED['flange_quat'])).magnitude() < 1e-6

    def worst(cap, pivot, axis):          # a held turn from a cap grasp, 0 -> 78 deg about the bottom edge
        flange = np.asarray(cap)-GRASP_HAND.apply([0, 0, PAD_CENTRE_TCP_Z])
        targets = []
        for deg in np.arange(0, 78.01, 1.):
            turn = R.from_rotvec(np.radians(deg)*np.asarray(axis, dtype=float))
            Tt = np.eye(4)
            Tt[:3, :3], Tt[:3, 3] = (turn*GRASP_HAND).as_matrix(), pivot+turn.apply(flange-pivot)
            targets.append(Tt)
        return min(m for _, m, _, _ in k.track(GRASP_Q, targets))
    assert worst([.539, .0295, .0308], np.array([.340, .03, .006]), [0, -1, 0]) < 0          # cap +x: infeasible
    assert worst([.31, .03, .0308], np.array([.509, .03, .006]), [0, 1, 0]) > k.NUC_JOINT_MARGIN  # cap toward robot


def test_held_turn_uses_the_object_angle_not_the_hand_tilt():
    # grasp TCP and the TCP at a release, pivot and axis as a model used them
    pivot, axis = np.array([.5699, .0211, .0061]), np.array([-.0323, .9995, 0.])
    grasp_tcp, release_tcp = np.array([.374, .012, .032]), np.array([.488, .014, .189])
    phi = arc_angle_deg(axis, pivot, grasp_tcp, release_tcp)
    assert phi == pytest.approx(58.6, abs=1.)            # the hand had turned 66.7 deg (8.6 deg twist)
    grasp_quat = (GRASP_HAND*R.from_quat(BackendConfig().tcp_quaternion)).as_quat()
    q = R.from_quat(rebased_quat((grasp_tcp, grasp_quat), axis, pivot, release_tcp))
    turned = q*R.from_quat(grasp_quat).inv()
    assert math.degrees(turned.magnitude()) == pytest.approx(phi, abs=.5)
    assert np.allclose(turned.as_rotvec()/turned.magnitude(), axis/np.linalg.norm(axis), atol=1e-3)
