"""Posture model of the force zero and its one-time calibration, reused across episodes (synthetic plant)."""
import json

import numpy as np
import pytest

from forceharness.real import posture_bias as pb
from forceharness.real import posture_calibration as cal

# One logged still state: joints, the external joint torques, and the wrench the robot side solved from them.
LOGGED = dict(joints=[0.02726062759757042, 0.24772903323173523, 0.02015535533428192, -2.321366548538208,
                      0.025355558842420578, 2.535597324371338, 0.8477513790130615],
              motor_torques_external=[0.5959728956222534, -0.838927686214447, 0.5801296234130859, 0.6042605042457581,
                                      0.14677752554416656, 0.40923893451690674, 0.15302522480487823],
              wrench=[-0.351672676264795, 1.412531041961281, 0.98308776729091, 0.04298907196754132,
                      -0.3593389519496064, -0.1677481633838547])


def test_wrench_mapping_matches_the_robot_side_reader():
    s = LOGGED
    w = pb.wrench_from_tau(s['joints'], s['motor_torques_external'])
    assert np.allclose(w, s['wrench'], atol=1e-3)


def test_fit_recovers_a_synthetic_posture_bias_and_one_pose_zero_check():
    rng = np.random.default_rng(3)
    beta = np.zeros(pb.N_PARAMS)
    beta[4*7:4*7+4] = [.08, .004, -.003, .006]        # hand + camera mass / first-moment error
    beta[4*5+2] = .01
    base = np.array([0., .3, 0., -2.2, 0., 2.5, .8])
    qs = [base+rng.normal(0, .25, 7) for _ in range(60)]
    offsets = {'day1': rng.normal(0, .2, 7), 'day2': rng.normal(0, .2, 7)}
    samples = [dict(q=q, tau=offsets[f'day{1+i % 2}']+pb.regressor(q)@beta, session=f'day{1+i % 2}')
               for i, q in enumerate(qs)]
    fitted, _, residual = pb.fit(samples, ridge=1e-6)
    assert np.abs(residual).max() < 1e-3
    # A new power-on: one still pose gives the seven offsets, every other pose is predicted.
    new = rng.normal(0, .2, 7)
    q0, q1 = base, base+np.array([.2, -.1, .1, .3, -.2, -.6, .4])
    d = pb.session_offset(q0, new+pb.regressor(q0)@beta, fitted)
    truth = pb.wrench_from_tau(q1, new+pb.regressor(q1)@beta)
    assert np.allclose(pb.predicted_wrench(q1, fitted, d), truth, atol=1e-3)


def test_calibration_runs_once_and_is_reused(tmp_path):
    from forceharness.real.runner import FreePrimitives, demo
    b, _, _ = demo()
    path = tmp_path/'posture_bias_fr3.json'
    assert cal.load(path) is None                        # first use: nothing to reuse yet
    with b:
        p = FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9')
        start = b.eef_pos.copy()
        doc = cal.run_calibration(p, path, out_dir=tmp_path/'run', log=lambda m: None)
        assert np.linalg.norm(b.eef_pos-start) < .01      # back where the episode will start
    samples = json.loads((tmp_path/'run/posture_samples.json').read_text())
    assert len(samples) == len(cal.STATION_OFFSETS)*len(cal.TILTS)
    assert {tuple(s['tilt']) for s in samples} >= {(0, 0), (60, 0), (-30, 0), (0, 30), (0, -30)}
    assert doc['use_model'] is False                     # demo port has no joint torques
    reused = cal.load(path)
    assert reused is not None and reused.meta['created'] == doc['created']


def test_stuck_wrist_turn_stops_with_a_tool_result_before_the_robot_side_stall_fault():
    from forceharness.real.runner import FreePrimitives, demo
    b, _, _ = demo()
    with b:
        p = FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9')
        b.port.driver.frozen = True                     # joint 6 held by static friction
        rep = p.call('turn', axis=[0, 1, 0], pivot=b.eef_pos.tolist(), angle_deg=20)
        assert rep.termination == 'stalled' and not b.fault
        assert 3.2 <= rep.duration < 4.
        assert rep.derived['rotation_progress']['version'] == 'r33c'
        b.port.driver.frozen = False                    # the session is still usable
        rep = p.call('turn', axis=[0, 1, 0], pivot=b.eef_pos.tolist(), angle_deg=10)
        assert rep.termination == 'completed' and not b.fault
