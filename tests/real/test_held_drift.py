"""FR3 force drift while holding: the held tare drifts by 2-4 N once the hand moves (a 6 g eraser read hundreds of
grams after a grasp and lift, and the tared drift faked a contact above a board), and a held turn whose pivot sits at the
TCP must not rebase from a noise angle. Offline only: synthetic plant."""
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from forceharness.real import runner as free
from forceharness.real.checks import arc_angle_deg

from real_helpers import PROMPTS


def angle_deg(q0, q1):
    return math.degrees((R.from_quat(q1)*R.from_quat(q0).inv()).magnitude())


def test_no_rebase_when_the_pivot_sits_at_the_tcp():
    # a logged turn: grasp TCP, pivot = current TCP (0.1 mm apart), axis +x
    grasp = (np.array([.5303, -.1345, .0265]), R.from_euler('x', 180, degrees=True).as_quat())
    pivot, now = np.array([.5313, .0744, .0532]), np.array([.5313, .0744, .0531])
    assert abs(arc_angle_deg([1, 0, 0], pivot, grasp[0], now)) > 30          # the noise angle the bug used
    current = R.from_euler('x', 181, degrees=True).as_quat()
    start, record = free.held_turn_start(grasp, [1, 0, 0], pivot, now, current)
    assert np.allclose(start, current) and not record['applied'] and '2 cm' in record['reason']


def test_rebase_kept_for_a_held_object_turning_about_its_support():
    # a bottle grasped at the cap, pivot at its base 0.18 m below, turned 10 deg about +y
    pivot = np.array([.45, 0., .02])
    grasp = (np.array([.45, 0., .20]), R.from_euler('x', 180, degrees=True).as_quat())
    turned = R.from_rotvec([0, math.radians(10), 0])
    now = pivot+turned.apply(grasp[0]-pivot)
    for twist, applied in ((2., True), (8., False)):
        current = (R.from_rotvec([0, math.radians(10+twist), 0])*R.from_quat(grasp[1])).as_quat()
        start, record = free.held_turn_start(grasp, [0, 1, 0], pivot, now, current)
        assert record['applied'] is applied and record['offset_deg'] == pytest.approx(twist, abs=.2)
        expected = (turned*R.from_quat(grasp[1])).as_quat() if applied else current
        assert angle_deg(start, expected) < .1


def held_bottle(b, world):
    """Demo: grasp bottle A at z 0.16, carry it up 3 cm (the carry tare is taken there)."""
    p = free.FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9',
                            interface='force')
    p.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
    x, y = world.objects[0]['xy']
    for target in ((.425, y/2, .25), (x, y, .25), (x, y, .16)):
        assert p.call('move_to', x=target[0], y=target[1], z=target[2]).termination == 'completed'
    assert p.call('grasp').termination == 'completed' and b._closed
    assert p.call('move_to', x=x, y=y, z=.19).termination == 'completed'
    assert p.carry_bias[2] == pytest.approx(-.32*free.G, abs=.1)                # the bottle's weight
    return p, x, y


@pytest.mark.parametrize('guarded', [True, False])
def test_held_turn_with_the_pivot_at_the_tcp_turns_only_the_requested_angle(monkeypatch, guarded):
    if not guarded:                                    # without the guard: always rebase
        monkeypatch.setattr(free, 'held_turn_start', lambda grasp_ref, axis, pivot, eef_pos, eef_quat: (
            free.rebased_quat(grasp_ref, axis, pivot, eef_pos), dict(applied=True)))
    b, world, _ = free.demo()
    with b:
        p, x, y = held_bottle(b, world)
        assert p.call('move_to', x=x, y=y-.05, z=.19).termination == 'completed'
        pivot = np.asarray(b.eef_pos, dtype=float)+[0, 0, .002]
        assert abs(arc_angle_deg([1, 0, 0], pivot, p.grasp_ref[0], b.eef_pos)) > 30
        q0 = np.asarray(b.eef_quat, dtype=float).copy()
        if not guarded:
            try:
                p.call('turn', axis=[1, 0, 0], pivot=pivot.tolist(), angle_deg=20)
            except ValueError as exc:                 # at 0.15 rad/s the overshoot reaches the envelope
                assert 'envelope' in str(exc)
                return
            assert angle_deg(q0, b.eef_quat) > 25      # the overshoot, reproduced
            return
        rep = p.call('turn', axis=[1, 0, 0], pivot=pivot.tolist(), angle_deg=20)
        assert rep.termination == 'completed' and not b.fault
        assert rep.derived['held_turn_rebase']['applied'] is False
        assert angle_deg(q0, b.eef_quat) == pytest.approx(20, abs=1.5)
        assert p.held_angle_deg(np.array([1., 0, 0]), pivot) is None


@pytest.mark.parametrize('rezero', [True, False])
def test_held_contact_search_zeroes_in_the_air(monkeypatch, rezero):
    b, world, _ = free.demo()
    with b:
        p, x, y = held_bottle(b, world)
        b.port.driver.bias[2] += 4.                    # the estimate drifts 4 N up after the tare
        if not rezero:                                 # without the held re-zero: keep the stale tare
            monkeypatch.setattr(free, 'held_reading', lambda backend: p.carry_bias.copy())
        rep = p.call('move_until_contact', direction=[0, 0, -1], max_dist=.02, f_threshold=3.)
        if rezero:
            # the bottle hangs 15 cm above the synthetic table: no contact within 2 cm
            assert rep.termination != 'contact' and not any(e.kind == 'contact_start' for e in rep.events)
            assert rep.derived['held_contact_zero']['shift_N'][2] == pytest.approx(4., abs=.1)
        else:
            assert rep.termination == 'contact'        # the false contact, reproduced


def test_receipt_and_prompts_state_the_real_force_error(tmp_path):
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'),
                      '--instruction', 'pick up the left bottle, weigh it and put it back'])
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert code == 0 and 'runner_fixes' not in r
    lifted = next(x for x in r['actions'] if x.get('weight_since_grasp_g'))
    text = lifted['receipt']['text']
    assert f'error of up to ±{free.WEIGHT_ERROR_G} g' in text
    force = (PROMPTS/'free_task_force.md').read_text()
    baseline = (PROMPTS/'free_task_baseline.md').read_text()
    assert '±400 g' in force and 're-zeroes' in force and 'raw force limit of 25 N' in force
    for prompt in (force, baseline):
        assert 'check by height' in prompt
    assert '±400 g' not in baseline and 'move_until_contact' not in baseline


def test_a_closed_gripper_at_the_start_is_named_before_any_motion(tmp_path, monkeypatch):
    # an object still held (19 mm) from a stopped episode: the second action used to fail with a misleading
    # "Verified force calibration is missing" message
    original = free.demo

    def holding():
        b, world, calls = original()
        b.port.driver.width = .019
        return b, world, calls
    monkeypatch.setattr(free, 'demo', holding)
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--instruction', 'x'])
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert code != 0 and r['actions'] == [] and 'The gripper is not open at the start (width 19 mm' in r['fault']
    assert not b_moved(r)


def b_moved(record):
    return bool(record.get('scene_updates')) or bool(record.get('zero_check'))
