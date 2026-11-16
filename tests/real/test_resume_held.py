"""Holding at the start and the torque cap: a held lift's torque estimate can climb to the 3 N*m cap with nothing
stuck; the motion stops and the episode goes on with a tool result. --resume-held starts an episode holding what the
operator names. Offline only: synthetic plant."""
import json

import pytest

from forceharness.real import runner as free

UP, DOWN = dict(dx=0, dy=0, dz=.01), dict(dx=0, dy=0, dz=-.01)
DONE = dict(name='done', args=dict(success=True, note='synthetic scene only'))


def run(tmp_path, monkeypatch, calls, width=None, extra=()):
    original = free.demo

    def plant():
        b, world, _ = original()
        if width is not None:
            b.port.driver.width = width          # closed on something, as left by a stopped episode
        return b, world, calls
    monkeypatch.setattr(free, 'demo', plant)
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'),
                      '--instruction', 'wipe the writing off the board', *extra])
    return code, json.loads((tmp_path/'ep'/'episode.json').read_text())


def test_resume_held_starts_holding_and_tells_the_model(tmp_path, monkeypatch):
    calls = [dict(name='move_by', args=UP), dict(name='grasp', args={}), dict(name='release', args={}), DONE]
    code, r = run(tmp_path, monkeypatch, calls, width=.0196, extra=('--resume-held', 'eraser'))
    assert code == 0 and r['fault'] is None and 'runner_fixes' not in r
    held = r['resumed_holding']
    assert held['object'] == 'eraser' and held['width_m'] == pytest.approx(.0196) and len(held['carry_bias_N']) == 3
    moved, grasp, release = r['actions'][:3]
    assert moved['termination'] == 'completed'
    assert 'relative to the hand at the start, which already held the eraser' in moved['receipt']['text']
    assert grasp['invalid'] == 'the gripper is already closed'                 # held, not unknown: no second close
    assert release['termination'] == 'completed' and r['held_state'] is False
    first = (tmp_path/'relay'/'obs_0.md').read_text()
    assert 'At the start the gripper is already closed, holding "eraser"' in first and 'closed width 20 mm' in first


def test_resume_held_needs_a_grasp_width(tmp_path, monkeypatch):
    code, r = run(tmp_path, monkeypatch, [DONE], width=.0005, extra=('--resume-held', 'eraser'))
    assert code != 0 and r['actions'] == [] and 'A gripper width of 0.5 mm does not look like holding something (it must be between 1 and 75 mm)' in r['fault']
    assert not r.get('zero_check') and not r.get('scene_updates')       # refused before any motion or photo


def test_resume_held_accepts_a_thin_flat_grip(tmp_path, monkeypatch):
    code, r = run(tmp_path, monkeypatch, [DONE], width=.0019, extra=('--resume-held', 'peeler'))
    assert code == 0 and r['fault'] is None and r['resumed_holding']['width_m'] == pytest.approx(.0019)


def test_resume_held_with_an_open_gripper_starts_empty(tmp_path, monkeypatch):
    code, r = run(tmp_path, monkeypatch, [DONE], extra=('--resume-held', 'eraser'))
    assert code == 0 and 'resumed_holding' not in r and 'nothing held' in r['resume_held_ignored']
    assert 'At the start the gripper is already closed' not in (tmp_path/'relay'/'obs_0.md').read_text()


def test_a_closed_start_without_the_operator_names_the_option(tmp_path, monkeypatch):
    code, r = run(tmp_path, monkeypatch, [DONE], width=.0196)
    assert code != 0 and 'The gripper is not open at the start (width 20 mm' in r['fault'] and '--resume-held' in r['fault']


@pytest.mark.parametrize('kind,condition', [('torque_cap', 'A'), ('torque_cap', 'B'), ('force_cap', 'A')])
def test_torque_cap_stops_the_motion_and_the_episode_goes_on(tmp_path, monkeypatch, kind, condition):
    original = free.FreePrimitives.call

    def capped(self, name, **kwargs):
        rep = original(self, name, **kwargs)
        if name == 'move_by' and kwargs.get('dz', 0) > 0:
            rep.termination, rep.peak_torque = kind, 3.008     # a held lift as logged
        return rep
    monkeypatch.setattr(free.FreePrimitives, 'call', capped)
    calls = [dict(name='move_by', args=UP), dict(name='move_by', args=DOWN), DONE]
    code, r = run(tmp_path, monkeypatch, calls, extra=('--condition', condition))
    first = r['actions'][0]
    if kind == 'torque_cap':
        assert code == 0 and r['fault'] is None and r['finished'] and len(r['actions']) == 3
        note = free.TORQUE_STOP_NOTE if condition == 'A' else free.TORQUE_STOP_NOTE_B
        assert first['torque_stop_Nm'] == pytest.approx(3.008) and note in first['receipt']['text']
        assert r['actions'][1]['termination'] == 'completed'
    else:
        assert code != 0 and r['fault'].startswith('BackendError: Primitive stopped: force_cap') and len(r['actions']) == 1
