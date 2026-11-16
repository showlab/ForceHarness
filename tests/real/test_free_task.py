"""The runner with the free task (the operator types the task; every primitive of the condition is on the menu).

Offline only: the synthetic plant with two synthetic bottles, and the live branch of main() with the SSH port, the
cameras and the relay replaced by fixtures (synthetic captures)."""
import copy
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forceharness.real import runner as free
from forceharness.agent.llm import LLMResponse, ToolCall
from forceharness.real import scene_objects
from forceharness.real.primitives import ContinuousPrimitives
from forceharness.real.types import BackendError
from real_helpers import CALIBRATION, make_scene, photographer_from

FORCE_ONLY = {'move_until_contact', 'probe', 'push', 'slide', 'search', 'insert', 'lift', 'pull'}


def run(tmp_path, *args):
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), *args])
    return code, json.loads((tmp_path/'ep'/'episode.json').read_text())


def test_menus_open_every_primitive_and_keep_the_baseline_without_force():
    a, b = free.CONDITIONS['A']['menu'], free.CONDITIONS['B']['menu']
    assert set(a) == set(free.PRIMITIVES)-set(free.TASK_ONLY) and FORCE_ONLY-set(free.TASK_ONLY) <= set(a)
    assert not FORCE_ONLY & set(b) and {'move_to', 'turn', 'point_tool', 'grasp', 'release', 'done'} <= set(b)
    for name in a+list(free.TASK_ONLY):          # task-only primitives join a menu through a task preset
        assert name in ('scene', 'locate', 'done') or callable(getattr(ContinuousPrimitives, name, None))
    assert 'stop_on_resistance_drop' in free.signature('turn', 'A') and 'stop_on_resistance_drop' not in free.signature('turn', 'B')
    text = free.menu_text('B', free.GRIP_RULES['auto'])
    assert 'move_until_contact' not in text and '{grasp_rule}' not in text and 'Close the gripper' in text


def test_scripted_demo_finds_the_weight_in_A_and_gives_no_force_in_B(tmp_path):
    code, r = run(tmp_path/'A', '--instruction', 'pick up the left bottle, weigh it and put it back')
    assert code == 0 and r['fault'] is None and r['finished'] and r['version'] == 'free_task/1'
    assert r['instruction'] == 'pick up the left bottle, weigh it and put it back' and r['condition'] == 'phys2_force'
    lifted = next(x for x in r['actions'] if x['name'] == 'move_by' and x.get('weight_since_grasp_g'))
    assert lifted['weight_since_grasp_g'] == pytest.approx(320, abs=5)           # synthetic bottle A, 0.32 kg
    # the weight reference is the empty hand at the grasp pose, read before the close
    assert next(x for x in r['actions'] if x['name'] == 'grasp')['weight_reference'] == 'before_grasp'
    obs = (tmp_path/'A'/'relay'/'obs_0.md').read_text()
    assert 'pick up the left bottle, weigh it and put it back' in obs and 'move_until_contact(' in obs and '#2 center' in obs
    code, r = run(tmp_path/'B', '--condition', 'B', '--instruction', 'pick up the left bottle and put it down again')
    assert code == 0 and r['finished'] and r['condition'] == 'none_guard'
    relay = ''.join(p.read_text() for p in (tmp_path/'B'/'relay').glob('obs_*.md'))
    assert 'still reading' not in relay and 'grams' not in relay and 'net force' not in relay
    assert 'move_until_contact(' not in relay


def test_prompt_is_filled_in_one_pass(tmp_path):
    code, r = run(tmp_path, '--instruction', 'literal {menu} and {table_z} stay', '--max-actions', '1')
    system = (tmp_path/'relay'/'obs_0.md').read_text()
    assert 'literal {menu} and {table_z} stay' in system and '- `scene()`' in system


def demo_backend():
    b, world, _ = free.demo()
    return b, world


def test_validate_refuses_before_any_motion():
    b, _ = demo_backend()
    with b:
        assert free.validate('move_by', dict(dx=.05, dy=0, dz=0), b, 'A')[0] == dict(dx=.05, dy=0, dz=0)
        for name, args, message in (
                ('move_to', dict(x=.60, y=0, z=.25), 'per move'),
                ('move_to', dict(x=.40, y=0), 'missing'),
                ('move_to', dict(x=.40, y=0, z=.25, force=1), 'unexpected'),
                ('move_until_contact', dict(direction=[0, 0, -1], max_dist=.05, f_threshold=12), 'f_threshold'),
                ('move_until_contact', dict(direction=[0, 0, 0], max_dist=.05), 'zero vector'),
                ('turn', dict(axis=[1, 0, 0], pivot=[.4, 0, .1], angle_deg=45), '|angle_deg|'),
                ('point_tool', dict(direction=[1, 0, 0]), 'per call'),
                ('release', {}, 'already open'),
                ('done', dict(note='x'), 'missing'),
                ('fly', {}, 'not on the menu')):
            with pytest.raises(ValueError, match=message.replace('|', r'\|')):
                free.validate(name, args, b, 'A')
        with pytest.raises(ValueError, match='not on the menu'):
            free.validate('lift', dict(height=.02), b, 'B')
        args, ignored = free.validate('turn', dict(axis=[1, 0, 0], pivot=[.4, 0, .1], angle_deg=10,
                                                   stop_on_resistance_drop=True), b, 'B')
        assert ignored == ['stop_on_resistance_drop'] and 'stop_on_resistance_drop' not in args


def test_reach_walks_the_real_arm_from_the_logged_joints():
    # a logged state: the FR3 joints reproduce the logged flange, so the IK walk runs
    state = dict(joints=[-0.0008282056078314781, -0.5011396408081055, 0.013265701942145824, -2.6334707736968994,
                         0.007745702285319567, 2.1053194999694824, 0.8444894552230835],
                 flange_pos=[0.37455424666404724, 0.004877594765275717, 0.40578076243400574],
                 flange_quat=[0.9134523868560791, -0.4067210257053375, -0.012349549680948257, 0.005497400648891926])
    b = SimpleNamespace(_state=state)
    flange = np.asarray(state['flange_pos'])
    feasible, why = free.translation_reach(b, [0, 0, .20])                         # past the z < 0.55 box edge
    assert 'workspace' in why and feasible == pytest.approx(.55-flange[2], abs=.006)
    feasible, why = free.translation_reach(b, [.40, 0, 0])                         # a joint would travel > 0.85 rad
    assert '0.85 rad' in why and .1 < feasible < .4
    assert free.translation_reach(b, [0, 0, -.05]) == (pytest.approx(.05), None)
    feasible, why = free.rotation_reach(b, [0, 0, 1], state['flange_pos'], 30)
    assert why is None and feasible == pytest.approx(30)


def test_regrasp_after_a_verified_open_and_an_empty_close_is_not_a_fault(tmp_path, monkeypatch):
    original = free.demo

    def scripted():
        b, world, _ = original()
        x, y = world.objects[1]['xy']
        calls = [dict(name='grasp', args={}),                                  # nothing between the fingers
                 dict(name='grasp', args={}),                                  # already closed: refused, no motion
                 dict(name='release', args={}),
                 dict(name='move_to', args=dict(x=x, y=y/2, z=.25)),
                 dict(name='move_to', args=dict(x=x, y=y, z=.25)),
                 dict(name='move_to', args=dict(x=x, y=y, z=.16)),
                 dict(name='grasp', args={}),                                  # second grasp, on bottle B
                 dict(name='move_by', args=dict(dx=0, dy=0, dz=.03)),
                 dict(name='move_by', args=dict(dx=0, dy=0, dz=-.03)),
                 dict(name='release', args={}),
                 dict(name='done', args=dict(success=True))]
        return b, world, calls
    monkeypatch.setattr(free, 'demo', scripted)
    code, r = run(tmp_path, '--instruction', 'test')
    assert code == 0 and r['fault'] is None and r['backend_fault'] is None
    first, again, *_ = r['actions']
    assert 'Empty grasp' in first['receipt']['text'] and again['invalid'] == 'the gripper is already closed'
    assert not any(x.get('invalid') for x in r['actions'][2:])
    second = r['actions'][6]
    assert second['termination'] == 'completed' and 'Holding something: closed width 65 mm' in second['receipt']['text']
    assert r['actions'][7]['weight_since_grasp_g'] == pytest.approx(150, abs=5)    # bottle B, 0.15 kg


def test_a_stuck_wrist_rotation_is_stopped_as_a_receipt():
    # static friction of a joint can hold an in-place turn; stop before the robot side's 4 s stall fault
    b, _ = demo_backend()
    with b:
        p = free.FreePrimitives(b, confirm=lambda kind, state: True, tick=lambda: None, stop_protocol='r9', interface='force')
        p.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
        b.port.driver.frozen = True                              # the plant ignores every new target
        rep = p.call('rotate_z', angle_deg=20, rate_deg_s=2.)          # 8 deg/s would finish before 3.2 s
        assert rep.termination == 'stalled' and rep.derived['relay_guard']['kind'] == 'stalled' and not b.fault
        assert 3.2 <= rep.duration < 4.
        assert rep.derived['rotation_progress']['version'] == 'r33c'
        b.port.driver.frozen = False
        assert p.call('rotate_z', angle_deg=5).termination == 'completed'


def test_closes_per_episode_are_not_limited():
    # screwing needs release / turn back / grip cycles: the runner's port has no trial budget
    b, _ = demo_backend()
    with b:
        assert b.grasps_left() is None
        b.port.trial, b._grasp_attempted, b._closed = 6, True, False
        free.validate('grasp', {}, b, 'A')
        b.port.released = True
        b.port.next_weigh_trial(7)                              # a seventh trial after a confirmed release
        with pytest.raises(BackendError, match='Sequential'):
            b.port.next_weigh_trial(9)                          # still one at a time
        b.port.max_trials = 6                                   # a port with a budget is still enforced
        b.port.trial = 6
        assert b.grasps_left() == 0
        with pytest.raises(ValueError, match='6 times'):
            free.validate('grasp', {}, b, 'A')


def test_score_keeps_the_first_operator_outcome(tmp_path):
    run(tmp_path, '--instruction', 'x', '--max-actions', '1')
    assert free.main(['score', '--out', str(tmp_path/'ep'), '--outcome', 'partial', '--note', 'half']) == 0
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert r['outcome'] == 'partial' and r['success'] is False and r['score_note'] == 'half'
    with pytest.raises(SystemExit):
        free.main(['score', '--out', str(tmp_path/'ep'), '--outcome', 'success'])


def test_objects_on_a_synthetic_capture(tmp_path):
    folder = make_scene(tmp_path/'scene', boxes=[dict(centre=(.45, .04), size=(.20, .065), height=.06)])
    found = scene_objects.detect(folder, free.TCP_TRANSLATION)
    assert len(found['objects']) == 1
    bottle = found['objects'][0]
    assert bottle['centre'][:2] == pytest.approx([.45, .04], abs=.01)
    assert bottle['length'] == pytest.approx(.20, abs=.03) and bottle['height'] == pytest.approx(.06, abs=.005)
    assert found['table_z'] == pytest.approx(.005, abs=.003)
    assert '#1 center' in scene_objects.describe(found)


class FakeClient:
    """Stands in for the relay: records what the model would read and answers from a script."""
    script, last = [], None

    def __init__(self, relay_dir, timeout_s=600, action_prompt=None, **_):
        self.dir = Path(relay_dir)
        self.dir.mkdir(parents=True, exist_ok=False)
        self.turn, self.events, self.seen = 0, [], []
        self.wait_tick = lambda: None
        FakeClient.last = self

    def chat(self, messages, tools):
        self.seen.append(copy.deepcopy(messages))
        name, args = FakeClient.script[self.turn]
        self.turn += 1
        return LLMResponse('reason '+name, [ToolCall(str(self.turn), name, args)], 0, 0, 0.)

    def finish(self):
        (self.dir/'done.json').write_text('{}')


def site_file(tmp_path):
    """A filled-in robot site file (placeholders replaced) with the example hand-eye matrix."""
    from real_helpers import ROOT
    hand_eye = tmp_path/'hand_eye.txt'
    hand_eye.write_text((ROOT/'configs/real/hand_eye.example.txt').read_text())
    site = json.loads((ROOT/'configs/real/robot.example.json').read_text())
    site['robot_pc'].update(ssh_host='robot-pc-test', conda_prefix='/opt/polymetis', python='/opt/polymetis/bin/python',
                            hardware_config='/opt/franka_hardware.yaml')
    site['cameras'].update(exterior_serial='000000000001', wrist_serial='000000000002', hand_eye_file=str(hand_eye))
    path = tmp_path/'robot.json'
    path.write_text(json.dumps(site))
    return path


@pytest.mark.parametrize('camera_fails', [False, True])
def test_live_branch_shows_the_detected_objects_every_turn(tmp_path, monkeypatch, camera_fails):
    from forceharness.real import backend as backend_mod, transport, wrist_overlay
    from forceharness.real.plant import DemoPort
    monkeypatch.setattr(wrist_overlay, 'HAND_EYE_FILE', None)     # the runner sets it; restored afterwards

    class Port(DemoPort):
        def next_weigh_trial(self, n):
            pass

    class Synthetic(backend_mod.ContinuousBackend):
        def __init__(self, port, **kw):
            super().__init__(port, clock=port.clock, sleep=port.sleep, **kw)

        def calibration_valid(self):
            return self._closed is not None

    def photographer(out, backend, cameras=None):
        n = iter(range(1, 100))

        def photograph(phase, tick):
            if camera_fails:
                raise RuntimeError('wrist camera unavailable')
            folder = out/'photos'/f'{next(n):02d}_{phase}'
            make_scene(folder, boxes=[dict(centre=(.448, .042), size=(.20, .065), height=.06)])
            return dict(wrist_images=[str(folder/'scene_wrist.png')])
        return photograph
    port_kwargs = {}

    def port(cfg, folder, **kw):
        assert cfg.f_cap == 25
        port_kwargs.update(kw)
        return Port(cfg, payload=0)
    monkeypatch.setattr(transport, 'RobotPort', port)
    monkeypatch.setattr(backend_mod, 'ContinuousBackend', Synthetic)
    monkeypatch.setattr(free, 'live_photographer', photographer)
    monkeypatch.setattr(free, 'RelayClient', FakeClient)
    monkeypatch.setattr(free.posture_cal, 'load', lambda *a, **k: SimpleNamespace(meta=dict(use_model=False), d_tau=None))
    FakeClient.script = [('scene', {}), ('hold', dict(seconds=.2)), ('done', dict(success=False, note='test'))]
    code = free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--live', '--execute',
                      '--auto-confirm', '--calibration', str(CALIBRATION), '--robot-config', str(site_file(tmp_path)),
                      '--instruction', 'set the bottle upright'])
    r = json.loads((tmp_path/'ep'/'episode.json').read_text())
    assert code == 0 and r['fault'] is None and r['mode'] == 'live' and r['model'] == 'relay'
    if camera_fails:
        assert 'scene capture failed' in r['actions'][0]['invalid']
        assert 'wrist camera unavailable' in r['actions'][0]['receipt']['text']
        assert not (tmp_path/'relay'/'obs_0_wrist.png').exists()
        return
    assert [u['turn'] for u in r['scene_updates']] == [0, 1, 2] and all(len(u['objects']) == 1 for u in r['scene_updates'])
    assert all(u['perception_version'] == 'scene_objects/angle1' and 'candidates' in u for u in r['scene_updates'])
    first = FakeClient.last.seen[0]
    assert 'set the bottle upright' in first[0]['text'] and 'z_table ≈ 0.005' in first[0]['text']
    centre = re.search(r'#1 center \[([-\d.]+), ([-\d.]+)\]', first[1]['parts'][1]['text'])
    assert centre and [float(v) for v in centre.groups()] == pytest.approx([.448, .042], abs=.006)
    assert 'obs_0_wrist.png' in first[1]['parts'][1]['text']
    assert (tmp_path/'relay'/'obs_0_wrist.png').is_file() and (tmp_path/'ep'/'photos'/'01_turn00'/'scene_wrist_free.png').is_file()
    receipt = json.loads(FakeClient.last.seen[2][-1]['parts'][0]['text'])
    assert 'still reading' in receipt['text'] and '#1 center' in receipt['text'] and receipt['states']['pointing']
    assert r['actions'][1]['note'] == 'reason hold'
    assert port_kwargs['grasp_min_width_m'] == .008 and port_kwargs['max_trials'] is None
    assert port_kwargs['camera_factory'].func is free.GripperCameras and port_kwargs['robot'].robot_pc.ssh_host == 'robot-pc-test'


def test_live_needs_a_filled_in_site_file(tmp_path, monkeypatch):
    from real_helpers import ROOT
    with pytest.raises(SystemExit):
        free.main(['--out', str(tmp_path/'ep'), '--relay-dir', str(tmp_path/'relay'), '--live', '--execute',
                   '--auto-confirm', '--calibration', str(CALIBRATION),
                   '--robot-config', str(ROOT/'configs/real/robot.example.json'), '--instruction', 'x'])
    assert not (tmp_path/'ep').exists()
