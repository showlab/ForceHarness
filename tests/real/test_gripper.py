"""The Franka Hand path of the public release (not yet validated on hardware): the robot-side command, the
workstation port with its ledger and width checks, and the operator script. Fakes only: no SSH, no Polymetis."""
import importlib.util
import json
import sys
from types import SimpleNamespace

import pytest

from forceharness.real.config import BackendConfig, load_robot_config
from forceharness.real.gripper import RemoteGripper, check_result
from forceharness.real.transport import RobotPort
from forceharness.real.types import BackendError
from real_helpers import ROOT, robot_side


class State:
    def __init__(self, width, moving, grasped=True, ok=True):
        self.width, self.is_moving, self.is_grasped, self.prev_command_successful, self.error_code = width, moving, grasped, ok, 0


class Hand:
    def __init__(self, states):
        self.states, self.calls = list(states), []

    def get_state(self):
        return self.states.pop(0) if len(self.states) > 1 else self.states[0]

    def grasp(self, **kwargs):
        self.calls.append(('grasp', kwargs))

    def goto(self, **kwargs):
        self.calls.append(('goto', kwargs))


def test_the_robot_side_command_waits_until_the_hand_has_moved():
    # grasp(blocking=True) returns when the server has accepted the command, before the hand moves
    g = robot_side('gripper')
    queued = [State(.08, False)]*3 + [State(.06, True), State(.03, True), State(.0272, False)]
    s, settled = g.wait_done(Hand(queued), .08, poll=0)
    assert settled and s.width == .0272
    s, settled = g.wait_done(Hand([State(.08, False)]), .08, timeout=.05, poll=0)
    assert not settled and s.width == .08
    opening = [State(.03, False)]*2 + [State(.05, True), State(.0801, False)]
    s, settled = g.wait_done(Hand(opening), .03, poll=0)
    assert settled and s.width == .0801


def test_the_robot_side_command_prints_one_json_line(monkeypatch, capsys):
    g = robot_side('gripper')
    hand = Hand([State(.08, False), State(.05, True), State(.029, False)])
    monkeypatch.setitem(sys.modules, 'polymetis', SimpleNamespace(GripperInterface=lambda **kw: hand))
    assert g.main(['close', '--force', '60', '--speed', '0.05']) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out['op'] == 'close' and out['force_N'] == 60 and out['width_m'] == .029 and out['settled']
    assert hand.calls == [('grasp', dict(speed=.05, force=60., blocking=True))]
    with pytest.raises(SystemExit):
        g.main(['close', '--force', '141'])


def site():
    return load_robot_config(ROOT/'configs/real/robot.example.json')


def test_the_remote_command_runs_the_bundle_script_with_the_configured_force():
    argv = RemoteGripper(site()).command('close')
    assert argv[:5] == ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=5']
    remote = argv[-1]
    assert 'sha256sum --status -c SHA256SUMS' in remote and 'gripper.py close' in remote
    assert '--force 60' in remote and '--port 50052' in remote and remote.startswith('cd "$HOME"/')
    assert '--width 0.08' in ' '.join(RemoteGripper(site()).command('open'))
    assert '--force 30' in RemoteGripper(site()).command('close', 30)[-1]


class FakeProc:
    def __init__(self, out, code=0, polls=2):
        self.out, self.returncode, self.polls = out, code, polls

    def poll(self):
        self.polls -= 1
        return None if self.polls > 0 else self.returncode

    def communicate(self, timeout=None):
        return self.out, ''

    def kill(self):
        pass

    def wait(self, timeout=None):
        pass


def test_the_remote_gripper_polls_until_done_and_parses_the_reply():
    t = [0.]
    reply = json.dumps(dict(op='close', width_m=.029, prev_command_successful=True))
    polled = []
    g = RemoteGripper(site(), popen=lambda *a, **k: FakeProc(reply), clock=lambda: t[0],
                      sleep=lambda s: t.__setitem__(0, t[0]+s))
    assert g.run('close', poll=lambda: polled.append(t[0]), min_s=3.2)['width_m'] == .029
    assert len(polled) >= 64 and t[0] >= 3.2                         # polled every 50 ms for at least 3.2 s
    bad = RemoteGripper(site(), popen=lambda *a, **k: FakeProc('', code=1), clock=lambda: 0., sleep=lambda s: None)
    with pytest.raises(BackendError, match='Gripper command failed'):
        bad.run('open')
    stuck = RemoteGripper(site(), popen=lambda *a, **k: FakeProc('', polls=10**9), clock=lambda: t[0],
                          sleep=lambda s: t.__setitem__(0, t[0]+s))
    with pytest.raises(BackendError, match='timed out; outcome unknown'):
        stuck.run('close')


@pytest.mark.parametrize('closed,width,ok,error', [
    (True, .029, True, None), (True, .0081, True, None), (True, .008, True, 'plausible grasp'),
    (True, .075, True, 'plausible grasp'), (True, .029, False, 'did not succeed'),
    (False, .08, True, None), (False, .077, True, 'did not open')])
def test_acceptance_of_one_action(closed, width, ok, error):
    if error is None:
        check_result(closed, dict(prev_command_successful=ok), width, .008)
    else:
        with pytest.raises(RuntimeError, match=error):
            check_result(closed, dict(prev_command_successful=ok), width, .008)


class FakeGripper:
    def __init__(self, width_after, ok=True):
        self.width_after, self.ok, self.calls = width_after, ok, []

    def run(self, op, poll=None, min_s=0.):
        self.calls.append(op)
        poll()
        self.port.width = self.width_after
        return dict(op=op, width_m=self.width_after, prev_command_successful=self.ok)


def port(tmp_path, width_after, mode='holding', running=True, ok=True, minimum=.008):
    tmp_path.mkdir(parents=True, exist_ok=True)              # the transport's log folder (open() makes it)
    gripper = FakeGripper(width_after, ok)
    p = RobotPort(BackendConfig(motion_enabled=True), tmp_path, robot=site(), grasp_min_width_m=minimum, gripper=gripper)
    gripper.port = p
    p.width = .08
    p.request = lambda op: dict(session=dict(mode=mode), state=dict(policy_running=running, gripper_width=p.width,
                                                                     error_code=0, timestamp_ns=1))
    p.next_weigh_trial(1)
    return p, gripper


def test_one_close_per_trial_and_the_interval_between_commands(tmp_path):
    p, g = port(tmp_path, .029)
    p.grip(True)
    assert g.calls == ['close'] and p.ledger['grasp_attempts'] == 1 and not p.released
    with pytest.raises(RuntimeError, match='interval'):
        p.grip(False)
    p.ledger['last_gripper_time'] = 0
    with pytest.raises(RuntimeError, match='retry'):
        p.grip(True)
    g.width_after = .08
    p.grip(False)
    assert g.calls == ['close', 'open'] and p.released
    p.next_weigh_trial(2)                                         # a new trial only after a confirmed release
    assert p.ledger['grasp_attempts'] == 0


def test_an_empty_or_implausible_close_is_an_error_the_runner_can_classify(tmp_path):
    for width in (.0, .008, .075):
        p, _ = port(tmp_path/str(width), width)
        with pytest.raises(RuntimeError, match='plausible grasp'):
            p.grip(True)


def test_a_foreign_controller_or_an_active_primitive_refuses_before_any_command(tmp_path):
    for kwargs, match in ((dict(mode='idle'), 'Another controller'), (dict(mode='active'), 'own hold')):
        p, g = port(tmp_path/kwargs['mode'], .029, **kwargs)
        with pytest.raises(BackendError, match=match):
            p.grip(True)
        assert g.calls == []


def test_a_failed_hardware_command_is_not_accepted(tmp_path):
    p, _ = port(tmp_path, .029, ok=False)
    with pytest.raises(RuntimeError, match='did not succeed'):
        p.grip(True)


def test_no_trial_no_gripper(tmp_path):
    p = RobotPort(BackendConfig(motion_enabled=True), tmp_path, robot=site())
    with pytest.raises(BackendError, match='no gripper trial'):
        p.grip(True)


@pytest.mark.parametrize('minimum', [.007, .026, float('nan')])
def test_invalid_minimum_rejected_without_connection(tmp_path, minimum):
    with pytest.raises(ValueError, match='8 and 25'):
        RobotPort(BackendConfig(), tmp_path, robot=site(), grasp_min_width_m=minimum)


def load_check_script():
    spec = importlib.util.spec_from_file_location('gripper_check', ROOT/'scripts/real/gripper_check.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_operator_script_logs_and_explains(tmp_path):
    check = load_check_script()

    class Gripper:
        def run(self, op, force=None):
            assert op == 'close' and force == 30
            return dict(op='close', width_before_m=.08, width_m=.0196, settled=True, force_N=30)
    assert check.main(['close', '--force', '30'], gripper=Gripper(), log=tmp_path/'log.jsonl') == 0
    record = json.loads((tmp_path/'log.jsonl').read_text().splitlines()[-1])
    assert record['command'] == 'close' and record['width_m'] == .0196 and '--resume-held' in record['verdict']
    assert 'nothing seems to be held' in check.verdict('close', .003) and 'Open (80 mm)' in check.verdict('open', .08)
    assert 'restart the Polymetis gripper server' in check.verdict('open', .056)
    with pytest.raises(SystemExit):
        check.main(['close', '--force', '150'], gripper=Gripper(), log=tmp_path/'log.jsonl')
