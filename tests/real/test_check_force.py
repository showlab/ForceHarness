"""The read-only force gate (scripts/real/check_force.py): statistics, sign trials and the known-mass check on
synthetic streams, and the read-only path through the real worker process serving the synthetic plant."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from forceharness.real import force_check as fc
from forceharness.real.config import BackendConfig, load_robot_config
from forceharness.real.transport import SSHPort
from real_helpers import CALIBRATION, ROOT

G = 9.81


def rows(n, wrench, hz=20., t0=0., noise=0., drift=None, seed=0, residual=.2, smin=.1):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        w = np.asarray(wrench, dtype=float)+rng.normal(0, noise, 6)
        if drift is not None:
            w = w+np.asarray(drift)*i/n
        out.append(dict(t=t0+i/hz, timestamp_ns=int((t0+i/hz)*1e9), wrench=w.tolist(), residual=residual, smin=smin,
                        gripper_width=.08))
    return out


def test_still_statistics_rate_noise_and_drift():
    stream = rows(600, [1., -2., .5, .1, 0., 0.], noise=.03, drift=[0, 0, -.6, 0, 0, 0])      # 30 s at 20 Hz
    s = fc.stream_stats(stream, force_sign=-1)
    assert s['samples'] == 600 and s['request_rate_hz'] == pytest.approx(20, abs=.1)
    assert s['unique_robot_timestamps'] == 600 and s['robot_rate_hz'] == pytest.approx(20, abs=.1)
    assert all(.02 < v < .04 for v in s['force_std_N'][:2])
    assert s['drift_N'][2] == pytest.approx(.6*25/30, abs=.05)          # reaction sign: raw -0.6 N reads +0.6
    assert s['drift_window_s'] == 5. and s['force_norm_std_N'] > 0
    assert s['residual_Nm'] == dict(mean=.2, max=.2) and s['sigma_min'] == dict(mean=.1, min=.1)
    short = fc.stream_stats(rows(60, [0]*6), force_sign=1)              # 3 s: windows of a third
    assert short['drift_window_s'] == pytest.approx(59/20/3, abs=1e-3)
    with pytest.raises(ValueError):
        fc.stream_stats(rows(5, [0]*6), force_sign=1)


@pytest.mark.parametrize('press,raw_dz,matches', [('up', -6., True), ('up', 6., False), ('down', 6., True),
                                                  ('up', -.2, False)])
def test_sign_trial(press, raw_dz, matches):
    rest = rows(30, [0., 0., .5, 0, 0, 0])
    pressed = rows(60, [0., 0., .5+raw_dz, 0, 0, 0], t0=2.)
    result = fc.press_result(rest, pressed, force_sign=-1, press=press)
    assert result['delta_z_N'] == pytest.approx(-raw_dz) and result['matches'] is matches


def test_known_mass_with_and_without_the_closure_correction():
    empty = rows(100, [0., 0., .3, 0, 0, 0])
    loaded = rows(100, [0., 0., .3+.3207*G, 0, 0, 0])            # raw +z when the hand carries a weight (sign -1)
    got = fc.mass_result(empty, loaded, force_sign=-1, known_grams=320.7)
    assert got['estimated_g'] == pytest.approx(320.7, abs=.1) and got['error_g'] == pytest.approx(0, abs=.1)
    c = [0., 0., 1.1439, 0, 0, 0]
    closed = rows(100, [0., 0., .3+.3207*G+1.1439, 0, 0, 0])     # closing the gripper adds its own offset
    got = fc.mass_result(empty, closed, force_sign=-1, known_grams=320.7, closure_correction=c)
    assert got['estimated_corrected_g'] == pytest.approx(320.7, abs=.1) and got['estimated_g'] > 400
    assert 'closure_correction_estimate_N' not in got                     # only when the gripper closed here


def test_a_known_mass_gripped_here_estimates_the_closure_correction():
    empty = rows(100, [0., 0., .3, 0, 0, 0])
    closed = rows(100, [-.6, -.2, .3+.3207*G+1.1439, 0, 0, 0])
    got = fc.mass_result(empty, closed, force_sign=-1, known_grams=320.7, closed=True)
    assert got['closure_correction_estimate_N'] == pytest.approx([-.6, -.2, 1.1439], abs=1e-4)   # raw sign
    again = fc.mass_result(empty, closed, force_sign=-1, known_grams=320.7, closed=True,
                           closure_correction=got['closure_correction_estimate_N']+[0, 0, 0])
    assert again['estimated_corrected_g'] == pytest.approx(320.7, abs=.1)


def test_still_statistics_report_the_raw_bias_and_the_pose():
    stream = rows(100, [1., -2., .5, .3, .1, -.2])
    for r in stream:
        r.update(flange_pos=[.5, 0., .3], flange_quat=[1., 0., 0., 0.])
    s = fc.stream_stats(stream, force_sign=-1)
    assert s['raw_force_mean_N'] == pytest.approx([1., -2., .5]) and s['raw_torque_mean_Nm'] == pytest.approx([.3, .1, -.2])
    assert s['flange_pos_m'] == pytest.approx([.5, 0., .3]) and s['flange_quat'] == [1., 0., 0., 0.]


class Port:
    """Answers `state` like the read-only worker; the plant wrench follows `phase`."""
    limits = dict(f_cap=25., t_cap=3.)
    metadata = dict(hz=1000, frame='world', reference='panda_link8', controller='test')

    def __init__(self):
        self.phase, self.t = 'rest', 0.

    def request(self, op):
        assert op == 'state'
        self.t += .05
        z = {'rest': .5, 'press': .5-6., 'loaded': .5+.32*G}[self.phase]
        return dict(state=dict(timestamp_ns=int(self.t*1e9), wrench=[0., 0., z, 0., 0., 0.], residual=.1, smin=.1,
                               gripper_width=.08 if self.phase != 'loaded' else .029))


class Clock:
    def __init__(self):
        self.t = 0.

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += max(s, .01)


def quiet(**kw):
    clock = Clock()
    return dict(clock=clock, sleep=clock.sleep, say=lambda m: None, **kw)


def args(**kw):
    base = dict(sign=False, mass=False, seconds=1., hz=50., repeats=2, press='up', known_grams=None,
                close_gripper=False, calibration=CALIBRATION)
    return SimpleNamespace(**dict(base, **kw))


def calibration():
    return json.loads(CALIBRATION.read_text())


def test_still_mode_reports_the_guard_caps_and_the_reference(monkeypatch):
    out = fc_script().run_checks(args(), Port(), **quiet(calibration=calibration()))
    assert out['mode'] == 'still' and out['guard_limits']['f_cap'] == 25. and out['reference_FR3']
    assert out['still']['force_std_N'] == [0., 0., 0.] and len(out['still']['raw_minus_calibration_bias_N']) == 3


def test_sign_mode_never_moves_and_reads_the_press():
    port = Port()

    def say(message):
        if 'NOW' in message:
            port.phase = 'press'
        elif 'release' in message:
            port.phase = 'rest'
    clock = Clock()
    out = fc_script().run_checks(args(sign=True), port, ask=lambda m: '', say=say, clock=clock, sleep=clock.sleep,
                                 calibration=calibration())
    assert out['sign']['all_match'] and [t['sign_z'] for t in out['sign']['trials']] == [1, 1]


def test_mass_mode_without_the_gripper_moves_nothing():
    port = Port()

    def ask(message):
        if 'Place or hang' in message:
            port.phase = 'loaded'
        return ''
    out = fc_script().run_checks(args(mass=True, known_grams=320.), port, ask=ask, **quiet(calibration=calibration()))
    assert 'gripper_close' not in out and 'gripper_open' not in out
    m = out['mass']
    assert m['estimated_g'] == pytest.approx(320., abs=1.) and m['estimated_corrected_g'] is not None


def test_mass_mode_closes_the_gripper_only_after_an_explicit_close():
    port, calls = Port(), []
    gripper = SimpleNamespace(run=lambda op: calls.append(op) or dict(op=op, width_m=.029))
    out = fc_script().run_checks(args(mass=True, close_gripper=True), port, ask=lambda m: 'close',
                                 **quiet(gripper=gripper, calibration=calibration()))
    assert calls == [] and out['mass'] == dict(aborted='gripper close not confirmed')
    answers = iter(['', 'CLOSE', '', 'no'])
    out = fc_script().run_checks(args(mass=True, close_gripper=True), port, ask=lambda m: next(answers),
                                 **quiet(gripper=gripper, calibration=calibration()))
    assert calls == ['close'] and out['gripper_open'] == 'left closed by the operator'


def fc_script():
    spec = importlib.util.spec_from_file_location('check_force_script', ROOT/'scripts/real/check_force.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_read_only_worker_serves_the_stream(tmp_path, monkeypatch):
    actual = subprocess.Popen
    fixture = Path(__file__).with_name('worker_fixture.py')
    seen = []

    def spawn(argv, **kw):
        seen.append(argv)
        return actual([sys.executable, str(fixture)], **kw)              # no --enable-motion: read-only
    monkeypatch.setattr(subprocess, 'Popen', spawn)
    port = SSHPort(BackendConfig(motion_enabled=False), tmp_path, robot=load_robot_config(ROOT/'configs/real/robot.example.json'))
    port.open()
    try:
        stream = fc.sample(port, 1., 20.)
        with pytest.raises(Exception, match='Read-only worker'):
            port.request('begin', budget=1.)
    finally:
        port.close()
    assert '--enable-motion' not in seen[0][-1]
    s = fc.stream_stats(stream, force_sign=-1)
    assert s['samples'] >= 5 and s['force_std_N'] == [0., 0., 0.]
    rows_logged = [json.loads(line) for line in (tmp_path/'nuc.jsonl').read_text().splitlines()]
    assert not any(r['type'] in ('begin', 'target') for r in rows_logged)
