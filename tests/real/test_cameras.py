"""Camera capture contract and failure cleanup without devices or robot access (a fake pyrealsense2)."""
import json
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from forceharness.real import cameras as free_camera

CAMERAS = (('exterior', '000000000001', False), ('wrist', '000000000002', True))
SITE = SimpleNamespace(exterior_serial='000000000001', wrist_serial='000000000002', capture_python='python3',
                       capture_timeout_s=free_camera.CAPTURE_TIMEOUT_S)


class FakeRS:
    stream = SimpleNamespace(color='color', depth='depth')
    format = SimpleNamespace(rgb8='rgb8', z16='z16')

    def __init__(self, fail_wrist=False):
        self.configs, self.pipes = [], []
        self.fail_wrist = fail_wrist

    def config(self):
        cfg = SimpleNamespace(streams=[])
        cfg.enable_device = lambda serial: setattr(cfg, 'serial', serial)
        cfg.enable_stream = lambda kind, *args: cfg.streams.append(kind)
        self.configs.append(cfg)
        return cfg

    def pipeline(self):
        pipe = SimpleNamespace(stopped=False)

        def start(cfg):
            pipe.cfg = cfg
            return SimpleNamespace(get_device=lambda: SimpleNamespace(
                first_depth_sensor=lambda: SimpleNamespace(get_depth_scale=lambda: .001)))

        def wait(ms):
            assert ms in (1000, 5000)
            time.sleep(.001)
            if self.fail_wrist and pipe.cfg.serial == CAMERAS[1][1]:
                raise RuntimeError('Frame did not arrive')
            intr = SimpleNamespace(fx=400, fy=400, ppx=3, ppy=2, coeffs=[0]*5)
            color = SimpleNamespace(get_data=lambda: np.full((4, 6, 3), 127, np.uint8),
                                    profile=SimpleNamespace(as_video_stream_profile=lambda: SimpleNamespace(intrinsics=intr)))
            depth = SimpleNamespace(get_data=lambda: np.full((4, 6), 200, np.uint16))
            return SimpleNamespace(get_color_frame=lambda: color, get_depth_frame=lambda: depth)

        pipe.start, pipe.wait_for_frames = start, wait
        pipe.stop = lambda: setattr(pipe, 'stopped', True)
        self.pipes.append(pipe)
        return pipe

    def align(self, stream):
        assert stream == 'color'
        return SimpleNamespace(process=lambda frames: frames)


def test_exterior_depth_is_not_requested_and_wrist_depth_stays_in_metres(tmp_path):
    rs = FakeRS()
    state = dict(flange_pos=[.4, 0, .3], flange_quat=[1, 0, 0, 0], gripper_width=.08)
    free_camera.capture(tmp_path, state, CAMERAS, rs)
    assert [c.streams for c in rs.configs] == [['color'], ['color', 'depth']]
    assert all(p.stopped for p in rs.pipes)
    with np.load(tmp_path/'scene.npz') as arrays:
        assert set(arrays.files) == {'exterior_rgb', 'wrist_rgb', 'wrist_depth'}
        assert np.allclose(arrays['wrist_depth'], .2)
    meta = json.loads((tmp_path/'scene.json').read_text())
    assert meta['flange_pos'] == state['flange_pos']
    assert meta['wrist']['depth_enabled'] and not meta['exterior']['depth_enabled']
    assert (tmp_path/'scene_exterior.png').is_file() and (tmp_path/'scene_wrist.png').is_file()
    with pytest.raises(FileExistsError):
        free_camera.capture(tmp_path, state, CAMERAS, rs)


def test_wrist_depth_failure_is_explicit_and_camera_resources_are_released(tmp_path):
    rs = FakeRS(fail_wrist=True)
    with pytest.raises(RuntimeError, match='wrist camera.*Frame did not arrive'):
        free_camera.capture(tmp_path, {}, CAMERAS, rs)
    assert all(p.stopped for p in rs.pipes)
    assert not (tmp_path/'scene.npz').exists()


def test_photographer_reports_child_error_not_release_confirmation(tmp_path, monkeypatch):
    def popen(argv, stdout, stderr, cwd=None):
        assert argv[:3] == ['python3', '-m', 'forceharness.real.cameras'] and json.loads(argv[-1])[1][1] == '000000000002'
        stdout.write('RuntimeError: wrist camera unavailable\n')
        return SimpleNamespace(poll=lambda: 1, returncode=1)
    monkeypatch.setattr(free_camera.subprocess, 'Popen', popen)
    photograph = free_camera.live_photographer(tmp_path, SimpleNamespace(_state={}), SITE)
    with pytest.raises(RuntimeError, match='wrist camera unavailable') as error:
        photograph('test', lambda: None)
    assert 'release confirmation' not in str(error.value)


def test_photographer_timeout_stops_only_its_own_camera_child(tmp_path, monkeypatch):
    events = []
    proc = SimpleNamespace(poll=lambda: None, terminate=lambda: events.append('terminate'),
                           wait=lambda timeout: events.append('wait'))
    monkeypatch.setattr(free_camera.subprocess, 'Popen', lambda *a, **k: proc)
    clock = iter([0., free_camera.CAPTURE_TIMEOUT_S+1.])
    monkeypatch.setattr(free_camera.time, 'monotonic', lambda: next(clock))
    with pytest.raises(RuntimeError, match='exceeded'):
        free_camera.live_photographer(tmp_path, SimpleNamespace(_state={}), SITE)('test', lambda: None)
    assert events == ['terminate', 'wait']


def test_gripper_camera_stream_checks_and_saves_fresh_before_after_evidence(tmp_path):
    rs = FakeRS()
    with free_camera.GripperCameras(CAMERAS, rs) as cameras:
        cameras.save(tmp_path, 'before', {'gripper': [.08]}, {})
        for _ in range(5):
            cameras.check()
            time.sleep(.01)
        cameras.save(tmp_path, 'after', {'gripper': [.03]}, {})
        assert [c.streams for c in rs.configs] == [['color'], ['color', 'depth']]
    assert all(p.stopped for p in rs.pipes)
    for label in ('before', 'after'):
        with np.load(tmp_path/f'{label}.npz') as arrays:
            assert set(arrays.files) == {'exterior_rgb', 'wrist_rgb', 'wrist_depth'}
        assert json.loads((tmp_path/f'{label}.json').read_text())['capture_version'] == 'free_camera/r42'


def test_gripper_camera_failure_closes_streams_without_gripper_actuation():
    rs = FakeRS(fail_wrist=True)
    with pytest.raises(RuntimeError, match='wrist camera'):
        with free_camera.GripperCameras(CAMERAS, rs):
            pytest.fail('Missing mandatory wrist depth must prevent entry')
    assert all(p.stopped for p in rs.pipes)


@pytest.mark.parametrize('age, error', [(0.51, None), (0.1, 'camera disconnected')])
def test_gripper_freshness_guard_is_not_bypassed(age, error):
    cameras = free_camera.GripperCameras(CAMERAS)
    cameras.lock = threading.Lock()
    cameras.error = error
    cameras.latest = ({}, {}, time.monotonic()-age)
    with pytest.raises(RuntimeError, match='stale|disconnected'):
        cameras.check()


def test_capture_process_runs_as_a_module_with_the_configured_python():
    from forceharness.real.config import CameraConfig
    cfg = CameraConfig(exterior_serial='1', wrist_serial='2')
    assert cfg.capture_python == __import__('sys').executable          # default: the runner's own interpreter
    assert CameraConfig(exterior_serial='1', wrist_serial='2', python='/opt/rs/bin/python').capture_python == '/opt/rs/bin/python'
    assert free_camera.camera_specs(cfg) == (('exterior', '1', False), ('wrist', '2', True))


def test_missing_pyrealsense_names_the_interpreter(monkeypatch):
    import builtins
    real_import = builtins.__import__

    def no_realsense(name, *args, **kwargs):
        if name == 'pyrealsense2':
            raise ImportError('No module named pyrealsense2')
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_realsense)
    with pytest.raises(RuntimeError, match='pyrealsense2 is not installed for .*cameras.python'):
        free_camera._realsense()
