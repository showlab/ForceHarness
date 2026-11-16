"""RealSense captures for the real-robot runner: exterior RGB for context, wrist RGB-D for measured geometry.

The exterior camera's depth is never used (scene_objects and locate measure only on the wrist image), so it is
not requested. This module never connects to robot control. Per-turn photos run in a separate capture process
(`python -m forceharness.real.cameras <folder> <state json> <camera json>`, started in the repository root), so a
hung or crashed camera driver cannot take the runner down and a capture can time out; that process needs numpy,
Pillow and pyrealsense2. Its interpreter is cameras.python in configs/real/robot.json (default: the runner's own
interpreter). This module imports nothing else from forceharness at module level.
"""
from datetime import datetime
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[2]
CAPTURE_TIMEOUT_S = 20


def camera_specs(cameras_config):
    """(name, serial, with_depth) for both cameras, from the site configuration."""
    return (('exterior', cameras_config.exterior_serial, False), ('wrist', cameras_config.wrist_serial, True))


def _realsense():
    try:
        import pyrealsense2
    except ImportError as exc:
        raise RuntimeError('pyrealsense2 is not installed for '+sys.executable
                           +' (install it, or set cameras.python in configs/real/robot.json)') from exc
    return pyrealsense2


class GripperCameras:
    """Live camera checks and before/after evidence around every gripper action.

    Keep the 0.5 s freshness check throughout gripper motion. External depth is
    never requested; wrist RGB-D remains mandatory. No robot APIs are called here.
    """
    def __init__(self, cameras=None, rs=None):
        if cameras is None:
            from forceharness.real.config import load_robot_config
            cameras = camera_specs(load_robot_config().cameras)
        self.camera_list = tuple(cameras)
        self.rs = rs

    def __enter__(self):
        if self.rs is None:
            self.rs = _realsense()
        rs = self.rs
        self.resources = ExitStack()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.latest = self.error = None
        self.cameras = []
        try:
            for name, serial, with_depth in self.camera_list:
                pipe, cfg = rs.pipeline(), rs.config()
                cfg.enable_device(serial)
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                if with_depth:
                    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                profile = pipe.start(cfg)
                self.resources.callback(pipe.stop)
                scale = profile.get_device().first_depth_sensor().get_depth_scale() if with_depth else None
                align = rs.align(rs.stream.color) if with_depth else None
                self.cameras.append((name, serial, pipe, align, scale))
            self.thread = threading.Thread(target=self._read, daemon=True)
            self.thread.start()
            deadline = time.monotonic()+12
            while self.latest is None and time.monotonic() < deadline:
                if self.error:
                    raise RuntimeError(self.error)
                time.sleep(.05)
            self.check()
            return self
        except BaseException:
            self.close()
            raise

    def _read(self):
        import numpy as np
        try:
            count = 0
            while not self.stop_event.is_set():
                data, meta, sampled = {}, {}, []
                for name, serial, pipe, align, scale in self.cameras:
                    try:
                        frames = pipe.wait_for_frames(5000 if count == 0 else 1000)
                        if align is not None:
                            frames = align.process(frames)
                        color = frames.get_color_frame()
                        if not color:
                            raise RuntimeError('Missing color frame')
                        intr = color.profile.as_video_stream_profile().intrinsics
                        data[name+'_rgb'] = np.asanyarray(color.get_data()).copy()
                        meta[name] = dict(serial=serial, fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                                          coeffs=list(intr.coeffs), time_unix=time.time(), depth_enabled=align is not None)
                        if align is not None:
                            depth = frames.get_depth_frame()
                            if not depth:
                                raise RuntimeError('Missing wrist depth frame')
                            data[name+'_depth'] = np.asanyarray(depth.get_data()).copy()*scale
                        sampled.append(time.monotonic())
                    except Exception as exc:
                        raise RuntimeError(f'{name} camera ({serial}): {exc}') from exc
                count += 1
                if count >= 20:
                    with self.lock:
                        # Both images must remain fresh, not just the last camera read.
                        self.latest = data, meta, min(sampled)
        except BaseException as exc:
            self.error = str(exc)

    def check(self):
        if self.error:
            raise RuntimeError('Camera acquisition failed: '+self.error)
        with self.lock:
            if self.latest is None or time.monotonic()-self.latest[2] > .5:
                raise RuntimeError('Camera acquisition stale (exterior RGB / wrist RGB-D)')

    def save(self, root, label, state, health):
        import numpy as np
        from PIL import Image
        self.check()
        with self.lock:
            data, camera_meta, _ = self.latest
            meta = dict(state, **camera_meta, health=health, capture_version='free_camera/r42')
        root = Path(root)
        outputs = [root/f'{label}{suffix}' for suffix in ('.npz', '.json', '_exterior.png', '_wrist.png')]
        if any(path.exists() for path in outputs):
            raise FileExistsError('Capture evidence already exists')
        np.savez_compressed(outputs[0], **data)
        for name, _, _ in self.camera_list:
            Image.fromarray(data[name+'_rgb']).save(root/f'{label}_{name}.png')
        outputs[1].write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
        return meta

    def close(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join(timeout=6)
        self.resources.close()

    def __exit__(self, *args):
        self.close()


def capture(folder, state, cameras, rs=None):
    """Save fresh images and only the depth actually measured, in metres."""
    import numpy as np
    from PIL import Image
    if rs is None:
        rs = _realsense()
    folder = Path(folder)
    outputs = ['scene.npz', 'scene.json', 'scene_exterior.png', 'scene_wrist.png']
    if any((folder/name).exists() for name in outputs):
        raise FileExistsError('Capture evidence already exists')
    arrays, meta = {}, dict(state)
    meta['capture_version'] = 'free_camera/r41'
    for name, serial, with_depth in cameras:
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
        if with_depth:
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        started = False
        try:
            profile = pipe.start(cfg)
            started = True
            for _ in range(20):
                frames = pipe.wait_for_frames(5000)
            if with_depth:
                frames = rs.align(rs.stream.color).process(frames)
            color = frames.get_color_frame()
            if not color:
                raise RuntimeError('Missing color frame')
            intr = color.profile.as_video_stream_profile().intrinsics
            arrays[name+'_rgb'] = np.asanyarray(color.get_data()).copy()
            meta[name] = dict(serial=serial, fx=intr.fx, fy=intr.fy, ppx=intr.ppx, ppy=intr.ppy,
                              coeffs=list(intr.coeffs), time_unix=time.time(), depth_enabled=with_depth)
            if with_depth:
                depth = frames.get_depth_frame()
                if not depth:
                    raise RuntimeError('Missing wrist depth frame')
                scale = profile.get_device().first_depth_sensor().get_depth_scale()
                arrays[name+'_depth'] = np.asanyarray(depth.get_data()).copy()*scale
        except Exception as exc:
            raise RuntimeError(f'{name} camera ({serial}): {exc}') from exc
        finally:
            if started:
                pipe.stop()
    # Complete captures only; never reuse a previous frame or invent exterior depth.
    np.savez_compressed(folder/'scene.npz', **arrays)
    for name, _, _ in cameras:
        Image.fromarray(arrays[name+'_rgb']).save(folder/f'scene_{name}.png')
    (folder/'scene.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')


def live_photographer(out, backend, cameras_config):
    """photograph(phase, tick): one fresh capture of both cameras in a separate process; tick() runs while waiting."""
    sequence = 0
    python = cameras_config.capture_python
    cameras = json.dumps([list(c) for c in camera_specs(cameras_config)])
    timeout = cameras_config.capture_timeout_s

    def photograph(phase, tick):
        nonlocal sequence
        sequence += 1
        folder = out/'photos'/f'{sequence:02d}_{phase}'
        folder.mkdir(parents=True, exist_ok=False)
        with (folder/'capture.log').open('x') as log:
            proc = subprocess.Popen([python, '-m', 'forceharness.real.cameras', str(folder.resolve()),
                                     json.dumps(backend._state), cameras], stdout=log, stderr=log, cwd=str(ROOT))
            deadline = time.monotonic()+timeout
            try:
                while proc.poll() is None:
                    tick()
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'Camera capture exceeded {timeout:g} s; see {folder}/capture.log')
                if proc.returncode:
                    log.flush()
                    detail = (folder/'capture.log').read_text()[-2000:].strip().splitlines()
                    reason = detail[-1] if detail else f'exit code {proc.returncode}'
                    raise RuntimeError(f'Camera capture failed: {reason}; see {folder}/capture.log')
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=5)
        for name in ('scene_wrist.png', 'scene_exterior.png', 'scene.npz', 'scene.json'):
            if not (folder/name).is_file():
                raise RuntimeError(f'Fresh capture missing {name}; see {folder}/capture.log')
        return dict(wrist_images=[str((folder/'scene_wrist.png').resolve())],
                    captured_at=datetime.now().astimezone().isoformat())

    return photograph


if __name__ == '__main__':
    capture(Path(sys.argv[1]), json.loads(sys.argv[2]), [tuple(c) for c in json.loads(sys.argv[3])])
