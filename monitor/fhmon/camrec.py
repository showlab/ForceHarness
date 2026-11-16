"""Record one RealSense colour stream to MP4 for the monitor (python -m fhmon starts it per session).

Runs under the camera interpreter (pyrealsense2 + Pillow), not the monitor's, and imports nothing
from fhmon or forceharness:
    python -I camrec.py --serial <SERIAL> --out runs/videos/<id>.mp4 [--width 1280 --height 720 --fps 15]
                        [--refuse SERIAL ...] [--info JSON] [--max-s 14400]
    python -I camrec.py --test-source --out DIR/<id>.mp4 --max-s 3      synthetic frames, opens no camera
Next to <id>.mp4 it keeps:
  <id>.json          status, camera, first-frame wall time, frame counts (rewritten every 2 s and at the end)
  <id>.preview.jpg   the newest frame, about once a second
While recording, frames go to <id>.part.mp4 (fragmented MP4, readable up to the last fragment even after
a crash); a clean stop remuxes it into <id>.mp4 so players can seek. SIGINT/SIGTERM stop it, and so does
the monitor going away. A RealSense serves one program at a time, so a camera the harness opens must be
listed with --refuse; the recorder then exits before touching any device.
Video time t corresponds to wall time first_frame_unix + t: a camera hiccup repeats the previous frame
instead of shortening the video (counted as `duplicated`).
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

STOP = False


def _stop(*_):
    global STOP
    STOP = True


def now_iso():
    return datetime.now().astimezone().isoformat(timespec='seconds')


def write_json(path, data):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)


class TestSource:
    """Grey frames with a moving bar, paced to the frame rate on the wall clock; opens no device."""
    clock = 'host_wall'

    def __init__(self, width, height, fps):
        self.w, self.h, self.fps, self.n = width, height, fps, 0
        self.info = dict(name='test pattern (synthetic, no camera)')

    def start(self):
        self.t0 = time.time()

    def read(self):
        due = self.t0+self.n/self.fps
        if due > time.time():
            time.sleep(due-time.time())
        frame = bytearray(b'\x60'*(self.w*self.h*3))
        x = (self.n*8) % self.w
        bar = b'\xff'*(min(16, self.w-x)*3)
        for y in range(self.h):
            off = (y*self.w+x)*3
            frame[off:off+len(bar)] = bar
        self.n += 1
        return bytes(frame), due

    def stop(self):
        pass


class RealSenseSource:
    clock = None

    def __init__(self, serial, width, height, fps, retry_s):
        self.serial, self.w, self.h, self.fps, self.retry_s = serial, width, height, fps, retry_s
        self.pipe, self.info = None, dict(serial=serial)

    def start(self):
        import pyrealsense2 as rs
        self.rs = rs
        deadline = time.monotonic()+self.retry_s
        while True:
            try:
                return self._open()
            except RuntimeError:
                # e.g. the previous session's recorder is still finishing with the same camera
                if STOP or time.monotonic() > deadline:
                    raise
                time.sleep(1.0)

    def _open(self):
        rs = self.rs
        found = {}
        for dev in rs.context().query_devices():
            try:
                found[dev.get_info(rs.camera_info.serial_number)] = dev
            except RuntimeError:
                continue
        dev = found.get(self.serial)
        if dev is None:
            raise RuntimeError(f'No camera with serial {self.serial} is connected (connected now: {", ".join(sorted(found)) or "none"})')
        for key in ('name', 'usb_type_descriptor', 'firmware_version'):
            info = getattr(rs.camera_info, key)
            self.info[key] = dev.get_info(info) if dev.supports(info) else None
        pipe, cfg = rs.pipeline(), rs.config()
        cfg.enable_device(self.serial)
        cfg.enable_stream(rs.stream.color, self.w, self.h, rs.format.rgb8, self.fps)
        try:
            pipe.start(cfg)
        except RuntimeError as exc:
            raise RuntimeError(f'Cannot open camera {self.serial}: {exc} (another program may be using it, '
                               f'or this camera may not support {self.w}x{self.h} @ {self.fps} fps)') from exc
        self.pipe = pipe

    def read(self):
        rs = self.rs
        color = self.pipe.wait_for_frames(5000).get_color_frame()
        if not color:
            raise RuntimeError('The received frames contain no color image')
        if self.clock is None:
            # global/system time is the host clock in ms; anything else falls back to arrival time
            ts = color.get_timestamp()/1000
            synced = color.get_frame_timestamp_domain() in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time)
            self.clock = 'camera_global_time' if synced and abs(ts-time.time()) < 2 else 'host_arrival'
        ts = color.get_timestamp()/1000 if self.clock == 'camera_global_time' else time.time()
        return bytes(color.get_data()), ts

    def stop(self):
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except RuntimeError:
                pass
            self.pipe = None


def write_preview(path, data, width, height):
    try:
        from PIL import Image
    except ImportError:
        return
    img = Image.frombytes('RGB', (width, height), data)
    img.thumbnail((640, 640))
    tmp = path.with_name(path.name+'.tmp')
    img.save(tmp, 'JPEG', quality=72)
    os.replace(tmp, path)


def probe_duration(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)],
                             capture_output=True, text=True, timeout=60)
        return round(float(json.loads(out.stdout)['format']['duration']), 3)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None


def finalize(part, out):
    """Remux the fragmented recording for seeking; keep the fragmented file if that fails."""
    if not part.exists() or part.stat().st_size == 0:
        return None
    try:
        done = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-n', '-i', str(part),
                               '-c', 'copy', '-movflags', '+faststart', str(out)], timeout=600).returncode == 0
    except (OSError, subprocess.SubprocessError):
        done = False
    if done and out.exists() and out.stat().st_size > 0:
        part.unlink()
        return out
    return part


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--serial', help='RealSense serial number of the camera to record')
    src.add_argument('--test-source', action='store_true', help='synthetic frames; opens no camera')
    ap.add_argument('--out', type=Path, required=True, help='final <id>.mp4; must not exist yet')
    ap.add_argument('--width', type=int, default=1280)
    ap.add_argument('--height', type=int, default=720)
    ap.add_argument('--fps', type=int, default=15)
    ap.add_argument('--refuse', action='append', default=[], help='serial that must never be opened (repeatable)')
    ap.add_argument('--label', default='', help='name shown for this camera, e.g. "front D435"')
    ap.add_argument('--info', default='{}', help='JSON stored as "session" in <id>.json')
    ap.add_argument('--max-s', type=float, default=4*3600, help='stop after this much video')
    ap.add_argument('--open-retry-s', type=float, default=20, help='keep trying a busy camera this long')
    a = ap.parse_args(argv)

    out = a.out.resolve()
    meta_path, part = out.with_suffix('.json'), out.with_name(out.stem+'.part.mp4')
    preview = out.with_name(out.stem+'.preview.jpg')
    if any(p.exists() for p in (out, part, meta_path)):
        print(f'refusing to overwrite an existing recording: {out}', flush=True)
        return 3
    out.parent.mkdir(parents=True, exist_ok=True)
    w, h, fps = a.width, a.height, a.fps
    meta = dict(kind='fhmon_video', version=1, id=out.stem, file=None, status='starting', pid=os.getpid(),
                camera=dict(source='test' if a.test_source else 'realsense', label=a.label, serial=a.serial, width=w,
                            height=h, fps=fps),
                session=json.loads(a.info), started=now_iso())
    write_json(meta_path, meta)

    def fail(message, code=2):
        meta.update(status='failed', error=message, ended=now_iso())
        write_json(meta_path, meta)
        print(message, flush=True)
        return code

    if not a.test_source and a.serial in a.refuse:
        return fail(f'Refusing to open camera {a.serial}: the harness uses it (a RealSense can only be opened by one program at a time)')
    if w <= 0 or h <= 0 or fps <= 0:
        return fail('Resolution and frame rate must be positive')
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    parent = os.getppid()
    source = TestSource(w, h, fps) if a.test_source else RealSenseSource(a.serial, w, h, fps, a.open_retry_s)
    try:
        source.start()
    except Exception as exc:
        source.stop()
        return fail(str(exc))
    meta['camera'].update(source.info)
    # own session: a Ctrl+C meant for this recorder must not cut the encoder off mid-file
    ffmpeg = subprocess.Popen(
        ['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}',
         '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '23',
         '-pix_fmt', 'yuv420p', '-g', str(2*fps), '-movflags', '+frag_keyframe+empty_moov+default_base_moof',
         '-flush_packets', '1', '-f', 'mp4', str(part)],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, start_new_session=True)
    size = w*h*3
    written = duplicated = dropped = 0
    last = ts0 = None
    next_preview = next_meta = 0.0
    error = None
    try:
        while not STOP:
            data, ts = source.read()
            if len(data) != size:
                raise RuntimeError(f'Wrong frame size: {len(data)} bytes, expected {size}')
            if ts0 is None:
                ts0 = ts
                meta.update(status='recording', clock=source.clock, first_frame_unix=round(ts, 3),
                            first_frame_at=datetime.fromtimestamp(ts).astimezone().isoformat(timespec='milliseconds'))
                write_json(meta_path, meta)
            index = round((ts-ts0)*fps)
            if index-written > 10*fps:
                # a stall ends at the 5 s frame timeout, so this is a clock step: rebase, don't pad minutes
                ts0 += (index-written)/fps
                index = written
                meta['clock_steps'] = meta.get('clock_steps', 0)+1
            if index < written:
                dropped += 1
                continue
            if last is not None and index > written:
                for _ in range(index-written):
                    ffmpeg.stdin.write(last)
                duplicated += index-written
                written = index
            ffmpeg.stdin.write(data)
            written += 1
            last = data
            now = time.monotonic()
            if now >= next_preview:
                write_preview(preview, data, w, h)
                next_preview = now+1.0
            if now >= next_meta:
                meta.update(frames=written, duplicated=duplicated, dropped=dropped, recorded_s=round(written/fps, 1),
                            updated=now_iso())
                write_json(meta_path, meta)
                next_meta = now+2.0
            if os.getppid() != parent:
                meta['stop_reason'] = 'The Monitor process is gone'
                break
            if written/fps >= a.max_s:
                meta['stop_reason'] = f'Reached the maximum recording time of {a.max_s/3600:g} h' if a.max_s >= 3600 else f'Reached the maximum recording time of {a.max_s:g} s'
                break
    except BrokenPipeError:
        error = 'The ffmpeg encoder exited early (see the .log file with the same name)'
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
    finally:
        source.stop()
        if STOP:
            meta.setdefault('stop_reason', 'Received a stop signal')
        meta.update(status='finalizing', frames=written, duplicated=duplicated, dropped=dropped,
                    recorded_s=round(written/fps, 1), updated=now_iso())
        write_json(meta_path, meta)
        try:
            ffmpeg.stdin.close()
        except BrokenPipeError:
            pass
        try:
            ffmpeg.wait(timeout=120)
        except subprocess.TimeoutExpired:
            ffmpeg.kill()
            ffmpeg.wait()
    final = finalize(part, out) if written else None
    if final is None:
        if part.exists() and not written:
            part.unlink()                     # an empty container, nothing recorded
        return fail(error or 'No frames were recorded')
    sha = hashlib.sha256()                    # chunked, not hashlib.file_digest: that needs Python 3.11
    with final.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            sha.update(chunk)
    digest = sha.hexdigest()
    meta.update(status='partial' if error else 'complete', error=error, file=final.name, ended=now_iso(),
                duration_s=probe_duration(final), bytes=final.stat().st_size, sha256=digest,
                ffmpeg_returncode=ffmpeg.returncode)
    write_json(meta_path, meta)
    return 1 if error else 0


if __name__ == '__main__':
    raise SystemExit(main())
