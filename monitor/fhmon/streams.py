"""Force streams the harness logs while an episode runs (layer 1, force sensing, seen from outside).

- <episode>/nuc/nuc.jsonl: the robot PC's stream on the real robot. 'sample' rows at ~57 Hz with the
  raw 6-axis wrench (base frame, flange reference, NOT tared, sign not corrected).
- <episode>/motion.jsonl: state rows with 'wrench' (raw) and optional 'phase' and 'trial' labels.

'timestamp_ns' comes from the robot PC clock, which need not be synced with the workstation
(minutes apart are common). The stream is anchored with the file's mtime: the newest line was
written moments before it. Demo ports use a synthetic clock that starts near zero; those streams
are plotted on their own time base.

Every point is [x, fx, fy, fz, |F|, |T|, gripper_width] with x in seconds since the
episode start (real clock) or since the first sample (synthetic clock).
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

NUC_SAMPLE = re.compile(rb'\{\s*"type"\s*:\s*"sample"')
SYNTHETIC_BEFORE = 1e9          # seconds; real robot timestamps are Unix time (~1.79e9)


# Where a runner logs the force stream inside its episode folder, in order of preference, and the row format.
STREAM_FILES = ((("nuc", "nuc.jsonl"), "nuc"),          # robot-PC stream: forceharness/real/transport.py writes <episode>/nuc/nuc.jsonl
                (("motion.jsonl",), "motion"))


def find_stream(episode_dir: Path):
    """(path, kind) of the force stream for this episode, or (None, None)."""
    for parts, kind in STREAM_FILES:
        path = episode_dir.joinpath(*parts)
        if path.is_file():
            return path, kind
    return None, None


def _parse(raw: bytes, kind: str):
    if kind == "nuc":
        if not NUC_SAMPLE.match(raw):
            return None
    elif b'"wrench"' not in raw:
        return None
    try:
        d = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    w = d.get("wrench")
    ts = d.get("timestamp_ns")
    if not isinstance(w, list) or len(w) < 6 or not isinstance(ts, (int, float)):
        return None
    try:
        fx, fy, fz, tx, ty, tz = (float(v) for v in w[:6])
    except (TypeError, ValueError):
        return None
    grip = d.get("gripper_width")
    return (ts / 1e9, fx, fy, fz, tx, ty, tz,
            float(grip) if isinstance(grip, (int, float)) else None,
            d.get("phase"), d.get("trial"))


def _point(s, x) -> list:
    t, fx, fy, fz, tx, ty, tz, grip = s[:8]
    return [round(x, 3), round(fx, 3), round(fy, 3), round(fz, 3),
            round(math.sqrt(fx * fx + fy * fy + fz * fz), 3),
            round(math.sqrt(tx * tx + ty * ty + tz * tz), 4),
            None if grip is None else round(grip, 4)]


def _fn(s) -> float:
    return s[1] * s[1] + s[2] * s[2] + s[3] * s[3]


class Clock:
    """Maps robot data time to x (seconds since the episode start)."""

    def __init__(self, first_t: float, last_t: float, file_mtime: float, episode_start: Optional[float]):
        self.synthetic = first_t < SYNTHETIC_BEFORE
        self.offset = None if self.synthetic else file_mtime - last_t      # wall = data + offset
        self.origin = first_t if (self.synthetic or episode_start is None) else episode_start - self.offset

    def x(self, t: float) -> float:
        return t - self.origin

    def wall_to_x(self, wall: Optional[float]) -> Optional[float]:
        if wall is None or self.synthetic:
            return None
        return wall - self.offset - self.origin

    def to_dict(self) -> dict:
        return dict(synthetic=self.synthetic, offset_s=self.offset)


_cache: "OrderedDict[tuple, dict]" = OrderedDict()
_cache_lock = threading.Lock()


def overview(path: Path, kind: str, episode_start: Optional[float], max_points: int = 2400) -> dict:
    """Whole stream, decimated to <= max_points by keeping each bin's largest |F| sample."""
    try:
        st = os.stat(path)
    except OSError:
        return dict(points=[], phases=[], clock=None)
    key = (str(path), st.st_mtime_ns, st.st_size, episode_start, max_points)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    samples = []
    with open(path, "rb") as f:
        for raw in f:
            s = _parse(raw, kind)
            if s:
                samples.append(s)
    if not samples:
        result = dict(points=[], phases=[], clock=None, kind=kind)
    else:
        clock = Clock(samples[0][0], samples[-1][0], st.st_mtime, episode_start)
        step = max(1, math.ceil(len(samples) / max_points))
        points = []
        for i in range(0, len(samples), step):
            best = max(samples[i:i + step], key=_fn)
            points.append(_point(best, clock.x(best[0])))
        phases = []
        for s in samples:
            label = s[8]
            if label is None:
                continue
            x = round(clock.x(s[0]), 3)
            if phases and phases[-1]["label"] == label and phases[-1]["trial"] == s[9]:
                phases[-1]["x1"] = x
            else:
                phases.append(dict(label=label, trial=s[9], x0=x, x1=x))
        result = dict(points=points, phases=phases, clock=clock.to_dict(), kind=kind,
                      n_samples=len(samples), rate_hz=_rate(samples))
        result["_clock"] = clock
    with _cache_lock:
        _cache[key] = result
        while len(_cache) > 6:
            _cache.popitem(last=False)
    return result


def _rate(samples) -> Optional[float]:
    if len(samples) < 2:
        return None
    span = samples[-1][0] - samples[0][0]
    return round((len(samples) - 1) / span, 1) if span > 0 else None


class StreamTail:
    """Follows a growing stream; poll() returns the new points, at most ~25 per second
    of data, keeping each 40 ms bin's largest |F| so short peaks survive."""

    def __init__(self, path: Path, kind: str, episode_start: Optional[float],
                 backlog_bytes: int = 1_500_000, bin_s: float = 0.04):
        self.path, self.kind = Path(path), kind
        self.episode_start = episode_start
        self.backlog, self.bin_s = backlog_bytes, bin_s
        self.pos: Optional[int] = None
        self.buf = b""
        self.clock: Optional[Clock] = None
        self.pending: List[tuple] = []
        self.phases: List[dict] = []            # labelled motion phases seen so far, like overview()
        self.phases_changed = False

    def _decimate(self, samples) -> List[list]:
        out = []
        for s in samples:
            if self.pending and s[0] - self.pending[0][0] >= self.bin_s:
                best = max(self.pending, key=_fn)
                out.append(_point(best, self.clock.x(best[0])))
                self.pending = []
            self.pending.append(s)
        return out

    def poll(self, max_bytes: int = 4_000_000) -> List[list]:
        try:
            st = os.stat(self.path)
        except OSError:
            return []
        if self.pos is None or st.st_size < self.pos:
            self.pos = max(0, st.st_size - self.backlog)
            self.buf = b""
            skip_partial = self.pos > 0
        else:
            skip_partial = False
        if st.st_size == self.pos:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            chunk = f.read(max_bytes)
        self.pos += len(chunk)
        data = self.buf + chunk
        lines = data.split(b"\n")
        self.buf = lines.pop()
        if skip_partial and lines:
            lines = lines[1:]
        samples = [s for s in (_parse(raw, self.kind) for raw in lines) if s]
        if not samples:
            return []
        if self.clock is None:
            # The newest complete sample was written just before the file's mtime.
            self.clock = Clock(samples[0][0], samples[-1][0], st.st_mtime, self.episode_start)
        for smp in samples:
            if smp[8] is None:
                continue
            x = round(self.clock.x(smp[0]), 3)
            last = self.phases[-1] if self.phases else None
            if last and last["label"] == smp[8] and last["trial"] == smp[9]:
                last["x1"] = x
            else:
                self.phases.append(dict(label=smp[8], trial=smp[9], x0=x, x1=x))
                self.phases_changed = True
        return self._decimate(samples)

    def clock_info(self) -> Optional[dict]:
        return self.clock.to_dict() if self.clock else None


def peak(points: List[list]) -> Dict[str, Optional[float]]:
    if not points:
        return dict(force=None, torque=None)
    return dict(force=max(p[4] for p in points), torque=max(p[5] for p in points))
