"""Session videos from one global camera (config: monitor/video.json, made from video.example.json;
recorder: camrec.py).

One recorder subprocess per session whose mode the config lists (live by default). It starts right
after the runner and stops tail_s seconds after the runner exits. The runner never waits for the
camera: a camera that is missing or busy only marks that video failed. The recorder refuses the
cameras the harness opens (`never_open`): a RealSense serves one program at a time, so recording the
exterior or wrist camera would make the harness's own photos fail mid-episode.
Every video stays in one folder (default runs/videos/) as <session id>.mp4 with .json (status, camera,
first-frame wall time), .preview.jpg and .log beside it. The monitor never deletes them.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import quote

from .util import is_within, now_iso, read_json, write_json

CAMREC = Path(__file__).with_name("camrec.py")
FINAL = ("complete", "partial", "failed")
VID = re.compile(r"^[A-Za-z0-9_.-]{1,120}$")


class VideoManager:
    def __init__(self, conf_path: Path, runs_root: Path, harness_root: Path, paths, enabled: bool = True):
        conf = read_json(conf_path) or {}
        cam = conf.get("camera") or {}
        self.camera = dict(serial=str(cam.get("serial") or ""), label=str(cam.get("label") or ""),
                           width=int(cam.get("width") or 1280), height=int(cam.get("height") or 720),
                           fps=int(cam.get("fps") or 15), test_source=bool(cam.get("test_source")))
        self.never_open = {str(k): str(v) for k, v in (conf.get("never_open") or {}).items()}
        self.modes = list(conf.get("modes") or ["live"])
        self.tail_s = float(conf.get("tail_s", 3))
        self.max_s = float(conf.get("max_hours", 4)) * 3600
        self.python = str(conf.get("python") or sys.executable)
        folder = str(conf.get("dir") or "runs/videos")
        self.dir = ((runs_root / folder[5:]) if folder.startswith("runs/") else (harness_root / folder)).resolve()
        self.harness_root, self.paths = harness_root, paths
        serial = self.camera["serial"]
        if not conf:
            self.error = f"No recording config: {conf_path.name}"
        elif not enabled:
            self.error = "The Monitor was started with --no-video, so nothing is recorded this time"
        elif not conf.get("enabled"):
            self.error = f"enabled is false in {conf_path.name}"
        elif not serial and not self.camera["test_source"]:
            self.error = f"{conf_path.name} has no camera serial number"
        elif serial in self.never_open:
            self.error = f"Camera {serial} is used by the harness ({self.never_open[serial]}), so it cannot record video"
        elif not is_within(self.dir, runs_root):
            self.error = "The video folder must be inside the runs folder for the page to play it"
        elif not self.camera["test_source"] and not Path(self.python).exists():
            self.error = f"Python for the camera not found: {self.python}"
        else:
            self.error = None
        self.enabled = self.error is None
        self.lock = threading.Lock()
        self.recs: Dict[str, dict] = {}

    def public_config(self) -> dict:
        return dict(enabled=self.enabled, error=self.error, modes=self.modes, dir=self._shown(self.dir),
                    camera={k: v for k, v in self.camera.items() if k != "test_source"})

    def wants(self, mode: str) -> bool:
        return self.enabled and mode in self.modes

    # -- recording ---------------------------------------------------------------
    def start(self, vid: str, info: dict) -> Optional[str]:
        """Start recording; returns the video id (normally the session id) or None."""
        if not VID.match(vid):
            return None
        base, n = vid, 1
        while any((self.dir / f"{vid}{ext}").exists() for ext in (".mp4", ".part.mp4", ".json")):
            n += 1
            vid = f"{base}_{n}"                # never record over an earlier video
        with self.lock:
            for rec in self.recs.values():
                rec["cut"].set()               # a new session gets the camera now: end the previous tail
        self.dir.mkdir(parents=True, exist_ok=True)
        cam = self.camera
        cmd = [self.python, "-I", str(CAMREC), "--out", str(self.dir / f"{vid}.mp4"), "--width", str(cam["width"]),
               "--height", str(cam["height"]), "--fps", str(cam["fps"]), "--max-s", str(self.max_s),
               "--label", cam["label"], "--info", json.dumps(info, ensure_ascii=False)]
        cmd += ["--test-source"] if cam["test_source"] else ["--serial", cam["serial"]]
        for serial in self.never_open:
            cmd += ["--refuse", serial]
        try:
            with open(self.dir / f"{vid}.log", "ab") as log:
                proc = subprocess.Popen(cmd, cwd=self.dir, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
        except OSError as exc:
            write_json(self.dir / f"{vid}.json", dict(kind="fhmon_video", version=1, id=vid, status="failed", session=info,
                                                      started=now_iso(), ended=now_iso(), error=f"The recorder did not start: {exc}"))
            return vid
        with self.lock:
            self.recs[vid] = dict(proc=proc, cut=threading.Event())
        return vid

    def stop(self, vid: Optional[str]) -> None:
        """Keep recording tail_s more seconds (the arm coming to rest), then stop; returns at once."""
        with self.lock:
            rec = self.recs.get(vid) if vid else None
        if rec is not None:
            threading.Thread(target=self._finish, args=(vid, rec), name=f"video:{vid}", daemon=True).start()

    def _finish(self, vid: str, rec: dict) -> None:
        rec["cut"].wait(self.tail_s)
        self._interrupt(rec["proc"])
        try:
            rec["proc"].wait(timeout=600)      # the final remux of a long video takes a while
        except subprocess.TimeoutExpired:
            rec["proc"].kill()
            rec["proc"].wait()
        self._reconcile(vid, rec["proc"].returncode)
        with self.lock:
            if self.recs.get(vid) is rec:
                del self.recs[vid]

    @staticmethod
    def _interrupt(proc) -> None:
        if proc.poll() is None:
            try:
                os.kill(proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass

    def _reconcile(self, vid: str, code) -> None:
        """The recorder writes its own final status; if it died before that, record the failure."""
        path = self.dir / f"{vid}.json"
        meta = read_json(path) or dict(kind="fhmon_video", version=1, id=vid)
        if meta.get("status") not in FINAL:
            meta.update(status="failed", ended=now_iso(), error=f"The recorder exited abnormally (exit {code}); see {vid}.log")
            write_json(path, meta)

    def shutdown(self, timeout: float = 120.0) -> None:
        with self.lock:
            recs = list(self.recs.items())
        for _, rec in recs:
            rec["cut"].set()
            self._interrupt(rec["proc"])
        deadline = time.time() + timeout
        for vid, rec in recs:
            try:
                rec["proc"].wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                continue                       # it still finishes the file after the monitor is gone
            self._reconcile(vid, rec["proc"].returncode)

    # -- reading -----------------------------------------------------------------
    def list(self) -> List[dict]:
        out = []
        if self.dir.is_dir():
            for path in self.dir.glob("*.json"):
                meta = read_json(path)
                if isinstance(meta, dict) and meta.get("kind") == "fhmon_video" and VID.match(str(meta.get("id") or "")):
                    out.append(self.public(meta))
        out.sort(key=lambda v: v.get("started") or "", reverse=True)
        return out

    def for_episode(self, ep_dir: Optional[Path]) -> Optional[dict]:
        if ep_dir is None:
            return None
        target = str(Path(ep_dir).resolve())
        return next((v for v in self.list() if (v.get("session") or {}).get("episode_dir") == target), None)

    def preview(self, vid: str) -> Optional[Path]:
        path = self.dir / f"{vid}.preview.jpg"
        return path if VID.match(vid or "") and path.is_file() else None

    def public(self, meta: dict) -> dict:
        v = {k: val for k, val in meta.items() if k != "pid"}
        vid = v["id"]
        with self.lock:
            rec = self.recs.get(vid)
        v["active"] = bool(rec and rec["proc"].poll() is None)
        part = self.dir / f"{vid}.part.mp4"
        if v.get("status") not in FINAL and not v["active"]:
            v["status"] = "interrupted"        # the recorder was killed before it could finish the file
            v["file"] = part.name if part.is_file() else None
        f = self.dir / v["file"] if v.get("file") else None
        v["url"] = self.paths.url(f) if f and f.is_file() and v["status"] != "recording" else None
        v["path"] = self._shown(f) if f else None
        v["preview_url"] = f"/api/video/preview?id={quote(vid)}" if self.preview(vid) else None
        log = self.dir / f"{vid}.log"
        v["log_url"] = self.paths.url(log) if log.is_file() else None
        ep_dir = (v.get("session") or {}).get("episode_dir")
        v["episode_key"] = self.paths.rel(Path(ep_dir)) if ep_dir and (Path(ep_dir) / "episode.json").is_file() else None
        return v

    def _shown(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.harness_root.resolve()).as_posix()
        except ValueError:
            return str(path)
