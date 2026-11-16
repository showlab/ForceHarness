"""Session videos: the recorder (synthetic frames only, no camera), the manager's lifecycle around a
session, and byte-range serving for the browser's player."""
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from fhmon import camrec
from fhmon.episodes import Paths
from fhmon.server import App, Config, Server
from fhmon.sessions import SessionManager
from fhmon.video import VideoManager

pytestmark = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="needs ffmpeg")
CAMREC = Path(camrec.__file__)
EXTERIOR, WRIST = "000000000001", "000000000002"           # placeholder serials, no camera is opened


def conf(tmp_path, **over) -> Path:
    data = dict(enabled=True, modes=["live"], dir="runs/videos", python=sys.executable, tail_s=0.2, max_hours=1,
                camera=dict(serial="", test_source=True, label="test", width=160, height=120, fps=10),
                never_open={EXTERIOR: "exterior", WRIST: "wrist"})
    data.update(over)
    path = tmp_path / "video.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return path


def manager(tmp_path, runs, **over) -> VideoManager:
    return VideoManager(conf(tmp_path, **over), runs, tmp_path, Paths([runs]))


def wait_for(cond, timeout=20.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = cond()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("timed out")


def test_recorder_writes_a_seekable_mp4_and_stops_on_sigint(tmp_path):
    out = tmp_path / "v" / "a.mp4"
    proc = subprocess.Popen([sys.executable, "-I", str(CAMREC), "--test-source", "--out", str(out), "--width", "160",
                             "--height", "120", "--fps", "10", "--info", json.dumps(dict(session="a"))])
    meta = out.with_suffix(".json")
    wait_for(lambda: (json.loads(meta.read_text()).get("frames") or 0) >= 10 if meta.exists() else False)
    proc.send_signal(signal.SIGINT)
    assert proc.wait(30) == 0
    m = json.loads(meta.read_text())
    assert m["status"] == "complete" and m["file"] == "a.mp4" and m["session"] == dict(session="a")
    assert m["first_frame_unix"] > 0 and m["duration_s"] >= 1.0 and len(m["sha256"]) == 64
    assert out.is_file() and not out.with_name("a.part.mp4").exists()
    assert out.with_name("a.preview.jpg").is_file()
    head = out.read_bytes()[:64]
    assert b"ftyp" in head and b"moov" in out.read_bytes()[:4096]     # faststart: index at the front


def test_recorder_refuses_harness_cameras_before_touching_a_device(tmp_path):
    # the monitor's interpreter has no pyrealsense2: reaching the device code would raise ImportError
    out = tmp_path / "b.mp4"
    code = camrec.main(["--serial", EXTERIOR, "--refuse", EXTERIOR, "--refuse", WRIST, "--out", str(out)])
    m = json.loads(out.with_suffix(".json").read_text())
    assert code == 2 and m["status"] == "failed" and "harness" in m["error"] and not out.exists()


def test_recorder_never_overwrites_a_recording(tmp_path):
    out = tmp_path / "c.mp4"
    out.write_bytes(b"old")
    assert camrec.main(["--test-source", "--out", str(out), "--max-s", "1"]) == 3
    assert out.read_bytes() == b"old"


def test_config_cannot_point_at_a_harness_camera(tmp_path, runs):
    vm = manager(tmp_path, runs, camera=dict(serial=EXTERIOR, width=640, height=480, fps=15))
    assert not vm.enabled and "harness" in vm.error and not vm.wants("live")
    assert not manager(tmp_path, runs, dir="/tmp/elsewhere").enabled
    assert VideoManager(conf(tmp_path), runs, tmp_path, Paths([runs]), enabled=False).error.startswith("The Monitor was started with --no-video")


def test_manager_records_a_session_and_finds_it_by_episode(tmp_path, runs):
    vm = manager(tmp_path, runs)
    ep_dir = runs / "monitor" / "s1" / "episode"
    vid = vm.start("s1", dict(session="s1", task="free", mode="live", episode_dir=str(ep_dir.resolve())))
    assert vid == "s1"
    wait_for(lambda: (vm.for_episode(ep_dir) or {}).get("status") == "recording")
    v = vm.for_episode(ep_dir)
    assert v["active"] and v["url"] is None and v["episode_key"] is None      # no episode.json yet
    assert wait_for(lambda: vm.for_episode(ep_dir)["preview_url"])
    ep_dir.mkdir(parents=True)
    (ep_dir / "episode.json").write_text("{}")
    vm.stop("s1")
    # "complete" is written before the recorder thread clears "active", so wait for both
    v = wait_for(lambda: (lambda x: x if x["status"] == "complete" and not x["active"] else None)(vm.for_episode(ep_dir)))
    assert v["url"].startswith("/api/file?f=") and v["path"].endswith("runs/videos/s1.mp4")
    assert v["episode_key"] == "0:monitor/s1/episode" and not v["active"]
    assert [x["id"] for x in vm.list()] == ["s1"]
    # a second session with the same id never records over the first video
    assert vm.start("s1", dict(session="s1")) == "s1_2"
    vm.shutdown(timeout=30)
    assert json.loads((vm.dir / "s1_2.json").read_text())["status"] in ("complete", "failed")


def test_recorder_that_dies_is_reported_failed(tmp_path, runs):
    dies = tmp_path / "dies"
    dies.write_text("#!/bin/sh\nexit 1\n")
    dies.chmod(0o755)
    vm = manager(tmp_path, runs, python=str(dies))
    vm.camera["test_source"] = True
    vm.start("s2", dict(session="s2"))
    vm.stop("s2")
    v = wait_for(lambda: next((x for x in vm.list() if x["id"] == "s2" and x["status"] == "failed"), None))
    assert "exit" in v["error"]


class _Cfg:
    allow_live = True


def test_sessions_record_only_listed_modes(tmp_path, runs):
    vm = manager(tmp_path, runs, modes=["demo"])
    sm = SessionManager(_Cfg(), {}, None, None, None, videos=vm)

    class S:  # the parts of a Session that recording touches
        def __init__(self, sid, mode):
            self.id, self.mode, self.started, self.video = sid, mode, "2026-10-29T02:00:00+00:00", None
            self.task, self.condition = dict(id="free", name="Free task"), dict(id="A", label="Force harness", arm="harness")
            self.dir = runs / "monitor" / sid
            self.episode_dir = self.dir / "episode"

    demo, live = S("d1", "demo"), S("l1", "live")
    sm._record(demo)
    sm._record(live)
    assert demo.video == "d1" and live.video is None
    vm.stop(demo.video)
    v = wait_for(lambda: (lambda x: x if x and x["status"] == "complete" else None)(vm.for_episode(demo.episode_dir)))
    assert v["session"]["task_name"] == "Free task" and v["session"]["mode"] == "demo"


@pytest.fixture
def served(tmp_path, runs):
    root = tmp_path / "fh"
    (root / "scripts").mkdir(parents=True)
    cfg = Config(root, runs, runs / "monitor", Path(sys.executable), "127.0.0.1", 0, allow_live=False,
                 video_config=conf(tmp_path))
    app = App(cfg)
    srv = Server(app)
    port = srv.server_address[1]
    app.hosts = {f"127.0.0.1:{port}"}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def fetch(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def test_video_files_are_served_with_byte_ranges(served, runs):
    app, base = served
    body = bytes(range(256)) * 40
    (runs / "videos").mkdir()
    (runs / "videos" / "x.mp4").write_bytes(body)
    url = base + app.store.paths.url(runs / "videos" / "x.mp4")
    status, headers, data = fetch(url)
    assert status == 200 and data == body and headers["Accept-Ranges"] == "bytes" and headers["Content-Type"] == "video/mp4"
    status, headers, data = fetch(url, {"Range": "bytes=100-199"})
    assert status == 206 and data == body[100:200] and headers["Content-Range"] == f"bytes 100-199/{len(body)}"
    assert fetch(url, {"Range": "bytes=10000-"})[2] == body[10000:]
    assert fetch(url, {"Range": "bytes=-16"})[2] == body[-16:]
    status, headers, _ = fetch(url, {"Range": f"bytes={len(body)}-"})
    assert status == 416 and headers["Content-Range"] == f"bytes */{len(body)}"


def test_video_api_lists_recordings_and_serves_the_preview(served, runs):
    app, base = served
    assert json.loads(fetch(base + "/api/info")[2])["video"]["enabled"]
    app.videos.start("s3", dict(session="s3", episode_dir=str((runs / "monitor" / "s3" / "episode").resolve())))
    wait_for(lambda: app.videos.preview("s3"))
    status, headers, data = fetch(base + "/api/video/preview?id=s3")
    assert status == 200 and data[:2] == b"\xff\xd8" and headers["Cache-Control"] == "no-store"
    assert fetch(base + "/api/video/preview?id=../x")[0] == 404
    key = "0:monitor/s3/episode"
    video = json.loads(fetch(base + "/api/video?key=" + urllib.request.quote(key))[2])["video"]
    assert video["id"] == "s3" and video["status"] == "recording"
    app.videos.shutdown(timeout=30)
    listing = json.loads(fetch(base + "/api/videos")[2])
    assert [v["id"] for v in listing["videos"]] == ["s3"] and listing["videos"][0]["status"] == "complete"
    assert listing["config"]["dir"].endswith("runs/videos")
