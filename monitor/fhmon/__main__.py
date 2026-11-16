"""python -m fhmon  (run from <repository>/monitor, or use monitor/run.sh)."""
from __future__ import annotations

import argparse
import os
import re
import shutil
import signal
import sys
import threading
from pathlib import Path
from typing import Optional

from .server import LOOPBACK, App, Config, Server, default_video_config

HERE = Path(__file__).resolve().parents[1]          # <repository>/monitor


def load_env_file(path: Path) -> list:
    """KEY=VALUE lines (with ${VAR} references) into os.environ, like `set -a; . .env`.
    Returns the variable names; values are never printed."""
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        line = line[len("export "):] if line.startswith("export ") else line
        name, sep, value = line.partition("=")
        if not sep or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name.strip()):
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}", lambda m: os.environ.get(m.group(1), ""), value)
        os.environ[name.strip()] = value
        names.append(name.strip())
    return names


def runner_python(value) -> Optional[Path]:
    """The interpreter that starts the runner scripts: a path, or a command name looked up on PATH.
    Symlinks are kept as they are: a virtual environment's python is a symlink and must stay one."""
    path = Path(value).expanduser()
    if path.is_file():
        return path.absolute()
    found = shutil.which(str(value))
    return Path(found) if found else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="fhmon", description="ForceHarness monitor: web console for real-robot episodes")
    ap.add_argument("--harness-root", type=Path, default=HERE.parent,
                    help="ForceHarness repository root (default: parent of monitor/)")
    ap.add_argument("--runs", type=Path, help="run folder to index (default: <harness-root>/runs)")
    ap.add_argument("--data-dir", type=Path, help="where monitor sessions go (default: <runs>/monitor)")
    ap.add_argument("--python", type=Path,
                    help="interpreter for the runner scripts (default: $FH_PYTHON, else the interpreter running the monitor)")
    ap.add_argument("--codex-bin", type=Path,
                    help="Codex CLI executable for the codex-cli model source (default: $CODEX_BIN, else codex on PATH)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--allow-live", action="store_true",
                    help="allow starting REAL robot episodes from the page (loopback only; runner checks still apply)")
    ap.add_argument("--env-file", type=Path, help="load API keys from this .env into the monitor process only")
    ap.add_argument("--video-config", type=Path,
                    help="camera settings for session videos (default: monitor/video.json, else monitor/video.example.json)")
    ap.add_argument("--no-video", action="store_true", help="do not record sessions with the camera of the video config")
    a = ap.parse_args(argv)

    if a.allow_live and a.host not in LOOPBACK:
        ap.error("--allow-live only on 127.0.0.1/localhost: robot episodes must be started on the operator's machine "
                 "(use ssh -L to look from elsewhere)")
    if a.env_file:
        names = load_env_file(a.env_file)
        print(f"loaded {len(names)} variables from {a.env_file} (values not shown)")
    root = a.harness_root.resolve()
    if not (root / "scripts").is_dir():
        ap.error(f"{root} does not look like the ForceHarness repository (no scripts/)")
    runs = (a.runs or root / "runs").resolve()
    data = (a.data_dir or runs / "monitor").resolve()
    python = runner_python(a.python or os.environ.get("FH_PYTHON") or sys.executable)
    if python is None:
        ap.error(f"runner interpreter not found: {a.python or os.environ.get('FH_PYTHON')} (check --python or FH_PYTHON)")
    if a.codex_bin:
        os.environ["CODEX_BIN"] = str(a.codex_bin)
    video_config = (a.video_config or default_video_config()).resolve()

    cfg = Config(root, runs, data, Path(python), a.host, a.port, a.allow_live, video=not a.no_video,
                 video_config=video_config)
    app = App(cfg)
    server = Server(app)

    def shutdown(*_):
        if server.closing:
            return
        server.closing = True
        print("\nstopping: interrupting running episodes (soft stop through the runner) ...", flush=True)

        def run():
            app.sessions.shutdown()
            server.shutdown()
        threading.Thread(target=run, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    shown = "localhost" if a.host in LOOPBACK else a.host
    print(f"ForceHarness monitor {app.info()['version']}")
    print(f"  harness : {root}")
    print(f"  runs    : {runs}")
    print(f"  sessions: {data}")
    print(f"  python  : {python}")
    print(f"  live    : {'ENABLED (--allow-live)' if a.allow_live else 'off (demo / watch only)'}")
    video = app.videos.public_config()
    if video["enabled"]:
        cam = video["camera"]
        print(f"  video   : {cam['label'] or 'camera'} {cam['serial']} -> {video['dir']} ({'/'.join(video['modes'])} sessions)")
    else:
        print(f"  video   : off ({video['error']}; config {video_config.name})")
    if a.host not in LOOPBACK:
        print("  WARNING : listening beyond loopback; anyone who can open the page can start demo episodes")
    print(f"  open    : http://{shown}:{a.port}/", flush=True)
    server.serve_forever(poll_interval=0.5)
    server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
