"""HTTP server: JSON API, the live event stream (SSE) and the static web app. Standard library only.

It can start robot episodes, so:
- it binds 127.0.0.1 by default and refuses --allow-live on any other address;
- every POST must carry the per-start token embedded in the page (no cross-site requests);
- on loopback the Host header must name this server (no DNS rebinding);
- /api/file serves only images, text and MP4 video from the run roots, by extension.
"""
from __future__ import annotations

import hashlib
import json
import queue
import re
import secrets
import sys
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from . import __version__, episodes, live, llm, streams
from . import settings as settings_mod
from .bridge import BRIDGE_SYSTEM
from .labels import Labels
from .sessions import LaunchError, SessionManager
from .settings import Settings, SettingsError
from .util import is_within, read_json
from .video import VideoManager

WEB = Path(__file__).resolve().parents[1] / "web"
TASKS = Path(__file__).resolve().parents[1] / "tasks.json"
# Site-specific camera settings: video.json when it exists (copy video.example.json and fill it in),
# else the example, which ships with recording disabled.
VIDEO = Path(__file__).resolve().parents[1] / "video.json"
VIDEO_EXAMPLE = VIDEO.with_name("video.example.json")
FILE_TYPES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".md": "text/plain; charset=utf-8",
              ".json": "application/json; charset=utf-8", ".txt": "text/plain; charset=utf-8",
              ".log": "text/plain; charset=utf-8", ".mp4": "video/mp4"}
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
                ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".png": "image/png", ".ico": "image/x-icon"}
TEXT_LIMIT = 6_000_000
LOOPBACK = ("127.0.0.1", "localhost", "::1")


def default_video_config() -> Path:
    return VIDEO if VIDEO.exists() else VIDEO_EXAMPLE


class Config:
    def __init__(self, harness_root: Path, runs_root: Path, data_dir: Path, python: Path, host: str, port: int,
                 allow_live: bool, video: bool = True, video_config: Optional[Path] = None):
        self.harness_root, self.runs_root, self.data_dir = harness_root, runs_root, data_dir
        self.python, self.host, self.port, self.allow_live = python, host, port, allow_live
        self.video, self.video_config = video, video_config or default_video_config()


def load_registry(harness_root: Path) -> dict:
    """tasks.json by id. A fixed task's goal is the prompt file its runner reads at start, relative to the
    repository root; a missing file is reported when the task is launched (records stay viewable)."""
    data = read_json(TASKS) or {}
    for task in data.get("tasks") or []:
        if task.get("fixed_instruction_file"):
            path = Path(harness_root) / task["fixed_instruction_file"]
            task["fixed_instruction_path"] = str(path)     # the page may edit it: sessions.fixed_instruction re-reads
            task["fixed_instruction"] = path.read_text(encoding="utf-8").strip() if path.is_file() else None
    return {t["id"]: t for t in data.get("tasks") or []}


class App:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.token = secrets.token_urlsafe(24)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        roots = [cfg.runs_root] + ([] if is_within(cfg.data_dir, cfg.runs_root) else [cfg.data_dir])
        self.store = live.Store(roots)
        self.hub = live.Hub()
        self.watchers = live.Watchers(self.store, self.hub)
        self.settings = Settings(cfg.data_dir / "settings.json")
        self.registry = load_registry(cfg.harness_root)
        self.videos = VideoManager(cfg.video_config, cfg.runs_root, cfg.harness_root, self.store.paths, enabled=cfg.video)
        self.sessions = SessionManager(cfg, self.registry, self.settings, self.hub, self.store, videos=self.videos)
        self.store.bridge_log = self.sessions.attach_log
        self.labels = Labels(cfg.data_dir / "labels.json")
        self.store.labels = self.labels
        self.model_cache: dict = {}
        port = cfg.port
        self.loopback = cfg.host in LOOPBACK
        self.hosts = {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}

    # -- read-only views ---------------------------------------------------------
    def info(self) -> dict:
        tasks = []
        for t in self.registry.values():
            self.sessions.fixed_instruction(t)
            tasks.append({k: t.get(k) for k in ("id", "name", "short", "summary", "max_actions", "guards", "torque_cap",
                                                 "conditions", "demo", "live", "live_external", "score", "model_images",
                                                 "instruction", "fixed_instruction", "fixed_instruction_file")})
        return dict(version=__version__, harness_root=str(self.cfg.harness_root), runs_root=str(self.cfg.runs_root),
                    data_dir=str(self.cfg.data_dir), python=str(self.cfg.python), allow_live=self.cfg.allow_live,
                    host=self.cfg.host, port=self.cfg.port, tasks=tasks, providers=llm.PROVIDERS,
                    bridge_system=BRIDGE_SYSTEM, video=self.videos.public_config(), now=time.time())

    def episode_force(self, key: str) -> dict:
        ep = self.store.load(key, with_text=False)
        if ep is None:
            raise LaunchError("Episode not found")
        stream = ep.get("stream") or {}
        turn_bands = [dict(x0=t["x0"], x1=t["x1"], n=t["n"], label=(t.get("action") or {}).get("name"))
                      for t in ep["turns"] if t.get("x0") is not None and t.get("x1") is not None]
        events = []
        for t in ep["turns"]:
            rep = t.get("report") or {}
            if t.get("x1") is not None and rep.get("duration_s"):
                start = t["x1"] - rep["duration_s"]
                for e in rep.get("events") or []:
                    if isinstance(e.get("t"), (int, float)):
                        events.append(dict(x=round(start + e["t"], 3), kind=e.get("kind"), force=e.get("force"), n=t["n"]))
        path = self.store.paths.resolve(stream["file"]) if stream.get("file") else None
        if path is not None:
            ov = streams.overview(path, stream["kind"], ep.get("started_epoch"))
            clock = ov.get("clock") or {}
            synthetic = clock.get("synthetic")
            return dict(source="stream", kind=ov.get("kind"), points=ov.get("points", []), phases=ov.get("phases", []),
                        clock=clock, rate_hz=ov.get("rate_hz"), n_samples=ov.get("n_samples"),
                        bands=[] if synthetic else turn_bands, events=[] if synthetic else events,
                        raw=True)
        # No logged stream (demo port): lay the primitives' own traces end to end.
        points, bands, events, x = [], [], [], 0.0
        ep_dir = self.store.ep_dir(key)
        for t in ep["turns"]:
            if not t.get("report"):
                continue
            rows = episodes.report_rows(ep_dir / f"action_{t['n']:02d}.json") or {}
            pts = rows.get("points") or []
            if not pts:
                continue
            t0 = pts[0][0]
            for p in pts:
                points.append([round(x + p[0] - t0, 3)] + p[1:6] + [None])
            span = pts[-1][0] - t0
            bands.append(dict(x0=round(x, 3), x1=round(x + span, 3), n=t["n"], label=rows.get("primitive")))
            for e in rows.get("events") or []:
                if isinstance(e.get("t"), (int, float)):
                    events.append(dict(x=round(x + e["t"] - t0, 3), kind=e.get("kind"), force=e.get("force"), n=t["n"]))
            x += span + 0.6
        return dict(source="actions", kind=None, points=points, phases=[], bands=bands, events=events,
                    clock=dict(synthetic=True, concatenated=True), raw=False)

    def model_list(self, body: dict) -> dict:
        """Dropdown content for the settings form: models the key can use (cached 10 min) plus
        common names. The form may be unsaved; a stored key is only used for the same host."""
        stored = self.settings.get(str(body.get("id") or ""))
        form = body.get("profile") or {}
        provider = form.get("provider") or (stored or {}).get("provider") or "openai"
        if provider not in llm.PROVIDERS:
            raise LaunchError("Unknown interface")
        base = str(form.get("base_url") or "").strip() or llm.DEFAULT_BASE[provider]
        if not re.match(r"^https?://\S+$", base):
            raise LaunchError("Base URL must start with http:// or https://")
        if provider in llm.KEYLESS:          # Codex CLI: the plan's models from its own cache, no key
            return dict(models=llm.list_models(dict(provider=provider), ""), presets=[], source="codex-login")
        prices = settings_mod.load_prices(self.cfg.harness_root / settings_mod.PRICE_TABLE)
        presets = settings_mod.presets(provider, prices)
        key, source = self.settings.key_for_host(stored, provider, base, str(body.get("key") or ""),
                                                 str(form.get("key_env") or ""))
        if not key:
            return dict(models=[], presets=presets, source=None)
        cache_key = (provider, base.rstrip("/"), hashlib.sha256(key.encode()).hexdigest()[:16])
        hit = self.model_cache.get(cache_key)
        if hit and time.time() - hit[0] < 600 and not body.get("refresh"):
            models = hit[1]
        else:
            try:
                models = llm.list_models(dict(provider=provider, base_url=base), key)
            except llm.LLMError as exc:
                return dict(models=[], presets=presets, source=source, error=str(exc))
            self.model_cache[cache_key] = (time.time(), models)
        for m in models:
            pr = settings_mod.price_for(m["id"], prices)
            m["price_in"], m["price_out"] = (pr if pr else (None, None))
        return dict(models=models, presets=presets, source=source)

    def action_rows(self, key: str, n: int) -> dict:
        ep_dir = self.store.ep_dir(key)
        if ep_dir is None:
            raise LaunchError("Episode not found")
        rows = episodes.report_rows(ep_dir / f"action_{int(n):02d}.json")
        if rows is None:
            raise LaunchError("This step has no primitive report")
        return rows

    def stream_events(self, key: str):
        watcher = self.watchers.ensure(key)
        first = watcher.initial_events()
        for s in self.sessions.sessions.values():
            if s.key == key:
                first.append(dict(type="session", session=s.public()))
                if s.bridge is not None:
                    first.append(dict(type="model", kind="state", **s.bridge.state))
        attached = self.sessions.attached.get(key)
        if attached is not None:
            first.append(dict(type="model", kind="state", **attached.state, attached=True))
        return first


class Handler(BaseHTTPRequestHandler):
    server_version = f"fhmon/{__version__}"
    protocol_version = "HTTP/1.0"

    @property
    def app(self) -> App:
        return self.server.app  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # quiet: only errors go to stderr
        pass

    # -- helpers -------------------------------------------------------------
    def _send(self, status: int, body: bytes, ctype: str, extra: dict = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, status: int = 200):
        self._send(status, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, status: int, message: str):
        self._json(dict(error=message), status)

    def _host_ok(self) -> bool:
        if not self.app.loopback:
            return True
        return (self.headers.get("Host") or "").strip().lower() in self.app.hosts

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length > 2_000_000:
            raise LaunchError("Request too large")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise LaunchError("Request is not valid JSON") from None
        if not isinstance(data, dict):
            raise LaunchError("Request must be a JSON object")
        return data

    # -- dispatch --------------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        url = urlparse(self.path)
        q = {k: v[-1] for k, v in parse_qs(url.query).items()}
        try:
            route = url.path
            if route == "/" or route == "/index.html":
                return self._index()
            if route.startswith("/static/"):
                return self._static(route[len("/static/"):])
            if route == "/api/info":
                return self._json(self.app.info())
            if route == "/api/episodes":
                return self._json(dict(episodes=self.app.store.scan()))
            if route == "/api/episode":
                ep = self.app.store.load(q.get("key", ""))
                return self._json(ep) if ep else self._error(404, "Episode not found")
            if route == "/api/episode/force":
                return self._json(self.app.episode_force(q.get("key", "")))
            if route == "/api/episode/action":
                return self._json(self.app.action_rows(q.get("key", ""), int(q.get("n", "0"))))
            if route == "/api/file":
                return self._file(q.get("f", ""))
            if route == "/api/compare":
                return self._json(live.compare(self.app.store.scan(), q.get("task", "free"), q.get("mode", "all"),
                                               q.get("instruction")))
            if route == "/api/settings":
                return self._json(dict(profiles=self.app.settings.public(), providers=llm.PROVIDERS,
                                       bridge_system=BRIDGE_SYSTEM))
            if route == "/api/sessions":
                return self._json(dict(sessions=self.app.sessions.list(),
                                       attached=[dict(key=k, state=b.state) for k, b in self.app.sessions.attached.items()]))
            if route == "/api/inputs":
                return self._json(self.app.sessions.live_inputs(q.get("task", "")))
            if route == "/api/stream":
                return self._stream(q.get("key", ""))
            if route == "/api/videos":
                return self._json(dict(videos=self.app.videos.list(), config=self.app.videos.public_config()))
            if route == "/api/video":
                return self._json(dict(video=self.app.videos.for_episode(self.app.store.paths.resolve(q.get("key", "")))))
            if route == "/api/video/preview":
                path = self.app.videos.preview(q.get("id", ""))
                return self._send(200, path.read_bytes(), "image/jpeg") if path else self._error(404, "No frame yet")
            return self._error(404, "not found")
        except (LaunchError, SettingsError, ValueError) as exc:
            return self._error(400, str(exc))
        except (BrokenPipeError, ConnectionResetError):
            return None
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return self._error(500, f"{type(exc).__name__}: {exc}")

    def do_POST(self):
        if not self._host_ok():
            return self._error(403, "unexpected Host header")
        if self.headers.get("X-FHMon-Token") != self.app.token:
            return self._error(403, "Missing or wrong page token (please reload the page)")
        origin = self.headers.get("Origin")
        if origin and self.app.loopback and urlparse(origin).netloc.lower() not in self.app.hosts:
            return self._error(403, "cross-origin request refused")
        url = urlparse(self.path)
        try:
            body = self._body()
            app, sm = self.app, self.app.sessions
            route = url.path
            if route == "/api/settings/profile":
                profile = app.settings.upsert(body.get("profile") or {}, key=body.get("key") or None,
                                              clear_key=bool(body.get("clear_key")))
                return self._json(dict(profile=profile, profiles=app.settings.public()))
            if route == "/api/settings/delete":
                app.settings.delete(str(body.get("id") or ""))
                return self._json(dict(profiles=app.settings.public()))
            if route == "/api/settings/test":
                return self._json(self._test_profile(body))
            if route == "/api/settings/models":
                return self._json(self.app.model_list(body))
            if route == "/api/sessions/preview":
                planned = sm.plan(body, preview=True)
                return self._json(dict(sessions=[dict(id=s.id, command=s.command, key=s.key,
                                                      condition=s.condition["id"]) for s in planned]))
            if route == "/api/sessions/launch":
                return self._json(dict(sessions=sm.launch(body)))
            if route == "/api/tasks/instruction":
                tid = str(body.get("task") or "")
                return self._json(dict(task=tid, fixed_instruction=sm.save_fixed_instruction(
                    tid, str(body.get("text") or ""), str(body.get("base") or ""))))
            if route == "/api/sessions/stop":
                return self._json(dict(session=sm.stop(str(body.get("id")), str(body.get("how")))))
            if route == "/api/bridge/manual":
                sm.manual(str(body.get("target")), int(body.get("turn")), str(body.get("text") or ""))
                return self._json(dict(ok=True))
            if route == "/api/bridge/retry":
                sm.retry(str(body.get("target")))
                return self._json(dict(ok=True))
            if route == "/api/bridge/finish":
                sm.finish_attached(str(body.get("target")))
                return self._json(dict(ok=True))
            if route == "/api/attach":
                return self._json(sm.attach(str(body.get("key")), body.get("model") or {},
                                            str(body.get("instruction") or "")))
            if route == "/api/detach":
                sm.detach(str(body.get("key")))
                return self._json(dict(ok=True))
            if route == "/api/label":
                key = str(body.get("key") or "")
                if not app.store.ep_dir(key):
                    return self._error(404, "Episode not found")
                return self._json(dict(key=key, label=app.labels.set(key, body.get("success"), str(body.get("note") or ""))))
            if route == "/api/score":
                return self._json(dict(output=sm.score(str(body.get("key")), str(body.get("value")))))
            return self._error(404, "not found")
        except (LaunchError, SettingsError, ValueError, TypeError) as exc:
            return self._error(400, str(exc))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            return self._error(500, f"{type(exc).__name__}: {exc}")

    # -- endpoints -------------------------------------------------------------
    def _test_profile(self, body: dict) -> dict:
        settings = self.app.settings
        profile = settings.get(str(body.get("id") or ""))
        if profile is None:
            raise LaunchError("Save this profile before testing it")
        key = settings.key_for(profile)
        turns = [dict(role="user", parts=[dict(type="text", text="Connectivity test: reply with OK only. Reply with the single word OK.")])]
        probe = dict(profile, max_tokens=max(64, min(int(profile.get("max_tokens") or 256), 1024)))
        try:
            result = llm.call(probe, key, "You are a connectivity check.", turns, timeout=90)
        except llm.LLMError as exc:
            return dict(ok=False, error=str(exc))
        return dict(ok=True, text=(result.get("text") or "")[:300], latency_s=result.get("latency_s"),
                    usage_in=result.get("usage_in"), usage_out=result.get("usage_out"),
                    thinking=bool(result.get("thinking")))

    def _index(self):
        html = (WEB / "index.html").read_text(encoding="utf-8")
        html = html.replace("{{TOKEN}}", self.app.token).replace("{{VERSION}}", __version__)
        self._send(200, html.encode("utf-8"), "text/html; charset=utf-8",
                   {"Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                                               "script-src 'self'; connect-src 'self'; frame-ancestors 'none'",
                    "Referrer-Policy": "no-referrer"})

    def _static(self, rel: str):
        path = (WEB / rel).resolve()
        if not is_within(path, WEB) or not path.is_file() or path.suffix not in STATIC_TYPES:
            return self._error(404, "not found")
        self._send(200, path.read_bytes(), STATIC_TYPES[path.suffix])

    def _file(self, token: str):
        path = self.app.store.paths.resolve(token)
        if path is None or not path.is_file() or path.suffix.lower() not in FILE_TYPES:
            return self._error(404, "not found")
        ctype = FILE_TYPES[path.suffix.lower()]
        if ctype.startswith("video/"):
            return self._send_ranged(path, ctype)
        if ctype.startswith("text/") or ctype.startswith("application/json"):
            if path.stat().st_size > TEXT_LIMIT:
                return self._error(413, "File too large to show in the page")
        self._send(200, path.read_bytes(), ctype, {"Cache-Control": "private, max-age=60"})

    def _send_ranged(self, path: Path, ctype: str):
        """Videos stream from disk and honour one byte range, which the browser needs to seek."""
        size = path.stat().st_size
        start, end, status = 0, size - 1, 200
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", (self.headers.get("Range") or "").strip())
        if m and (m[1] or m[2]):
            if m[1]:
                start, end = int(m[1]), (min(int(m[2]), size - 1) if m[2] else size - 1)
            else:
                start = max(0, size - int(m[2]))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "private, max-age=60")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command == "HEAD":
            return
        with path.open("rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def _stream(self, key: str):
        if not key or self.app.store.paths.resolve(key) is None:
            return self._error(400, "Invalid key")
        hub = self.app.hub
        q = hub.subscribe(key)
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()

            def send(event: dict):
                data = json.dumps(event, ensure_ascii=False, default=str)
                self.wfile.write(f"data: {data}\n\n".encode("utf-8"))

            for ev in self.app.stream_events(key):
                send(ev)
            self.wfile.flush()
            last_ping = time.time()
            while True:
                try:
                    ev = q.get(timeout=2.0)
                    send(ev)
                    # drain a burst (token deltas, force batches) before flushing
                    for _ in range(200):
                        try:
                            send(q.get_nowait())
                        except queue.Empty:
                            break
                    self.wfile.flush()
                except queue.Empty:
                    pass
                if time.time() - last_ping > 5:
                    send(dict(type="ping", t=time.time()))
                    self.wfile.flush()
                    last_ping = time.time()
                if getattr(self.server, "closing", False):
                    return
        except (BrokenPipeError, ConnectionResetError, OSError):
            return
        finally:
            hub.unsubscribe(key, q)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, app: App):
        self.app = app
        self.closing = False
        super().__init__((app.cfg.host, app.cfg.port), Handler)
