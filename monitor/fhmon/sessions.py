"""Episodes started from the monitor: one harness runner subprocess plus, when a model is
needed, one relay bridge.

Folder per session (default <runs>/monitor/<sid>/):
  session.json   what was asked: task, condition, mode, model label, instruction, command
  episode/       the runner's --out (the runner creates it and refuses to reuse a folder)
  relay/         the runner's --relay-dir (likewise)
  bridge.jsonl   one line per model turn: reply, thinking, latency, tokens, errors
  runner.log     the runner's stdout and stderr
Live robot sessions need `--allow-live` on the server plus the operator checklist; the
runner script still enforces every check and guard of its own.
Sessions whose mode the video config lists (monitor/video.json) also get a global-camera video
(fhmon/video.py), kept with all the others in one folder (default runs/videos/<sid>.mp4).
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from . import llm
from .bridge import BRIDGE_SYSTEM, Bridge
from .util import now_iso, read_json, slug, write_json

TERMINAL = ("ended", "failed", "stopped", "cancelled")


class LaunchError(ValueError):
    pass


class Session:
    def __init__(self, **kw):
        self.proc: Optional[subprocess.Popen] = None
        self.bridge: Optional[Bridge] = None
        self.next: Optional["Session"] = None
        self.stopped_by: Optional[str] = None
        self.interrupt_at: Optional[float] = None
        self.exit_code: Optional[int] = None
        self.started: Optional[str] = None
        self.ended: Optional[str] = None
        self.score_note: Optional[str] = None
        self.video: Optional[str] = None
        self.resume_held: Optional[str] = None
        self.__dict__.update(kw)

    def public(self) -> dict:
        return dict(
            kind="fhmon_session", id=self.id, task=self.task["id"], task_name=self.task["name"],
            condition=self.condition["id"], condition_label=self.condition["label"], arm=self.condition["arm"],
            mode=self.mode, model=self.model_public, instruction=self.instruction, resume_held=self.resume_held, name=self.id,
            pair_id=self.pair_id, pair_role=self.pair_role, command=self.command, key=self.key,
            episode_dir=str(self.episode_dir), relay_dir=str(self.relay_dir), status=self.status,
            exit_code=self.exit_code, pid=self.proc.pid if self.proc else None, started=self.started,
            ended=self.ended, stopped_by=self.stopped_by, checklist=self.checklist, inputs=self.inputs,
            log_url=self.log_url, video=self.video,
            bridge=self.bridge.state if self.bridge else None, score_note=self.score_note,
            bridge_system=BRIDGE_SYSTEM if self.model_public.get("source") == "api" else None)


class SessionManager:
    def __init__(self, cfg, registry: Dict[str, dict], settings, hub, store, videos=None):
        self.cfg, self.registry, self.settings, self.hub, self.store = cfg, registry, settings, hub, store
        self.videos = videos
        self.sessions: Dict[str, Session] = {}
        self.attached: Dict[str, Bridge] = {}
        self.lock = threading.Lock()

    # -- registry helpers --------------------------------------------------------
    def task(self, tid: str) -> dict:
        if tid not in self.registry:
            raise LaunchError(f"Unknown task: {tid}")
        return self.registry[tid]

    def fixed_instruction(self, task: dict) -> Optional[str]:
        """A fixed task's goal is the prompt file its runner reads at start; read it fresh, the page may edit it.
        None when that file is missing."""
        if task.get("fixed_instruction_path"):
            path = Path(task["fixed_instruction_path"])
            task["fixed_instruction"] = path.read_text(encoding="utf-8").strip() if path.is_file() else None
        return task.get("fixed_instruction")

    def save_fixed_instruction(self, tid: str, text: str, base: str) -> str:
        """Write the edited goal back to the prompt file: the next runner start reads it, A and B alike."""
        task = self.task(tid)
        if not task.get("fixed_instruction_path"):
            raise LaunchError("This task has no fixed task description")
        text = text.strip()
        if not text:
            raise LaunchError("The task description cannot be empty")
        if len(text) > 4000:
            raise LaunchError("The task description can be at most 4000 characters (the runner's limit)")
        with self.lock:
            current = self.fixed_instruction(task)
            if current is None:
                raise LaunchError(f"{task['fixed_instruction_file']} is missing")
            if current != base.strip():
                raise LaunchError(f"{task['fixed_instruction_file']} was changed elsewhere: reload the page to see the latest text, then edit it")
            if any(s.task["id"] == tid and s.status in ("queued", "starting") for s in self.sessions.values()):
                raise LaunchError("An episode of this task is still queued (the second episode of a pair reads the task description when it starts); save after it has started")
            if text != current:
                path = Path(task["fixed_instruction_path"])
                tmp = path.with_name(path.name + ".tmp")
                tmp.write_text(text + "\n", encoding="utf-8")
                os.replace(tmp, path)
                task["fixed_instruction"] = text
        return text

    def live_inputs(self, tid: str) -> Dict[str, List[dict]]:
        live = (self.task(tid).get("live") or {}).get("inputs") or {}
        root = self.cfg.harness_root
        out = {}
        for name, spec in live.items():
            seen, files = set(), []
            for pattern in spec.get("glob") or []:
                # "runs/..." means the run folder in use, which --runs may put elsewhere
                base, pat = (self.cfg.runs_root, pattern[5:]) if pattern.startswith("runs/") else (root, pattern)
                for p in base.glob(pat):
                    try:
                        rel = p.relative_to(root).as_posix()
                    except ValueError:
                        rel = str(p)
                    if p.is_file() and rel not in seen:
                        seen.add(rel)
                        files.append(dict(path=rel, mtime=p.stat().st_mtime))
            files.sort(key=lambda f: f["mtime"], reverse=True)
            out[name] = dict(label=spec.get("label", name), default=spec.get("default"),
                             required=spec.get("required", False), files=files)
        return out

    # -- planning ----------------------------------------------------------------
    def _model(self, task: dict, mode: str, req: dict, external_ok: bool = True):
        source = (req or {}).get("source") or "mock"
        if source == "scripted":
            if mode == "demo" and (task.get("demo") or {}).get("scripted"):
                return dict(source="scripted", label="Scripted demo (no model)"), None
            raise LaunchError("Scripted demo is only available in Demo (no robot) mode")
        if source == "api":
            profile = self.settings.get((req or {}).get("profile") or "")
            if not profile:
                raise LaunchError("API profile not found")
            if not profile.get("model"):
                raise LaunchError(f'Profile "{profile["name"]}" has no model name yet')
            key = self.settings.key_for(profile)
            if key is None or (not key and profile.get("provider") not in llm.KEYLESS):
                raise LaunchError(f'Profile "{profile["name"]}" has no API key yet (enter it in Settings or set the environment variable {profile.get("key_env")})')
            label = profile["model"]+(" (Codex subscription)" if profile.get("provider") == "codex-cli" else "")
            public = dict(source="api", label=label, profile_id=profile["id"], profile_name=profile["name"],
                          provider=profile["provider"], model=profile["model"], base_url=profile["base_url"],
                          image_turns=profile.get("image_turns", 2))
            return public, dict(public, profile=profile, key=key)
        if source == "mock":
            if mode == "live":
                raise LaunchError("The real robot cannot use the mock model")
            public = dict(source="mock", label="Mock model")
            return public, public
        if source == "manual":
            public = dict(source="manual", label="Manual (operator gives actions)")
            return public, public
        if source == "external" and external_ok:
            return dict(source="external", label="External relay"), None
        raise LaunchError("Unknown model source")

    def plan(self, req: dict, preview: bool = False) -> List[Session]:
        """Validate a launch request and build its sessions; preview skips only the operator checklist."""
        task = self.task(req.get("task") or "")
        mode = req.get("mode") or "demo"
        if mode not in ("demo", "live"):
            raise LaunchError("mode must be demo or live")
        cond_ids = req.get("conditions") or []
        conds = [c for cid in cond_ids for c in task["conditions"] if c["id"] == cid]
        if not conds or len(conds) != len(cond_ids) or len(conds) > 2:
            raise LaunchError("Please choose one condition, or a pair of two conditions")
        public_model, full_model = self._model(task, mode, req.get("model") or {})
        instruction = str(req.get("instruction") or "").strip()[:4000]
        task_text = task.get("instruction")      # free task: the instruction IS the task, the runner gets it
        fixed = self.fixed_instruction(task)
        if task.get("fixed_instruction_file") and not fixed:
            raise LaunchError(f"{task['fixed_instruction_file']} is missing or empty: the runner reads this task's goal from it")
        if fixed:
            if instruction and instruction != fixed:
                raise LaunchError("The task description is fixed in the task file; to change it, edit and save it in the \"Fixed task\" box above")
            instruction = fixed
        elif task_text:
            if task_text.get("required") and not instruction:
                raise LaunchError("Please write the task for this episode (the instruction sent to the model)")
        elif instruction and public_model["source"] not in ("api", "manual", "mock"):
            raise LaunchError("Extra instruction can only be delivered through the Monitor's model relay; the model of this episode does not go through the Monitor")
        # the operator names what the still-closed gripper holds; the runner then starts holding it
        resume_held = str(req.get("resume_held") or "").strip()
        if resume_held:
            if mode != "live" or not (task.get("live") or {}).get("resume_held"):
                raise LaunchError("\"Resume from the last episode\" is only for real-robot episodes of this task")
            if len(resume_held) > 40 or "\n" in resume_held or "\r" in resume_held:
                raise LaunchError("Keep the held object short: one line, at most 40 characters")
        inputs, checklist = {}, None
        if mode == "live":
            live = task.get("live")
            if not self.cfg.allow_live:
                raise LaunchError("The server was not started with --allow-live, so real-robot mode is off")
            if not live:
                raise LaunchError("This task cannot start the real robot from the Monitor: " + ((task.get("live_external") or {}).get("note") or ""))
            if len(conds) != 1:
                raise LaunchError("Run one real-robot episode at a time: between episodes the scene must be reset and re-measured by hand")
            checklist = req.get("checklist") or []
            if not preview and (len(checklist) != len(live.get("checklist") or []) or not all(checklist)):
                raise LaunchError("Please tick every item on the real-robot checklist")
            choices = self.live_inputs(task["id"])
            for name, spec in choices.items():
                value = (req.get("inputs") or {}).get(name) or spec.get("default")
                if spec.get("required") and not value:
                    raise LaunchError(f"Missing {spec['label']}")
                if value and value not in {f["path"] for f in spec["files"]}:
                    raise LaunchError(f"{spec['label']} must be one of the listed files")
                inputs[name] = value
            with self.lock:
                if any(s.mode == "live" and s.status not in TERMINAL for s in self.sessions.values()):
                    raise LaunchError("A real-robot episode is already running")
        base = slug(req.get("name") or "") or f"{datetime.now():%Y%m%d-%H%M%S}_{task['id']}_{mode}"
        pair_id = base if len(conds) == 2 else None
        out: List[Session] = []
        for i, cond in enumerate(conds):
            sid = f"{base}_{cond['id']}" if pair_id else base
            sdir = self.cfg.data_dir / sid
            if sdir.exists() or sid in self.sessions:
                raise LaunchError(f"An episode with this name already exists: {sid}")
            # both halves of a pair share one synthetic scene (the same hidden assignment)
            pair_fixture = ((sdir if i == 0 else out[0].dir) / "pair_fixture.json") if pair_id else None
            command = self._command(task, cond, mode, sdir, inputs, pair_fixture,
                                    relay_model=public_model["source"] != "scripted", instruction=instruction,
                                    resume_held=resume_held)
            s = Session(id=sid, task=task, condition=cond, mode=mode, model_public=public_model, model_full=full_model,
                        instruction=instruction, resume_held=resume_held or None,
                        dir=sdir, episode_dir=sdir / "episode", relay_dir=sdir / "relay",
                        command=command, pair_id=pair_id, pair_role=("first" if i == 0 else "second") if pair_id else None,
                        checklist=(task.get("live") or {}).get("checklist") if mode == "live" else None,
                        inputs=inputs, pair_fixture=pair_fixture, status="queued",
                        key=self.store.key_of(sdir / "episode"), log_url=self.store.paths.url(sdir / "runner.log"))
            out.append(s)
        return out

    def _command(self, task, cond, mode, sdir: Path, inputs: dict, pair_fixture: Optional[Path],
                 relay_model: bool = True, instruction: str = "", resume_held: str = "") -> List[str]:
        ctx = dict(episode_dir=str(sdir / "episode"), relay_dir=str(sdir / "relay"), session_dir=str(sdir),
                   instruction=instruction, resume_held=resume_held,
                   pair_fixture=str(pair_fixture or ""), condition=cond["id"],
                   plan=str(self.cfg.harness_root / inputs["plan"]) if inputs.get("plan") else "",
                   calibration=str(self.cfg.harness_root / inputs["calibration"]) if inputs.get("calibration") else "")
        block = task.get("demo") if mode == "demo" else task.get("live")
        args = list(cond.get("args") or []) + list(block.get("args") or [])
        for name, extra in (block.get("optional_args") or {}).items():
            if inputs.get(name):                   # e.g. --plan only when a file was chosen
                args += extra
        if relay_model and block.get("relay_args"):
            args += block["relay_args"]            # demo: let the relay model, not the scripted demo, decide
        if pair_fixture and block.get("pair_args"):
            args += block["pair_args"]
        if task.get("instruction") and instruction:
            args += task["instruction"]["args"]      # e.g. --instruction "{instruction}" (one argv item, no shell)
        if resume_held and block.get("resume_held"):
            args += block["resume_held"]["args"]     # --resume-held={resume_held}, likewise one argv item
        args += task.get("output_args") or []
        return [str(self.cfg.python), str(self.cfg.harness_root / task["script"])] + [a.format(**ctx) for a in args]

    # -- running -----------------------------------------------------------------
    def launch(self, req: dict) -> List[dict]:
        planned = self.plan(req)
        with self.lock:
            for s in planned:
                self.sessions[s.id] = s
        for a, b in zip(planned, planned[1:]):
            a.next = b
        self._start(planned[0])
        return [s.public() for s in planned]

    def _env(self) -> dict:
        env = dict(os.environ)
        for k in list(env):
            if k.endswith("_API_KEY"):
                env.pop(k)             # relay runners never call an API themselves
        root = str(self.cfg.harness_root)
        env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("MPLCONFIGDIR", str(self.cfg.harness_root / ".mpl"))
        return env

    def _save(self, s: Session) -> None:
        if s.dir.exists():
            write_json(s.dir / "session.json", s.public())
        self.hub.publish(s.key, dict(type="session", session=s.public()))

    def _start(self, s: Session) -> None:
        s.dir.mkdir(parents=True, exist_ok=False)
        s.started, s.status = now_iso(), "starting"
        self._save(s)
        log = open(s.dir / "runner.log", "ab")
        try:
            s.proc = subprocess.Popen(s.command, cwd=self.cfg.harness_root, env=self._env(), stdin=subprocess.DEVNULL,
                                      stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as exc:
            log.close()
            s.status, s.ended = "failed", now_iso()
            s.score_note = f"Could not start: {exc}"
            self._save(s)
            return
        if s.model_full and s.model_public["source"] in ("api", "mock", "manual"):
            s.bridge = Bridge(s.relay_dir, s.model_full, publish=lambda ev, k=s.key: self.hub.publish(k, ev),
                              log_path=s.dir / "bridge.jsonl",
                              instruction="" if s.task.get("instruction") or s.task.get("fixed_instruction") else s.instruction,
                              graceful_done=s.task.get("graceful_done"),
                              image_turns=s.model_public.get("image_turns", 2)).start()
        self._record(s)
        s.status = "running"
        self._save(s)
        threading.Thread(target=self._wait, args=(s, log), name=f"session:{s.id}", daemon=True).start()

    def _record(self, s: Session) -> None:
        """Global-camera video; whatever happens to the camera, the runner is already running."""
        if not (self.videos and self.videos.wants(s.mode)):
            return
        try:
            s.video = self.videos.start(s.id, dict(
                session=s.id, task=s.task["id"], task_name=s.task["name"], condition=s.condition["id"],
                condition_label=s.condition["label"], arm=s.condition["arm"], mode=s.mode, started=s.started,
                episode_dir=str(s.episode_dir.resolve()), session_dir=str(s.dir.resolve())))
        except Exception as exc:
            print(f"video for {s.id} not started: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    def _wait(self, s: Session, log) -> None:
        code = s.proc.wait()
        log.close()
        if self.videos:
            self.videos.stop(s.video)
        s.exit_code, s.ended = code, now_iso()
        wrote_episode = (s.episode_dir / "episode.json").exists()
        if s.stopped_by and s.stopped_by != "operator_finish":
            s.status = "stopped"
        elif code in (0, 2) and wrote_episode:
            s.status = "ended"         # the runners exit 2 for a completed-but-failed episode
        else:
            s.status = "failed"        # e.g. argparse refused the command (also exit 2) or a crash
        if s.bridge:
            time.sleep(1.0)
            s.bridge.stop()
        if s.mode == "demo" and (s.task.get("demo") or {}).get("autoscore_from_fixture") and s.status == "ended":
            self._autoscore(s)
        self._save(s)
        nxt = s.next
        if nxt is not None and nxt.status == "queued":
            if s.stopped_by:
                nxt.status = "cancelled"
                self._save(nxt)
            else:
                self._start(nxt)

    def _autoscore(self, s: Session) -> None:
        """Demo only: the synthetic plant's assignment is the truth; score with the runner's own subcommand."""
        fixture = s.pair_fixture or (s.dir / "episode_fixture.json")
        data = read_json(fixture) or {}
        truth = data.get("assignment")
        if not truth:
            s.score_note = "The demo's ground-truth file does not exist, so the episode was not scored automatically"
            return
        try:
            s.score_note = self._run_score(s.task, s.episode_dir, truth)
        except LaunchError as exc:
            s.score_note = f"Automatic scoring failed: {exc}"

    def stop(self, sid: str, how: str) -> dict:
        s = self.sessions.get(sid)
        if s is None:
            raise LaunchError("No such episode")
        if s.next is not None and s.next.status == "queued":
            s.next.status = "cancelled"
            self._save(s.next)
        if s.status in TERMINAL or s.proc is None:
            if s.status == "queued":
                s.status = "cancelled"
                self._save(s)
            return s.public()
        if how == "finish":
            if not s.bridge:
                raise LaunchError("The model of this episode does not go through the Monitor, so \"Finish next turn\" is not available; use \"Interrupt\"")
            s.bridge.request_finish()
            s.stopped_by = "operator_finish"
        elif how == "interrupt":
            # Only the runner itself: its children (the link to the robot PC, the camera process) must stay up
            # while it stops the controller through them. A terminal Ctrl+C would hit them all.
            s.stopped_by, s.interrupt_at = "operator_interrupt", time.time()
            os.kill(s.proc.pid, signal.SIGINT)
        elif how == "kill":
            if not s.interrupt_at or time.time() - s.interrupt_at < 8:
                raise LaunchError("Use \"Interrupt\" first; if the runner has not exited after 8 s, use \"Force quit\"")
            s.stopped_by = "operator_kill"
            os.killpg(s.proc.pid, signal.SIGKILL)
        else:
            raise LaunchError("Unknown stop method")
        self._save(s)
        return s.public()

    def bridge_for(self, sid_or_key: str) -> Bridge:
        s = self.sessions.get(sid_or_key)
        bridge = s.bridge if s else self.attached.get(sid_or_key)
        if bridge is None or not bridge.alive():
            raise LaunchError("This episode has no Monitor model relay running")
        return bridge

    def manual(self, sid_or_key: str, turn: int, text: str) -> None:
        err = self.bridge_for(sid_or_key).submit_manual(int(turn), text)
        if err:
            raise LaunchError(err)

    def retry(self, sid_or_key: str) -> None:
        self.bridge_for(sid_or_key).retry()

    def finish_attached(self, key: str) -> None:
        self.bridge_for(key).request_finish()

    def list(self) -> List[dict]:
        with self.lock:
            return [s.public() for s in sorted(self.sessions.values(), key=lambda s: s.started or "", reverse=True)]

    # -- attaching to a runner started elsewhere -----------------------------------
    def attach(self, key: str, model_req: dict, instruction: str = "") -> dict:
        ep = self.store.load(key, with_text=False)
        if not ep:
            raise LaunchError("Episode not found")
        if ep["status"] in ("ended", "stale"):
            raise LaunchError("This episode has already ended")
        relay = self.store.paths.resolve(ep["relay"]["dir"]) if (ep.get("relay") or {}).get("dir") else None
        if relay is None or not relay.is_dir():
            raise LaunchError("This episode has no relay folder (the runner has not started yet, or it is not in relay mode)")
        if key in self.attached and self.attached[key].alive():
            raise LaunchError("This episode has already been taken over")
        task = next((t for t in self.registry.values() if t["id"] == ep["family"]), None)
        if task is None:
            raise LaunchError("Unknown task")
        public, full = self._model(task, ep.get("mode") or "live", model_req, external_ok=False)
        log = self.cfg.data_dir / "attach" / f"{slug(key.replace(':', '_').replace('/', '__'), 120)}.bridge.jsonl"
        log.parent.mkdir(parents=True, exist_ok=True)
        bridge = Bridge(relay, full, publish=lambda ev, k=key: self.hub.publish(k, ev), log_path=log,
                        instruction=instruction, graceful_done=task.get("graceful_done"),
                        image_turns=public.get("image_turns", 2)).start()
        self.attached[key] = bridge
        write_json(log.with_suffix(".meta.json"), dict(key=key, model=public, instruction=instruction,
                                                       started=now_iso(), episode=ep.get("name")))
        return dict(key=key, model=public, state=bridge.state)

    def detach(self, key: str) -> None:
        bridge = self.attached.pop(key, None)
        if bridge:
            bridge.stop()

    def attach_log(self, key: str) -> Optional[Path]:
        log = self.cfg.data_dir / "attach" / f"{slug(key.replace(':', '_').replace('/', '__'), 120)}.bridge.jsonl"
        return log if log.exists() else None

    # -- scoring (operator verdict, written by the runner's own score subcommand) ---------
    def score(self, key: str, value: str) -> str:
        ep = self.store.load(key, with_text=False)
        if not ep:
            raise LaunchError("Episode not found")
        task = next((t for t in self.registry.values() if t["id"] == ep["family"]), None)
        if not task or not task.get("score"):
            raise LaunchError("This task has no score command")
        if ep["status"] != "ended":
            raise LaunchError("This episode has not ended yet")
        if (ep.get("result") or {}).get("scored"):
            raise LaunchError("Already scored (the harness's score command does not allow overwriting)")
        return self._run_score(task, self.store.ep_dir(key), value)

    def _run_score(self, task: dict, ep_dir: Path, value: str) -> str:
        sc = task["score"]
        if value not in {o["value"] for o in sc["options"]}:
            raise LaunchError("Invalid score option")
        args = [a.format(episode_dir=str(ep_dir), outcome=value, truth=value) for a in sc["args"]]
        cmd = [str(self.cfg.python), str(self.cfg.harness_root / task["script"])] + args
        try:
            proc = subprocess.run(cmd, cwd=self.cfg.harness_root, env=self._env(), capture_output=True, text=True,
                                  timeout=120, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise LaunchError("The score command timed out") from None
        if proc.returncode != 0:
            raise LaunchError((proc.stderr or proc.stdout).strip()[-600:] or f"exit {proc.returncode}")
        return proc.stdout.strip()[-600:]

    # -- shutdown ----------------------------------------------------------------
    def shutdown(self, timeout: float = 15.0) -> None:
        running = [s for s in self.sessions.values() if s.proc is not None and s.proc.poll() is None]
        for s in running:
            s.stopped_by = "monitor_shutdown"
            try:
                os.kill(s.proc.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.time() + timeout
        for s in running:
            try:
                s.proc.wait(max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                continue
            # the waiter thread dies with the process; record the end here
            s.exit_code, s.ended, s.status = s.proc.returncode, now_iso(), "stopped"
            self._save(s)
        for s in self.sessions.values():
            if s.bridge:
                s.bridge.stop()
        for b in self.attached.values():
            b.stop()
        if self.videos:
            self.videos.shutdown()
