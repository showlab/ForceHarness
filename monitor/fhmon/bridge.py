"""Relay bridge: plays the model side of the harness relay protocol.

A real-robot runner started with --relay-dir writes obs_<n>.md (+ obs_<n>_*.png) and pending.json
{"turn": n, "waiting": true}, then waits for cmd_<n>.json {"calls": [...], "note": "..."}. Anything
that writes that file can be the model: a person, another agent, or this bridge, which sends the
observation text and images to a model and writes its answer back.

It reads only the relay directory, the runner's public side, so anything kept in the
private episode directory (truth, fixtures, raw logs) never reaches the model.
Sources: "api" (a settings profile), "mock" (offline plumbing check, no network),
"manual" (the operator types the JSON in the browser).
"""
from __future__ import annotations

import base64
import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

from . import llm
from .receipts import DONE_TAKES_SUCCESS, OBS_MOVE_HEAD
from .util import append_jsonl, read_json, read_text

BRIDGE_SYSTEM = (
    "You are the decision model in a robot experiment and interact with the robot through a file relay. Each user "
    "message is the full text of one observation file obs_<n>.md; the attached images are the scene images the "
    "observation mentions. Reply strictly as the \"YOUR MOVE\" section at the end of the observation asks, and output "
    "only the JSON it asks for; you may briefly give the reason for your decision in the JSON field note.\n"
    "You are the decision model of a robot experiment, connected through a file relay. Each user message is the full "
    "text of one observation file; the attached images are the files it mentions. Answer exactly as its YOUR MOVE "
    "section asks (JSON only); you may add a short reason in the JSON field \"note\"."
)
INSTRUCTION_HEAD = "[additional operator instruction]"
FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)
# A 429 rate limit (shared ChatGPT plan, API RPM/TPM) usually clears within a minute; a used-up plan or
# API quota does not come back during an episode. The runner keeps the arm holding while the bridge waits.
RATE_LIMIT_WAITS_S = (20, 40, 80)
# The last alternative is codex_cli's stall message: a codex that printed nothing for minutes is retried like a 429.
RATE_LIMIT_RE = re.compile(r"\b429\b|too many requests|rate.?limit|exceeded retry limit|produced no output for", re.I)
QUOTA_RE = re.compile(r"usage.?limit|insufficient_quota|exceeded your current quota", re.I)


def retry_wait(message: str, attempts: int) -> Optional[float]:
    """Seconds to wait before the next model attempt, or None to give up on this turn."""
    if QUOTA_RE.search(message):
        return None
    if RATE_LIMIT_RE.search(message):
        return RATE_LIMIT_WAITS_S[attempts - 1] if attempts <= len(RATE_LIMIT_WAITS_S) else None
    return 3 * attempts if attempts < 3 else None


def _objects(text: str):
    """Top-level JSON objects in free text as (start, end, obj); nested ones are skipped."""
    decoder, out, i = json.JSONDecoder(), [], 0
    while True:
        i = text.find("{", i)
        if i < 0:
            return out
        try:
            obj, j = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            i += 1
            continue
        out.append((i, j, obj))
        i = j


def extract_action(text: str):
    """(cmd dict, leftover text) from a model reply, or (None, None). Takes the last action
    object: a {"calls": [...]} or a single {"name", "args"} call, fenced or bare."""
    text = text or ""
    for m in reversed(list(FENCE_RE.finditer(text))):
        found = [(i, j, normalize(obj)) for i, j, obj in _objects(m.group(1))]
        found = [f for f in found if f[2]]
        if found:
            return found[-1][2], (text[:m.start()] + text[m.end():]).strip()
    found = [(i, j, normalize(obj)) for i, j, obj in _objects(text)]
    found = [f for f in found if f[2]]
    if not found:
        return None, None
    i, j, cmd = found[-1]
    return cmd, (text[:i] + text[j:]).strip()


def normalize(obj) -> Optional[dict]:
    if not isinstance(obj, dict):
        return None
    if isinstance(obj.get("calls"), list):
        raw_calls = obj["calls"]
    elif isinstance(obj.get("name"), str):
        raw_calls = [obj]
    else:
        return None
    calls = [dict(name=c["name"], args=c.get("args") if isinstance(c.get("args"), dict) else {})
             for c in raw_calls if isinstance(c, dict) and isinstance(c.get("name"), str)]
    if not calls:
        return None
    note = obj.get("note")
    return dict(calls=calls, note=note if isinstance(note, str) else None)


def example_action(obs_text: str) -> Optional[dict]:
    """The sample JSON every relay obs ends with ("JSON: {"calls":[...]}")."""
    tail = (obs_text or "").rsplit(OBS_MOVE_HEAD, 1)[-1]
    found = [normalize(obj) for _, _, obj in _objects(tail) if isinstance(obj, dict) and "calls" in obj]
    found = [f for f in found if f]
    return found[-1] if found else None


class Bridge:
    """One thread per relay directory. publish(event) goes to the episode's live stream."""

    def __init__(self, relay_dir: Path, spec: dict, publish: Callable[[dict], None], log_path: Path,
                 instruction: str = "", graceful_done: Optional[dict] = None, image_turns: int = 2):
        self.relay = Path(relay_dir)
        self.spec = spec                      # {source, label, profile?, key?}
        self.publish = publish
        self.log_path = Path(log_path)
        self.instruction = (instruction or "").strip()
        self.graceful_done = graceful_done or {"name": "done", "args": {}}
        self.image_turns = max(0, int(image_turns))
        self.replies: Dict[int, str] = {}     # turn -> raw assistant text (ours)
        self.failed: Dict[int, str] = {}
        self.manual: Dict[int, str] = {}
        self.finish_requested = False
        self.state = dict(status="waiting", turn=None, error=None, source=spec.get("source"), label=spec.get("label"))
        self._stop = threading.Event()
        self._cancel = threading.Event()
        self._wake = threading.Event()
        self._img_cache: Dict[str, str] = {}
        self.thread = threading.Thread(target=self._run, name=f"bridge:{self.relay.name}", daemon=True)

    # -- control ------------------------------------------------------------
    def start(self):
        self.thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._cancel.set()
        self._wake.set()

    def request_finish(self):
        self.finish_requested = True
        self._cancel.set()      # stop a reply in progress; the turn is answered with done instead
        self._wake.set()

    def retry(self):
        self.failed.clear()
        self._wake.set()

    def submit_manual(self, turn: int, text: str) -> Optional[str]:
        cmd, _ = extract_action(text)
        if cmd is None:
            return "Could not parse: expected JSON like {\"calls\": [{\"name\": ..., \"args\": {...}}]}"
        self.manual[turn] = json.dumps(cmd, ensure_ascii=False)
        self._wake.set()
        return None

    def alive(self) -> bool:
        return self.thread.is_alive()

    # -- loop ----------------------------------------------------------------
    def _set(self, **kw):
        self.state.update(kw)
        self.publish(dict(type="model", kind="state", **self.state))

    def _run(self):
        self._set(status="waiting")
        while not self._stop.is_set():
            if (self.relay / "done.json").exists():
                self._set(status="finished", turn=None)
                return
            pending = read_json(self.relay / "pending.json") if self.relay.is_dir() else None
            turn = pending.get("turn") if isinstance(pending, dict) else None
            if (isinstance(turn, int) and pending.get("waiting") and turn not in self.replies
                    and not (self.relay / f"cmd_{turn}.json").exists()
                    and (turn not in self.failed or self.finish_requested)):
                try:
                    self._answer(turn)
                except Exception as exc:  # never let one turn kill the bridge
                    self.failed[turn] = f"{type(exc).__name__}: {exc}"
                    self._set(status="error", turn=turn, error=self.failed[turn])
            self._wake.wait(0.25)
            self._wake.clear()
        self._set(status="stopped", turn=None)

    # -- one turn ------------------------------------------------------------
    def _answer(self, turn: int):
        obs_path = self.relay / f"obs_{turn}.md"
        deadline = time.time() + 5
        while not obs_path.exists() and time.time() < deadline and not self._stop.is_set():
            time.sleep(0.1)
        obs_text = read_text(obs_path) or ""
        record = dict(turn=turn, t=time.time(), source=self.spec.get("source"), label=self.spec.get("label"),
                      obs_chars=len(obs_text), images=[p.name for p in self._images(turn)],
                      instruction_included=bool(self.instruction) and turn == 0)
        if self.finish_requested:
            cmd = dict(calls=[self.graceful_done], note="[monitor] The operator ended this episode (not the model's decision)")
            self._write(turn, cmd, record, text=json.dumps(cmd, ensure_ascii=False), operator=True)
            return
        source = self.spec.get("source")
        if source == "manual":
            self._answer_manual(turn, obs_text, record)
        elif source == "mock":
            self._answer_mock(turn, obs_text, record)
        else:
            self._answer_api(turn, obs_text, record)

    def _answer_manual(self, turn, obs_text, record):
        example = example_action(obs_text)
        example = json.dumps({"calls": example["calls"]}, ensure_ascii=False) if example else '{"calls": []}'
        # the example rides in the state too, so a browser that connects later still gets the editor
        self._set(status="manual_wait", turn=turn, error=None, example=example)
        self.publish(dict(type="model", kind="manual_wait", turn=turn, example=example))
        while turn not in self.manual and not self._stop.is_set() and not self.finish_requested:
            if (self.relay / f"cmd_{turn}.json").exists():
                return
            self._wake.wait(0.25)
            self._wake.clear()
        if self.finish_requested and turn not in self.manual:
            return self._answer(turn)
        if self._stop.is_set():
            return
        text = self.manual.pop(turn)
        cmd, _ = extract_action(text)
        self._write(turn, cmd, record, text=text)

    def _answer_mock(self, turn, obs_text, record):
        """Offline plumbing check: the obs example action twice, then done."""
        self._set(status="thinking", turn=turn, error=None)
        first = read_text(self.relay / "obs_0.md") or ""
        if turn < 2 and example_action(obs_text):
            cmd = example_action(obs_text)
            cmd["note"] = "mock: replying with the example action from the observation"
        else:
            args = {"success": False, "note": "mock model: pipeline check only"} if DONE_TAKES_SUCCESS in first else {}
            cmd = dict(calls=[dict(name="done", args=args)], note="mock: finished")
        text = f"(mock model, turn {turn})\n" + json.dumps(cmd, ensure_ascii=False)
        for i in range(0, len(text), 12):
            if self._stop.is_set():
                return
            self.publish(dict(type="model", kind="delta", turn=turn, channel="text", text=text[i:i + 12]))
            time.sleep(0.03)
        record.update(latency_s=0.0, usage_in=0, usage_out=0)
        self._write(turn, cmd, record, text=text)

    def _answer_api(self, turn, obs_text, record):
        profile, key = self.spec.get("profile") or {}, self.spec.get("key")
        turns = self._conversation(turn)
        record.update(images_sent=[p["name"] for t in turns for p in t["parts"] if p["type"] == "image"],
                      provider=profile.get("provider"), model=profile.get("model"))
        attempts, errors = 0, []
        while not self._stop.is_set():
            if self.finish_requested:
                return self._answer(turn)
            attempts += 1
            self._cancel.clear()
            self._set(status="thinking", turn=turn, error=None)
            self.publish(dict(type="model", kind="request", turn=turn, attempt=attempts,
                              images=record["images_sent"], obs_chars=len(obs_text), model=profile.get("model")))
            buf = dict(text=[], thinking=[], t=time.time())

            def on_delta(channel, piece):
                buf[channel].append(piece)
                if time.time() - buf["t"] > 0.08:
                    self._flush(turn, buf)

            trace = (self.log_path.parent / f"codex_turn{turn:02d}_try{attempts}.jsonl"
                     if profile.get("provider") == "codex-cli" else None)
            try:
                result = llm.call(profile, key, BRIDGE_SYSTEM, turns, on_delta, self._cancel, trace=trace)
            except llm.Cancelled:
                if self.finish_requested:
                    return self._answer(turn)
                return
            except llm.LLMError as exc:
                self._flush(turn, buf)
                errors.append(str(exc))
                wait = retry_wait(str(exc), attempts)
                if wait is None:
                    message = str(exc) + (" (plan or API quota used up: click \"Retry turn\" once the quota is back, "
                                          "or switch to another model profile)"
                                          if QUOTA_RE.search(str(exc)) else "")
                    self.failed[turn] = message
                    record.update(error=message, attempts=attempts)
                    append_jsonl(self.log_path, record)
                    self._set(status="error", turn=turn, error=message)
                    self.publish(dict(type="model", kind="error", turn=turn, message=message, final=True))
                    return
                self.publish(dict(type="model", kind="error", turn=turn, final=False, retry_in_s=wait,
                                  message=str(exc) + (f" (rate limited, retrying in {wait:.0f} s)" if RATE_LIMIT_RE.search(str(exc))
                                                      else "")))
                self._cancel.wait(wait)     # request_finish() or stop() cuts the wait short
                if self._stop.is_set():
                    return
                continue
            self._flush(turn, buf)
            cmd, rest = extract_action(result["text"])
            if cmd is None and attempts < 3:
                # One correction round, recorded in the log; the relay never sees it.
                turns = turns + [dict(role="assistant", parts=[dict(type="text", text=result["text"] or "(empty)")]),
                                 dict(role="user", parts=[dict(type="text", text=(
                                     "Your reply contains no parsable JSON action. Reply with a single JSON object only, "
                                     'in the form {"calls": [{"name": "...", "args": {...}}]}.'))])]
                errors.append("reply had no parsable JSON action")
                self.publish(dict(type="model", kind="error", turn=turn, message="The reply has no parsable JSON; asking the model to answer again",
                                  final=False))
                continue
            if cmd is None:
                cmd = dict(calls=[], note="[monitor] The model's reply could not be parsed as an action")
            elif not cmd.get("note") and rest:
                cmd["note"] = rest[:1500]
            record.update(text=result["text"], thinking=result.get("thinking") or "", latency_s=result.get("latency_s"),
                          usage_in=result.get("usage_in"), usage_out=result.get("usage_out"),
                          stop_reason=result.get("stop_reason"), attempts=attempts, errors=errors,
                          cost_usd=_cost(profile, result.get("usage_in"), result.get("usage_out")))
            self._write(turn, cmd, record, text=result["text"])
            return

    def _flush(self, turn, buf):
        for channel in ("text", "thinking"):
            if buf[channel]:
                self.publish(dict(type="model", kind="delta", turn=turn, channel=channel, text="".join(buf[channel])))
                buf[channel] = []
        buf["t"] = time.time()

    # -- conversation ----------------------------------------------------------
    def _images(self, turn: int) -> List[Path]:
        if not self.relay.is_dir():
            return []
        imgs = [p for p in self.relay.glob(f"obs_{turn}_*") if p.suffix.lower() in (".png", ".jpg", ".jpeg")]
        return sorted(imgs, key=lambda p: ("wrist" not in p.name, p.name))

    def _image_part(self, path: Path) -> Optional[dict]:
        key = str(path)
        if key not in self._img_cache:
            try:
                self._img_cache[key] = base64.b64encode(path.read_bytes()).decode()
            except OSError:
                return None
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        return dict(type="image", mime=mime, b64=self._img_cache[key], name=path.name)

    def _conversation(self, turn: int) -> List[dict]:
        """Rebuild the chat from the relay files: every earlier obs as a user message and every
        earlier cmd (ours, or whoever answered it) as the assistant reply. Images ride along
        only for the newest `image_turns` observations."""
        turns: List[dict] = []
        for t in range(turn + 1):
            obs = read_text(self.relay / f"obs_{t}.md") or "(observation file missing)"
            parts = [dict(type="text", text=f"[obs_{t}.md]\n{obs}")]
            if t == 0 and self.instruction:
                parts.append(dict(type="text", text=f"{INSTRUCTION_HEAD}\n{self.instruction}"))
            for img in self._images(t):
                if turn - t < self.image_turns:
                    part = self._image_part(img)
                    if part:
                        parts.append(dict(type="text", text=f"[image {img.name}]"))
                        parts.append(part)
                else:
                    parts.append(dict(type="text", text=f"[image {img.name}: image from an earlier turn, omitted]"))
            turns.append(dict(role="user", parts=parts))
            if t < turn:
                reply = self.replies.get(t) or read_text(self.relay / f"cmd_{t}.json") or "(no reply recorded)"
                turns.append(dict(role="assistant", parts=[dict(type="text", text=reply)]))
        return turns

    # -- output --------------------------------------------------------------
    def _write(self, turn: int, cmd: Optional[dict], record: dict, text: str, operator: bool = False):
        cmd = cmd or dict(calls=[], note="[monitor] Empty action")
        path = self.relay / f"cmd_{turn}.json"
        tmp = self.relay / f".cmd_{turn}.json.monitor-tmp"
        tmp.write_text(json.dumps(cmd, ensure_ascii=False), encoding="utf-8")
        try:
            os.link(tmp, path)          # fails if someone else answered this turn meanwhile
            written = True
        except FileExistsError:
            written = False
        except OSError:                 # no hard links on this filesystem
            written = not path.exists()
            if written:
                os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
        self.replies[turn] = text
        record.update(calls=cmd.get("calls"), note=cmd.get("note"), written=written, operator=operator,
                      text=record.get("text", text))
        append_jsonl(self.log_path, record)
        self._set(status="waiting", turn=None, error=None, example=None)
        self.publish(dict(type="model", kind="done", turn=turn, calls=cmd.get("calls"), note=cmd.get("note"),
                          written=written, operator=operator, latency_s=record.get("latency_s"),
                          usage_in=record.get("usage_in"), usage_out=record.get("usage_out"),
                          cost_usd=record.get("cost_usd"), text=record.get("text"), thinking=record.get("thinking")))


def _cost(profile: dict, usage_in, usage_out) -> Optional[float]:
    pin, pout = profile.get("price_in_per_m"), profile.get("price_out_per_m")
    if pin in (None, "") or pout in (None, "") or usage_in is None or usage_out is None:
        return None
    return round((float(usage_in) * float(pin) + float(usage_out) * float(pout)) / 1e6, 5)
