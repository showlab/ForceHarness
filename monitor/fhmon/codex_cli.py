"""The model through the local Codex CLI (`codex exec`), billed to the ChatGPT plan it is signed in
with instead of per-token API billing, which makes long real-robot sessions much cheaper to run.

The executable is $CODEX_BIN (set by `python -m fhmon --codex-bin PATH`) or `codex` on PATH; it reads its
login and configuration from ~/.codex (or $CODEX_HOME).
One `codex exec` per turn, stateless like the API clients: the bridge's provider-neutral turns
(every earlier observation and reply, images of the newest turns) are flattened into one prompt on
stdin, the images go in with -i, the run is read-only and ephemeral in an empty temp directory, and
the final message comes back through -o. --json events give the live view its reasoning and usage.
Every *_API_KEY variable is removed from Codex's environment so it cannot fall back to API billing.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

HEAD = ("Through the monitor you are making the decisions for a real-robot experiment (one step at a time). Below is the "
        "full conversation so far: the observations the experiment sent and your earlier replies. "
        "Everything you need is in this message and the attached images: do not run commands and do not read or write "
        "files; reply directly as the \"YOUR MOVE\" section of the last observation asks, with a single JSON object only.")
STALL_S = 150.0     # no event at all for this long: stop codex (the bridge waits like a 429 and retries)


class CodexError(Exception):
    pass


class CodexCancelled(Exception):
    pass


def binary() -> str:
    """The Codex CLI executable: $CODEX_BIN, else `codex` on PATH. It must be OpenAI's Codex CLI, signed in
    (`codex login`) with the ChatGPT plan that should be used."""
    env = os.environ.get("CODEX_BIN")
    if env:
        return env
    path = shutil.which("codex")
    if path:
        return path
    raise CodexError("Codex CLI not found: install it so that `codex` is on PATH, or give its path with "
                     "--codex-bin or the environment variable CODEX_BIN")


def available() -> bool:
    try:
        binary()
        return True
    except CodexError:
        return False


def models() -> List[dict]:
    """Models the logged-in plan lists, from Codex's own cache (no network): [{id, label, created}]."""
    home = Path(os.environ.get("CODEX_HOME") or Path.home() / ".codex")
    try:
        data = json.loads((home / "models_cache.json").read_text())
    except (OSError, json.JSONDecodeError):
        return []
    out = []
    for m in (data.get("models") if isinstance(data, dict) else None) or []:
        mid = isinstance(m, dict) and (m.get("slug") or m.get("id"))
        if mid and m.get("visibility", "list") == "list":
            out.append(dict(id=str(mid), label=str(m.get("display_name") or mid), created=None))
    return out


def prompt(system: str, turns: List[dict]) -> tuple:
    """(prompt text, [(name, mime, b64)]) with the images in the order they appear."""
    lines, images = [HEAD, "", "## System instructions", system or "", ""], []
    for t in turns:
        lines.append("## Your earlier reply" if t.get("role") == "assistant" else "## Message from the experiment")
        for p in t.get("parts", []):
            if p.get("type") == "text":
                lines.append(p["text"])
            elif p.get("type") == "image":
                images.append((p.get("name") or f"image_{len(images)}.png", p.get("mime") or "image/png", p["b64"]))
                lines.append(f"(attached image {len(images)}: {p.get('name') or ''})")
        lines.append("")
    if images:
        lines.append("Images attached to this message: %d, in the same order as the \"attached image n\" marks above." % len(images))
    return "\n".join(lines), images


def _safe(name: str, i: int, mime: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(name).stem)[:60] or "image"
    return f"{i:02d}_{stem}" + (".jpg" if "jpeg" in mime else ".png")


def call(profile: dict, system: str, turns: List[dict], on_delta, cancel=None, timeout: float = 420.0,
         trace: Optional[Path] = None, stall_s: float = STALL_S) -> Dict[str, object]:
    """One codex exec turn. `trace` (optional) receives every event line and the stderr tail with times.
    A codex that prints nothing for `stall_s` seconds is stopped: a turn can hang for minutes on the server
    side while the same request, sent again, answers in seconds."""
    text_in, images = prompt(system, turns)
    with tempfile.TemporaryDirectory(prefix="fhmon-codex-") as tmp:
        work = Path(tmp)
        last = work / "last_message.txt"
        cmd = [binary(), "exec", "--skip-git-repo-check", "--ephemeral", "--sandbox", "read-only", "--json",
               "-C", str(work), "-o", str(last)]
        if profile.get("model"):
            cmd += ["-m", str(profile["model"])]
        if profile.get("reasoning_effort"):
            cmd += ["-c", f'model_reasoning_effort="{profile["reasoning_effort"]}"']
        cmd += ["-c", 'model_reasoning_summary="auto"']
        for i, (name, mime, b64) in enumerate(images):
            path = work / _safe(name, i, mime)
            path.write_bytes(base64.b64decode(b64))
            cmd += ["-i", str(path)]
        env = {k: v for k, v in os.environ.items() if not k.endswith("_API_KEY")}
        try:
            proc = subprocess.Popen(cmd, cwd=work, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace")
        except OSError as exc:
            raise CodexError(f"codex failed to start: {exc}") from None
        lines: "queue.Queue[Optional[str]]" = queue.Queue()
        err_tail: List[str] = []

        def pump_out():
            for line in proc.stdout:
                lines.put(line)
            lines.put(None)

        def pump_err():
            for line in proc.stderr:
                err_tail.append(line.rstrip())
                del err_tail[:-20]
        for fn in (pump_out, pump_err):
            threading.Thread(target=fn, daemon=True).start()
        try:
            proc.stdin.write(text_in)
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        texts, thinking, usage, errors, failed = [], [], {}, [], None
        t0 = time.time()
        deadline, last_event = t0 + timeout, t0
        trace_fh = open(trace, "a", encoding="utf-8") if trace else None
        try:
            while True:
                if cancel is not None and cancel.is_set():
                    raise CodexCancelled()
                if time.time() > deadline:
                    raise CodexError(f"codex did not finish within {timeout:.0f} s")
                if time.time() - last_event > stall_s:
                    raise CodexError(f"codex produced no output for {stall_s:.0f} s (probably queued behind a rate limit or stuck on the server); this call was stopped")
                try:
                    line = lines.get(timeout=0.25)
                except queue.Empty:
                    continue
                if line is None:
                    break
                last_event = time.time()
                if trace_fh:
                    trace_fh.write(json.dumps(dict(t=round(last_event - t0, 2), line=line.rstrip()[:4000]),
                                              ensure_ascii=False) + "\n")
                    trace_fh.flush()
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind = ev.get("type") or ""
                item = ev.get("item") or {}
                if kind == "item.completed" and item.get("type") == "reasoning" and item.get("text"):
                    thinking.append(item["text"].strip())
                    on_delta("thinking", item["text"].strip() + "\n")
                elif kind == "item.completed" and item.get("type") == "agent_message":
                    texts.append(item.get("text") or "")
                    on_delta("text", item.get("text") or "")
                elif kind == "item.started" and item.get("type") == "command_execution":
                    on_delta("thinking", f"[codex running a command: {item.get('command')}]\n")
                elif kind == "turn.completed":
                    usage = ev.get("usage") or {}
                elif kind == "turn.failed":
                    failed = str((ev.get("error") or {}).get("message") or ev.get("error") or "turn failed")
                elif kind == "error":
                    errors.append(str(ev.get("message") or ev))
                    on_delta("thinking", f"[codex: {errors[-1][:300]}]\n")     # e.g. its own retry notices
            proc.wait(timeout=30)
        except BaseException:
            _stop(proc)
            raise
        finally:
            if trace_fh:
                trace_fh.write(json.dumps(dict(t=round(time.time() - t0, 2), stderr_tail=err_tail[-20:],
                                               returncode=proc.poll()), ensure_ascii=False) + "\n")
                trace_fh.close()
        final = last.read_text(encoding="utf-8", errors="replace").strip() if last.exists() else ""
        final = final or (texts[-1] if texts else "")
    if failed or (proc.returncode and not final):
        detail = failed or (errors[-1] if errors else "") or " / ".join(x for x in err_tail[-3:] if x) or "no output"
        raise CodexError(f"codex failed (exit code {proc.returncode}): {detail[:400]}")
    return dict(text=final, thinking="\n".join(thinking), stop_reason="completed",
                usage_in=usage.get("input_tokens"), usage_out=usage.get("output_tokens"),
                usage_cached=usage.get("cached_input_tokens"))


def _stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
