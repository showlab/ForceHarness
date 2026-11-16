"""The Codex CLI as a model source (ChatGPT plan instead of API billing), with a fake `codex` binary:
it records its argv, environment, stdin prompt and images, and answers in codex exec --json form."""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

from fhmon import codex_cli, llm, settings as settings_mod
from fhmon.bridge import Bridge

from test_receipts_bridge import _relay, _wait

FAKE = r'''#!{python}
import json, os, sys, time
argv = sys.argv[1:]
out = argv[argv.index("-o") + 1]
images = [argv[i + 1] for i, a in enumerate(argv) if a == "-i"]
log = dict(argv=argv, prompt=sys.stdin.read(), env_keys=sorted(k for k in os.environ if k.endswith("_API_KEY")),
           images=[os.path.getsize(p) for p in images], cwd=os.getcwd())
open(os.environ["FAKE_CODEX_LOG"], "a").write(json.dumps(log) + "\n")
mode = os.environ.get("FAKE_CODEX_MODE", "ok")
print(json.dumps({{"type": "thread.started", "thread_id": "t-1"}}), flush=True)
if mode == "sleep":
    time.sleep(30)
if mode == "fail":
    print(json.dumps({{"type": "turn.failed", "error": {{"message": "You've hit your usage limit."}}}}), flush=True)
    sys.exit(1)
print(json.dumps({{"type": "item.completed", "item": {{"id": "r", "type": "reasoning", "text": "Look at the wrist image first"}}}}), flush=True)
reply = os.environ.get("FAKE_CODEX_REPLY", '{{"calls":[{{"name":"rotate_z","args":{{"angle_deg":-88}}}}],"note":"align the angle first"}}')
print(json.dumps({{"type": "item.completed", "item": {{"id": "m", "type": "agent_message", "text": reply}}}}), flush=True)
open(out, "w").write(reply)
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 1200, "cached_input_tokens": 800, "output_tokens": 40}}}}), flush=True)
'''


@pytest.fixture
def fake_codex(tmp_path, monkeypatch):
    exe = tmp_path / "codex"
    exe.write_text(FAKE.format(python=sys.executable))
    exe.chmod(0o755)
    log = tmp_path / "codex_calls.jsonl"
    monkeypatch.setenv("CODEX_BIN", str(exe))
    monkeypatch.setenv("FAKE_CODEX_LOG", str(log))
    monkeypatch.setenv("FH_API_KEY", "sk-should-not-reach-codex")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-should-not-reach-codex")

    def calls():
        return [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return calls


PROFILE = dict(id="codex", provider="codex-cli", model="gpt-6-astra", reasoning_effort="medium")


def test_one_turn_runs_a_read_only_ephemeral_codex_with_the_images(fake_codex):
    turns = [dict(role="user", parts=[dict(type="text", text="[obs_0.md]\nobservation 0"),
                                      dict(type="image", mime="image/png", b64="iVBORw0KGgo=", name="obs_0_wrist.png")]),
             dict(role="assistant", parts=[dict(type="text", text='{"calls":[{"name":"scene","args":{}}]}')]),
             dict(role="user", parts=[dict(type="text", text="[obs_1.md]\nobservation 1")])]
    deltas = []
    r = llm.call(PROFILE, "", "system instructions S", turns, lambda kind, text: deltas.append((kind, text)))
    assert json.loads(r["text"])["calls"][0] == {"name": "rotate_z", "args": {"angle_deg": -88}}
    assert r["usage_in"] == 1200 and r["usage_out"] == 40 and r["thinking"] == "Look at the wrist image first" and r["latency_s"] >= 0
    (call,) = fake_codex()
    argv = call["argv"]
    assert argv[0] == "exec" and "--ephemeral" in argv and argv[argv.index("--sandbox") + 1] == "read-only"
    assert argv[argv.index("-m") + 1] == "gpt-6-astra" and 'model_reasoning_effort="medium"' in argv
    assert argv.count("-i") == 1 and call["images"] == [8] and argv[-2] == "-i"     # images last, prompt on stdin
    assert call["env_keys"] == []                                  # no API key: the plan, never API billing
    p = call["prompt"]
    assert p.index("system instructions S") < p.index("observation 0") < p.index('"scene"') < p.index("observation 1")
    assert "(attached image 1: obs_0_wrist.png)" in p and p.startswith(codex_cli.HEAD)
    assert ("thinking", "Look at the wrist image first\n") in deltas and any(k == "text" for k, _ in deltas)
    assert not Path(call["cwd"]).exists()                          # the temp dir is removed afterwards


def test_a_failed_turn_and_a_cancel_surface_like_the_api_clients(fake_codex, monkeypatch):
    monkeypatch.setenv("FAKE_CODEX_MODE", "fail")
    with pytest.raises(llm.LLMError, match="usage limit"):
        llm.call(PROFILE, "", "S", [dict(role="user", parts=[dict(type="text", text="x")])])
    monkeypatch.setenv("FAKE_CODEX_MODE", "sleep")
    cancel = threading.Event()
    threading.Timer(1.0, cancel.set).start()
    t0 = time.time()
    with pytest.raises(llm.Cancelled):
        llm.call(PROFILE, "", "S", [dict(role="user", parts=[dict(type="text", text="x")])], cancel=cancel)
    assert time.time() - t0 < 10


def test_the_bridge_answers_a_turn_through_codex_without_a_key(tmp_path, fake_codex):
    relay = _relay(tmp_path)
    spec = dict(source="api", label="gpt-6-astra (Codex subscription)", key="", profile=dict(PROFILE, name="Codex"))
    events = []
    b = Bridge(relay, spec, events.append, tmp_path / "bridge.jsonl").start()
    try:
        cmd = _wait(relay / "cmd_0.json", timeout=20)
    finally:
        b.stop()
    assert cmd["calls"] == [{"name": "rotate_z", "args": {"angle_deg": -88}}]
    (call,) = fake_codex()
    assert call["argv"].count("-i") == 2                           # wrist and exterior photo of turn 0
    log = [json.loads(line) for line in (tmp_path / "bridge.jsonl").read_text().splitlines()]
    assert log[0]["provider"] == "codex-cli" and log[0]["cost_usd"] is None and log[0]["written"]


def test_settings_offer_codex_without_a_key(tmp_path, fake_codex, monkeypatch):
    home = tmp_path / "codex_home"
    home.mkdir()
    (home / "models_cache.json").write_text(json.dumps({"models": [
        {"slug": "gpt-6-astra", "visibility": "list"}, {"slug": "codex-auto-review", "visibility": "hide"}]}))
    monkeypatch.setenv("CODEX_HOME", str(home))
    old = tmp_path / "settings.json"                 # saved before the Codex option existed
    old.write_text(json.dumps({"profiles": [dict(settings_mod.default_profiles()[0])]}))
    s = settings_mod.Settings(old)
    codex = [p for p in s.public() if p["provider"] == "codex-cli"]
    assert len(codex) == 1 and codex[0]["has_key"] and codex[0]["key_source"] == "codex-login"
    assert s.key_for(codex[0]) == ""
    saved = s.upsert(dict(codex[0], reasoning_effort="high"))
    assert saved["provider"] == "codex-cli" and saved["reasoning_effort"] == "high"
    assert [m["id"] for m in llm.list_models(dict(provider="codex-cli"), "")] == ["gpt-6-astra"]


def test_the_executable_is_codex_bin_else_codex_on_path(tmp_path, monkeypatch):
    monkeypatch.delenv("CODEX_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    assert not codex_cli.available()
    with pytest.raises(codex_cli.CodexError, match="--codex-bin"):
        codex_cli.binary()
    exe = tmp_path / "codex"
    exe.write_text("#!/bin/sh\n")
    exe.chmod(0o755)
    assert codex_cli.binary() == str(exe)
    monkeypatch.setenv("CODEX_BIN", "/opt/elsewhere/codex")
    assert codex_cli.binary() == "/opt/elsewhere/codex"


def test_a_silent_codex_is_stopped_and_traced(fake_codex, tmp_path, monkeypatch):
    # a turn can hang for minutes with no output while the same request, sent again, answers in seconds
    monkeypatch.setenv("FAKE_CODEX_MODE", "sleep")
    turns = [dict(role="user", parts=[dict(type="text", text="[obs_0.md]\nobservation 0")])]
    trace = tmp_path / "codex_turn00_try1.jsonl"
    t0 = time.time()
    with pytest.raises(codex_cli.CodexError, match="produced no output for"):
        codex_cli.call(PROFILE, "system", turns, lambda kind, text: None, trace=trace, stall_s=0.5)
    assert time.time() - t0 < 10
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert json.loads(rows[0]["line"])["type"] == "thread.started" and "stderr_tail" in rows[-1]
