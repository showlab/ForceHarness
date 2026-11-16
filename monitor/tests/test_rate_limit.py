"""Model-call retries: a 429 waits longer before the next attempt, a used-up quota stops."""
import json
import time

from conftest import obs_text
from fhmon import bridge, llm
from fhmon.bridge import Bridge, retry_wait

CODEX_429 = "exceeded retry limit, last status: 429 Too Many Requests"


def test_retry_wait_by_kind():
    assert [retry_wait(CODEX_429, n) for n in (1, 2, 3, 4)] == [20, 40, 80, None]
    assert retry_wait("HTTP 429: Rate limit reached for gpt-6-astra", 1) == 20
    assert retry_wait("You've hit your usage limit. Try again at 23:05.", 1) is None
    assert retry_wait("HTTP 429: You exceeded your current quota (insufficient_quota)", 1) is None
    assert [retry_wait("Connection timed out", n) for n in (1, 2, 3)] == [3, 6, None]       # other errors unchanged
    stalled = "codex produced no output for 150 s (probably queued behind a rate limit or stuck on the server); this call was stopped"
    assert retry_wait(stalled, 1) == 20                                          # waits like a 429
    assert retry_wait("codex produced no output for 150 s", 1) == 20             # by the stall wording alone
    assert retry_wait("codex failed (exit code 1): no output", 1) == 3           # a plain failure is not a stall


def _relay(tmp_path):
    relay = tmp_path / "relay"
    relay.mkdir()
    (relay / "obs_0.md").write_text(obs_text(0, "## SYSTEM\nTask"), encoding="utf-8")
    (relay / "pending.json").write_text(json.dumps({"turn": 0, "waiting": True}))
    return relay


def _run(tmp_path, monkeypatch, failures):
    monkeypatch.setattr(bridge, "RATE_LIMIT_WAITS_S", (0.05, 0.05, 0.05))
    calls = []

    def fake_call(profile, key, system, turns, on_delta=None, cancel=None, timeout=300.0, trace=None):
        calls.append(time.time())
        if len(calls) <= len(failures):
            raise llm.LLMError(failures[len(calls) - 1])
        return dict(text='{"calls":[{"name":"scene","args":{}}]}', thinking="", usage_in=1, usage_out=1,
                    latency_s=0.01, stop_reason="stop")
    monkeypatch.setattr(llm, "call", fake_call)
    events, log = [], tmp_path / "bridge.jsonl"
    spec = dict(source="api", label="t", key="k", profile=dict(id="t", provider="codex-cli", model="m"))
    b = Bridge(_relay(tmp_path), spec, events.append, log).start()
    t0 = time.time()
    while time.time() - t0 < 5 and not log.exists():
        time.sleep(0.02)
    b.stop()
    return calls, events, json.loads(log.read_text().splitlines()[-1])


def test_429_is_retried_until_it_clears(tmp_path, monkeypatch):
    calls, events, record = _run(tmp_path, monkeypatch, [CODEX_429, CODEX_429, CODEX_429])
    assert len(calls) == 4 and record["attempts"] == 4 and len(record["errors"]) == 3
    waits = [e for e in events if e.get("kind") == "error"]
    assert all(not e["final"] and "rate limited" in e["message"] for e in waits)
    assert json.loads((tmp_path / "relay" / "cmd_0.json").read_text())["calls"][0]["name"] == "scene"


def test_used_up_quota_stops_at_once(tmp_path, monkeypatch):
    calls, events, record = _run(tmp_path, monkeypatch, ["You've hit your usage limit. Try again at 23:05."])
    assert len(calls) == 1 and record["attempts"] == 1 and "quota used up" in record["error"]
    assert [e["final"] for e in events if e.get("kind") == "error"] == [True]
    assert not (tmp_path / "relay" / "cmd_0.json").exists()
