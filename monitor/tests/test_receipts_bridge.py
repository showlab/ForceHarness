import json
import time

from conftest import PNG, RECEIPT_A, RECEIPT_B, obs_text
from fhmon import receipts
from fhmon.bridge import BRIDGE_SYSTEM, Bridge, example_action, extract_action


# -- receipt translation ------------------------------------------------------------
def test_harness_receipt_marks_force_lines():
    rows = receipts.split_lines(RECEIPT_A)
    force = [r["text"].strip() for r in rows if r["kind"] == "force"]
    assert force[0].startswith("contact events (t relative")
    assert force[1].startswith("t=1.50s resistance_drop")
    assert any(f.startswith("contact state at end") for f in force)
    assert any(f.startswith("still reading") for f in force)
    assert force[-1].startswith("net force peak")
    motion = [r["text"] for r in rows if r["kind"] == "motion"]
    assert any(m.startswith("end position") for m in motion)          # both conditions get this line
    assert rows[-1]["kind"] == "scene"


def test_closed_width_note_is_force_and_the_object_lines_are_scene():
    # Wording of forceharness/real/runner.py (CLOSED_WIDTH_NOTE) and scene_objects.py
    text = ("result: completed; duration 1.00 s\n(Closed width 41 mm is not the 29 mm of the force calibration: force "
            "readings while holding it use the calibrated gripper correction as an approximation, and may be off by about 1 N)\n"
            "#2 center [0.40, -0.10], highest point [0.40, -0.10, 0.08]\n  Grasp angle: grip across its narrow side\n"
            "(No reliable object coordinates in this frame.)")
    assert [r["kind"] for r in receipts.split_lines(text)] == ["motion", "force", "scene", "scene", "scene"]


def test_measurement_block_belongs_to_force():
    text = ("result: contact\nmeasurements (turn):\n  contact point   0.0004 ± 0.0003 m\n      [t=0.38 s]\n"
            "      ref: earlier in this episode\nturned 0.2 deg of the requested 1 deg")
    kinds = [r["kind"] for r in receipts.split_lines(text)]
    assert kinds == ["motion", "force", "force", "force", "force", "motion"]


def test_baseline_receipt_has_no_force():
    t = receipts.translation(RECEIPT_B)
    assert not t["has_force"] and t["force_text"] == ""


# -- reply parsing -------------------------------------------------------------------
def test_extract_action_variants():
    cmd, rest = extract_action('Let me think.\n```json\n{"calls":[{"name":"lift","args":{"height":0.05}}],"note":"n"}\n```\nOK')
    assert cmd == {"calls": [{"name": "lift", "args": {"height": 0.05}}], "note": "n"} and "```" not in rest
    cmd, _ = extract_action('Example {"calls":[{"name":"hold","args":{}}]} I choose {"name":"grasp","args":{}}')
    assert cmd["calls"] == [{"name": "grasp", "args": {}}]                       # the last action wins
    cmd, _ = extract_action('{"calls":[{"name":"a","args":{"k":{"x":1}}},{"name":"b","args":{}}]}')
    assert [c["name"] for c in cmd["calls"]] == ["a", "b"]                         # nested args stay whole
    assert extract_action("no json here") == (None, None)
    assert extract_action('{"foo": 1}') == (None, None)


def test_example_action_from_obs():
    assert example_action(obs_text(3, "body"))["calls"][0]["name"] == "hold"


# -- bridge against a relay folder ------------------------------------------------------
def _relay(tmp_path, turn=0, images=True, body="## SYSTEM\nTask"):
    relay = tmp_path / "relay"
    relay.mkdir(exist_ok=True)
    for t in range(turn + 1):
        (relay / f"obs_{t}.md").write_text(obs_text(t, body if t == 0 else "## TOOL RESULT\n{}"), encoding="utf-8")
        if images:
            (relay / f"obs_{t}_wrist.png").write_bytes(PNG)
            (relay / f"obs_{t}_exterior.png").write_bytes(PNG)
    (relay / "pending.json").write_text(json.dumps({"turn": turn, "waiting": True}))
    return relay


def _wait(path, timeout=6.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if path.exists():
            return json.loads(path.read_text())
        time.sleep(0.05)
    raise AssertionError(f"{path.name} was not written")


def _spec(api, **profile):
    return dict(source="api", label="t", key="k-123",
                profile=dict(id="t", provider="openai", base_url=api.url, model="m", max_tokens=100, **profile))


def test_bridge_api_sends_obs_images_and_instruction(tmp_path, fake_api):
    api = fake_api(['I have looked at it. {"calls":[{"name":"turn","args":{"angle_deg":20}}],"note":"turn"}'])
    relay = _relay(tmp_path)
    events = []
    b = Bridge(relay, _spec(api), events.append, tmp_path / "bridge.jsonl", instruction="no single move over 6 cm").start()
    try:
        cmd = _wait(relay / "cmd_0.json")
    finally:
        b.stop()
    assert cmd == {"calls": [{"name": "turn", "args": {"angle_deg": 20}}], "note": "turn"}
    req = api.requests[0]
    assert req["auth"] == "Bearer k-123"
    user = [m for m in req["body"]["messages"] if m["role"] == "user"][0]["content"]
    texts = [p["text"] for p in user if p["type"] == "text"]
    assert texts[0].startswith("[obs_0.md]") and any("no single move over 6 cm" in t for t in texts)
    assert req["body"]["messages"][0] == {"role": "system", "content": BRIDGE_SYSTEM}
    assert sum(p["type"] == "image_url" for p in user) == 2
    log = [json.loads(line) for line in (tmp_path / "bridge.jsonl").read_text().splitlines()]
    assert log[0]["usage_in"] == 10 and log[0]["written"] and "k-123" not in json.dumps(log)
    assert any(e.get("kind") == "delta" for e in events) and any(e.get("kind") == "done" for e in events)


def test_bridge_rebuilds_history_and_limits_images(tmp_path, fake_api):
    api = fake_api(['{"calls":[{"name":"hold","args":{"seconds":1}}]}'])
    relay = _relay(tmp_path, turn=2)
    (relay / "cmd_0.json").write_text('{"calls":[{"name":"a","args":{}}]}')
    (relay / "cmd_1.json").write_text('{"calls":[{"name":"b","args":{}}]}')
    b = Bridge(relay, _spec(api), lambda e: None, tmp_path / "b.jsonl", image_turns=1).start()
    try:
        _wait(relay / "cmd_2.json")
    finally:
        b.stop()
    msgs = api.requests[0]["body"]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user", "assistant", "user"]
    assert '"name":"a"' in msgs[2]["content"]                                    # someone else's earlier reply
    images = [sum(p["type"] == "image_url" for p in m["content"]) for m in msgs if m["role"] == "user"]
    assert images == [0, 0, 2]                                                   # only the newest obs carries images
    assert "[image obs_0_wrist.png: image from an earlier turn, omitted]" in [p.get("text") for p in msgs[1]["content"]]


def test_bridge_retries_unparsable_reply(tmp_path, fake_api):
    api = fake_api(["I think it should turn a little.", '{"calls":[{"name":"turn","args":{}}]}'])
    relay = _relay(tmp_path, images=False)
    b = Bridge(relay, _spec(api), lambda e: None, tmp_path / "b.jsonl").start()
    try:
        cmd = _wait(relay / "cmd_0.json")
    finally:
        b.stop()
    assert cmd["calls"][0]["name"] == "turn" and len(api.requests) == 2
    assert api.requests[1]["body"]["messages"][-1]["content"][0]["text"].startswith("Your reply contains no parsable JSON action")


def test_bridge_never_overwrites_an_answered_turn(tmp_path, fake_api):
    api = fake_api(['{"calls":[{"name":"x","args":{}}]}'])
    relay = _relay(tmp_path, images=False)
    (relay / "cmd_0.json").write_text('{"calls":[{"name":"mine","args":{}}]}')
    b = Bridge(relay, _spec(api), lambda e: None, tmp_path / "b.jsonl").start()
    time.sleep(0.8)
    b.stop()
    assert json.loads((relay / "cmd_0.json").read_text())["calls"][0]["name"] == "mine"
    assert api.requests == []


def test_manual_and_graceful_finish(tmp_path):
    relay = _relay(tmp_path, images=False)
    events = []
    b = Bridge(relay, dict(source="manual", label="m"), events.append, tmp_path / "b.jsonl",
               graceful_done={"name": "done", "args": {"success": False, "note": "op"}}).start()
    try:
        t0 = time.time()
        while not any(e.get("kind") == "manual_wait" for e in events) and time.time() - t0 < 5:
            time.sleep(0.05)
        assert b.state["status"] == "manual_wait" and json.loads(b.state["example"])["calls"][0]["name"] == "hold"
        assert b.submit_manual(0, "garbage") is not None
        assert b.submit_manual(0, '{"calls":[{"name":"grasp","args":{}}]}') is None
        assert _wait(relay / "cmd_0.json")["calls"][0]["name"] == "grasp"
        (relay / "obs_1.md").write_text(obs_text(1, "x"), encoding="utf-8")
        (relay / "pending.json").write_text(json.dumps({"turn": 1, "waiting": True}))
        b.request_finish()
        done = _wait(relay / "cmd_1.json")
        assert done["calls"] == [{"name": "done", "args": {"success": False, "note": "op"}}]
        assert "operator" in done["note"]
    finally:
        b.stop()


def test_mock_replays_the_example_then_finishes(tmp_path):
    # Runner wording (forceharness/real/receipt_text.py): the tool menu line the runners write for done
    relay = _relay(tmp_path, images=False, body="## SYSTEM\n- `done(success, note)`: end the episode")
    b = Bridge(relay, dict(source="mock", label="mock"), lambda e: None, tmp_path / "b.jsonl").start()
    try:
        assert _wait(relay / "cmd_0.json")["calls"] == [{"name": "hold", "args": {"seconds": 0.5}}]
        for turn in (1, 2):
            (relay / f"obs_{turn}.md").write_text(obs_text(turn, "## TOOL RESULT\n{}"), encoding="utf-8")
            (relay / "pending.json").write_text(json.dumps({"turn": turn, "waiting": True}))
            cmd = _wait(relay / f"cmd_{turn}.json")
        assert cmd["calls"] == [{"name": "done", "args": {"success": False, "note": "mock model: pipeline check only"}}]
    finally:
        b.stop()


def test_finish_during_retry_wait(tmp_path):
    relay = _relay(tmp_path, images=False)
    dead = dict(source="api", label="t", key="k",
                profile=dict(id="t", provider="openai", base_url="http://127.0.0.1:9/v1", model="m"))
    b = Bridge(relay, dead, lambda e: None, tmp_path / "b.jsonl",
               graceful_done={"name": "done", "args": {}}).start()
    try:
        time.sleep(0.6)                 # first attempt fails at once (nothing listens on port 9)
        b.request_finish()
        assert _wait(relay / "cmd_0.json", timeout=3)["calls"] == [{"name": "done", "args": {}}]
    finally:
        b.stop()
