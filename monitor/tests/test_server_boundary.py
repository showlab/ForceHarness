import json
import re
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from conftest import MONITOR, fake_repository, make_free
from fhmon.server import App, Config, Server
from fhmon.sessions import LaunchError

HARNESS = MONITOR.parent


@pytest.fixture
def server(tmp_path, runs):
    make_free(runs)
    root = fake_repository(tmp_path / "fh")
    cfg = Config(root, runs, runs / "monitor", Path(sys.executable), "127.0.0.1", 0, allow_live=False)
    app = App(cfg)
    srv = Server(app)
    port = srv.server_address[1]
    app.hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield app, f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


def _get(url, headers=None):
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _post(url, body, token=None, headers=None):
    h = {"Content-Type": "application/json", **(headers or {})}
    if token:
        h["X-FHMon-Token"] = token
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_page_carries_token_and_api_lists_episodes(server):
    app, base = server
    status, html = _get(base + "/")
    assert status == 200 and app.token.encode() in html
    status, body = _get(base + "/api/episodes")
    eps = json.loads(body)["episodes"]
    assert status == 200 and [e["name"] for e in eps] == ["free_test_A"]


def test_posts_need_the_page_token_and_a_local_origin(server):
    app, base = server
    req = {"task": "blackboard_wipe", "conditions": ["A", "B"], "mode": "demo"}
    assert _post(base + "/api/sessions/preview", req)[0] == 403
    assert _post(base + "/api/sessions/preview", req, token=app.token, headers={"Origin": "http://evil.example"})[0] == 403
    status, body = _post(base + "/api/sessions/preview", req, token=app.token)
    assert status == 200 and len(body["sessions"]) == 2
    cmd = body["sessions"][0]["command"]
    assert cmd[0] == str(app.cfg.python) and cmd[1].endswith("scripts/real/blackboard_wipe.py")
    assert cmd[2:4] == ["--condition", "A"] and "--out" in cmd and "--relay-dir" in cmd and "--live" not in cmd


def test_host_header_and_file_traversal(server):
    _, base = server
    assert _get(base + "/api/info", {"Host": "evil.example"})[0] == 403
    assert _get(base + "/api/file?f=0%3A..%2F..%2Fetc%2Fpasswd")[0] == 404
    assert _get(base + "/api/file?f=0%3Afree%2Ffree_test_A_relay%2Fobs_0_wrist.png")[0] == 200
    assert _get(base + "/api/file?f=0%3Afree%2Ffree_test_A%2Fepisode.json")[0] == 200
    assert _get(base + "/static/../fhmon/server.py")[0] == 404


def test_live_is_gated(server):
    app, base = server
    req = {"task": "blackboard_wipe", "conditions": ["A"], "mode": "live", "model": {"source": "manual"}}
    status, body = _post(base + "/api/sessions/preview", req, token=app.token)
    assert status == 400 and "allow-live" in body["error"]
    app.cfg.allow_live = True
    good = {"calibration": "configs/real/force_calibration.json"}
    req["inputs"] = good
    with pytest.raises(LaunchError, match="checklist"):
        app.sessions.plan(req)
    assert "--live" in app.sessions.plan(req, preview=True)[0].command      # preview may show the command
    status, body = _post(base + "/api/sessions/launch", req, token=app.token)
    assert status == 400 and "checklist" in body["error"]                          # launching still needs every box
    req["checklist"] = [True, True, True]
    req["inputs"] = {"calibration": "/etc/passwd"}
    with pytest.raises(LaunchError, match="listed files"):
        app.sessions.plan(req)
    req["inputs"] = good
    planned = app.sessions.plan(req)
    assert "--live" in planned[0].command and "--auto-confirm" in planned[0].command
    with pytest.raises(LaunchError, match="mock"):
        app.sessions.plan(dict(req, model={"source": "mock"}))
    with pytest.raises(LaunchError, match="one real-robot episode"):
        app.sessions.plan(dict(req, conditions=["A", "B"]))


def test_live_uses_the_default_force_calibration(server):
    app, _ = server
    app.cfg.allow_live = True
    req = {"task": "blackboard_wipe", "conditions": ["A"], "mode": "live", "model": {"source": "manual"},
           "checklist": [True] * 3}
    assert [f["path"] for f in app.sessions.live_inputs("blackboard_wipe")["calibration"]["files"]] == \
        ["configs/real/force_calibration.json"]
    cmd = app.sessions.plan(req)[0].command
    assert cmd[cmd.index("--calibration") + 1] == str(app.cfg.harness_root / "configs/real/force_calibration.json")


def test_settings_never_store_keys(server, tmp_path):
    app, base = server
    status, body = _post(base + "/api/settings/profile",
                         {"profile": {"id": "p", "name": "P", "provider": "anthropic", "model": "m"}, "key": "sk-secret-999"},
                         token=app.token)
    assert status == 200
    saved = (app.cfg.data_dir / "settings.json").read_text()
    assert "sk-secret-999" not in saved
    prof = next(p for p in body["profiles"] if p["id"] == "p")
    assert prof["has_key"] and prof["key_source"] == "memory" and "sk-secret-999" not in json.dumps(body)
    status, body = _post(base + "/api/settings/profile", {"profile": {"name": "x", "provider": "nope"}}, token=app.token)
    assert status == 400


def test_monitor_and_harness_do_not_import_each_other():
    for path in (MONITOR / "fhmon").glob("*.py"):
        assert not re.search(r"^\s*(from|import)\s+forceharness", path.read_text(), re.M), path
    for folder in ("forceharness", "scripts", "robot_side"):
        for path in (HARNESS / folder).rglob("*.py"):
            assert "fhmon" not in path.read_text(errors="ignore"), path


def test_demo_model_choice_maps_to_runner_flags(server):
    app, _ = server
    plan = lambda task, cond, source, mode="demo": app.sessions.plan({"task": task, "conditions": [cond], "mode": mode,
                                                                      "model": {"source": source}})[0]
    scripted = plan("blackboard_wipe", "A", "scripted")
    assert "--demo-model" not in scripted.command and scripted.model_full is None
    assert "--demo-model" in plan("blackboard_wipe", "B", "mock").command      # runner waits for the relay model
    with pytest.raises(LaunchError, match="Scripted demo"):
        plan("blackboard_wipe", "A", "scripted", mode="live")


def test_new_profiles_get_unique_ids(server):
    app, _ = server
    a = app.settings.upsert({"name": "My model", "provider": "openai"})
    b = app.settings.upsert({"name": "My model", "provider": "openai"})
    assert a["id"] != b["id"]
    assert app.settings.upsert({"id": a["id"], "name": "Renamed", "provider": "openai"})["id"] == a["id"]


def test_runner_that_writes_nothing_is_a_failed_session(server):
    import time
    app, _ = server
    out = app.sessions.launch({"task": "blackboard_wipe", "conditions": ["A"], "mode": "demo",
                               "model": {"source": "scripted"}, "name": "no_episode"})
    sid = out[0]["id"]
    for _ in range(100):
        if app.sessions.sessions[sid].status not in ("starting", "running"):
            break
        time.sleep(0.05)
    s = app.sessions.sessions[sid]
    assert s.status == "failed" and s.exit_code == 0 and s.public()["log_url"].startswith("/api/file?f=")


def test_model_list_and_key_stays_with_its_host(server, fake_api):
    from fhmon import llm
    app, base = server
    api = fake_api([])
    assert [m["id"] for m in llm.list_models({"provider": "openai", "base_url": api.url}, "k")] == ["gpt-new", "gpt-old"]
    # a profile with a key typed into memory
    app.settings.upsert({"id": "p1", "name": "P1", "provider": "openai", "base_url": api.url, "model": "gpt-new"}, key="k-1")
    status, body = _post(base + "/api/settings/models", {"id": "p1", "profile": {"provider": "openai", "base_url": api.url}},
                         token=app.token)
    assert status == 200 and body["source"] == "memory" and [m["id"] for m in body["models"]] == ["gpt-new", "gpt-old"]
    assert api.requests[-1]["auth"] == "Bearer k-1"
    # the same profile pointed at another host in the form: the stored key is not sent there
    n = len(api.requests)
    status, body = _post(base + "/api/settings/models",
                         {"id": "p1", "profile": {"provider": "openai", "base_url": "http://127.0.0.1:9/v1"}}, token=app.token)
    assert status == 200 and body["source"] is None and body["models"] == [] and len(api.requests) == n
    # saving the profile with a new Base URL forgets the memory key
    saved = app.settings.upsert({"id": "p1", "name": "P1", "provider": "openai", "base_url": "http://127.0.0.1:9/v1"})
    assert saved["key_dropped"] and app.settings.key_for(app.settings.get("p1")) is None


def test_presets_come_from_the_price_table(server):
    app, base = server
    table = app.cfg.harness_root / "forceharness" / "eval" / "prices.yaml"
    table.parent.mkdir(parents=True)
    # Runner wording (forceharness/real/receipt_text.py): the placeholder marker of the price table comments
    table.write_text("gpt-6-astra:    {in: 10.0, out: 50.0}\ngemini-3: {in: 0.5, out: 3.0}   # Placeholder, not checked\n"
                     "claude-opus: {in: 4.0, out: 20.0}\n")
    status, body = _post(base + "/api/settings/models", {"profile": {"provider": "openai", "key_env": ""}}, token=app.token)
    assert status == 200 and body["source"] is None
    assert [m["id"] for m in body["presets"]] == ["gpt-6-astra"]                 # placeholder and claude prefixes left out
    assert body["presets"][0]["price_in"] == 10.0
    status, body = _post(base + "/api/settings/models", {"profile": {"provider": "anthropic", "key_env": ""}}, token=app.token)
    opus = next(m for m in body["presets"] if m["id"] == "claude-opus-5-5")
    assert opus["price_out"] == 20.0


def test_settings_page_shows_the_whole_relay_system_prompt(server):
    from fhmon.bridge import BRIDGE_SYSTEM
    app, base = server
    assert json.loads(_get(base + "/api/settings")[1])["bridge_system"] == BRIDGE_SYSTEM
    assert app.info()["bridge_system"] == BRIDGE_SYSTEM


def test_a_missing_task_prompt_blocks_launching_only(server):
    app, base = server
    Path(app.registry["screw_insert"]["fixed_instruction_path"]).unlink()
    task = next(t for t in app.info()["tasks"] if t["id"] == "screw_insert")
    assert task["fixed_instruction"] is None                                   # the page still loads
    with pytest.raises(LaunchError, match="missing"):
        app.sessions.plan({"task": "screw_insert", "conditions": ["A"], "mode": "demo", "model": {"source": "external"}})
    assert _get(base + "/api/episodes")[0] == 200


def test_tasks_json_names_files_that_exist_in_the_repository():
    """The launch contract with scripts/real/ and forceharness/prompts/real/; checked once the runners are there."""
    if not any((HARNESS / "scripts" / "real").glob("*.py")):
        pytest.skip("no runner scripts in scripts/real/ in this checkout")
    tasks = json.loads((MONITOR / "tasks.json").read_text(encoding="utf-8"))["tasks"]
    missing = [t["script"] for t in tasks if not (HARNESS / t["script"]).is_file()]
    missing += [t["fixed_instruction_file"] for t in tasks
                if t.get("fixed_instruction_file") and not (HARNESS / t["fixed_instruction_file"]).is_file()]
    assert not missing, missing
