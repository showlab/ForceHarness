import json

from fhmon import live
from test_server_boundary import _get, _post, server  # noqa: F401  (fixture)


def _episodes(base):
    return json.loads(_get(base + "/api/episodes")[1])["episodes"]


def test_hand_label_overrides_score_and_can_be_cleared(server):  # noqa: F811
    app, base = server
    key = _episodes(base)[0]["key"]
    before = _episodes(base)[0]["result"]

    status, body = _post(base + "/api/label", {"key": key, "success": True, "note": "Done after all; the runner scored it wrong"}, token=app.token)
    assert status == 200 and body["label"]["success"] is True
    r = _episodes(base)[0]["result"]
    assert r["manual"] and r["scored"] and r["success"] is True
    assert r["label"]["note"] == "Done after all; the runner scored it wrong" and r["harness"]["scored"] == before["scored"]
    ep = json.loads(_get(base + "/api/episode?key=" + key)[1])
    assert ep["result"]["success"] is True
    arms = live.compare(app.store.scan(), "free")["arms"]
    assert sum(a["n_success"] for a in arms.values()) >= 1

    # note only: keeps the harness result
    _post(base + "/api/label", {"key": key, "success": None, "note": "Just a note"}, token=app.token)
    r = _episodes(base)[0]["result"]
    assert not r.get("manual") and r["scored"] == before["scored"] and r["label"]["note"] == "Just a note"

    # both empty: the label is gone, and it survives a restart of the store
    _post(base + "/api/label", {"key": key, "success": None, "note": ""}, token=app.token)
    assert _episodes(base)[0]["result"]["label"] is None
    assert json.loads((app.cfg.data_dir / "labels.json").read_text()) == {}


def test_label_needs_token_and_a_real_episode(server):  # noqa: F811
    app, base = server
    key = _episodes(base)[0]["key"]
    assert _post(base + "/api/label", {"key": key, "success": True})[0] == 403
    assert _post(base + "/api/label", {"key": "nope/x", "success": True}, token=app.token)[0] == 404
    assert _post(base + "/api/label", {"key": key, "success": "yes"}, token=app.token)[0] == 400
