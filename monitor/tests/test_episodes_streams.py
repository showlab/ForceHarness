import json
import os
import time

from conftest import make_free
from fhmon import episodes, live, streams


def test_free_adapter_joins_relay_actions_and_reports(runs):
    ep_dir = make_free(runs)
    paths = episodes.Paths([runs])
    key = paths.rel(ep_dir)
    ep = episodes.adapter_for(json.loads((ep_dir / "episode.json").read_text())).load(ep_dir, key, paths)
    assert ep["family"] == "free" and ep["arm"] == "harness" and ep["status"] == "ended"
    assert ep["model"] == "External relay" and ep["metrics"]["actions"] == 2 and ep["metrics"]["decision_latency_s"] == 15.0
    t0, t1 = ep["turns"]
    assert t0["decision"]["note"] == "lift to weigh" and t0["action"]["termination"] == "completed"
    assert [i["label"] for i in t0["obs"]["images"]] == ["wrist", "exterior"]
    assert t0["translation"]["has_force"] and t0["report"]["peak_force"] == 4.3 and len(t0["report"]["spark"]) > 2
    assert t0["report"]["derived"]["payload_kg"] == 0.32
    assert t1["action"]["status"] == "done" and t1["receipt"] is None
    assert ep["result"]["scored"] is False and ep["result"]["model_claim"] is True


def test_baseline_arm_and_running_status(runs):
    ep_dir = make_free(runs, name="free_test_B", condition="none_guard", ended=False)
    relay = ep_dir.with_name(ep_dir.name + "_relay")
    (relay / "obs_2.md").write_text("# turn 2\n")
    (relay / "pending.json").write_text(json.dumps({"turn": 2, "waiting": True}))
    paths = episodes.Paths([runs])
    ep = episodes.FreeAdapter().load(ep_dir, paths.rel(ep_dir), paths)
    assert ep["arm"] == "baseline" and ep["status"] == "waiting_model"
    assert not ep["turns"][0]["translation"]["has_force"]
    assert ep["turns"][2]["decision"] is None and ep["turns"][2]["obs"]["name"] == "obs_2.md"


def test_fixed_tasks_and_unknown_tasks(runs):
    record = dict(version="free_task/1", task_id="screw_insert")
    assert episodes.adapter_for(record).family == "screw_insert"
    assert episodes.adapter_for(dict(record, task_id="free")).family == "free"
    assert episodes.adapter_for(dict(record, task_id="not_a_task")) is None      # never counted as a free task
    assert episodes.adapter_for(dict(version="something_else")) is None


def test_paths_refuse_traversal(runs):
    paths = episodes.Paths([runs])
    assert paths.resolve("0:../../etc/passwd") is None
    assert paths.resolve("7:x") is None and paths.resolve("nope") is None
    assert paths.resolve("0:free") == (runs / "free").resolve()


def test_store_scan_and_compare(runs):
    make_free(runs, scored="success")
    make_free(runs, name="free_test_B", condition="none_guard", scored="failed")
    make_free(runs, name="wipe_A", task_id="blackboard_wipe", scored="success")
    (runs / "free" / "not_an_episode").mkdir()
    (runs / "free" / "not_an_episode" / "episode.json").write_text('{"version": "an_older_runner"}')
    store = live.Store([runs])
    found = store.scan()
    assert sorted(e["family"] + "/" + e["arm"] for e in found) == ["blackboard_wipe/harness", "free/baseline", "free/harness"]
    cmp = live.compare(found, "free")
    assert cmp["arms"]["harness"]["n_success"] == 1 and cmp["arms"]["harness"]["ci_low"] > 0
    assert cmp["arms"]["baseline"]["n_scored"] == 1 and cmp["arms"]["baseline"]["n_success"] == 0


def test_wilson_interval():
    p, lo, hi = live.wilson(4, 4)
    assert p == 1.0 and abs(lo - 0.51) < 0.01 and hi == 1.0
    assert live.wilson(0, 0) == (None, None, None)


def _nuc_line(ts_ns, f):
    return json.dumps({"type": "sample", "timestamp_ns": ts_ns, "wrench": [f, 0.0, 0.0, 0.0, 0.0, 0.1],
                       "gripper_width": 0.08, "flange_pos": [0.4, 0, 0.3]}) + "\n"


def test_stream_overview_aligns_robot_clock_with_file_time(tmp_path):
    path = tmp_path / "nuc.jsonl"
    robot0 = 1_793_000_000.0                      # robot PC clock, 8 min behind this computer
    with path.open("w") as f:
        f.write("Warning: torch noise\n")
        f.write('{"type":"ready"}\n')
        for i in range(200):
            f.write(_nuc_line(int((robot0 + i * 0.02) * 1e9), 1.0 if i != 100 else 9.0))
    wall_end = robot0 + 199 * 0.02 + 480.0
    os.utime(path, (wall_end, wall_end))
    episode_start = robot0 + 480.0 - 1.0          # the episode began 1 s before the first sample
    ov = streams.overview(path, "nuc", episode_start, max_points=50)
    assert not ov["clock"]["synthetic"] and abs(ov["clock"]["offset_s"] - 480.0) < 1e-3
    assert abs(ov["points"][0][0] - 1.0) < 0.05
    assert max(p[4] for p in ov["points"]) == 9.0  # decimation keeps the spike


def test_stream_tail_follows_growth(tmp_path):
    path = tmp_path / "motion.jsonl"
    rows = lambda a, b: "".join(json.dumps({"timestamp_ns": int(i * 0.05e9), "wrench": [0, 0, i / 10, 0, 0, 0],
                                            "phase": "lift_to_H" if i < 30 else "loaded_measure", "trial": 1}) + "\n"
                                for i in range(a, b))
    path.write_text(rows(0, 20))
    tail = streams.StreamTail(path, "motion", None)
    first = tail.poll()
    assert first and tail.clock_info()["synthetic"]
    with path.open("a") as f:
        f.write(rows(20, 40))
    more = tail.poll()
    assert more and more[0][0] > first[-1][0]
    assert [p["label"] for p in tail.phases] == ["lift_to_H", "loaded_measure"]
    assert tail.poll() == []


def test_watcher_publishes_snapshot(runs):
    ep_dir = make_free(runs)
    store = live.Store([runs])
    hub = live.Hub()
    key = store.key_of(ep_dir)
    q = hub.subscribe(key)
    w = live.Watchers(store, hub).ensure(key)
    try:
        ev = q.get(timeout=3)
        assert ev["type"] == "snapshot" and ev["episode"]["key"] == key
    finally:
        w.stopped.set()
        hub.unsubscribe(key, q)
    time.sleep(0.1)
