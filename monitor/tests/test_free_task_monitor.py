"""Free task (runner scripts/real/free_task.py): the operator's text is the task, every primitive is open.

The record (conftest.make_free) mirrors what the runner writes (episode.json free_task/1, relay obs/cmd files,
the object list in each tool result); the monitor never imports the runner."""
import json

import pytest

from conftest import RECEIPT_A, TASK, make_free
from fhmon import episodes, live, receipts
from fhmon.sessions import LaunchError
from test_server_boundary import server  # noqa: F401  (fixture)


def test_receipt_split_knows_the_still_reading_and_the_object_list():
    rows = receipts.split_lines(RECEIPT_A)

    def kind(head):
        return next(r["kind"] for r in rows if r["text"].strip().startswith(head))
    assert kind("still reading") == "force" and kind("net force peak") == "force"
    # Runner wording (forceharness/real/receipt_text.py): the object list heads of the runner
    assert kind("#1 center") == "scene" and kind("Scene objects") == "scene" and kind("Table z") == "scene"
    assert kind("[Photos this turn") == "scene"
    assert kind("primitive=move_by") == "motion"


def test_free_adapter_reads_the_task_text_and_the_objects_of_each_turn(runs):
    ep_dir = make_free(runs, scored="success")
    paths = episodes.Paths([runs])
    ep = episodes.adapter_for(json.loads((ep_dir / "episode.json").read_text())).load(ep_dir, paths.rel(ep_dir), paths)
    assert ep["family"] == "free" and ep["task_text"] == TASK and "plan" not in ep
    assert ep["result"]["success"] is True and ep["result"]["outcome"] == "success"
    t0, t1 = ep["turns"]
    assert t0["scene"]["objects"][0]["id"] == 1 and t0["scene"]["synthetic"] is True
    assert t0["decision"]["note"] == "lift to weigh" and t0["translation"]["has_force"]
    assert "weighs about 320 g" in t0["translation"]["force_text"] and t1["action"]["status"] == "done"


def test_free_task_needs_its_text_and_passes_it_as_one_argument(server):  # noqa: F811
    app, _ = server
    req = {"task": "free", "conditions": ["A"], "mode": "demo", "model": {"source": "mock"}}
    with pytest.raises(LaunchError, match="task for this episode"):
        app.sessions.plan(req)
    s = app.sessions.plan(dict(req, instruction="--live " + TASK))[0]
    assert "--instruction=--live " + TASK in s.command and "--demo-model" in s.command   # braces untouched, never a flag
    assert s.instruction.startswith("--live")
    assert s.command[1].endswith("scripts/real/free_task.py")
    ext = app.sessions.plan(dict(req, instruction=TASK, model={"source": "external"}))[0]   # the runner carries it
    assert "--instruction=" + TASK in ext.command
    with pytest.raises(LaunchError, match="is fixed in the task file"):                      # a fixed task keeps its goal
        app.sessions.plan({"task": "screw_insert", "conditions": ["A"], "mode": "demo", "model": {"source": "external"},
                           "instruction": "x"})


def test_compare_keeps_a_and_b_on_the_same_task_text(runs):
    make_free(runs, "free_1_A", scored="success")
    make_free(runs, "free_1_B", condition="none_guard", scored="failed")
    make_free(runs, "free_2_A", task="Stand the bottle up", scored="failed")
    store = live.Store([runs])
    all_rows = live.compare(store.scan(), "free")
    assert {t["text"]: t["n"] for t in all_rows["instructions"]} == {TASK: 2, "Stand the bottle up": 1}
    one = live.compare(store.scan(), "free", instruction=TASK)
    assert one["arms"]["harness"]["n_success"] == 1 and one["arms"]["baseline"]["n_scored"] == 1
    assert len(one["episodes"]) == 2
