"""Offline tests of the Code-as-Policies baseline (forceharness/baselines/code_as_policies.py): FakeSim and a fake model;
no network, no API key, no robosuite."""
import json
from pathlib import Path

import pytest
import yaml

from forceharness.agent.loop import Episode, EpisodeConfig
from forceharness.baselines.code_as_policies import (TRUTH_KEYS, FakeProgramLLM, ProgramEpisode, ProgramRejected, api_doc, check_program,
                                        extract_program, receipt, scrub_truth)
from forceharness.baselines.expert import ScriptedAgent
from forceharness.agent.tools import tool_schemas
from forceharness.primitives.report import ContactEvent, ContactReport
from forceharness.eval.runner import expand_matrix, run_matrix, run_one
from forceharness.translation.base import kinematic_summary
from forceharness.backends.fake import FakeSim
from forceharness.hiddenphys.tasks import get_task

PROG = Path(__file__).parent / "fixtures" / "programs"
FORCE_KEYS = {"events", "quantities", "derived", "wrench_rows", "pos_rows", "final_wrench", "final_wrench_net", "peak_force_N",
              "peak_torque_Nm", "mean_force_in_contact_N", "torque_z_peak_Nm", "payload_estimate_kg"}


def _keys(o, acc=None):
    acc = set() if acc is None else acc
    if isinstance(o, dict):
        for k, v in o.items():
            acc.add(k)
            _keys(v, acc)
    elif isinstance(o, list):
        for v in o:
            _keys(v, acc)
    return acc


def _ep(tmp_path, interface="force", seed=1, program=None, client=None, **kw):
    sim = FakeSim(get_task("weight_sort"), seed=seed, scene_noise_pos=0.002)
    cfg = EpisodeConfig(task="weight_sort", model="fake", fmt="cap", vision=False, interface=interface, seed=seed, run_name="t",
                        out_dir=str(tmp_path), **kw)
    if client is None and program is None:
        client = FakeProgramLLM("weight_sort", interface, program_dirs=[PROG])
    return ProgramEpisode(sim, client, cfg, harness="cap", program=program)


# ----------------------------------------------------------------------------- parsing and static checks
def test_extract_program_variants():
    assert extract_program("Here:\n```python\nx = 1\n```\n") == ("x = 1", 1)
    code, n = extract_program("old:\n```python\na = 1\n```\nfull program:\n```python\na = 1\nb = 2\n```")
    assert code == "a = 1\nb = 2" and n == 2
    assert extract_program("```\ny = 2\n```")[0] == "y = 2"
    assert extract_program("import math\nx = math.pi")[0].startswith("import math")
    assert extract_program("I cannot do that.")[0] is None
    assert extract_program("```python\nx = 1\ny = 2")[0] == "x = 1\ny = 2"          # output truncated, no closing fence


@pytest.mark.parametrize("bad", ["robot.move_to.__closure__", "import os", "from sys import path", "x = robot._ep",
                                 "().__class__", "import forceharness"])
def test_check_program_rejects(bad):
    with pytest.raises(ProgramRejected):
        check_program(bad)


def test_check_program_allows_common_code():
    check_program("import numpy as np\nfrom math import pi\nimport json, collections\n"
                  "class A:\n    def __init__(self):\n        self._x = 1\n\n"
                  "if __name__ == '__main__':\n    print(np.zeros(3), pi, A())\n")


# ----------------------------------------------------------------------------- results: pose has kinematics only, force adds force fields; neither has ground truth
def _rep(**kw):
    rows = [[i * 0.05, 0.1, -0.2, 6.0, 0.01, 0.02, 0.0] for i in range(30)]
    d = dict(primitive="move_to", params={"x": 0.1, "y": 0.0, "z": 0.95}, termination="force_cap", t_start=0.0, t_end=1.5,
             start_pos=[0.1, 0.0, 1.0], end_pos=[0.1, 0.0, 0.97], commanded_disp=[0, 0, -0.05], actual_disp=[0, 0, -0.03],
             events=[ContactEvent(0.5, "contact_start", 6.0, "reaction force mainly along +z", [0.1, 0, 0.97])],
             wrench_rows=rows, pos_rows=[[0.1, 0.0, 0.97]] * 30, final_wrench=[0.1, -0.2, 26.0, 0.01, 0.02, 0.0], peak_force=26.0,
             peak_torque=0.02, mean_force_in_contact=12.0, n_steps=30, gripper_opening=0.04, yaw_deg=0.0,
             derived={"torque_z_peak_Nm": 0.3, "payload_estimate_kg": 0.6, "rotated_deg": 0.0, "note": "n", "opening_m": 0.04},
             quantities=[{"name": "payload_mass", "value": 0.6, "unit": "kg"}],
             truth={"contact_fraction": 0.5, "task_state": {"label": "heavy", "mass": 0.6}})
    d.update(kw)
    return ContactReport(**d)


def test_pose_receipt_is_kinematic_only_and_matches_none_summary():
    rep = _rep()
    r = receipt(rep, "pose", "minimal")
    assert r["result"] == "stopped early (motion blocked, safety stop)"
    assert not (FORCE_KEYS & set(r)) and not (TRUTH_KEYS & _keys(r))
    assert "heavy" not in json.dumps(r) and "26.0" not in json.dumps(r)
    # item by item the same as the none|pose tool result: derived keys written by kinematic_summary == derived keys in the receipt
    summ = kinematic_summary(rep, hide_force=True, proprio="minimal")
    shown = {k for k in rep.derived if any(ln.startswith(k + "=") for ln in summ.splitlines())}
    assert shown == {k for k in rep.derived if k in r} == {"rotated_deg", "note"}
    assert r["short_of_target_m"] == pytest.approx(0.02, abs=1e-6)
    grasp = receipt(_rep(primitive="grasp", params={}, termination="completed", derived={"opening_m": 0.04, "closed_on_something": True}),
                    "pose", "minimal")
    assert grasp["closed_on_something"] is True and grasp["opening_m"] == 0.04


def test_force_receipt_adds_force_fields_but_no_truth():
    r = receipt(_rep(), "force", "minimal")
    assert r["result"] == "force_cap" and r["derived"]["payload_estimate_kg"] == 0.6
    assert r["quantities"][0]["name"] == "payload_mass" and r["final_wrench"][2] == 26.0 and len(r["wrench_rows"]) == 30
    assert r["events"][0]["kind"] == "contact_start"
    assert not (TRUTH_KEYS & _keys(r)) and "heavy" not in json.dumps(r)


def test_scrub_truth():
    fb = {"calls": [{"primitive": "lift", "truth": {"x": 1}, "derived": {"payload_estimate_kg": 0.5, "label": "heavy"}}], "ok": 1}
    assert sorted(scrub_truth(fb)) == ["/calls[0]/derived/label", "/calls[0]/truth"] and fb["ok"] == 1


# ----------------------------------------------------------------------------- API documentation
def test_api_doc_matches_tool_schemas():
    for iface in ("pose", "force"):
        tools = tool_schemas(iface, True, ("point_tool", "pull", "turn"))
        doc = api_doc(tools)
        for t in tools:
            if t["name"] not in ("scene", "done"):
                assert f"robot.{t['name']}(" in doc
        assert "robot.scene()" in doc and "robot.done(" in doc and "robot.state()" in doc
    pose = api_doc(tool_schemas("pose", True, ("pull",)))
    assert "robot.push(" not in pose and "robot.pull(" not in pose and "robot.lift(" not in pose
    assert "robot.move_to(x, y, z, speed=None)" in pose
    assert "robot.lift(height=None, hold_time=None)" in api_doc(tool_schemas("force", True))


def test_system_prompt_per_interface(tmp_path):
    sp = _ep(tmp_path, "pose", 1, program="pass").system_prompt()
    assert "robot.move_to(x, y, z, speed=None)" in sp and "robot.lift(" not in sp and "$" not in sp
    assert "No force or torque information" in sp and "quantities" not in sp
    sf = _ep(tmp_path, "force", 1, program="pass").system_prompt()
    assert "robot.lift(height=None, hold_time=None)" in sf and "quantities" in sf and "$" not in sf


# ----------------------------------------------------------------------------- one episode
def _episode_keys(tmp_path):
    sim = FakeSim(get_task("weight_sort"), seed=1)
    cfg = EpisodeConfig(task="weight_sort", model="scripted", fmt="none", vision=False, interface="force", seed=1, run_name="k",
                        out_dir=str(tmp_path))
    return set(Episode(sim, ScriptedAgent("weight_sort", "force"), cfg).run())


def test_cap_force_program_succeeds_with_one_model_call(tmp_path):
    want = _episode_keys(tmp_path)
    for seed in (1, 2, 3, 4):
        ep = _ep(tmp_path, "force", seed)
        rec = ep.run()
        assert rec["stop_reason"] == "done", (rec["program_error"], rec["program_stdout"])
        assert rec["success"] is True and rec["verdict_correct"] is True and rec["error"] is None
        assert rec["n_model_calls"] == 1 and rec["tokens_in"] > 0 and rec["n_tool_calls"] >= 8
        assert f"{rec['fmt']}|{rec['interface']}" == "cap|force" and rec["condition"] == "weight_sort|fake|cap|novision|force"
        assert want <= set(rec), want - set(rec)
        assert "payload_estimate_kg" in rec["program_stdout"] and rec["program"].startswith("#")
        prompt = ep.messages[1]["parts"][0]["text"]
        assert prompt.startswith("TASK:") and "scene:" in prompt and "label" not in prompt
    assert (ep.out / "program.py").exists() and (ep.out / "record.json").exists() and (ep.out / "visible.json").exists()


def test_cap_pose_program_gets_no_force(tmp_path):
    outcome = {}
    for seed in (1, 2):                       # FakeSim weight_sort: odd seeds heavy, even seeds light; a pose program can only guess light
        ep = _ep(tmp_path, "pose", seed)
        rec = ep.run()
        assert rec["stop_reason"] == "done", rec["program_error"]
        outcome[seed] = rec["success"]
        assert not (FORCE_KEYS & _keys(ep.visible)) and not (TRUTH_KEYS & _keys(ep.visible))
        assert f"{rec['fmt']}|{rec['interface']}" == "cap|pose"
    assert outcome == {1: False, 2: True}


def test_budget_calls_and_swallowed_stop(tmp_path):
    rec = _ep(tmp_path, "force", 1, program=(PROG / "budget_calls.py").read_text(), max_tool_calls=12).run()
    assert rec["stop_reason"] == "max_tool_calls" and rec["n_tool_calls"] == 12 and rec["n_model_calls"] == 0
    swallow = ("n = 0\nwhile n < 50:\n    try:\n        robot.hold(0.05)\n    except BaseException:\n        n += 1\n"
               "print('swallowed', n)\n")
    rec = _ep(tmp_path, "force", 2, program=swallow, max_tool_calls=5).run()
    assert rec["stop_reason"] == "max_tool_calls" and rec["n_tool_calls"] == 5 and "swallowed 50" in rec["program_stdout"]


def test_budget_sim_time(tmp_path):
    rec = _ep(tmp_path, "force", 1, program=(PROG / "budget_sim.py").read_text(), max_sim_time=50.0).run()
    assert rec["stop_reason"] == "max_sim_time" and rec["sim_time_s"] > 50.0 and rec["n_tool_calls"] == 3


def test_wall_clock_alarm_stops_pure_python_loop(tmp_path):
    rec = _ep(tmp_path, "force", 1, program="while True:\n    pass\n", max_wall_time=1.0).run()
    assert rec["stop_reason"] == "max_wall_time" and rec["wall_time_s"] < 10


def test_program_exception_is_a_failure_not_an_error(tmp_path):
    ep = _ep(tmp_path, "force", 1, program=(PROG / "weight_sort__force__keyerror.py").read_text())
    rec = ep.run()
    assert rec["stop_reason"] == "program_error" and rec["success"] is False and rec["error"] is None
    assert "KeyError" in rec["program_error"] and "payload_kg" in ep.traceback_program and "line " in ep.traceback_program
    assert "forceharness" not in ep.traceback_program and "Traceback" in rec["traceback"]


def test_rejected_unavailable_and_bad_calls(tmp_path):
    rec = _ep(tmp_path, "pose", 1, program="import os\nos.system('echo hi')\n").run()
    assert rec["stop_reason"] == "program_error" and rec["program_error"].startswith("rejected")
    rec = _ep(tmp_path, "pose", 2, program="robot.lift(0.1)\n").run()            # no lift on the pose interface
    assert rec["stop_reason"] == "program_error" and "not available" in rec["program_error"]
    rec = _ep(tmp_path, "force", 3, program="r = robot.move_to(0.0, 0.0)\n").run()      # missing argument
    assert rec["stop_reason"] == "program_error" and rec["program_error"].startswith("RobotError")
    rec = _ep(tmp_path, "force", 4, program="open('/tmp/x', 'w')\n").run()             # no open()
    assert rec["stop_reason"] == "program_error" and "NameError" in rec["program_error"]


def test_results_are_read_only_but_json_friendly(tmp_path):
    prog = ("import json\nr = robot.hold(0.1)\ntry:\n    r['result'] = 'x'\nexcept TypeError:\n    print('read-only ok')\n"
            "print('json', len(json.dumps(r)) > 10)\nrobot.done(success=False, verdict='light')\n")
    rec = _ep(tmp_path, "force", 2, program=prog).run()
    assert rec["stop_reason"] == "done" and "read-only ok" in rec["program_stdout"] and "json True" in rec["program_stdout"]
    assert rec["claimed_success"] is False and rec["verdict"] == "light"


def test_no_program_in_reply(tmp_path):
    class Silent:
        def chat(self, messages, tools):
            from forceharness.agent.llm import LLMResponse
            return LLMResponse(text="I would rather not.", tool_calls=[], usage_in=10, usage_out=5, latency_s=0.0)

    rec = _ep(tmp_path, "force", 1, client=Silent()).run()
    assert rec["stop_reason"] == "no_program" and rec["n_model_calls"] == 1 and rec["error"] is None and not rec["success"]


# ----------------------------------------------------------------------------- matrix runner
def _fake_model(harness, dirs):
    return dict(name=f"fake-{harness}", provider="fake", harness=harness, fake_programs=dirs)


def test_run_one_dispatches_cap_harness(tmp_path):
    mx = dict(run_name="t", n_episodes=1, seed_base=2,
              cells=[dict(task="weight_sort", model=_fake_model("cap", str(PROG)), fmt="none", vision=False, interface="force")])
    jobs = expand_matrix(mx)
    assert jobs[0]["cfg"]["fmt"] == "cap"                     # the fmt slot carries the harness name, so conditions cannot collide with none|force
    jobs[0]["cfg"]["out_dir"] = str(tmp_path)
    rec = run_one(jobs[0]["cfg"], jobs[0]["model"], backend="fake")
    assert rec["success"] and rec["harness"] == "cap" and rec["n_model_calls"] == 1, rec.get("error")
    assert rec["condition"] == "weight_sort|fake-cap|cap|novision|force"


def test_run_matrix_rejects_unknown_harness(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text(yaml.safe_dump(dict(run_name="t", n_episodes=1, seed_base=2, cells=[
        dict(task="weight_sort", model=_fake_model("other", str(PROG)), fmt="other", vision=False, interface="force")])))
    with pytest.raises(SystemExit):
        run_matrix(str(p), workers=1, backend="fake", out_dir=str(tmp_path))


class _ClientLike:
    """Mimics LLMClient: has its own .chat and also a .client that is not callable (the OpenAI SDK object in a real
    LLMClient)."""

    def __init__(self, inner):
        self.inner = inner
        self.client = object()
        self.n = 0

    def chat(self, messages, tools):
        self.n += 1
        return self.inner.chat(messages, tools)


def test_real_client_is_not_unwrapped_twice(tmp_path):
    """Regression: an LLMClient must be used as is, and an LLMAgent unwrapped exactly once. Unwrapping an LLMClient once
    more reaches the SDK object and fails with "'Chat' object is not callable"."""
    from forceharness.agent.loop import LLMAgent

    for wrap in (lambda c: c, LLMAgent):
        cl = _ClientLike(FakeProgramLLM("weight_sort", "force", program_dirs=[PROG]))
        rec = _ep(tmp_path, "force", 1, client=wrap(cl)).run()
        assert rec["error"] is None and rec["stop_reason"] == "done", (rec["error"], rec["program_error"])
        assert rec["n_model_calls"] == 1 and cl.n == 1
