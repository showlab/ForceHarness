"""Offline checks of the eight ForceBench task adapters: call describe_scene with fake privileged geometry and compare fields
and geometry with ForceBench's specifications; build the scripted agent for every task; check the RGB-only variants, the
ForceBench-only tools, the paper settings read by scripts/forcebench/batch.py, and the defaults of the scripts."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")

from forceharness.backends.forcebench import prims as prims_fb
from forceharness.backends.forcebench.fbsim import FBSim  # noqa: E402
from forceharness.backends.forcebench.scripted import FBScriptedAgent  # noqa: E402
from forceharness.backends.forcebench.session import Session  # noqa: E402
from forceharness.backends.forcebench.tasks import TASKS, get_fb_task  # noqa: E402

ENTS = {
    "weight_sort": {"box": dict(kind="rigid", pos=[0.69, -0.05, 0.003], rpy=[0, 0, math.pi / 2])},
    "rattle_sort": {"box": dict(kind="rigid", pos=[0.44, 0.02, 0.002], rpy=[0, 0, math.pi / 2])},
    "faucet_turn": {"faucet_valve": dict(kind="articulation", pos=[0.653, 0.015, -0.037], rpy=[0, 0, 0],
                                         joint_names=["valve_joint"], joint_pos=[0.0], body_names=[], body_pos=[])},
    "faucet_force_turn": {"faucet_valve": dict(kind="articulation", pos=[0.653, 0.015, -0.037], rpy=[0, 0, 0],
                                               joint_names=["valve_joint"], joint_pos=[0.0], body_names=[], body_pos=[])},
    "stove_turn_on": {"stove_knob": dict(kind="articulation", pos=[0.4433, -0.0001, 0.0205], rpy=[0, 0, 0],
                                         joint_names=["stove_knob_joint"], joint_pos=[0.0], body_names=[], body_pos=[])},
    "stove_press_turn": {"stove_knob": dict(kind="articulation", pos=[0.4433, -0.0001, 0.0205], rpy=[0, 0, 0],
                                            joint_names=["stove_press_joint", "stove_turn_joint"], joint_pos=[0.0, 0.0],
                                            body_names=[], body_pos=[])},
    "radmanso_drawer_sort_table": {"cabinet": dict(kind="articulation", pos=[0.62, -0.275, 0.0], rpy=[0, 0, math.pi],
                                                   joint_names=["PrismaticJoint_middle"], joint_pos=[-0.02], body_names=[], body_pos=[])},
    "nut_thread": {"nut": dict(kind="rigid", pos=[0.49, 0.0, 0.010], rpy=[0, 0, 0]),
                   "target_bolt": dict(kind="rigid", pos=[0.63, 0.05, 0.0], rpy=[0, 0, 0])},
}


def _sim(task_name: str) -> FBSim:
    img = np.zeros((4, 4, 3), np.uint8)
    req = {"step": 0, "rollout_id": f"{task_name}:seed=300",
           "observation": {"state": np.array([0.5, 0, 0.18, math.pi, 0, 0, 1.0], np.float32), "images": {"main": img, "wrist": img},
                           "eef_wrench": np.zeros(6, np.float32), "prompt": "x"},
           "privileged": {"entities": ENTS[task_name]}}
    sess = Session(req["rollout_id"], req, lambda s: None)
    return FBSim(sess, get_fb_task(task_name), seed=300, scene_noise_pos=0.0, scene_noise_yaw_deg=0.0)


@pytest.mark.parametrize("name", sorted(TASKS))
def test_scene_and_agent(name):
    sim = _sim(name)
    scene = sim.scene()
    assert scene and all("name" in o for o in scene)
    assert "Task details" in get_fb_task(name).instruction
    FBScriptedAgent(name, "force")


def test_geometry_matches_spec():
    lever = _sim("faucet_turn").scene()[0]
    assert np.allclose(lever["pivot"], [0.663, 0.015, -0.047], atol=1e-4)
    assert np.allclose(lever["grasp_point"], [0.66291, 0.00113, 0.043], atol=2e-4)         # hinge + (-0.09 mm, -13.9 mm, 9 cm)
    h = _sim("radmanso_drawer_sort_table").scene()[0]
    assert np.allclose(h["pos"], [0.6202, -0.1611 + 0.02, 0.2188], atol=2e-4)              # 2 cm open -> handle 2 cm further toward +y
    bolt = [o for o in _sim("nut_thread").scene() if o["name"] == "bolt"][0]
    assert abs(bolt["target_nut_bottom_z"] - 0.032) < 1e-9 and abs(bolt["shank_top_z"] - 0.035) < 1e-9


# ------------------------------------------------------------------ RGB only
from forceharness.backends.forcebench.tasks import TASKS_RGB  # noqa: E402

_CALIB = {"main": {"K": [[256.5, 0, 112], [0, 256.5, 112], [0, 0, 1]], "pos": [0.955, 0.0, 0.595],
                   "R_world_cam": [[0.0, 0.8141, -0.5807], [1.0, 0.0, 0.0], [0.0, -0.5807, -0.8141]], "width": 224, "height": 224},
          "wrist": {"K": [[192.4, 0, 112], [0, 192.4, 112], [0, 0, 1]], "pos": [0.66, 0.0, 0.25],
                    "R_world_cam": [[0.0, -0.8678, -0.4969], [-1.0, 0.0, 0.0], [0.0, 0.4969, -0.8678]], "width": 224, "height": 224}}


def _rgb_sim(task_name: str) -> FBSim:
    img = np.zeros((224, 224, 3), np.uint8)
    req = {"step": 0, "rollout_id": f"{task_name}:seed=300",
           "observation": {"state": np.array([0.5, 0, 0.18, math.pi, 0, 0, 1.0], np.float32), "images": {"main": img, "wrist": img},
                           "eef_wrench": np.zeros(6, np.float32), "prompt": "x"},
           "camera_calibration": _CALIB, "diag_truth": {"entities": ENTS[task_name]}}
    sess = Session(req["rollout_id"], req, lambda s: None)
    return FBSim(sess, get_fb_task(task_name, "rgb"), seed=300)


@pytest.mark.parametrize("name", sorted(TASKS_RGB))
def test_rgb_task_has_no_coordinates(name):
    sim = _rgb_sim(name)
    task = get_fb_task(name, "rgb")
    assert task.extra.get("perception") == "rgb" and "locate" in task.extra.get("fb_extra_tools", ())
    for o in sim.scene():                     # only the table height and the camera note: no object positions
        assert not ({"pos", "grasp_point", "pivot"} & set(o)), o
    assert "No positions are given" in task.instruction or name == "weight_sort"
    FBScriptedAgent(name, "force", perception="rgb", sim=sim)


def test_locate_planes_roundtrip():
    sim = _rgb_sim("rattle_sort")
    for cam, p in (("main", [0.62, -0.10, 0.03]), ("wrist", [0.60, 0.02, 0.0])):
        uv = sim.project(cam, p)
        assert np.allclose(sim.locate_on_plane(cam, uv[0], uv[1], p[2]), p, atol=2e-4)           # horizontal plane z (test rotations have 4 decimals)
        assert np.allclose(sim.locate_on_plane(cam, uv[0], uv[1], p[1], axis=1), p, atol=2e-4)   # vertical plane y
        assert np.allclose(sim.locate_on_plane(cam, uv[0], uv[1], p[0], axis=0), p, atol=2e-4)   # vertical plane x


def test_projection_fit_recovers_box():
    from forceharness.backends.forcebench.vision import _hull_mask, box_outline, fit_projection

    sim = _rgb_sim("rattle_sort")
    box = box_outline((0.12, 0.054, 0.0233), 0.0)
    truth = np.array([0.452, 0.013])
    uv = np.array([sim.project("wrist", p) for p in box(*truth)])
    mask = _hull_mask((224, 224), uv)
    xy, iou = fit_projection(sim, "wrist", mask, box, truth + np.array([0.012, -0.008]))
    assert iou > 0.95 and np.linalg.norm(xy - truth) < 0.0015


def test_locate_tool_planes(tmp_path):
    """In strict mode the model fills in every field (unused ones with 0): `plane` decides."""
    from forceharness.agent.llm import ToolCall
    from forceharness.agent.loop import EpisodeConfig
    from forceharness.backends.forcebench.episode import FBEpisode

    sim = _rgb_sim("radmanso_drawer_sort_table")
    sim.step = lambda *a, **k: sim.wrench()          # no evaluator

    class _Agent:
        kind = "llm"

    cfg = EpisodeConfig(task="fb_drawer", model="x", fmt="phys2", vision=True, interface="force", seed=300,
                        out_dir=str(tmp_path), run_name="t")
    ep = FBEpisode(sim, _Agent(), cfg)
    assert "locate" in [t["name"] for t in ep.tools]
    # what a model actually sent: {"z": 0.219, "x": 0, "y": 0} (no plane) -> horizontal plane z = 0.219
    out = ep.run_tool(ToolCall("a", "locate", {"camera": "main", "u": 35, "v": 111, "z": 0.219, "x": 0, "y": 0}))
    assert "on z = 0.219" in out[0]["text"]
    assert abs(ep.loc_log[-1]["p"][2] - 0.219) < 1e-4
    # strict mode, every field filled: with plane = 'z' the value is ignored
    out = ep.run_tool(ToolCall("b", "locate", {"camera": "main", "u": 35, "v": 111, "z": 0.219, "plane": "z", "value": 0}))
    assert "on z = 0.219" in out[0]["text"]
    # vertical plane: plane = 'y', z is ignored
    out = ep.run_tool(ToolCall("c", "locate", {"camera": "main", "u": 100, "v": 120, "z": 0, "plane": "y", "value": -0.16}))
    assert "on y = -0.160" in out[0]["text"]
    assert abs(ep.loc_log[-1]["p"][1] + 0.16) < 1e-4
    out = ep.run_tool(ToolCall("d", "locate", {"camera": "main", "u": 100, "v": 120, "plane": "x"}))
    assert "needs value" in out[0]["text"]
    out = ep.run_tool(ToolCall("e", "locate", {"camera": "main", "u": 100, "v": 120}))
    assert "on z = 0.000" in out[0]["text"]


def test_blank_images_guard(tmp_path):
    from forceharness.agent.loop import EpisodeConfig
    from forceharness.backends.forcebench.episode import FBEpisode

    sim = _rgb_sim("stove_turn_on")                  # the test images are all zeros: they must count as blank
    sim.step = lambda *a, **k: sim.wrench()

    class _Agent:
        kind = "llm"

        def decide(self, ctx):                       # must not be called
            raise AssertionError("model must not be called on blank images")

    cfg = EpisodeConfig(task="fb_stove", model="x", fmt="phys2", vision=True, interface="force", seed=300,
                        out_dir=str(tmp_path), run_name="t")
    ep = FBEpisode(sim, _Agent(), cfg)
    assert len(ep.blank_images()) == 2
    rec = ep.run_safe()
    assert rec["stop_reason"] == "blank_images" and rec["n_model_calls"] == 0
    sim.images_raw["main"] = np.random.default_rng(0).integers(0, 255, (224, 224, 3), dtype=np.uint8)
    assert len(ep.blank_images()) == 1


def _ep(tmp_path, name, interface, fmt):
    from forceharness.agent.loop import EpisodeConfig
    from forceharness.backends.forcebench.episode import FBEpisode
    from forceharness.backends.forcebench.tasks import for_interface

    sim = _rgb_sim(name)
    sim.task = for_interface(sim.task, interface)
    sim.step = lambda *a, **k: sim.wrench()

    class _Agent:
        kind = "llm"

    cfg = EpisodeConfig(task=f"fb_{name}", model="x", fmt=fmt, vision=True, interface=interface, seed=300,
                        out_dir=str(tmp_path), run_name="t")
    return FBEpisode(sim, _Agent(), cfg)


def test_no_force_tools_and_text(tmp_path):
    """No force: no tools that close a loop on force, move_turn instead, and the instruction names no unavailable tool.
    With force: tool menu and instruction unchanged."""
    for name in ("stove_press_turn", "nut_thread"):
        ep = _ep(tmp_path, name, "pose", "none")
        tools = [t["name"] for t in ep.tools]
        assert "move_turn" in tools and "locate" in tools
        assert not ({"press_turn", "pull", "push", "lift", "move_until_contact", "search", "insert", "probe", "slide"} & set(tools))
        assert "press_turn" not in ep.task.instruction and "move_turn" in ep.task.instruction
        assert "by touch" not in ep.task.instruction
        ep = _ep(tmp_path, name, "force", "phys2")
        tools = [t["name"] for t in ep.tools]
        assert "press_turn" in tools
        assert ("move_turn" in tools) == (name == "nut_thread")      # Nut Thread: force motions are a superset of the control's; Stove Press: not given
        assert ep.task.instruction == get_fb_task(name, "rgb").instruction
    ep = _ep(tmp_path, "radmanso_drawer_sort_table", "pose", "none")
    tools = [t["name"] for t in ep.tools]
    assert "point_tool" in tools and "pull" not in tools and "move_turn" not in tools
    ep = _ep(tmp_path, "faucet_force_turn", "pose", "none")
    assert "turn" in [t["name"] for t in ep.tools]


def test_move_turn_holds_commanded_height():
    """move_turn presses the commanded height down and keeps it there while turning; it reads no force (all-zero readings
    give the same trajectory)."""
    from forceharness.backends.forcebench.prims import FBPrimitives

    sim = _rgb_sim("stove_press_turn")
    log = []

    def fake_step(dpos, drot=None, grip=None):
        log.append((np.asarray(dpos, dtype=float).copy(), None if drot is None else np.asarray(drot, dtype=float).copy()))
        sim.step_index += 1                          # the hand does not move (held up by the knob): the command stays below it
        return sim.wrench()

    sim.step = fake_step
    prims = FBPrimitives(sim, interface="pose", f_contact=3.0, f_cap=25.0, extra_tools=("move_turn",))
    rep = prims.call("move_turn", angle_deg=30.0, press_dz=-0.015, turn_dz=0.0, rate_deg_s=0, press_time=0)
    leads = np.array([d[0][2] for d in log])
    assert rep.termination == "completed"
    assert abs(leads.min() + 0.015) < 1e-6 and abs(leads[-1] + 0.015) < 1e-6        # 15 mm down, kept there to the end
    assert any(d[1] is not None and abs(d[1][2]) > 0 for d in log)                   # it does turn


def test_locate_object_fits_known_shape(tmp_path):
    """Draw a fake bolt: a click a few pixels off still gives its axis within 2 mm; an unknown object name gets a clear reply."""
    from PIL import Image, ImageDraw

    from forceharness.agent.llm import ToolCall
    from forceharness.backends.forcebench.vision import stack_outline, _hull_mask

    ep = _ep(tmp_path, "nut_thread", "force", "phys2")
    tool = next(t for t in ep.tools if t["name"] == "locate_object")
    assert tool["parameters"]["properties"]["object"]["enum"] == ["bolt", "nut"]
    sim = ep.sim
    parts = sim.task.extra["object_models"]["bolt"]
    truth = np.array([0.60, 0.02])
    uv = np.array([sim.project("wrist", p) for p in stack_outline(parts)(*truth)])
    img = np.full((224, 224, 3), (190, 165, 140), np.uint8)
    img[_hull_mask((224, 224), uv)] = (70, 72, 75)
    sim.images_raw["wrist"] = img
    centre = uv.mean(axis=0)
    out = ep.run_tool(ToolCall("a", "locate_object", {"camera": "wrist", "u": centre[0] + 4, "v": centre[1] - 3, "object": "bolt"}))
    assert "fit score" in out[0]["text"]
    assert np.linalg.norm(np.array(ep.loc_log[-1]["p"][:2]) - truth) < 0.002 and ep.loc_log[-1]["score"] > 0.8
    out = ep.run_tool(ToolCall("b", "locate_object", {"camera": "wrist", "u": 100, "v": 100, "object": "lemon"}))
    assert "must be one of" in out[0]["text"]


def test_nut_instruction_height_rule_matches_geometry():
    """The G + 0.022 / G + 0.025 height rule of the instruction must match the task constants (an earlier wording, 8 mm, did
    not match a mid-height grasp)."""
    from forceharness.backends.forcebench import tasks as T

    text = T.NUT_THREAD_RGB.instruction
    assert "8 mm below" not in text
    assert "G + 0.022" in text and "G + 0.025" in text
    assert T.NT_TARGET_BOTTOM_Z - T.NT_NUT_BOTTOM_Z == pytest.approx(0.022)
    assert T.NT_SHANK_TOP_Z - T.NT_NUT_BOTTOM_Z == pytest.approx(0.025)


def test_press_turn_zero_stop_depth_means_no_stop():
    """In strict mode a model that does not use stop_depth sends 0, which must not mean "stop after 0 m"."""
    src = (Path(prims_fb.__file__)).read_text(encoding="utf-8")
    assert "float(stop_depth) <= 0" in src


def test_press_turn_numbers_reach_the_model():
    """The phys2 tool result must show the resisting-torque numbers (they once stayed in `derived`, leaving only a note)."""
    from types import SimpleNamespace

    from forceharness.translation.quantities import PhysicsInjector

    rep = SimpleNamespace(primitive="press_turn", events=[], final_wrench=[0.0] * 6, final_wrench_net=[0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                          derived=dict(turned_deg=-40.0, press_depth_m=0.008, press_force_mean_N=8.02, resisting_torque_mean_Nm=0.112,
                                       resisting_torque_se_Nm=0.007, resisting_torque_n=33, resisting_torque_peak_Nm=0.19))
    rep.quantities = prims_fb._turn_quantities(rep)
    text = PhysicsInjector(level=2).render_force(rep, {})[0]["text"]
    assert "resisting torque" in text and "0.112" in text and "0.007" in text
    assert "press depth" in text and "8 mm" in text


def test_stove_press_force_note_only_in_force_text():
    """The force instruction adds only the note on the force reading and the tool; the no-force instruction has neither the
    note nor press_turn."""
    from forceharness.backends.forcebench import tasks as T

    force = T.for_interface(T.STOVE_PRESS_TURN_RGB, "force").instruction
    pose = T.for_interface(T.STOVE_PRESS_TURN_RGB, "pose").instruction
    assert T._SPT_READING_NOTE in force and T._SPT_READING_NOTE not in pose
    assert "press_turn" not in pose
    assert "stop_on_resistance_drop" not in prims_fb.PRESS_TURN_SCHEMA["parameters"]["properties"]


def test_press_turn_reports_resistance_per_side():
    """Per-side reading: the median for each side once measured, an explicit "not measured yet" otherwise, and no longer the
    mean of the current turn (models compared directions with it)."""
    from types import SimpleNamespace

    from forceharness.translation.quantities import PhysicsInjector

    rep = SimpleNamespace(primitive="press_turn", events=[], final_wrench=[0.0] * 6, final_wrench_net=[0.0, 0.0, 8.0, 0.0, 0.0, 0.0],
                          derived=dict(press_depth_m=0.008, press_force_mean_N=8.02, resisting_torque_mean_Nm=0.05, resisting_torque_peak_Nm=0.19,
                                       resistance_ccw_side_n=12, resistance_ccw_side_Nm=0.081, resistance_ccw_side_se_Nm=0.006, resistance_cw_side_n=0))
    rep.quantities = prims_fb._turn_quantities(rep)
    text = PhysicsInjector(level=2).render_force(rep, {})[0]["text"]
    assert "resistance ccw side" in text and "0.081" in text and "12 samples" in text
    assert "resistance cw side" in text and "n/a" in text and "not measured yet" in text
    assert "resisting torque" not in text


# ------------------------------------------------------------------ paper settings (scripts/forcebench/batch.py) and script defaults
_REPO = Path(__file__).resolve().parents[1]
_FB_SCRIPTS = _REPO / "scripts" / "forcebench"


def _batch_module():
    import importlib.util

    spec = importlib.util.spec_from_file_location("fb_batch", _FB_SCRIPTS / "batch.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_paper_settings_resolve_to_valid_jobs():
    """Every row and the example job list resolve to options the policy server knows; the per-task overrides name real
    primitive attributes; the faucet tasks get the paper's `turn` report."""
    import json
    import re

    pytest.importorskip("yaml")
    from forceharness.backends.forcebench.prims import FBPrimitives

    B = _batch_module()
    settings = B.load_settings(B.DEFAULT_SETTINGS)
    assert set(settings["tasks"]) == set(TASKS_RGB)
    flags = set(re.findall(r'add_argument\(\s*"(--[a-z-]+)"', (_FB_SCRIPTS / "policy_server.py").read_text(encoding="utf-8")))
    jobs = [j for row in settings["rows"] for j in B.row_jobs(settings, row, list(settings["tasks"]))]
    jobs += B.load_file(_REPO / "configs" / "forcebench" / "jobs_example.yaml")
    for job in jobs:
        r = B.resolve(settings, job)
        assert {a for a in r["args"] if a.startswith("--")} <= flags, r["args"]
        perception = [r["args"][i + 1] for i, a in enumerate(r["args"]) if a == "--perception"][-1]
        assert ("--privileged_geometry" in r["env"]["EVAL_ARGS"].split()) == (perception == "privileged"), job
    sim = _rgb_sim("faucet_turn")
    sim.step = lambda *a, **k: sim.wrench()          # no evaluator
    prims = FBPrimitives(sim, interface="force", f_contact=3.0, f_cap=25.0)
    for t in settings["tasks"].values():
        for k in (t or {}).get("prim_overrides") or {}:
            assert hasattr(prims, k), k
    r = B.resolve(settings, B.row_jobs(settings, "sol_force", ["faucet_force_turn"])[0])
    over = json.loads(r["args"][r["args"].index("--prim-overrides") + 1])
    assert over == {"max_lead_contact": 0.015, "turn_quantities": False}
    assert r["env"]["SEED"] == "300" and r["env"]["N"] == "10" and r["env"]["ENV_SPACING"] == "30"
    assert r["env"]["EVAL_ARGS"] == "--no-privileged_geometry"
    assert r["args"][r["args"].index("--interface") + 1] == "force" and r["args"][r["args"].index("--fmt") + 1] == "phys2"
    r = B.resolve(settings, B.row_jobs(settings, "astra_no_force", ["nut_thread"])[0])
    assert r["args"][r["args"].index("--interface") + 1] == "pose" and r["args"][r["args"].index("--fmt") + 1] == "none"


def test_script_defaults_are_the_paper_setting():
    """RGB-only perception and no privileged geometry by default; privileged mode stays available; no dev-only probe agent."""
    server = (_FB_SCRIPTS / "policy_server.py").read_text(encoding="utf-8")
    evaluator = (_FB_SCRIPTS / "eval_forcebench.py").read_text(encoding="utf-8")
    assert 'add_argument("--perception", default="rgb", choices=["privileged", "rgb"]' in server
    assert 'choices=["scripted", "llm"]' in server and "probe" not in server
    assert '"--privileged_geometry", action=argparse.BooleanOptionalAction, default=False' in evaluator
    assert "MIT License" in evaluator and "ForceBench contributors" in evaluator


def test_shell_scripts_never_accept_the_eula(tmp_path):
    """The scripts parse; they only check the NVIDIA EULA variables, never set them; the driver-check workaround is opt-in."""
    import os
    import re
    import shutil
    import subprocess

    if shutil.which("bash") is None:
        pytest.skip("bash not available")
    for name in ("env.sh", "install.sh", "run.sh", "simpy_kit.sh"):
        path = _FB_SCRIPTS / name
        assert subprocess.run(["bash", "-n", str(path)], capture_output=True).returncode == 0, name
        for line in path.read_text(encoding="utf-8").splitlines():
            code = line.split("#", 1)[0].strip()
            if code.startswith("echo"):
                continue
            assert not re.search(r"\b(ACCEPT_EULA|OMNI_KIT_ACCEPT_EULA|ISAACSIM_ACCEPT_EULA)=", code), (name, line)
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(tmp_path), "FB_BASE": str(tmp_path / "fb")}
    probe = '. "$1"; printf "%s|" "$FB_KIT_ARGS"; fb_require_eula 2>/dev/null; echo "$?"'
    out = subprocess.run(["bash", "-c", probe, "x", str(_FB_SCRIPTS / "env.sh")], env=env, capture_output=True, text=True)
    assert out.stdout.strip() == "|3", out
    out = subprocess.run(["bash", "-c", probe, "x", str(_FB_SCRIPTS / "env.sh")], capture_output=True, text=True,
                         env=dict(env, FB_SKIP_DRIVER_CHECK="1", OMNI_KIT_ACCEPT_EULA="YES"))
    assert out.stdout.strip() == "--/rtx/verifyDriverVersion/enabled=false|0" and "WARNING" in out.stderr, out
