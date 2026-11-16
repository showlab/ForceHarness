import json

from forceharness.backends.fake import FakeSim
from forceharness.hiddenphys.tasks import get_task
from forceharness.agent.loop import Episode, EpisodeConfig
from forceharness.baselines.expert import ScriptedAgent


def run(interface, fmt, seed=3, tmp_path=None):
    task = get_task("weight_sort")
    sim = FakeSim(task, seed=seed, scene_noise_pos=0.002)
    cfg = EpisodeConfig(task="weight_sort", model="scripted", fmt=fmt, vision=False, interface=interface, seed=seed,
                        run_name="t", out_dir=str(tmp_path), max_turns=20)
    ep = Episode(sim, ScriptedAgent("weight_sort", interface), cfg)
    return ep, ep.run()


def test_scripted_force_interface_succeeds(tmp_path):
    for seed in (1, 2, 3, 4):
        ep, rec = run("force", "signals", seed, tmp_path)
        assert rec["stop_reason"] == "done", rec
        assert rec["success"] is True and rec["verdict_correct"] is True, rec
        assert rec["n_tool_calls"] >= 8 and rec["tokens_in"] == 0
        kinds = [r.primitive for r in ep.reports]
        assert "lift" in kinds and "grasp" in kinds and "move_until_contact" in kinds
        lift = [r for r in ep.reports if r.primitive == "lift"][0]
        assert abs(lift.derived["payload_estimate_kg"] - rec["truth"]["mass"]) < 0.05
    assert (tmp_path / "t" / "transcripts").exists()


def test_pose_interface_without_force_only_guesses(tmp_path):
    outcomes = []
    for seed in range(1, 9):
        ep, rec = run("pose", "none", seed, tmp_path)
        assert rec["stop_reason"] == "done"
        outcomes.append(rec["verdict_correct"])
        assert all("move_until_contact" != r.primitive for r in ep.reports)
    assert not all(outcomes)   # without force the expert can only guess light, so the heavy seeds fail


def test_transcript_saved_json(tmp_path):
    ep, rec = run("force", "events", 5, tmp_path)
    t = json.loads((ep.out / "transcript.json").read_text())
    assert t[0]["role"] == "system" and any(m["role"] == "tool" for m in t)
    assert "contact" in json.dumps(t)
