"""Replay recorded demonstrations and check that they reproduce the recorded outcome. No model calls.

For each npz: build a new ForceSim with the recording's seed and episode settings (construction resets to the
recording's starting point), redo the pre-grasp first if the recording started after it (lock_or_stiff), then replay the
normalised actions in order, setting the stiffness kp and the hand weld to their recorded values before each tick; ticks
that came through ForceSim.step also run the task's step_hook. Finally compare success with the recorded success and the
final end-effector position with the recorded one.

    python scripts/hiddenphys/replay_demos.py demos/com_sort/s2000*.npz
"""
import _bootstrap  # noqa: F401
import argparse
import json
import sys

import numpy as np


def replay(path: str) -> dict:
    from forceharness.agent.loop import EpisodeConfig
    from forceharness.eval.runner import _make_sim
    from forceharness.backends.mujoco.recorder import load_demo

    d = load_demo(path)
    m = d["meta"]
    ep = dict(m["episode_cfg"])                    # render as during recording: the forward pass before rendering changes the solver's warm start, which grows over long contact-rich episodes
    cfg = EpisodeConfig(**ep)
    sim = _make_sim(m["task"], cfg, "robosuite")
    if m.get("start_after_pregrasp"):                  # lock_or_stiff: the recording starts after the pre-grasp, so redo it before replaying
        from forceharness.primitives.library import Primitives

        pa = dict(m["prims_args"])
        pr = Primitives(sim, interface=pa["interface"], f_contact=pa["f_contact"], f_cap=pa["f_cap"], extra_tools=tuple(pa["extra_tools"]),
                        pos_tol=pa["pos_tol"], slow_in_contact=pa["slow_in_contact"])
        sim.task.pregrasp_hook(sim, pr)
    try:
        mj, mm, dd = sim.mujoco, sim.m, sim.d
        act = getattr(dd, "eq_active", None)
        act = mm.eq_active if act is None else act
        # Start from the weld state the sim actually has and call set_held_weld only when the state changes: activating it
        # again would rewrite the constraint with the relative pose of that moment, which differs from the recording.
        weld_prev = any((mj.mj_id2name(mm, mj.mjtObj.mjOBJ_EQUALITY, i) or "").startswith("held_weld") and bool(act[i])
                        for i in range(mm.neq))
        kp_prev = None
        for i in range(len(d["action"])):
            kp = d["kp"][i]
            if kp_prev is None or not np.allclose(kp, kp_prev):
                sim.set_kp(kp.astype(float)); kp_prev = kp
            w = bool(d["weld"][i])
            if w != weld_prev:
                sim.set_held_weld(w)
                weld_prev = w
            a = d["action"][i][:sim.action_dim].astype(float)
            sim.obs = sim.env.step(a)[0]
            if d["via_step"][i]:                     # as in the second half of ForceSim.step: count, read force, peaks, task hook
                if sim.action_dim > 6:
                    sim.grip_cmd = float(a[6])
                sim.n_steps += 1
                wr = sim.wrench()
                sim.peak_wrench = max(sim.peak_wrench, wr.fmag)
                hook = getattr(sim.task, "step_hook", None)
                if hook is not None:
                    hook(sim, wr)
        ok = bool(sim.success())
        err_mm = float(np.linalg.norm(sim.eef_pos - d["eef_pos"][-1].astype(float)) * 1000)
        return dict(file=path, task=m["task"], seed=m["seed"], recorded_success=m["success"], replay_success=ok,
                    final_eef_err_mm=round(err_mm, 2), n_steps=len(d["action"]))
    finally:
        sim.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="recorded episodes (npz files written by record_demos*.py)")
    args = ap.parse_args()
    rows = [replay(p) for p in args.files]
    for r in rows:
        print(json.dumps(r, ensure_ascii=False), flush=True)
    n = len(rows)
    same = sum(r["replay_success"] == r["recorded_success"] for r in rows)
    print(f"SUMMARY replay: success matches {same}/{n}; final eef error median "
          f"{np.median([r['final_eef_err_mm'] for r in rows]) if rows else float('nan'):.2f} mm", flush=True)


if __name__ == "__main__":
    main()
