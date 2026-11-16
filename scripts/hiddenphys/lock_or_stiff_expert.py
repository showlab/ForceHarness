"""Reference scripts for lock_or_stiff: the expert force controller (F), position-only probes (N*), and a position script
that is told whether the door is locked (O). No model calls.

Every episode starts with the task's pre-grasp (forceharness.hiddenphys.tasks._ls_pregrasp), exactly as in model
episodes: the gripper already holds the handle.

  F         expert force controller: ``pull`` along the tangent at 12.5 N (15 cm, at most 5 times); the door moved ->
            open it and answer stiff; the pull force was reached and the door did not move (stalled) -> answer locked
  O         position + oracle: knows the class; stiff -> pull the door open with 5 mm ``move_by`` steps along the
            tangent + ``rotate_z``; locked -> answer locked without pulling
  N1        position only: treat every door as stiff, pull 2 cm per step, answer locked after 6 steps without motion
  N2        position only: answer locked for every door
  N3, N4    position only, stepped probe: from the same start, command growing distances along the tangent (1..5 mm for
            N3, 0.5..4 mm for N4); the door moved by >= 0.5 deg -> stiff, the limit reached -> locked.
            N3 ("0.5 deg rule") is the "Position only" row of the main table
  N3f, N4f  N3 / N4 with the finest step the scene shows (0.1 deg) as the "moved" threshold ("0.1 deg rule")

The arm stiffness is an experimental factor: kp 150 (soft arm, used for the main table) or kp 1500 (stiff arm, in the
supplement). A script episode succeeds only if the door ends in the right state, the handle is not damaged and the
answer (verdict) is right. The position-only variant used in the main table (N3) was chosen on development seeds before
the test seeds were run.

Usage:
    python scripts/hiddenphys/lock_or_stiff_expert.py F                   # soft arm, test seeds 9100-9149
    python scripts/hiddenphys/lock_or_stiff_expert.py N3 --kp 1500 --seed-start 9100 --n 20
    python scripts/hiddenphys/lock_or_stiff_expert.py O --seeds 9100 9101 --out runs

Prints one JSON line per episode and a summary, and writes the list of records to
<out>/<run-name>/lock_or_stiff_<strategy>_kp<kp>.json (run name ``expert_checks`` by default; FH_CHECK_OUT or
--out-file give an explicit path). scripts/hiddenphys/compare_table.py and expert_table.py read these files.
"""
import _bootstrap  # noqa: F401
import argparse
import math, json, os
from pathlib import Path

import numpy as np
from forceharness.agent.loop import EpisodeConfig
from forceharness.eval.runner import _make_sim
from forceharness.primitives.library import Primitives
from forceharness.hiddenphys.tasks import _ls_door_deg, _ls_pregrasp

EP = dict(scene_noise_pos=0.004, scene_noise_yaw_deg=2.0, f_contact=3.0, f_cap=25.0, force_noise_std=0.5, force_bias_std=0.3)
STRATEGIES = ("F", "O", "N1", "N2", "N3", "N4", "N3f", "N4f")


def find(sc, name):
    return next(o for o in sc if o.get("name") == name)


def tangent(sim, hg):
    rr = (sim.eef_pos - hg)[:2]; tg = np.array([-rr[1], rr[0]]) / np.linalg.norm(rr)
    if tg @ (-sim.eef_pos[:2]) < 0: tg = -tg
    return tg


def setup(seed, iface, kp):
    cfg = EpisodeConfig(task="lock_or_stiff", model="x", fmt="phys2", vision=False, interface=iface, seed=seed, kp=kp, **EP)
    sim = _make_sim("lock_or_stiff", cfg, "robosuite")
    pr = Primitives(sim, interface=iface, f_contact=3.0, f_cap=25.0, extra_tools=("point_tool", "pull"), pos_tol=0.0001, slow_in_contact=True)
    ok = _ls_pregrasp(sim, pr)["grasped"]                       # the same pre-grasp function as at the start of model episodes
    hg = np.array(find(sim.scene(), "door_hinge")["pos"])
    return sim, pr, hg, ok


def pose_open(pr, sim, hg, steps=40, d=0.005):
    """Open the door with the position interface: 5 mm along the current tangent per step, then turn the wrist to follow.
    With 2 cm steps the chord deviates from the arc by about 0.7 mm, which is 17 N of radial force on the stiff arm
    (about 25 N/mm)."""
    for _ in range(steps):
        tg = tangent(sim, hg)
        rad = float(np.linalg.norm((sim.eef_pos - hg)[:2]))
        sp = 0.001 if _ls_door_deg(sim) < 1.0 else 0.005      # pull slowly until the door moves: breakaway and damage are checked at 20 Hz, and at 5 mm/s the stiff arm gains about 6 N per tick
        pr.call("move_by", dx=float(d * tg[0]), dy=float(d * tg[1]), dz=0.0, stop_on_contact=False, speed=sp)
        if _ls_door_deg(sim) > 0.5:                               # while the door is stuck the tangent does not change, so do not turn the wrist (on the stiff arm, turning the wrist while holding the handle builds up force too)
            pr.call("rotate_z", angle_deg=math.degrees(d / rad))
        if _ls_door_deg(sim) > 20.0 or sim.task_state.get("damaged"):
            break


def run(strategy, seed, kp):
    iface = "force" if strategy == "F" else "pose"
    sim, pr, hg, ok = setup(seed, iface, kp)
    st = sim.task_state
    verdict, info = None, dict(broke_in_grasp=bool(st["broke_free"]), door_after_grasp=round(_ls_door_deg(sim), 2))
    if strategy == "F":
        terms = []
        for _ in range(5):
            rep = pr.call("pull", direction=list(tangent(sim, hg)) + [0.0], force=12.5, distance=0.15)
            terms.append(rep.termination)
            if _ls_door_deg(sim) > 20.0 or st["damaged"]:
                break
            if rep.termination == "stalled" and _ls_door_deg(sim) < 1.0:
                break
        verdict = "stiff" if _ls_door_deg(sim) > 1.0 else "locked"
        info.update(pull_terms=terms)
    elif strategy == "O":
        if st["label"] == "stiff":
            pose_open(pr, sim, hg); verdict = "stiff"
        else:
            verdict = "locked"
    elif strategy == "N1":
        moved = False
        for _ in range(6):
            tg = tangent(sim, hg)
            pr.call("move_by", dx=float(0.02 * tg[0]), dy=float(0.02 * tg[1]), dz=0.0, stop_on_contact=False, speed=0.005)
            if st["damaged"]:
                break
            if _ls_door_deg(sim) > 0.5:
                moved = True; break
        if moved:
            pose_open(pr, sim, hg)
        verdict = "stiff" if moved else "locked"
    elif strategy == "N2":
        verdict = "locked"
    else:                                                       # N3 / N4: from the same start, command growing distances along the tangent
        inc, cap = (0.001, 0.005) if strategy.startswith("N3") else (0.0005, 0.004)
        # N3f / N4f (added after the main checks, not registered beforehand): the "moved" threshold is the finest step the
        # scene shows (door_angle_deg has one decimal, so >= 0.1 deg counts as moved). With the 0.5 deg threshold the soft
        # arm's 5 mm probe had already pulled several stiff doors past breakaway, but they moved only 0.2-0.4 deg (the
        # arm's stored deflection runs out as soon as the door gives) and were judged locked.
        moved_deg = 0.1 if strategy.endswith("f") else 0.5
        start = sim.eef_pos.copy(); tg = tangent(sim, hg)
        s, moved = inc, False
        while s <= cap + 1e-9 and not st["damaged"]:
            tgt = start + np.array([tg[0], tg[1], 0.0]) * s
            pr.call("move_to", x=float(tgt[0]), y=float(tgt[1]), z=float(tgt[2]), speed=0.001)   # slow probe (as in pose_open)
            if round(_ls_door_deg(sim), 1) >= moved_deg:
                moved = True; break
            s += inc
        info.update(probe_cmd_mm=round(s * 1000, 1))
        if moved and not st["damaged"]:
            pose_open(pr, sim, hg)
        if not moved and not st["damaged"]:                    # judged locked: go back to the start instead of pushing on
            pr.call("move_to", x=float(start[0]), y=float(start[1]), z=float(start[2]), speed=0.005)
        verdict = "stiff" if moved else "locked"
    door_ok = bool(sim.success())                               # the door ends in the right state and is not damaged
    vc = verdict == st["label"]
    out = dict(strategy=strategy, kp=kp, seed=seed, label=st["label"], verdict=verdict, success=bool(door_ok and vc), door_ok=door_ok,
               verdict_ok=vc, damaged=bool(st["damaged"]), door=round(_ls_door_deg(sim), 1), f_b=st.get("breakaway_N"),
               peak_N=round(float(st["peak_handle_force_N"]), 1), hold_N=st.get("hold_N_max"), broke=st.get("broke_free"), grasp=ok, **info)
    sim.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy", choices=STRATEGIES, help="F = expert force controller, O = position + oracle, N* = position only")
    ap.add_argument("--kp", type=float, default=150.0, help="arm stiffness: 150 = soft arm (main table), 1500 = stiff arm")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit seeds (overrides --seed-start / --n)")
    ap.add_argument("--seed-start", type=int, default=9100, help="first seed (default: the test seeds 9100-9149)")
    ap.add_argument("--n", type=int, default=50, help="number of consecutive seeds")
    ap.add_argument("--out", default="runs", help="output root")
    ap.add_argument("--run-name", default="expert_checks", help="output folder under --out")
    ap.add_argument("--out-file", default=None,
                    help="explicit output path (default: $FH_CHECK_OUT, else <out>/<run-name>/lock_or_stiff_<strategy>_kp<kp>.json)")
    args = ap.parse_args()
    strategy, kp = args.strategy, args.kp
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.n))
    rows = []
    for seed in seeds:
        r = run(strategy, seed, kp); rows.append(r); print(json.dumps(r, ensure_ascii=False), flush=True)
    n = len(rows)
    print(f"SUMMARY lock_or_stiff {strategy} kp={kp:g}: success {sum(r['success'] for r in rows)}/{n}  damaged {sum(r['damaged'] for r in rows)}/{n}  "
          f"verdict_ok {sum(r['verdict_ok'] for r in rows)}/{n}  peak median {np.median([r['peak_N'] for r in rows]):.1f}", flush=True)
    out = Path(args.out_file or os.environ.get("FH_CHECK_OUT") or Path(args.out) / args.run_name / f"lock_or_stiff_{strategy}_kp{int(kp)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), indent=1)
    print(f"records: {out}", flush=True)


if __name__ == "__main__":
    main()
