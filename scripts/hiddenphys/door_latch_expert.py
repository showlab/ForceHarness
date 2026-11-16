"""Reference scripts for door_latch: the expert force controller (F), position-only probes (N*), and a position script
that is told the hidden unlock angle (O). No model calls.

Every strategy starts the same way: level the tool in front of the handle, move in, grasp the bar. After that each one
follows the realistic sequence for a lever handle with a latch: press the handle down, keep it pressed while cracking
the door open, turn the handle back, then pull the door open like a door without latch.

  F       expert force controller: ``turn`` the lever up to 45 deg, stopping when the resistance peaks and drops (the
          latch "click"); crack the door open with ``pull`` (8 N, 3 cm); turn the lever back; ``pull`` the door open
          (12 cm, at most 3 times)
  O       position + oracle: knows the unlock angle theta_u; ``turn`` the lever to theta_u + 3 deg, then crack, release
          and pull with ``move_by`` steps along the tangent + ``rotate_z`` (as all position-only strategies do)
  N1, N2  position only: press the lever to 25 deg (N1) or 35 deg (N2), then pull
  N3      position only: press 5 deg further at a time and try a 1 cm pull until the door moves
  N4      position only: press 2 deg further at a time and try a 2 mm pull (needs the arm's own stiffness, ~3 N per mm)
  N4tol   N4 with a 0.1 mm arrival tolerance (with the default 3 mm, a 2 mm test pull does not move the hand at all):
          the "stepped probe", which is the "Position only" row of the main table
  N5      position only, the strategy a model found without force: press to 15 deg, then alternate 3 deg more lever
          with a 1 deg turn of the hand about the door hinge (default turn rate 10 deg/s)
  N5slow  N5 at the model's turn rates (lever 2 deg/s, hinge 1 deg/s)
  N5half  N5slow with 0.5 deg hinge turns: the "alternating probe" of the supplement

The position-only variant used in the main table (N4tol) was chosen on development seeds before the test seeds were run.

Usage:
    python scripts/hiddenphys/door_latch_expert.py F                    # test seeds 9100-9149
    python scripts/hiddenphys/door_latch_expert.py N4tol --seed-start 9100 --n 20
    python scripts/hiddenphys/door_latch_expert.py O --seeds 9100 9101 --out runs

Prints one JSON line per episode and a summary, and writes the list of records to
<out>/<run-name>/door_latch_<strategy>.json (run name ``expert_checks`` by default; FH_CHECK_OUT or --out-file give an
explicit path). scripts/hiddenphys/compare_table.py and expert_table.py read these files. Episode settings follow the
protocol of the paper (EP below, kp 150); success requires the door open beyond 17 deg without handle damage.
"""
import _bootstrap  # noqa: F401
import argparse
import math, json, os
from pathlib import Path

import numpy as np
from forceharness.agent.loop import EpisodeConfig
from forceharness.eval.runner import _make_sim
from forceharness.primitives.library import Primitives

EP = dict(scene_noise_pos=0.004, scene_noise_yaw_deg=2.0, f_contact=3.0, f_cap=25.0, force_noise_std=0.5, force_bias_std=0.3)
R = 0.125
STRATEGIES = ("F", "O", "N1", "N2", "N3", "N4", "N4tol", "N5", "N5slow", "N5half")


def find(sc, name):
    return next(o for o in sc if o.get("name") == name)


POS_TOL = 0.003      # arrival tolerance of move_to; N4tol uses 0.1 mm (with 3 mm, test pulls of 2 mm or 0.5 mm never move the hand)


def setup(seed, iface, pos_tol=None):
    cfg = EpisodeConfig(task="door_latch", model="x", fmt="phys2", vision=False, interface=iface, seed=seed, **EP)
    sim = _make_sim("door_latch", cfg, "robosuite")
    pr = Primitives(sim, interface=iface, f_contact=3.0, f_cap=25.0, extra_tools=("point_tool", "pull", "turn"),
                    pos_tol=POS_TOL if pos_tol is None else pos_tol)
    sc = sim.scene()
    h = np.array(find(sc, "door_handle")["pos"]); hg = np.array(find(sc, "door_hinge")["pos"])
    lever = find(sc, "lever")
    r = (h - hg)[:2]; r /= np.linalg.norm(r); t = np.array([-r[1], r[0]])
    if t @ (-h[:2]) < 0: t = -t
    pre = h.copy(); pre[:2] += 0.12 * t
    pr.call("move_to", x=pre[0], y=pre[1], z=pre[2] + 0.10)
    pr.call("point_tool", direction=[-t[0], -t[1], 0.0])
    pr.call("move_to", x=pre[0], y=pre[1], z=pre[2])
    g = h.copy(); g[:2] -= 0.01 * t
    pr.call("move_to", x=g[0], y=g[1], z=g[2])
    ok = bool(pr.call("grasp").derived.get("closed_on_something"))
    return sim, pr, hg, (np.array(lever["pivot"]), np.array(lever["axis"])), ok


def cur_lever(sim):
    """Current lever axis (it moves with the door, so do not reuse the one read at the start)."""
    lv = find(sim.scene(), "lever")
    return np.array(lv["pivot"]), np.array(lv["axis"])


def lever_deg(sim):
    return float(find(sim.scene(), "lever")["angle_deg"])


def door_deg(sim):
    return float(find(sim.scene(), "door_angle_deg")["value"])


def tangent(sim, hg):
    rr = (sim.eef_pos - hg)[:2]; tg = np.array([-rr[1], rr[0]]) / np.linalg.norm(rr)
    if tg @ (-sim.eef_pos[:2]) < 0: tg = -tg
    return tg


def press_to(pr, lever, theta_deg, rate=None):
    """Turn the lever to theta (by the difference from the current angle, with turn; on the position interface the
    full difference is executed)."""
    pivot, axis = cur_lever(pr.sim)
    d = theta_deg - lever_deg(pr.sim)
    if abs(d) > 0.3:
        kw = {} if rate is None else dict(rate_deg_s=rate)
        pr.call("turn", axis=list(axis), pivot=list(pivot), angle_deg=float(d), **kw)


def pose_pull(pr, sim, hg, steps=10, d=0.02):
    for _ in range(steps):
        tg = tangent(sim, hg)
        rad = float(np.linalg.norm((sim.eef_pos - hg)[:2]))
        pr.call("move_by", dx=float(d * tg[0]), dy=float(d * tg[1]), dz=0.0, stop_on_contact=False)
        pr.call("rotate_z", angle_deg=math.degrees(d / rad))
        if door_deg(sim) > 20.0 or sim.task_state.get("damaged"):
            break


def release_lever(pr, lever):
    """Turn the lever back to 0 (the door has left the frame, so the latch cannot lock again); after that, pull like a
    door without latch."""
    th = lever_deg(pr.sim)
    if th > 0.5:
        pv, ax = cur_lever(pr.sim)
        pr.call("turn", axis=list(ax), pivot=list(pv), angle_deg=-float(th))


def crack_pose(pr, sim, hg, steps=2, d=0.015):
    for _ in range(steps):
        tg = tangent(sim, hg)
        rad = float(np.linalg.norm((sim.eef_pos - hg)[:2]))
        pr.call("move_by", dx=float(d * tg[0]), dy=float(d * tg[1]), dz=0.0, stop_on_contact=False)
        pr.call("rotate_z", angle_deg=math.degrees(d / rad))


def run(strategy, seed):
    """Realistic sequence: press the handle down -> keep it pressed while cracking the door open -> turn the handle
    back -> pull the door open like a door without latch."""
    iface = "force" if strategy == "F" else "pose"
    sim, pr, hg, lever, ok = setup(seed, iface, 0.0001 if strategy == "N4tol" else None)
    st = sim.task_state
    th_u = float(st["theta_u_deg"])
    info = {}
    if strategy == "F":
        rep = pr.call("turn", axis=list(lever[1]), pivot=list(lever[0]), angle_deg=45.0, stop_on_resistance_drop=True)
        info.update(press_term=rep.termination, lever_after_press=round(lever_deg(sim), 1))
        pr.call("pull", direction=list(tangent(sim, hg)) + [0.0], force=8.0, distance=0.03)      # crack the door open
        info.update(door_after_crack=round(door_deg(sim), 1))
        if not st.get("damaged"):
            release_lever(pr, lever)
            terms = []
            for _ in range(3):
                rep = pr.call("pull", direction=list(tangent(sim, hg)) + [0.0], force=8.0, distance=0.12)
                terms.append(rep.termination)
                if door_deg(sim) > 20.0 or st.get("damaged") or rep.termination == "completed":
                    break
            info.update(pull_term=terms)
    else:
        if strategy in ("O", "N1", "N2"):
            target = {"O": th_u + 3.0, "N1": 25.0, "N2": 35.0}[strategy]
            press_to(pr, lever, target)
            if not st.get("damaged"):
                crack_pose(pr, sim, hg)
        elif strategy in ("N5", "N5slow", "N5half"):        # the strategy a model used without force (found in model episodes, added to the checks afterwards):
            slow = strategy in ("N5slow", "N5half")         # press to the smallest unlock angle, then alternate 3 deg more lever with a 1 deg turn about the hinge (a small test pull along the arc)
            r_lever, r_hinge = (2.0, 1.0) if slow else (None, None)    # N5slow uses the model's rates: lever 2 deg/s, hinge 1 deg/s
            step_h = 0.5 if strategy == "N5half" else 1.0   # N5half: test turns of only 0.5 deg (what the model did in most episodes)
            press_to(pr, lever, 15.0, r_lever)
            hinge_axis, hinge_pv = [0.0, 0.0, 1.0], list(find(sim.scene(), "door_hinge")["pos"])
            hk = {} if r_hinge is None else dict(rate_deg_s=r_hinge)
            for _ in range(10):
                if st.get("damaged"):
                    break
                pr.call("turn", axis=hinge_axis, pivot=hinge_pv, angle_deg=step_h, **hk)
                if door_deg(sim) > 0.5:
                    info.update(found_at_deg=round(lever_deg(sim), 1))
                    for _ in range(4):
                        pr.call("turn", axis=hinge_axis, pivot=hinge_pv, angle_deg=1.0, **hk)
                    break
                press_to(pr, lever, lever_deg(sim) + 3.0, r_lever)
        else:                                               # N3 / N4: press a little more, try a pull, crack the door open once it moves
            inc, test = (5.0, 0.01) if strategy == "N3" else (2.0, 0.002)     # N4 and N4tol share parameters; N4tol has a 0.1 mm arrival tolerance
            theta = 0.0
            while theta < 50 and not st.get("damaged"):
                theta += inc
                press_to(pr, lever, theta)
                if st.get("damaged"):
                    break
                tg = tangent(sim, hg)
                pr.call("move_by", dx=float(test * tg[0]), dy=float(test * tg[1]), dz=0.0, stop_on_contact=False)
                if door_deg(sim) > 0.3:
                    info.update(found_at_deg=theta)
                    crack_pose(pr, sim, hg)
                    break
        info.update(lever_after_press=round(float(st.get("lever_max_deg", 0.0)), 1), door_after_crack=round(door_deg(sim), 1))
        if not st.get("damaged") and door_deg(sim) > 1.5:
            release_lever(pr, lever)
            pose_pull(pr, sim, hg)
    out = dict(strategy=strategy, seed=seed, grasp=ok, theta_u=th_u, success=bool(sim.success()), door=round(door_deg(sim), 1),
               damaged=bool(st.get("damaged")), peak_handle_N=round(float(st.get("peak_handle_force_N", 0.0)), 1),
               lever_max=round(float(st.get("lever_max_deg", 0.0)), 1), **info)
    sim.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("strategy", choices=STRATEGIES, help="F = expert force controller, O = position + oracle, N* = position only")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="explicit seeds (overrides --seed-start / --n)")
    ap.add_argument("--seed-start", type=int, default=9100, help="first seed (default: the test seeds 9100-9149)")
    ap.add_argument("--n", type=int, default=50, help="number of consecutive seeds")
    ap.add_argument("--out", default="runs", help="output root")
    ap.add_argument("--run-name", default="expert_checks", help="output folder under --out")
    ap.add_argument("--out-file", default=None,
                    help="explicit output path (default: $FH_CHECK_OUT, else <out>/<run-name>/door_latch_<strategy>.json)")
    args = ap.parse_args()
    strategy = args.strategy
    seeds = args.seeds or list(range(args.seed_start, args.seed_start + args.n))
    rows = []
    for seed in seeds:
        r = run(strategy, seed)
        rows.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    n = len(rows)
    print(f"SUMMARY {strategy}: success {sum(r['success'] for r in rows)}/{n}  damaged {sum(r['damaged'] for r in rows)}/{n}  "
          f"peak handle N median {np.median([r['peak_handle_N'] for r in rows]):.1f}", flush=True)
    out = Path(args.out_file or os.environ.get("FH_CHECK_OUT") or Path(args.out) / args.run_name / f"door_latch_{strategy}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"records: {out}", flush=True)


if __name__ == "__main__":
    main()
