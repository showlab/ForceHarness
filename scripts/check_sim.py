"""Check a MuJoCo (robosuite) installation end to end, without any API key.

For each task it builds the environment, saves the camera images, checks the force sign by moving the empty hand down
until it touches the table (with ``force_sign=-1`` the reaction on the hand must point up, +z), and then runs one episode of
the scripted expert controller through the full ForceHarness stack (force primitives + physics translation, ``phys2|force``).

    python scripts/check_sim.py                       # the five tasks the expert controller covers
    python scripts/check_sim.py --tasks weight_sort --seed 9100 --out runs
    python scripts/check_sim.py --no-render           # no OpenGL needed (camera images skipped)

Exit code 0 means every check passed; ``<out>/check_sim/summary.json`` has the details.
The two mechanism tasks (door_latch, lock_or_stiff) have their own expert scripts in ``scripts/hiddenphys/``.
"""
import _bootstrap  # noqa: F401
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from forceharness.agent.loop import Episode, EpisodeConfig
from forceharness.backends.mujoco.env import ForceSim
from forceharness.baselines.expert import ScriptedAgent
from forceharness.eval.runner import _make_sim as make_sim
from forceharness.hiddenphys.tasks import env_kwargs_for, get_task
from forceharness.primitives.library import Primitives

EXPERT_TASKS = ["weight_sort", "com_sort", "wipe_hard", "stack_gentle", "peg_in_hole_hard"]


def _task_with_env_kwargs(name: str, seed: int):
    task = get_task(name)
    if task.env_kwargs_fn is not None:
        task = dataclasses.replace(task, env_kwargs={**task.env_kwargs, **env_kwargs_for(task, seed)})
    return task


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="+", default=EXPERT_TASKS)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--seed", type=int, default=9100)
    ap.add_argument("--kp", type=float, default=150.0)
    ap.add_argument("--no-episode", action="store_true", help="only build the scene and run the force-sign check")
    ap.add_argument("--no-render", action="store_true",
                    help="skip camera images (no OpenGL needed; the paper's HiddenPhys episodes run without images)")
    args = ap.parse_args()
    out = Path(args.out) / "check_sim"
    out.mkdir(parents=True, exist_ok=True)
    summary, ok_all = {}, True
    for name in args.tasks:
        print(f"\n===== {name} =====", flush=True)
        row = summary.setdefault(name, {})
        try:
            t0 = time.time()
            sim = ForceSim(_task_with_env_kwargs(name, args.seed), seed=args.seed, kp=args.kp, render=not args.no_render,
                           image_size=256, force_sign=-1.0)
            row["build_s"] = round(time.time() - t0, 1)
            print(f"built in {row['build_s']} s; action_dim={sim.action_dim}; force sensor={sim.force_sensor_name}", flush=True)
            if not args.no_render:
                from PIL import Image
                for cam, img in sim.images().items():
                    Image.fromarray(img).save(out / f"{name}_{cam}.png")
            prims = Primitives(sim, interface="force")
            rep = prims.move_until_contact([0, 0, -1], max_dist=0.35, f_threshold=3.0)
            fz = float(rep.final_wrench[2])
            row.update(sign_check_termination=rep.termination, reaction_fz_N=round(fz, 2), contact_z=round(float(rep.end_pos[2]), 4))
            row["sign_ok"] = bool(rep.termination == "contact" and fz > 0)
            print(f"force-sign check: termination={rep.termination} reaction Fz={fz:+.2f} N -> "
                  f"{'OK' if row['sign_ok'] else 'FAIL (expected contact with Fz > 0)'}", flush=True)
            sim.close()
            if not args.no_episode and name in EXPERT_TASKS + [t + "_ood" for t in EXPERT_TASKS]:
                cfg = EpisodeConfig(task=name, model="scripted", fmt="phys2", vision=False, interface="force", seed=args.seed,
                                    run_name="check_sim", out_dir=args.out, max_turns=40, max_tool_calls=80, max_sim_time=300.0,
                                    kp=args.kp, force_sign=-1.0, scene_noise_pos=0.004, f_contact=3.0,
                                    force_noise_std=0.5, force_bias_std=0.3)
                sim = make_sim(name, cfg, "robosuite")      # same construction as the benchmark runner
                t0 = time.time()
                rec = Episode(sim, ScriptedAgent(name, "force"), cfg).run()
                row.update(expert_success=bool(rec["success"]), expert_stop=rec["stop_reason"], expert_error=rec["error"],
                           expert_tool_calls=rec["n_tool_calls"], expert_wall_s=round(time.time() - t0, 1))
                print(f"expert episode: success={rec['success']} tool_calls={rec['n_tool_calls']} stop={rec['stop_reason']} "
                      f"error={rec['error']}", flush=True)
                sim.close()
        except Exception as e:  # report and continue with the next task
            row["exception"] = f"{type(e).__name__}: {e}"
            print(f"ERROR: {row['exception']}", flush=True)
        row["ok"] = bool(row.get("sign_ok")) and "exception" not in row and row.get("expert_error") is None
        ok_all &= row["ok"]
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=str))
    print("\nSUMMARY")
    for name, row in summary.items():
        print(f"  {name:20s} {'PASS' if row['ok'] else 'FAIL'}  sign_ok={row.get('sign_ok')} expert_success={row.get('expert_success')}")
    print(f"details: {out / 'summary.json'}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
