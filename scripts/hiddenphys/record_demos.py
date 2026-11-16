"""Record expert demonstrations for the learned reference policies (Diffusion Policy, pi0.5, ForceVLA). No model calls.

One process per episode: build the sim (ForceSim resets on construction) -> DemoRecorder.start() -> the expert
(ScriptedAgent on the force interface) runs a full episode -> only successful episodes are saved as
<out>/<task>/s<seed>.npz. Every episode, failed ones included, adds one line to <out>/<task>/index.jsonl, and a rerun
skips the seeds already listed there. Episode settings come from the ``episode`` block of --matrix, i.e. the same
protocol as the reference rows. This script covers weight_sort, com_sort, wipe_hard, stack_gentle and peg_in_hole_hard;
record_demos_mechanisms.py records door_latch and lock_or_stiff in the same format.

Demonstration seeds start at 20000; the test seeds (9100-9149) and development seeds (9000-9019) must not be used. The
paper's training sets used 210 attempts per task (240 on peg_in_hole_hard), about 200 successful episodes each.

    python scripts/hiddenphys/record_demos.py --task com_sort --seed-start 20000 --n 210 --out demos --workers 8
    python scripts/hiddenphys/record_demos.py --task peg_in_hole_hard --n 240 --out demos --egl 0,1 --workers 8

Rendering is off-screen EGL (MUJOCO_GL=egl). --egl assigns MUJOCO_EGL_DEVICE_ID values round-robin to the workers; on a
multi-GPU machine the EGL device numbering can differ from the nvidia-smi order, so check which device is which first.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import yaml

_EGL = {"id": None}


def _init_worker(q):
    dev = q.get()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)     # must be set before this process builds its first robosuite env
    _EGL["id"] = dev


def record_one(task: str, seed: int, ep: dict, fmt: str, interface: str, image_size: int, out_dir: str, keep_failed: bool):
    from forceharness.agent.loop import Episode, EpisodeConfig
    from forceharness.baselines.expert import ScriptedAgent
    from forceharness.eval.runner import _make_sim
    from forceharness.backends.mujoco.recorder import DemoRecorder, install_primitive_tracker

    t0 = time.time()
    cfg = EpisodeConfig(**{**ep, "task": task, "model": "scripted", "fmt": fmt, "vision": True, "interface": interface,
                           "seed": seed, "image_size": image_size, "images_per_turn": False})
    sim = None
    try:
        sim = _make_sim(task, cfg, "robosuite")
        rec = DemoRecorder(sim)
        rec.start()
        install_primitive_tracker()
        res = Episode(sim, ScriptedAgent(task, interface), cfg).run()
        rec.stop()
        ok = bool(res.get("success"))
        line = dict(task=task, seed=seed, success=ok, verdict=res.get("verdict"), verdict_correct=res.get("verdict_correct"),
                    truth=res.get("truth"), sim_time_s=res.get("sim_time_s"), n_steps=len(rec.rows["action"]),
                    peak_wrench_N=res.get("peak_wrench_N"), stop_reason=res.get("stop_reason"), egl=_EGL["id"])
        if ok or keep_failed:
            sub = "" if ok else "failed/"
            path = Path(out_dir) / task / f"{sub}s{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(line, fmt=fmt, interface=interface, episode_cfg={k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
                        recorded_with="ScriptedAgent (expert force controller)", recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            line["bytes"] = rec.save(str(path), meta)
            line["file"] = str(path)
    except Exception as e:
        line = dict(task=task, seed=seed, success=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    finally:
        if sim is not None:
            sim.close()
    line["wall_s"] = round(time.time() - t0, 1)
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="weight_sort, com_sort, wipe_hard, stack_gentle or peg_in_hole_hard")
    ap.add_argument("--seed-start", type=int, default=20000, help="first demonstration seed (20000 and up)")
    ap.add_argument("--n", type=int, default=200, help="number of seeds to attempt (only successful episodes are saved)")
    ap.add_argument("--matrix", default="configs/hiddenphys/expert.yaml", help="config whose episode block gives the episode settings")
    ap.add_argument("--fmt", default="phys2", help="tool-result format seen by the expert (it reads only derived signals)")
    ap.add_argument("--interface", default="force", help="action interface of the expert")
    ap.add_argument("--image-size", type=int, default=128, help="camera image size in pixels (128, as in the LIBERO demonstrations)")
    ap.add_argument("--out", default="demos", help="output root; episodes go to <out>/<task>/")
    ap.add_argument("--egl", default="0", help="comma-separated MUJOCO_EGL_DEVICE_ID values, assigned round-robin to the workers")
    ap.add_argument("--workers", type=int, default=4, help="parallel episodes (one process each)")
    ap.add_argument("--keep-failed", action="store_true", help="also save failed episodes, under <out>/<task>/failed/")
    args = ap.parse_args()

    mpath = Path(args.matrix)
    if not mpath.exists() and (_bootstrap.ROOT / mpath).exists():       # allow running from outside the repository root
        mpath = _bootstrap.ROOT / mpath
    ep = dict(yaml.safe_load(mpath.read_text()).get("episode", {}))
    idx_path = Path(args.out) / args.task / "index.jsonl"
    idx_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if idx_path.exists():
        for l in idx_path.read_text().splitlines():
            try:
                done.add(int(json.loads(l)["seed"]))
            except Exception:
                pass
    seeds = [s for s in range(args.seed_start, args.seed_start + args.n) if s not in done]
    devs = [int(x) for x in args.egl.split(",") if x.strip()]
    print(f"[record] {args.task}: {len(seeds)} to run ({len(done)} done) → {idx_path.parent}; EGL {devs}", flush=True)
    ctx = mp.get_context("spawn")                    # spawn: each worker sets its EGL device before importing mujoco
    q = ctx.Queue()
    for i in range(args.workers):
        q.put(devs[i % len(devs)])
    t0, n_ok, n_bytes = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker, initargs=(q,)) as ex, \
            idx_path.open("a") as fh:
        futs = [ex.submit(record_one, args.task, s, ep, args.fmt, args.interface, args.image_size, args.out, args.keep_failed)
                for s in seeds]
        for f in as_completed(futs):
            line = f.result()
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n"); fh.flush()
            n_ok += bool(line.get("success")); n_bytes += int(line.get("bytes", 0) or 0)
            print(f"  s{line['seed']}: success={line.get('success')} steps={line.get('n_steps')} "
                  f"{(line.get('bytes') or 0)/1e6:.1f} MB wall={line['wall_s']}s err={line.get('error')}", flush=True)
    print(f"[record] {args.task}: {n_ok}/{len(seeds)} saved, {n_bytes/1e9:.2f} GB, {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
