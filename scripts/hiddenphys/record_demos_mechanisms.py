"""Record demonstrations of door_latch and lock_or_stiff with their expert force controllers (strategy F of
door_latch_expert.py and lock_or_stiff_expert.py), in the same format as record_demos.py. No model calls.

  door_latch: recording starts as soon as the sim is built; the script reaches for the handle, grasps it, presses and
    pulls (as in model episodes, all of this is the policy's job).
  lock_or_stiff: recording starts after the task's pre-grasp (task.pregrasp_hook grasps the handle for the policy),
    because the environment does that part in model episodes too; evaluation (scripts/baselines/dp/eval_policy.py) and
    replay (replay_demos.py) run the same pre-grasp first. Only the soft arm (kp 150) is recorded.

How: wrap the expert module's _make_sim / _ls_pregrasp (turn rendering on, attach the recorder, choose when recording
starts), then call its unchanged run("F", ...); the strategy itself is not modified. Successful episodes are saved as
<out>/<task>/s<seed>.npz, and every episode adds a line to <out>/<task>/index.jsonl (reruns skip listed seeds).

    python scripts/hiddenphys/record_demos_mechanisms.py --task door_latch --seed-start 20000 --n 210 --out demos --workers 16
    python scripts/hiddenphys/record_demos_mechanisms.py --task lock_or_stiff --n 210 --out demos --egl 0,1 --workers 8

Rendering is off-screen EGL; --egl assigns MUJOCO_EGL_DEVICE_ID values round-robin to the workers (see record_demos.py).
"""
import _bootstrap  # noqa: F401
import argparse
import dataclasses
import importlib
import json
import multiprocessing as mp
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_EGL = {"id": None}
_HOLD = {}
LS_PRIMS = dict(interface="force", f_contact=3.0, f_cap=25.0, extra_tools=["point_tool", "pull"], pos_tol=0.0001, slow_in_contact=True)


def _init_worker(q):
    dev = q.get()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(dev)
    _EGL["id"] = dev


def _patch(task):
    """Wrap the expert module once per process."""
    if _HOLD.get("patched"):
        return _HOLD["mod"]
    from forceharness.backends.mujoco.recorder import DemoRecorder, install_primitive_tracker

    mod = importlib.import_module("door_latch_expert" if task == "door_latch" else "lock_or_stiff_expert")
    orig_make = mod._make_sim

    def make(task_, cfg, backend):
        cfg = dataclasses.replace(cfg, vision=True, image_size=128, images_per_turn=False)
        sim = orig_make(task_, cfg, backend)
        rec = DemoRecorder(sim)
        _HOLD.update(rec=rec, cfg=cfg)
        if task == "door_latch":
            rec.start()
        return sim

    mod._make_sim = make
    if task == "lock_or_stiff":
        orig_pg = mod._ls_pregrasp

        def pg(sim, pr):
            r = orig_pg(sim, pr)
            _HOLD["rec"].start()
            _HOLD["pregrasp"] = r
            return r

        mod._ls_pregrasp = pg
    install_primitive_tracker()
    _HOLD.update(patched=True, mod=mod)
    return mod


def record_one(task, seed, kp, out_dir):
    t0 = time.time()
    try:
        mod = _patch(task)
        _HOLD.pop("pregrasp", None)
        res = mod.run("F", seed, kp) if task == "lock_or_stiff" else mod.run("F", seed)
        rec, cfg = _HOLD["rec"], _HOLD["cfg"]
        rec.stop()
        ok = bool(res.get("success"))
        line = dict(task=task, seed=seed, success=ok, n_steps=len(rec.rows["action"]), egl=_EGL["id"],
                    **{k: v for k, v in res.items() if k in ("verdict", "label", "verdict_ok", "door_ok", "damaged", "theta_u", "door")})
        if ok:
            path = Path(out_dir) / task / f"s{seed}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            meta = dict(line, fmt="phys2", interface="force", episode_cfg=dataclasses.asdict(cfg), check_result=res,
                        start_after_pregrasp=(task == "lock_or_stiff"), prims_args=(LS_PRIMS if task == "lock_or_stiff" else None),
                        recorded_with=f"scripts/hiddenphys/{mod.__name__}.py run('F') (expert force controller)",
                        recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            line["bytes"] = rec.save(str(path), meta)
    except Exception as e:
        line = dict(task=task, seed=seed, success=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    line["wall_s"] = round(time.time() - t0, 1)
    return line


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, choices=["door_latch", "lock_or_stiff"])
    ap.add_argument("--seed-start", type=int, default=20000, help="first demonstration seed (20000 and up)")
    ap.add_argument("--n", type=int, default=210, help="number of seeds to attempt (only successful episodes are saved)")
    ap.add_argument("--kp", type=float, default=150.0, help="arm stiffness for lock_or_stiff (the released data uses the soft arm, 150)")
    ap.add_argument("--out", default="demos", help="output root; episodes go to <out>/<task>/")
    ap.add_argument("--egl", default="0", help="comma-separated MUJOCO_EGL_DEVICE_ID values, assigned round-robin to the workers")
    ap.add_argument("--workers", type=int, default=8, help="parallel episodes (one process each)")
    args = ap.parse_args()
    idx = Path(args.out) / args.task / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["seed"] for l in idx.read_text().splitlines() if l.strip()} if idx.exists() else set()
    seeds = [s for s in range(args.seed_start, args.seed_start + args.n) if s not in done]
    devs = [int(x) for x in args.egl.split(",")]
    print(f"[record] {args.task}: {len(seeds)} to run ({len(done)} done) → {idx.parent}; EGL {devs}", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(args.workers):
        q.put(devs[i % len(devs)])
    t0, n_ok, n_bytes = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker, initargs=(q,)) as ex, idx.open("a") as fh:
        for f in as_completed([ex.submit(record_one, args.task, s, args.kp, args.out) for s in seeds]):
            line = f.result()
            fh.write(json.dumps(line, ensure_ascii=False, default=str) + "\n"); fh.flush()
            n_ok += bool(line.get("success")); n_bytes += int(line.get("bytes", 0) or 0)
            print(f"  s{line['seed']}: success={line.get('success')} steps={line.get('n_steps')} "
                  f"{(line.get('bytes') or 0) / 1e6:.1f} MB wall={line['wall_s']}s err={line.get('error')}", flush=True)
    print(f"[record] {args.task}: {n_ok}/{len(seeds)} saved, {n_bytes / 1e9:.2f} GB, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
