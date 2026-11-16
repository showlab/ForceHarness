"""Evaluate a trained policy (Diffusion Policy, pi0.5 or ForceVLA) on the HiddenPhys tasks. No API calls.

Same test seeds and same physical success check (TaskSpec.success) as every other row of the comparison.
The policy runs in another process (scripts/baselines/dp/dp_server.py or scripts/baselines/pi05/pi05_server.py); this client
only runs the simulator: every Ta ticks it sends the latest To frames, receives a Tp-tick action chunk and executes the first Ta.
Each tick: set the OSC stiffness from the kp part of the action; when the gripper command goes from closed to open, release the
grasp weld first (as the release primitive does); env.step; then, as ForceSim.step does, read the wrench, update the peak and
run the task's step_hook. Learned policies drive the controller directly: there is no 25 N safety stop, so the peak wrench and
the task's damage flag are recorded instead.
Every episode runs a fixed number of ticks (longest training demo x 1.5) before the final state is judged; it does not stop at
the first moment the success check holds.
Usage (from the repository root, in the simulator environment, with a policy server already listening on the port):
  python scripts/baselines/dp/eval_policy.py --task com_sort --name dp_force --ports 47001 --seeds 9100-9149 \
      --matrix configs/hiddenphys/expert.yaml --run runs/learned/dp_force_com_sort --egl 0 --workers 8
--egl takes comma-separated MUJOCO_EGL_DEVICE_ID values, assigned to the workers round-robin.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from collections import deque
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

AUTH = b"fh-dp"
_W = {"egl": None, "port": None}


def _init_worker(q):
    egl, port = q.get()
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(egl)
    _W.update(egl=egl, port=port)


def run_one(task, seed, ep, name, horizon_mult, policy_task=None):
    from multiprocessing.connection import Client

    from forceharness.agent.loop import EpisodeConfig
    from forceharness.eval.runner import _make_sim
    from forceharness.data.demos import encode_frame, pack, proprio, scene_vector, unpack

    t0 = time.time()
    ptask = policy_task or task                             # task name sent to the policy server (see --policy-task)
    conn = Client(("127.0.0.1", _W["port"]), authkey=AUTH)
    conn.send({"cmd": "config", "task": ptask})       # a multi-task server (pi0.5) answers per task; the DP server ignores it
    cfg = json.loads(conn.recv())
    ecfg = EpisodeConfig(**{**ep, "task": task, "model": name, "fmt": "none", "vision": True, "interface": "pose", "seed": seed,
                            "image_size": 128, "images_per_turn": False})
    sim = None
    rec = dict(episode_id=f"{task}__{name}__s{seed}", task=task, model=name, seed=seed, egl=_W["egl"], port=_W["port"],
               ckpt=cfg.get("ckpt"), ckpt_step=cfg.get("ckpt_step"), policy_task=ptask)
    try:
        sim = _make_sim(task, ecfg, "robosuite")
        if getattr(sim.task, "pregrasp_hook", None) is not None:      # pre-grasp done by the environment (lock_or_stiff: grasp the handle), as in model episodes and demos
            from forceharness.primitives.library import Primitives

            ex = sim.task.extra
            pr = Primitives(sim, interface="force", f_contact=float(ep.get("f_contact", 3.0)), f_cap=float(ep.get("f_cap", 25.0)),
                            extra_tools=tuple(ex.get("extra_tools", ())), pos_tol=float(ex.get("pos_tol", 0.003)),
                            slow_in_contact=bool(ex.get("slow_in_contact", False)))
            rec["pregrasp"] = sim.task.pregrasp_hook(sim, pr)
        sv = scene_vector(sim.scene(), cfg["scene_keys"])[0] if cfg.get("scene") else None
        osc_dim, To, Ta = int(cfg["osc_dim"]), int(cfg["To"]), int(cfg["Ta"])
        kp_lo = np.asarray(cfg["act_norm"]["lo"], np.float32)[osc_dim:osc_dim + 6]
        kp_hi = np.asarray(cfg["act_norm"]["hi"], np.float32)[osc_dim:osc_dim + 6]
        hook = getattr(sim.task, "step_hook", None)
        w = sim.wrench()

        def welded():
            mj, m, d = sim.mujoco, sim.m, sim.d
            act = getattr(d, "eq_active", None)
            act = m.eq_active if act is None else act
            return any((mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i) or "").startswith("held_weld") and bool(act[i]) for i in range(m.neq))

        def obs_now():
            o = sim.obs
            g = o.get("robot0_gripper_qpos")
            low = [proprio(o["robot0_eef_pos"], o["robot0_eef_quat"], g)]          # proprio pads zeros when there is no gripper
            if cfg.get("force"):
                low.append(np.concatenate([sim.force_sign * w.net_force_world, sim.force_sign * w.net_torque_world]).astype(np.float32))
            if sv is not None:
                low.append(sv)
            return {c: encode_frame(o[f"{c}_image"]) for c in cfg["cams"]}, np.concatenate(low).astype(np.float32)

        hist = deque([obs_now()] * To, maxlen=To)
        H = int(horizon_mult * int(cfg["max_T"]))
        weld, kp_prev, t, n_calls = welded(), None, 0, 0
        while t < H:
            conn.send({"cmd": "act", "task": ptask, "jpeg": {c: [h[0][c] for h in hist] for c in cfg["cams"]},
                       "low": pack(np.stack([h[1] for h in hist]))})
            acts = unpack(conn.recv())
            n_calls += 1
            for a in acts[:Ta]:
                osc = np.clip(a[:osc_dim].astype(float), -1, 1)
                kp = np.clip(a[osc_dim:osc_dim + 6], kp_lo, kp_hi).astype(float)
                if kp_prev is None or np.abs(kp - kp_prev).max() > 1.0:
                    sim.set_kp(kp); kp_prev = kp
                if osc_dim > 6:
                    if weld and osc[6] < 0:
                        sim.set_held_weld(False); weld = False
                    sim.grip_cmd = float(osc[6])
                sim.obs = sim.env.step(osc[:sim.action_dim])[0]
                sim.n_steps += 1
                w = sim.wrench()
                sim.peak_wrench = max(sim.peak_wrench, w.fmag)
                if hook is not None:
                    hook(sim, w)
                hist.append(obs_now())
                t += 1
                if t >= H:
                    break
        st = sim.task_state
        rec.update(success=bool(sim.success()), n_steps=t, horizon=H, n_policy_calls=n_calls, sim_time_s=round(sim.sim_time, 2),
                   peak_wrench_N=round(float(sim.peak_wrench), 2), damaged=bool(st.get("damaged", False)),
                   truth={k: v for k, v in st.items() if isinstance(v, (str, int, float, bool))})
    except Exception as e:
        rec.update(success=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc())
    finally:
        conn.close()
        if sim is not None:
            sim.close()
    rec["wall_time_s"] = round(time.time() - t0, 1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--name", required=True, help="name of this row in the results, e.g. dp, dp_force")
    ap.add_argument("--ports", required=True, help="comma-separated policy-server ports, assigned to the workers round-robin")
    ap.add_argument("--seeds", default="9100-9149")
    ap.add_argument("--matrix", default="configs/hiddenphys/expert.yaml", help="episode settings (noise, kp, ...) are read from its episode section")
    ap.add_argument("--run", required=True, help="results directory; appends to results.jsonl and skips episode_ids already there")
    ap.add_argument("--egl", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--horizon-mult", type=float, default=1.5)
    ap.add_argument("--policy-task", default=None,
                    help="task name sent to the policy server (default: --task). For a shifted variant <task>_ood pass the original task: a multi-task server uses it to pick the task config and the training prompt")
    args = ap.parse_args()
    a, b = (int(x) for x in args.seeds.split("-"))
    ep = dict(yaml.safe_load(Path(args.matrix).read_text()).get("episode", {}))
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    res = run / "results.jsonl"
    done = set()
    if res.exists():
        for l in res.read_text().splitlines():
            try:
                done.add(json.loads(l)["episode_id"])
            except Exception:
                pass
    seeds = [s for s in range(a, b + 1) if f"{args.task}__{args.name}__s{s}" not in done]
    ports, egls = [int(p) for p in args.ports.split(",")], [int(e) for e in args.egl.split(",")]
    print(f"[eval] {args.task} {args.name}: {len(seeds)} episodes → {res}", flush=True)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for i in range(args.workers):
        q.put((egls[i % len(egls)], ports[i % len(ports)]))
    t0, k = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=ctx, initializer=_init_worker, initargs=(q,)) as ex, res.open("a") as fh:
        for f in as_completed([ex.submit(run_one, args.task, s, ep, args.name, args.horizon_mult, args.policy_task) for s in seeds]):
            r = f.result()
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n"); fh.flush()
            k += bool(r.get("success"))
            print(f"  s{r['seed']}: success={r.get('success')} steps={r.get('n_steps')} peak={r.get('peak_wrench_N')} "
                  f"wall={r['wall_time_s']}s err={r.get('error')}", flush=True)
    print(f"[eval] {args.task} {args.name}: {k}/{len(seeds)} success, {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
