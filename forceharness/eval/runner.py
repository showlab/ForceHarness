"""Matrix runner: YAML -> list of conditions -> n episodes per condition -> results.jsonl.

Runs are resumable: episodes already in results.jsonl (by episode_id) are skipped.  Each episode builds its own
simulator in its own process (MuJoCo + EGL contexts are not shared across threads); model latency dominates the wall
time, so 4-8 workers are enough.
"""
from __future__ import annotations

import dataclasses
import itertools
import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from forceharness.agent.loop import Episode, EpisodeConfig, LLMAgent
from forceharness.eval.pricing import price_for


def _model_spec(m) -> Dict[str, Any]:
    return {"name": m} if isinstance(m, str) else dict(m)


def expand_matrix(mx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return [{"cfg": EpisodeConfig as a dict, "model": model spec}, ...] for every episode of the matrix.  ``cells``
    lists an explicit subset of conditions instead of the full product of tasks x models x formats x vision x
    interfaces."""
    run_name = mx.get("run_name", "dev")
    ep = dict(mx.get("episode", {}))
    n = int(mx.get("n_episodes", 1))
    seed_base = int(mx.get("seed_base", 1000))
    out = []
    if mx.get("cells"):
        combos = []
        for c in mx["cells"]:
            combos.append((c["task"], _model_spec(c["model"]), c["fmt"], bool(c.get("vision", True)), c["interface"]))
    else:
        combos = list(itertools.product(mx["tasks"], [_model_spec(m) for m in mx["models"]], mx["formats"],
                                        [bool(v) for v in mx.get("vision", [True])], mx["interfaces"]))
    for task, model, fmt, vision, interface in combos:
        if model.get("harness"):        # program-writing baselines (baselines/code_as_policies.py): the fmt slot of the condition name holds the harness name (e.g. cap|pose)
            fmt = model["harness"]
        pin, pout = price_for(model["name"])
        for i in range(n):
            cfg = EpisodeConfig(task=task, model=model["name"], fmt=fmt, vision=vision, interface=interface, seed=seed_base + i,
                                run_name=run_name, price_in_per_m=pin, price_out_per_m=pout, **ep)
            out.append({"cfg": dataclasses.asdict(cfg), "model": model})
    return out


def _make_sim(task_name: str, cfg: EpisodeConfig, backend: str):
    from forceharness.hiddenphys.tasks import get_task

    task = get_task(task_name)
    if backend == "fake":
        from forceharness.backends.fake import FakeSim

        return FakeSim(task, seed=cfg.seed, scene_noise_pos=cfg.scene_noise_pos, force_sign=cfg.force_sign)
    from forceharness.backends.mujoco.env import ForceSim

    from forceharness.hiddenphys.tasks import env_kwargs_for

    rng_kwargs = env_kwargs_for(task, cfg.seed)
    task = dataclasses.replace(task, env_kwargs={**task.env_kwargs, **rng_kwargs})
    noise = max(cfg.scene_noise_pos, float(task.extra.get("min_scene_noise_pos", 0.0)))
    return ForceSim(task, seed=cfg.seed, kp=cfg.kp, render=cfg.vision, image_size=cfg.image_size,
                    scene_noise_pos=noise, scene_noise_yaw_deg=cfg.scene_noise_yaw_deg, force_sign=cfg.force_sign,
                    force_noise_std=cfg.force_noise_std, force_bias_std=cfg.force_bias_std,
                    grip_force_scale=float(task.extra.get("grip_force_scale", 1.0)))


def _make_agent(kind: str, cfg: EpisodeConfig, model: Dict[str, Any]):
    if kind == "scripted":
        if model.get("policy") == "pos_only_peg":          # classical baseline: peg-in-hole with position control + compliance only (baselines/classical.py)
            from forceharness.baselines.classical import PositionOnlyPeg

            return PositionOnlyPeg(cfg.task, cfg.interface)
        from forceharness.baselines.expert import ScriptedAgent

        ag = ScriptedAgent(cfg.task, cfg.interface)
        ag.task_extra.update(model.get("override", {}))     # expert ablation: replace constants tuned on the task with values from the instruction or the literature (e.g. search_radius)
        return ag
    from forceharness.agent.llm import LLMClient

    client = LLMClient(model=model["name"], provider=model.get("provider"), base_url=model.get("base_url"),
                       api_key=os.environ.get(model["api_key_env"]) if model.get("api_key_env") else None,
                       temperature=model.get("temperature"), max_tokens=model.get("max_tokens"),
                       extra_body=model.get("extra_body"), reasoning_effort=model.get("reasoning_effort"))
    return LLMAgent(client)


def run_one(cfg_dict: Dict[str, Any], model: Dict[str, Any], backend: str = "robosuite", agent_kind: str = "llm") -> Dict[str, Any]:
    """Run one episode in the current process.  An exception becomes a failed record (stop_reason "exception")
    instead of propagating."""
    cfg = EpisodeConfig(**cfg_dict)
    sim = None
    try:
        sim = _make_sim(cfg.task, cfg, backend)
        if model.get("harness") and agent_kind == "llm":     # Code-as-Policies baseline: one model call per episode writes a complete program, which is then executed
            from forceharness.baselines.code_as_policies import run_program_episode

            rec = run_program_episode(sim, cfg, model)
        else:
            agent = _make_agent(agent_kind, cfg, model)
            rec = Episode(sim, agent, cfg).run()
    except Exception as e:
        rec = dict(episode_id=cfg.episode_id, condition=cfg.condition, task=cfg.task, model=cfg.model, fmt=cfg.fmt, vision=cfg.vision,
                   interface=cfg.interface, seed=cfg.seed, success=False, error=f"{type(e).__name__}: {e}", traceback=traceback.format_exc(),
                   stop_reason="exception", cfg=cfg_dict)
    finally:
        if sim is not None:
            sim.close()
    return rec


def _done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    ids = set()
    for line in path.read_text().splitlines():
        try:
            ids.add(json.loads(line)["episode_id"])
        except Exception:
            pass
    return ids


def run_matrix(matrix_path: str, workers: int = 4, backend: str = "robosuite", agent_kind: str = "llm",
               limit: Optional[int] = None, dry_run: bool = False, out_dir: str = "runs") -> Path:
    """Run (or resume) every episode of a matrix YAML, sequentially or with a process pool (workers > 1); returns the
    path of results.jsonl."""
    mx = yaml.safe_load(Path(matrix_path).read_text())
    jobs = expand_matrix(mx)
    if agent_kind == "llm" and any(j["model"].get("harness") not in (None, "cap") for j in jobs):
        raise SystemExit("unknown program-writing harness; run_matrix supports harness: cap (Code-as-Policies) only")
    for j in jobs:
        j["cfg"]["out_dir"] = out_dir
    run_dir = Path(out_dir) / mx.get("run_name", "dev")
    run_dir.mkdir(parents=True, exist_ok=True)
    results = run_dir / "results.jsonl"
    done = _done_ids(results)
    todo = [j for j in jobs if j["cfg"]["seed"] is not None and EpisodeConfig(**j["cfg"]).episode_id not in done]
    if limit:
        todo = todo[:limit]
    print(f"[matrix] {len(jobs)} episodes total, {len(done)} done, {len(todo)} to run; results → {results}")
    if dry_run:
        for j in todo[:20]:
            print("  ", EpisodeConfig(**j["cfg"]).episode_id)
        return results
    (run_dir / "matrix.yaml").write_text(yaml.safe_dump(mx, allow_unicode=True))
    t0 = time.time()
    with results.open("a") as fh:
        if workers <= 1:
            for j in todo:
                rec = run_one(j["cfg"], j["model"], backend, agent_kind)
                fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                print(f"  {rec['episode_id']}: success={rec.get('success')} calls={rec.get('n_model_calls')} stop={rec.get('stop_reason')} err={rec.get('error')}")
        else:
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futs = {ex.submit(run_one, j["cfg"], j["model"], backend, agent_kind): j for j in todo}
                for f in as_completed(futs):
                    rec = f.result()
                    fh.write(json.dumps(rec, default=str) + "\n"); fh.flush()
                    print(f"  {rec['episode_id']}: success={rec.get('success')} calls={rec.get('n_model_calls')} stop={rec.get('stop_reason')} err={rec.get('error')}")
    print(f"[matrix] finished {len(todo)} episodes in {(time.time()-t0)/60:.1f} min")
    return results
