#!/usr/bin/env python3
"""ForceHarness as a ForceBench policy server. It runs in the policy-server environment, not in the Isaac Sim environment.

One websocket connection = one episode: FBSim (evaluator observations -> the backend interface of the primitives) -> primitives
-> agent loop (scripted agent or a model through its API) -> one action per tick back to the evaluator. The evaluator is our
copy scripts/forcebench/eval_forcebench.py, run with --execute_chunk 1 (closed loop at every tick); scripts/forcebench/run.sh
starts both. The defaults are the paper's setting: RGB-only perception (two camera images + camera calibration, no object
coordinates) with the force interface and L2 tool results (phys2).

  python scripts/forcebench/policy_server.py --task weight_sort --agent scripted --port 8102 --run-name ws_scripted --out runs
  python scripts/forcebench/policy_server.py --task weight_sort --agent llm --model gpt-6.1-sol --provider openai-responses \\
      --reasoning-effort low --fmt phys2 --port 8102 --run-name ws_sol --out runs
API keys: --api-key-env names the variable (default FH_API_KEY, the OpenAI key; for Anthropic pass --provider anthropic
--api-key-env ANTHROPIC_API_KEY). The repository's .env file is read; variables already set in the environment take precedence.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")                 # API keys; does not override variables that are already set
except ImportError:
    pass

from forceharness.agent.loop import EpisodeConfig  # noqa: E402
from forceharness.eval.pricing import price_for  # noqa: E402
from forceharness.backends.forcebench.episode import FBEpisode  # noqa: E402
from forceharness.backends.forcebench.fbsim import FBSim  # noqa: E402
from forceharness.backends.forcebench.server import PolicyServer, parse_seed  # noqa: E402
from forceharness.backends.forcebench.tasks import for_interface, get_fb_task  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="ForceHarness policy server for ForceBench's evaluator (see docs/forcebench.md).")
    ap.add_argument("--task", required=True, help="ForceBench task, e.g. weight_sort or radmanso_drawer_sort_table")
    ap.add_argument("--agent", default="scripted", choices=["scripted", "llm"],
                    help="scripted: hand-written reference agent (no API calls); llm: a model through its API")
    ap.add_argument("--model", default="scripted")
    ap.add_argument("--provider", default=None)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--api-key-env", default="FH_API_KEY")
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--fmt", default="phys2")
    ap.add_argument("--interface", default="force", choices=["pose", "guard", "force"])
    ap.add_argument("--vision", action="store_true", default=False)
    ap.add_argument("--perception", default="rgb", choices=["privileged", "rgb"],
                    help="rgb (default, the paper's setting): no object coordinates, only the two camera images + camera "
                         "calibration; turns on --vision. privileged: noisy true object positions in the scene description; "
                         "the evaluator must then run with --privileged_geometry")
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--max-tool-calls", type=int, default=60)
    ap.add_argument("--max-steps", type=int, default=1500, help="ForceBench's step limit per episode (15 Hz)")
    ap.add_argument("--f-cap", type=float, default=25.0)
    ap.add_argument("--scene-noise-pos", type=float, default=0.004)
    ap.add_argument("--scene-noise-yaw-deg", type=float, default=2.0)
    ap.add_argument("--force-noise-std", type=float, default=0.0,
                    help="ForceBench's readings are already the raw simulated values; no extra noise by default")
    ap.add_argument("--force-bias-std", type=float, default=0.0)
    ap.add_argument("--wrench-frame", default="hand", choices=["hand", "world"])
    ap.add_argument("--stiffness-gain", default="{}",
                    help='JSON: MuJoCo OSC kp -> target scaling factor, e.g. {"40": 1.5, "150": 2.0}')
    ap.add_argument("--prim-overrides", default=None,
                    help='JSON: primitive instance attributes; replaces the default FBEpisode.prim_overrides '
                         '({"max_lead_contact": 0.015}) as a whole, so repeat max_lead_contact when overriding')
    ap.add_argument("--playbook", default="")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8102)
    ap.add_argument("--run-name", default="fb_dev")
    ap.add_argument("--out", default="runs")
    ap.add_argument("--max-episodes", type=int, default=0, help="reserved, currently unused (the server runs until it is stopped)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    task = for_interface(get_fb_task(args.task, args.perception), args.interface)
    if args.perception == "rgb":
        args.vision = True
    gains = {float(k): float(v) for k, v in json.loads(args.stiffness_gain).items()}
    pin, pout = price_for(args.model) if args.agent == "llm" else (0.0, 0.0)
    out_root = Path(args.out)
    done_count = [0]
    lock = threading.Lock()

    def make_agent(seed: int, cfg: EpisodeConfig):
        if args.agent == "scripted":
            from forceharness.backends.forcebench.scripted import FBScriptedAgent

            return FBScriptedAgent(args.task, args.interface, perception=args.perception)
        from forceharness.agent.llm import LLMClient
        from forceharness.agent.loop import LLMAgent

        client = LLMClient(model=args.model, provider=args.provider, base_url=args.base_url,
                           api_key=os.environ.get(args.api_key_env) if args.api_key_env else None,
                           reasoning_effort=args.reasoning_effort)
        return LLMAgent(client)

    def make_worker(rid, first_req):
        seed = parse_seed(rid)
        if seed is None:
            seed = 0

        def worker(sess):
            sim = FBSim(sess, task, seed, wrench_frame=args.wrench_frame, stiffness_gain=gains,
                        scene_noise_pos=args.scene_noise_pos, scene_noise_yaw_deg=args.scene_noise_yaw_deg,
                        force_noise_std=args.force_noise_std, force_bias_std=args.force_bias_std, max_steps=args.max_steps)
            cfg = EpisodeConfig(task=f"fb_{args.task}", model=args.model, fmt=args.fmt, vision=args.vision, interface=args.interface,
                                seed=seed, max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
                                max_sim_time=args.max_steps / 15.0 - 2.0, f_cap=args.f_cap, playbook=args.playbook,
                                run_name=args.run_name, out_dir=str(out_root), price_in_per_m=pin, price_out_per_m=pout,
                                scene_noise_pos=args.scene_noise_pos, scene_noise_yaw_deg=args.scene_noise_yaw_deg,
                                note=f"forcebench rollout {rid}")
            agent = make_agent(seed, cfg)
            if getattr(agent, "kind", "") == "scripted":
                agent.sim = sim
            ep = FBEpisode(sim, agent, cfg, prim_overrides=json.loads(args.prim_overrides) if args.prim_overrides else None)
            rec = ep.run_safe()
            rec["rollout_id"] = rid
            rec["fb_steps_used"] = sim.step_index
            run_dir = out_root / args.run_name
            run_dir.mkdir(parents=True, exist_ok=True)
            with lock:
                with open(run_dir / "results.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps({k: v for k, v in rec.items() if k != "cfg"}, default=str) + "\n")
                done_count[0] += 1
            logging.info("episode %s ended (%s): stop=%s calls=%s cost=$%.3f", rid, rec.get("ended_by"), rec.get("stop_reason"),
                         rec.get("n_model_calls"), rec.get("cost_usd") or 0.0)
            return {k: rec.get(k) for k in ("episode_id", "ended_by", "stop_reason", "verdict", "n_model_calls", "cost_usd")}

        return worker

    meta = dict(harness="force-harness", task=args.task, agent=args.agent, model=args.model, fmt=args.fmt,
                interface=args.interface, stiffness_gain=gains, wrench_frame=args.wrench_frame)
    srv = PolicyServer(make_worker, host=args.host, port=args.port, metadata=meta, log_dir=out_root / args.run_name)
    (out_root / args.run_name).mkdir(parents=True, exist_ok=True)
    (out_root / args.run_name / "server_args.json").write_text(json.dumps(vars(args), indent=1), encoding="utf-8")
    srv.serve_forever()


if __name__ == "__main__":
    main()
