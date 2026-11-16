"""Run one HiddenPhys episode and print its record as JSON.

    # offline, no simulator and no API key: the expert on the fake backend (weight_sort only)
    python scripts/run_episode.py --task weight_sort --agent scripted --backend fake
    # the expert force controller on MuJoCo with the protocol of the paper
    python scripts/run_episode.py --task com_sort --agent scripted --paper-defaults --seed 9100
    # ForceHarness (self-terminating force primitives + physics translation, phys2|force) with GPT-6 Astra
    python scripts/run_episode.py --task com_sort --model gpt-6-astra --provider openai-responses \\
        --reasoning-effort low --paper-defaults --seed 9100
    # the no-force control (none|pose) on the same episode
    python scripts/run_episode.py --task com_sort --model gpt-6-astra --provider openai-responses \\
        --reasoning-effort low --paper-defaults --fmt none --interface pose --seed 9100

Model episodes make paid API calls (keys come from .env, see .env.example). The transcript and record are written under
<out>/<run-name>/; batches of episodes from a YAML config are run by scripts/run_matrix.py.

Defaults: without --paper-defaults this script keeps its development defaults (fmt signals, camera images on, 25 turns,
and the EpisodeConfig defaults for everything else: a clean force signal, 60 tool calls, 150 s of simulated time).
--paper-defaults switches every option that is not given explicitly to the protocol of the HiddenPhys results
(PAPER_DEFAULTS below, the ``episode`` block of the configs in configs/hiddenphys/): fmt phys2, no images, force noise
0.5 N and residual bias 0.3 N, scene noise 4 mm / 2 deg, contact at 3 N, 25 N safety stop, 40 turns, 80 tool calls,
300 s of simulated time. Options given explicitly always win.
"""
import _bootstrap  # noqa: F401
import argparse
import json

from forceharness.eval.runner import run_one
from forceharness.eval.pricing import price_for
from forceharness.agent.loop import EpisodeConfig
import dataclasses


#: Episode protocol of the HiddenPhys results (the ``episode`` block of every config in configs/hiddenphys/).
PAPER_DEFAULTS = dict(fmt="phys2", vision=False, max_turns=40, max_tool_calls=80, max_sim_time=300.0, max_wall_time=1200.0,
                      scene_noise_pos=0.004, scene_noise_yaw_deg=2.0, f_contact=3.0, f_cap=25.0, force_noise_std=0.5,
                      force_bias_std=0.3, proprio="minimal")
#: This script's own defaults without --paper-defaults (all other fields keep the EpisodeConfig defaults).
DEV_DEFAULTS = dict(fmt="signals", vision=True, max_turns=25)
#: Options that --paper-defaults can supply when they are not given on the command line.
PROTOCOL_OPTIONS = ("fmt", "vision", "max_turns", "max_tool_calls", "max_sim_time", "scene_noise_pos", "f_contact",
                    "force_noise_std", "force_bias_std")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", required=True, help="task name, e.g. weight_sort (see forceharness/hiddenphys/tasks.py)")
    ap.add_argument("--model", default="scripted", help='model id, e.g. gpt-6-astra; "scripted" runs the expert controller')
    ap.add_argument("--provider", default=None, help="openai | openai-responses | anthropic (default: inferred from the model id)")
    ap.add_argument("--base-url", default=None, help="API base URL (default: the provider's, or FH_API_BASE)")
    ap.add_argument("--api-key-env", default=None,
                    help="environment variable holding the API key (default: FH_API_KEY or OPENAI_API_KEY; ANTHROPIC_API_KEY)")
    ap.add_argument("--fmt", default=None,
                    help="how force appears in tool results: none, events, plot, raw, signals, phys1, phys2, ... "
                         "(default: signals; phys2 with --paper-defaults)")
    ap.add_argument("--interface", default="force", choices=["pose", "guard", "force"],
                    help="action interface: force = self-terminating force primitives, pose = position-only primitives")
    ap.add_argument("--vision", dest="vision", action="store_true", default=None,
                    help="attach camera images (default: on; off with --paper-defaults)")
    ap.add_argument("--no-vision", dest="vision", action="store_false", help="no camera images")
    ap.add_argument("--seed", type=int, default=0, help="episode seed (test seeds 9100-9149, development seeds 9000-9019)")
    ap.add_argument("--agent", default=None, choices=[None, "scripted", "llm"],
                    help='default: "scripted" if --model is scripted, else "llm"')
    ap.add_argument("--backend", default="robosuite", choices=["robosuite", "fake"],
                    help="robosuite = MuJoCo benchmark; fake = offline test backend (weight_sort only)")
    ap.add_argument("--run-name", default="dev", help="output folder under --out")
    ap.add_argument("--out", default="runs", help="output root")
    ap.add_argument("--paper-defaults", action="store_true",
                    help="use the episode protocol of the HiddenPhys results for every option not given explicitly")
    ap.add_argument("--max-turns", type=int, default=None, help="model turns per episode (default: 25; 40 with --paper-defaults)")
    ap.add_argument("--max-tool-calls", type=int, default=None, help="tool calls per episode (default: 60; 80 with --paper-defaults)")
    ap.add_argument("--max-sim-time", type=float, default=None,
                    help="simulated seconds per episode (default: 150; 300 with --paper-defaults)")
    ap.add_argument("--scene-noise-pos", type=float, default=None,
                    help="per-episode offset of object positions in the scene, m (default: 0.003; 0.004 with --paper-defaults)")
    ap.add_argument("--f-contact", type=float, default=None, help="contact threshold, N (default: 2.0; 3.0 with --paper-defaults)")
    ap.add_argument("--force-noise-std", type=float, default=None,
                    help="per-tick force noise, N (default: 0; 0.5 with --paper-defaults)")
    ap.add_argument("--force-bias-std", type=float, default=None,
                    help="per-episode residual force bias after taring, N (default: 0; 0.3 with --paper-defaults)")
    ap.add_argument("--force-sign", type=float, default=-1.0, help="sign that turns the sensor reading into the reaction force")
    ap.add_argument("--kp", type=float, default=150.0, help="arm stiffness of the operational-space controller")
    ap.add_argument("--qa-every", type=int, default=0, help="ask a contact-state question every N turns (0 = never)")
    ap.add_argument("--reasoning-effort", default=None, help="reasoning effort passed to the API, e.g. low or high")
    ap.add_argument("--fmt-note", default="", help="free-text note (unused)")
    args = ap.parse_args()
    agent = args.agent or ("scripted" if args.model == "scripted" else "llm")
    pin, pout = price_for(args.model)
    ep = dict(PAPER_DEFAULTS) if args.paper_defaults else dict(DEV_DEFAULTS)
    for k in PROTOCOL_OPTIONS:
        v = getattr(args, k)
        if v is not None:
            ep[k] = v
    cfg = EpisodeConfig(task=args.task, model=args.model, interface=args.interface, seed=args.seed,
                        run_name=args.run_name, out_dir=args.out, force_sign=args.force_sign, kp=args.kp,
                        qa_every=args.qa_every, price_in_per_m=pin, price_out_per_m=pout, **ep)
    rec = run_one(dataclasses.asdict(cfg), dict(name=args.model, provider=args.provider, base_url=args.base_url,
                                                 api_key_env=args.api_key_env, reasoning_effort=args.reasoning_effort),
                  args.backend, agent)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("cfg", "traceback")}, indent=1, default=str))
    if rec.get("traceback"):
        print(rec["traceback"])


if __name__ == "__main__":
    main()
