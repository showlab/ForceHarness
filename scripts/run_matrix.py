"""Run a matrix of episodes from a YAML config and append one record per episode to <out>/<run_name>/results.jsonl.

configs/hiddenphys/ has one config per HiddenPhys result (see docs/hiddenphys.md); the config fixes tasks, models,
conditions (fmt|interface), seeds (seed_base + 0 .. n_episodes - 1) and the episode protocol.

    # quick smoke test with the expert, offline (no simulator, no API key)
    python scripts/run_matrix.py --matrix configs/hiddenphys/smoke.yaml --agent scripted --backend fake --workers 1
    # expert and position-only rows on MuJoCo (no API key)
    python scripts/run_matrix.py --matrix configs/hiddenphys/expert.yaml --agent scripted --workers 8
    # ForceHarness and the no-force control with GPT-6 Astra (paid API calls: --dry-run prints the episode count first)
    python scripts/run_matrix.py --matrix configs/hiddenphys/forceharness_astra.yaml --dry-run
    python scripts/run_matrix.py --matrix configs/hiddenphys/forceharness_astra.yaml --workers 6

Runs resume: episodes whose episode_id is already in results.jsonl are skipped, so raising n_episodes in a config and
running it again adds only the new seeds. Every episode builds its own simulator in its own process; for model rows the
API latency dominates, so 4-8 workers are enough.
"""
import _bootstrap  # noqa: F401
import argparse

from forceharness.eval.runner import run_matrix


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matrix", required=True, help="YAML matrix config, e.g. configs/hiddenphys/forceharness_astra.yaml")
    ap.add_argument("--workers", type=int, default=4, help="parallel episodes (one process each)")
    ap.add_argument("--agent", default="llm", choices=["llm", "scripted"],
                    help="llm: model rows; scripted: the expert / position-only scripts (expert*.yaml, smoke.yaml)")
    ap.add_argument("--backend", default="robosuite", choices=["robosuite", "fake"],
                    help="robosuite = MuJoCo benchmark; fake = offline test backend (weight_sort only)")
    ap.add_argument("--limit", type=int, default=None, help="run at most this many of the pending episodes")
    ap.add_argument("--dry-run", action="store_true", help="print the episode counts and the first pending episodes, run nothing")
    ap.add_argument("--out", default="runs", help="output root; records go to <out>/<run_name>/results.jsonl")
    args = ap.parse_args()
    run_matrix(args.matrix, workers=args.workers, backend=args.backend, agent_kind=args.agent, limit=args.limit,
               dry_run=args.dry_run, out_dir=args.out)


if __name__ == "__main__":
    main()
