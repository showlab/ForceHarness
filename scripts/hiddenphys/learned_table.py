"""Table of the learned reference policies on HiddenPhys (Diffusion Policy, pi0.5 and ForceVLA, with and without force).
No model calls.

Reads <run>/*/results.jsonl as written by scripts/baselines/dp/eval_policy.py (one folder per policy and task, e.g.
runs/learned/dp_force_weight_sort/) and prints one row per task and policy: successes k/n with a Wilson 95% interval,
median peak wrist force, episodes above the 25 N force cap (learned policies drive the controller directly and have no
safety stop, so this is reported separately), damaged handles, median simulated time, errors. lock_or_stiff is also
split by the true class: a policy that does nothing passes every locked door (the door stays shut and undamaged).

The record's model field (the --name given to eval_policy.py) maps to the paper's rows: dp = DP, dp_force = DP + force,
dpnoscene = DP, images only, dpnoscene_force = DP, images only + force, pi05_f0 = pi0.5, pi05_f2 = pi0.5 + force
(Prompt Fusion), forcevla = ForceVLA; dp50 / dp50_force are the 50-demonstration ablation. The paper evaluates on test
seeds 9100-9119 (DP also on 9100-9149); the shifted variants live in a separate folder (learned_shift).

    python scripts/hiddenphys/learned_table.py --runs runs                       # runs/learned
    python scripts/hiddenphys/learned_table.py --runs runs --run-name learned_shift
"""
import _bootstrap  # noqa: F401
import argparse
import json
import statistics
from pathlib import Path

from forceharness.eval.metrics import wilson

LEARNED_ROWS = {"dp": "DP", "dp_force": "DP + force", "dpnoscene": "DP, images only", "dpnoscene_force": "DP, images only + force",
                "pi05_f0": "pi0.5", "pi05_f2": "pi0.5 + force (Prompt Fusion)", "forcevla": "ForceVLA",
                "dp50": "DP, 50 demos", "dp50_force": "DP, 50 demos + force"}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="root folder of the run folders")
    ap.add_argument("--run-name", default="learned", help="folder under --runs with one subfolder per policy and task")
    ap.add_argument("--run", default=None, help="explicit folder (overrides --runs / --run-name)")
    ap.add_argument("--fcap", type=float, default=25.0, help="force cap of the harness primitives, N (reported, not enforced)")
    args = ap.parse_args()
    run = Path(args.run) if args.run else Path(args.runs) / args.run_name
    rows = []
    for f in sorted(run.glob("*/results.jsonl")):
        recs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        task, model = recs[0]["task"], recs[0]["model"]
        n = len(recs)
        k = sum(bool(r.get("success")) for r in recs)
        p, lo, hi = wilson(k, n)
        peaks = [r["peak_wrench_N"] for r in recs if r.get("peak_wrench_N") is not None]
        row = dict(task=task, model=model, n=n, k=k, rate=round(p, 3), ci=[round(lo, 3), round(hi, 3)],
                   peak_median=round(statistics.median(peaks), 1) if peaks else None,
                   over_fcap=sum(x > args.fcap for x in peaks), damaged=sum(bool(r.get("damaged")) for r in recs),
                   sim_median=round(statistics.median([r.get("sim_time_s", 0) for r in recs]), 1),
                   errors=sum(bool(r.get("error")) for r in recs))
        labels = sorted({(r.get("truth") or {}).get("label") for r in recs} - {None})
        if task == "lock_or_stiff" and labels:
            row["by_label"] = {lb: f"{sum(bool(r.get('success')) for r in recs if (r.get('truth') or {}).get('label') == lb)}/"
                                   f"{sum((r.get('truth') or {}).get('label') == lb for r in recs)}" for lb in labels}
        rows.append(row)
    if not rows:
        raise SystemExit(f"no results under {run}/*/results.jsonl")
    (run / "summary_learned.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"| task | policy | success | 95% CI | median peak wrist force (N) | episodes > {args.fcap:g} N | damaged | median sim time (s) | by door class | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        bl = "; ".join(f"{k} {v}" for k, v in (r.get("by_label") or {}).items()) or "—"
        name = LEARNED_ROWS.get(r["model"], r["model"])
        print(f"| {r['task']} | {name} | {r['k']}/{r['n']} | {r['ci'][0]:.2f}–{r['ci'][1]:.2f} | {r['peak_median']} | {r['over_fcap']} | "
              f"{r['damaged']} | {r['sim_median']} | {bl} | {r['errors']} |")


if __name__ == "__main__":
    main()
