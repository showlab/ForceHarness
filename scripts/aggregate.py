"""Aggregate the results.jsonl of one or more runs into a per-condition table (Markdown and JSON).

    python scripts/aggregate.py --run runs/forceharness_astra
    python scripts/aggregate.py --run runs/forceharness_other_models,runs/force_form_cheap --out runs/merged_cheap

One row per condition (task|model|fmt|vision|interface): success k/n with a Wilson 95% interval, over-claimed success,
verdict accuracy, calls, cost, simulated time, peak forces, errors. Prints the table and writes summary.md and
summary.json next to results.jsonl, or into --out when several runs are merged. The layouts of the paper's tables are
printed by scripts/hiddenphys/compare_table.py, expert_table.py, learned_table.py and force_form_table.py.
"""
import _bootstrap  # noqa: F401
import argparse
import json
from pathlib import Path

from forceharness.eval.metrics import aggregate, markdown


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="a runs/<run_name> folder or a results.jsonl path; several separated by commas are merged")
    ap.add_argument("--out", default=None, help="where to write the summary when several runs are merged (default: <first run>/../merged)")
    args = ap.parse_args()
    runs = [Path(r) for r in args.run.split(",")]
    if len(runs) == 1:
        p = runs[0]
        res = p if p.suffix == ".jsonl" else p / "results.jsonl"
        table = aggregate(str(res)); outdir = res.parent
    else:
        import tempfile
        lines = []
        for p in runs:
            res = p if p.suffix == ".jsonl" else p / "results.jsonl"
            if res.exists():
                lines += [l for l in res.read_text().splitlines() if l.strip()]
        tmp = Path(args.out or runs[0].parent / "merged"); tmp.mkdir(parents=True, exist_ok=True)
        (tmp / "results.jsonl").write_text("\n".join(lines) + "\n")
        table = aggregate(str(tmp / "results.jsonl")); outdir = tmp
    md = markdown(table)
    print(md)
    (outdir / "summary.md").write_text(md, encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
