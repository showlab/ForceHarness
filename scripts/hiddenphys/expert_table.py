"""Table of the scripted references on HiddenPhys (expert force controller, position-only controller, oracle position
script, and the expert with constants taken from the instruction), the layout of the supplement's script table. No model
calls.

Matrix runs (results.jsonl), one row per task and (model, interface) of the record (MATRIX_ROW):
  expert                    configs/hiddenphys/expert.yaml: "Expert force controller" (force interface) and "Position
                              only" (the same script on the pose interface) on five tasks
  expert_peg_position_only  configs/hiddenphys/expert_peg_position_only.yaml: "Position only" on peg_in_hole_hard
  expert_default_params_b   configs/hiddenphys/expert_default_params_b.yaml: expert with the 3 cm search radius stated in
                              the peg_in_hole_hard instruction
  expert_default_params     configs/hiddenphys/expert_default_params.yaml: expert with the 1 cm of the scene note
Script records (<checks-run>/*.json, default expert_checks): door_latch_<S>.json and lock_or_stiff_<S>_kp<kp>.json from
scripts/hiddenphys/door_latch_expert.py and lock_or_stiff_expert.py (CHECK_ROW names the strategies).

Each row: successes k/n with a Wilson 95% interval, correct verdicts (tasks with a verdict), whether the door ended right
(lock_or_stiff: the physical part alone, comparable with learned policies), damaged handles, median peak force (at the
wrist; at the handle for the mechanism scripts), median simulated time, errors, and successes on the first 20 test seeds
(the seeds of the main table). Writes <runs>/<first run>/summary_expert.json.

    python scripts/hiddenphys/expert_table.py --runs runs
    python scripts/hiddenphys/expert_table.py --runs runs --run-names expert --checks-run expert_checks
"""
import _bootstrap  # noqa: F401
import argparse
import json
import statistics
from pathlib import Path

from forceharness.eval.metrics import wilson

MATRIX_ROW = {("scripted", "force"): "Expert force controller", ("scripted", "pose"): "Position only",
              ("scripted_pos_only", "pose"): "Position only",
              ("scripted_r3cm", "force"): "Expert, search radius 3 cm (from the instruction)",
              ("scripted_r1cm", "force"): "Expert, search radius 1 cm (from the scene note)"}
CHECK_ROW = {"F": "Expert force controller", "O": "Oracle position script (knows the hidden variable)",
             "N3": "Position only, stepped probe, 0.5 deg rule (N3)", "N3f": "Position only, stepped probe, 0.1 deg rule (N3f)",
             "N4tol": "Position only, stepped probe, 2 deg + 2 mm (N4tol)", "N5half": "Position only, alternating probe (N5half)"}
FIRST20 = range(9100, 9120)


def _med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def _row(task, row, recs, peak_key, verdict_key=None, damaged_key="damaged", sim_key=None):
    n = len(recs)
    k = sum(bool(r.get("success")) for r in recs)
    p, lo, hi = wilson(k, n)
    out = dict(task=task, row=row, n=n, k=k, rate=round(p, 3), ci=[round(lo, 3), round(hi, 3)],
               peak_N_median=_med([r.get(peak_key) for r in recs]), errors=sum(bool(r.get("error")) for r in recs),
               first20=sum(bool(r.get("success")) for r in recs if int(r.get("seed", -1)) in FIRST20))
    if verdict_key:
        vs = [r.get(verdict_key) for r in recs if r.get(verdict_key) is not None]
        out["verdict_ok"] = f"{sum(bool(v) for v in vs)}/{len(vs)}" if vs else None
    if any(damaged_key in r for r in recs):
        out["damaged"] = sum(bool(r.get(damaged_key)) for r in recs)
    if sim_key:
        out["sim_s_median"] = _med([r.get(sim_key) for r in recs])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="root folder of the run folders")
    ap.add_argument("--run-names", nargs="+",
                    default=["expert", "expert_peg_position_only", "expert_default_params_b", "expert_default_params"],
                    help="matrix runs of the scripts (missing ones are skipped)")
    ap.add_argument("--checks-run", default="expert_checks", help="folder with the door_latch / lock_or_stiff script records")
    args = ap.parse_args()
    R = Path(args.runs)
    rows = []
    for name in args.run_names:
        res = R / name / "results.jsonl"
        if not res.exists():
            continue
        recs = [json.loads(l) for l in res.read_text().splitlines() if l.strip()]
        groups = {}
        for r in recs:
            groups.setdefault((r["task"], MATRIX_ROW.get((r["model"], r["interface"]), f"{r['model']} | {r['interface']}")), []).append(r)
        for (task, row), rs in sorted(groups.items()):
            rows.append(_row(task, row, rs, "peak_wrench_N", verdict_key="verdict_correct", sim_key="sim_time_s"))
    checks = R / args.checks_run
    for f in sorted(checks.glob("*.json")):
        rs = json.loads(f.read_text())
        stem = f.stem                                   # door_latch_F / lock_or_stiff_N3_kp150
        if stem.startswith("door_latch_"):
            task, strat = "door_latch", stem[len("door_latch_"):]
            rows.append(_row(task, CHECK_ROW.get(strat, f"Position only ({strat})"), rs, "peak_handle_N"))
        elif stem.startswith("lock_or_stiff_"):
            strat, kp = stem[len("lock_or_stiff_"):].rsplit("_kp", 1)
            arm = {"150": "soft arm, kp 150", "1500": "stiff arm, kp 1500"}.get(kp, f"kp {kp}")
            row = _row(f"lock_or_stiff ({arm})", CHECK_ROW.get(strat, f"Position only ({strat})"), rs, "peak_N", verdict_key="verdict_ok")
            row["door_ok"] = f"{sum(bool(r.get('door_ok')) for r in rs)}/{len(rs)}"     # the physical outcome alone (comparable with learned policies)
            rows.append(row)
    if not rows:
        raise SystemExit(f"no script results under {R} (expected {args.run_names[0]}/results.jsonl or {args.checks_run}/*.json)")
    out_dir = R / args.run_names[0] if (R / args.run_names[0]).is_dir() else checks
    (out_dir / "summary_expert.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print("| task | row | success | 95% CI | verdict correct | door correct | damaged | median peak force (N) | median sim time (s) | errors | first 20 seeds |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['task']} | {r['row']} | {r['k']}/{r['n']} | {r['ci'][0]:.2f}–{r['ci'][1]:.2f} | {r.get('verdict_ok') or '—'} | "
              f"{r.get('door_ok') or '—'} | {r.get('damaged', '—')} | {r['peak_N_median']} | {r.get('sim_s_median') or '—'} | {r['errors']} | "
              f"{r['first20']} |")


if __name__ == "__main__":
    main()
