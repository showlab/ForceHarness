"""Main HiddenPhys table: successes per task for every reference and ForceHarness row on the same test seeds, with damaged
handles and the distribution-shift totals (the layout of the paper's results table). No model calls.

Run folders read under --runs (run_name = config file name; each is produced by the config or script named here):

  expert/results.jsonl                   configs/hiddenphys/expert.yaml (run_matrix.py --agent scripted): force interface ->
                                           "Expert force controller", pose interface -> "Position only" (five tasks)
  expert_peg_position_only/results.jsonl configs/hiddenphys/expert_peg_position_only.yaml -> "Position only" on peg_in_hole_hard
  expert_checks/door_latch_<S>.json      scripts/hiddenphys/door_latch_expert.py <S>: F -> expert, N4tol -> position only,
                                           O -> "Oracle position script"
  expert_checks/lock_or_stiff_<S>_kp150.json
                                         scripts/hiddenphys/lock_or_stiff_expert.py <S> (soft arm): F, N3, O likewise
  learned/*/results.jsonl                scripts/baselines/dp/eval_policy.py --run runs/learned/<policy>_<task>; the record's
                                           model field names the row (LEARNED_ROWS: DP, DP + force, DP images only, pi0.5, ...)
  forceharness_astra/results.jsonl       configs/hiddenphys/forceharness_astra.yaml: "GPT-6 Astra + ForceHarness" (phys2|force)
                                           and "GPT-6 Astra, no force" (none|pose)
  forceharness_fable/results.jsonl       configs/hiddenphys/forceharness_fable.yaml: the same two rows for Claude Fable 5.2
  code_as_policies/results.jsonl         configs/hiddenphys/code_as_policies.yaml: Code-as-Policies without force (cap|pose)
                                           and with our force primitives (cap|force); 5 seeds per cell in the paper
Shift column (four tasks): expert_shift/ (configs/hiddenphys/expert_shift.yaml), forceharness_shift/
(configs/hiddenphys/forceharness_shift.yaml) and learned_shift/*/ (eval_policy.py on the *_ood tasks), each shown as
"in-distribution -> shifted" over weight_sort, com_sort, wipe_hard and peg_in_hole_hard.

Only seeds 9100 .. 9100+n-1 are counted (default n = 20, as in the main table): rows that ran 50 seeds (scripts, DP) are
cut to the same seeds, so episodes are paired across rows; --n 50 shows all 50 seeds of those rows. The position-only
variants of the two mechanism tasks (POSITION_ONLY_CHECK) were fixed beforehand as the best variant on development seeds.
lock_or_stiff uses the soft arm (kp 150) in every row; its script rows count success only with the right verdict, while
the success of model rows is the door outcome (their verdicts are in verdict_correct). Damage counts episodes whose handle
force passed the damage threshold (door_latch, lock_or_stiff).

    python scripts/hiddenphys/compare_table.py --runs runs
    python scripts/hiddenphys/compare_table.py --runs runs --n 5        # first five seeds, e.g. next to Code-as-Policies
"""
import _bootstrap  # noqa: F401
import argparse
import json
from collections import defaultdict
from pathlib import Path

TASKS = ["weight_sort", "com_sort", "wipe_hard", "stack_gentle", "peg_in_hole_hard", "door_latch", "lock_or_stiff"]
SHORT = {"weight_sort": "Weight", "com_sort": "CoM", "wipe_hard": "Wipe", "stack_gentle": "Stack", "peg_in_hole_hard": "Peg",
         "door_latch": "Latch", "lock_or_stiff": "Lock"}
SHIFT_BASE = ["weight_sort", "com_sort", "wipe_hard", "peg_in_hole_hard"]      # task X is shifted as X_ood
MECHANISMS = ("door_latch", "lock_or_stiff")
POSITION_ONLY_CHECK = {"door_latch": "N4tol", "lock_or_stiff": "N3"}      # fixed beforehand: best position-only variant on development seeds
EXPERT, POSITION_ONLY, ORACLE = "Expert force controller", "Position only", "Oracle position script"
LEARNED_ROWS = {"dp": "DP", "dp_force": "DP + force", "dpnoscene": "DP, images only", "dpnoscene_force": "DP, images only + force",
                "pi05_f0": "pi0.5", "pi05_f2": "pi0.5 + force (Prompt Fusion)", "forcevla": "ForceVLA",
                "dp50": "DP, 50 demos", "dp50_force": "DP, 50 demos + force"}
MODEL_NAMES = {"gpt-6-astra": "GPT-6 Astra", "gpt-6.1-sol": "GPT-6.1 Sol", "claude-fable-5-2": "Claude Fable 5.2"}


def _jsonl(p: Path):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def model_row(model: str, fmt: str, interface: str) -> str:
    """Paper name of a model row: phys2|force = ForceHarness, none|pose = the no-force control, cap = Code-as-Policies."""
    name = MODEL_NAMES.get(model, model)
    if fmt == "cap":
        return f"Code-as-Policies ({name})" + (" + force" if interface == "force" else ", no force")
    if (fmt, interface) == ("phys2", "force"):
        return f"{name} + ForceHarness"
    if (fmt, interface) == ("none", "pose"):
        return f"{name}, no force"
    return f"{name} {fmt}|{interface}"


def _damaged(r) -> bool:
    return bool((r.get("truth") or {}).get("damaged") or r.get("damaged"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="root folder that holds the run folders listed above")
    ap.add_argument("--n", type=int, default=20, help="count only seeds 9100 .. 9100+n-1")
    ap.add_argument("--expert-runs", nargs="+", default=["expert", "expert_peg_position_only"],
                    help="matrix runs of the expert scripts (force interface = expert, pose interface = position only)")
    ap.add_argument("--checks-run", default="expert_checks", help="folder with the door_latch / lock_or_stiff script records")
    ap.add_argument("--learned-run", default="learned", help="folder with one subfolder per learned policy and task")
    ap.add_argument("--harness-runs", nargs="+", default=["forceharness_astra", "forceharness_fable", "code_as_policies"],
                    help="model runs (ForceHarness, no-force control, Code-as-Policies)")
    ap.add_argument("--shift-runs", nargs="+", default=["expert_shift", "forceharness_shift"],
                    help="matrix runs on the *_ood tasks (expert rows and model rows)")
    ap.add_argument("--learned-shift-run", default="learned_shift", help="learned policies on the *_ood tasks")
    args = ap.parse_args()
    R = Path(args.runs)
    keep = lambda r: 9100 <= int(r.get("seed", -1)) < 9100 + args.n          # noqa: E731
    cell = defaultdict(lambda: defaultdict(lambda: [0, 0, None]))            # cell[row][task] = [k, n, damaged or None]
    cost = defaultdict(list)
    sources = []

    def add(row, task, success, damaged=None):
        c = cell[row][task]
        c[0] += bool(success); c[1] += 1
        if damaged is not None:
            c[2] = (c[2] or 0) + bool(damaged)

    def script_runs(names):
        for run in names:
            recs = [r for r in _jsonl(R / run / "results.jsonl") if keep(r)]
            sources.append((f"{run}/results.jsonl", len(recs)))
            for r in recs:
                task = r["task"]
                add(EXPERT if r["interface"] == "force" else POSITION_ONLY, task, r.get("success"),
                    _damaged(r) if task in MECHANISMS else None)

    def model_runs(names):
        for run in names:
            recs = [r for r in _jsonl(R / run / "results.jsonl") if keep(r)]
            sources.append((f"{run}/results.jsonl", len(recs)))
            for r in recs:
                row = model_row(r["model"], r["fmt"], r["interface"])
                add(row, r["task"], r.get("success"), _damaged(r) if r["task"] in MECHANISMS else None)
                if not r["task"].endswith("_ood"):
                    cost[row].append(float(r.get("cost_usd") or 0.0))

    def learned_runs(name):
        n = 0
        for d in sorted((R / name).glob("*/results.jsonl")):
            for r in (x for x in _jsonl(d) if keep(x)):
                add(LEARNED_ROWS.get(r["model"], r["model"]), r["task"], r.get("success"), bool(r.get("damaged")))
                n += 1
        sources.append((f"{name}/*/results.jsonl", n))

    script_runs(args.expert_runs)
    chk = R / args.checks_run
    for task, pre in (("door_latch", "door_latch_"), ("lock_or_stiff", "lock_or_stiff_")):
        for strat, row in (("F", EXPERT), (POSITION_ONLY_CHECK[task], POSITION_ONLY), ("O", ORACLE)):
            f = chk / (f"{pre}{strat}.json" if task == "door_latch" else f"{pre}{strat}_kp150.json")
            rs = [x for x in json.loads(f.read_text()) if keep(x)] if f.exists() else []
            sources.append((f"{args.checks_run}/{f.name}", len(rs)))
            if f.exists():
                cell[row].pop(task, None)                                     # the script records replace any matrix cell
                for x in rs:
                    add(row, task, x.get("success"), x.get("damaged"))
    learned_runs(args.learned_run)
    model_runs(args.harness_runs)
    # shifted variants: the same rows on the *_ood tasks
    for run in args.shift_runs:
        recs = [r for r in _jsonl(R / run / "results.jsonl") if keep(r)]
        sources.append((f"{run}/results.jsonl", len(recs)))
        for r in recs:
            row = EXPERT if r["model"].startswith("scripted") and r["interface"] == "force" else (
                POSITION_ONLY if r["model"].startswith("scripted") else model_row(r["model"], r["fmt"], r["interface"]))
            add(row, r["task"], r.get("success"))
    learned_runs(args.learned_shift_run)

    order = [EXPERT, POSITION_ONLY, ORACLE] + list(LEARNED_ROWS.values())
    named = list(MODEL_NAMES.values())
    by_model = lambda r: (next((i for i, nm in enumerate(named) if nm in r), len(named)), "no force" not in r, r)   # noqa: E731
    cap_rows = sorted((r for r in cell if r.startswith("Code-as-Policies")), key=by_model)
    harness_rows = sorted((r for r in cell if r not in order and r not in cap_rows), key=by_model)
    rows = [r for r in order + cap_rows + harness_rows if any(t in cell[r] for t in TASKS)]

    print(f"Successes on seeds 9100-{9100 + args.n - 1} (k/n per task). Damage: episodes past the handle's damage threshold. "
          "Shift: in-distribution -> shifted over four tasks.\n")
    print("| Method | " + " | ".join(SHORT[t] for t in TASKS) + " | Total | Damage | Shift |")
    print("|---|" + "---|" * (len(TASKS) + 3))
    for row in rows:
        c = cell[row]
        out, k_tot, n_tot, dmg = [], 0, 0, None
        for t in TASKS:
            if t in c:
                k, n, d = c[t]
                k_tot += k; n_tot += n
                if d is not None:
                    dmg = (dmg or 0) + d
                out.append(f"{k}/{n}")
            else:
                out.append("—")
        shift = "—"
        if all(t in c and t + "_ood" in c for t in SHIFT_BASE):
            a = sum(c[t][0] for t in SHIFT_BASE); na = sum(c[t][1] for t in SHIFT_BASE)
            b = sum(c[t + "_ood"][0] for t in SHIFT_BASE); nb = sum(c[t + "_ood"][1] for t in SHIFT_BASE)
            shift = f"{a}/{na} -> {b}/{nb}"
        print(f"| {row} | " + " | ".join(out) + f" | {k_tot}/{n_tot} | {'—' if dmg is None else dmg} | {shift} |")
    if cost:
        print("\nModel cost (list price, in-distribution tasks):")
    for r in (r for r in cap_rows + harness_rows if r in cost):
        cs = cost[r]
        print(f"  {r}: {len(cs)} episodes, US${sum(cs):.2f} in total, US${sum(cs) / max(1, len(cs)):.3f} per episode")
    print("\nSources (episodes read):")
    for name, n in sources:
        print(f"  {name}: {n}" + ("" if n else "  (missing or empty)"))


if __name__ == "__main__":
    main()
