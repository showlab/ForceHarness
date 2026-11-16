"""Form-of-force ablation table: the same force-primitive interface, only the way force is rendered in the tool results
changes (the paper's "Form of force" table). No model calls.

Forms (fmt): none = no force (none|pose, the no-force control), events = contact event words, plot = force plot,
raw = tared raw numbers, signals = derived signals, phys1 = physical quantities (L1), phys2 = quantities with
uncertainty and flags (L2, the ForceHarness default). Run folders read under --runs:

  forceharness_other_models  configs/hiddenphys/forceharness_other_models.yaml: none and phys2 for GPT-6.1 Sol and the
                               GPT-5.x models, seeds 9100-9104
  force_form_sol             configs/hiddenphys/force_form_sol.yaml: GPT-6.1 Sol, raw / events / plot / signals / phys1,
                               seeds 9100-9109
  force_form_sol_ext         configs/hiddenphys/force_form_sol_ext.yaml: GPT-6.1 Sol, none and phys2, seeds 9105-9109
  force_form_cheap           configs/hiddenphys/force_form_cheap.yaml: five GPT-5.x models, the five forms, seeds 9100-9104
Episodes are keyed by (model, form, task, seed), so the runs combine into GPT-6.1 Sol on all seven tasks x seeds
9100-9109 (70 episodes per form) and five cheaper models on five tasks (no peg_in_hole_hard, no door_latch) x seeds
9100-9104 (125 episodes per form). Decisions are scored on weight_sort, com_sort and lock_or_stiff. Paired comparisons
use the exact McNemar test on episodes with the same model, task and seed.

    python scripts/hiddenphys/force_form_table.py --runs runs
"""
import argparse
import collections
import json
import math
import os

SOL = "gpt-6.1-sol"
CHEAP = ["gpt-5.4", "gpt-5.2", "gpt-5.1", "gpt-5.4-mini", "gpt-5.6-luna"]
TASKS7 = ["weight_sort", "com_sort", "wipe_hard", "stack_gentle", "peg_in_hole_hard", "door_latch", "lock_or_stiff"]
TASKS5 = ["weight_sort", "com_sort", "wipe_hard", "stack_gentle", "lock_or_stiff"]
JUDGE = ["weight_sort", "com_sort", "lock_or_stiff"]
REPS = ["none", "events", "plot", "raw", "signals", "phys1", "phys2"]
FORM_NAMES = {"none": "No force (none)", "events": "Event words (events)", "plot": "Force plot (plot)", "raw": "Raw numbers (raw)",
              "signals": "Derived signals (signals)", "phys1": "Quantities, L1 (phys1)", "phys2": "+ uncertainty, L2 (phys2)"}
RUNS = ("forceharness_other_models", "force_form_sol", "force_form_sol_ext", "force_form_cheap")


def load(runs, name):
    p = os.path.join(runs, name, "results.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(line) for line in open(p)]


def damaged(x):
    t = x.get("truth") or {}
    return bool(t.get("damaged") or x.get("damaged"))


def mcnemar(a, b):
    """a, b: episodes where only the first / only the second succeeded; exact two-sided p."""
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def collect(runs):
    """{(model, fmt, task, seed): row}"""
    out = {}
    for name in RUNS:
        for x in load(runs, name):
            if x["model"] != SOL and x["model"] not in CHEAP:
                continue
            out[(x["model"], x["fmt"], x["task"], x["seed"])] = x
    return out


def block(data, models, tasks, seeds, title):
    print(f"\n## {title}")
    print("| form | success | rate | Wilson 95% | decisions correct | decision rate | damaged | mean tool calls | US$ per episode | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    res = {}
    for r in REPS:
        rows = [data[k] for k in ((m, r, t, s) for m in models for t in tasks for s in seeds) if k in data]
        n = len(rows)
        ok = sum(bool(x.get("success")) for x in rows)
        jr = [x for x in rows if x["task"] in JUDGE]
        v = sum(bool(x.get("verdict_correct")) for x in jr)
        lo, hi = wilson(ok, n)
        cost = sum(float(x.get("cost_usd") or 0) for x in rows) / max(n, 1)
        calls = sum(float(x.get("n_tool_calls") or 0) for x in rows) / max(n, 1)
        err = sum(bool(x.get("error")) for x in rows)
        dm = sum(damaged(x) for x in rows)
        res[r] = dict(n=n, ok=ok, judge_n=len(jr), judge_ok=v)
        print(f"| {FORM_NAMES[r]} | {ok}/{n} | {ok / max(n, 1):.2f} | {lo:.2f}–{hi:.2f} | {v}/{len(jr)} | {v / max(len(jr), 1):.2f} | {dm} | {calls:.1f} | {cost:.3f} | {err} |")
    return res


def by_task(data, models, tasks, seeds, title):
    print(f"\n### {title}: success per task")
    print("| form | " + " | ".join(tasks) + " |")
    print("|---|" + "---|" * len(tasks))
    for r in REPS:
        cells = []
        for t in tasks:
            rows = [data[k] for k in ((m, r, t, s) for m in models for s in seeds) if k in data]
            cells.append(f"{sum(bool(x.get('success')) for x in rows)}/{len(rows)}")
        print(f"| {FORM_NAMES[r]} | " + " | ".join(cells) + " |")


def by_model(data, models, tasks, seeds):
    print("\n### Success per model")
    print("| model | " + " | ".join(REPS) + " |")
    print("|---|" + "---|" * len(REPS))
    for m in models:
        cells = []
        for r in REPS:
            rows = [data[k] for k in ((m, r, t, s) for t in tasks for s in seeds) if k in data]
            cells.append(f"{sum(bool(x.get('success')) for x in rows)}/{len(rows)}")
        print(f"| {m} | " + " | ".join(cells) + " |")


def paired(data, models, tasks, seeds, a, b):
    only_a = only_b = both = 0
    for m in models:
        for t in tasks:
            for s in seeds:
                x, y = data.get((m, a, t, s)), data.get((m, b, t, s))
                if x is None or y is None:
                    continue
                xa, ya = bool(x.get("success")), bool(y.get("success"))
                only_a += xa and not ya
                only_b += ya and not xa
                both += 1
    return only_a, only_b, both, mcnemar(only_a, only_b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs", help="root folder that holds the run folders listed above")
    args = ap.parse_args()
    data = collect(args.runs)
    s10 = list(range(9100, 9110))
    s5 = list(range(9100, 9105))

    block(data, [SOL], TASKS7, s10, "GPT-6.1 Sol, 7 tasks, seeds 9100-9109")
    by_task(data, [SOL], TASKS7, s10, "GPT-6.1 Sol")
    block(data, CHEAP, TASKS5, s5, "Five cheaper GPT-5.x models pooled, 5 tasks, seeds 9100-9104")
    by_task(data, CHEAP, TASKS5, s5, "Cheaper models")
    by_model(data, CHEAP, TASKS5, s5)

    print("\n### Paired comparisons (same model, task and seed; exact McNemar test)")
    print("| comparison | Sol: only first / only second / p | cheaper models: only first / only second / p |")
    print("|---|---|---|")
    for a, b in (("phys2", "raw"), ("phys2", "events"), ("phys2", "plot"), ("phys2", "signals"), ("phys2", "phys1"), ("phys2", "none"), ("raw", "none")):
        p1 = paired(data, [SOL], TASKS7, s10, a, b)
        p2 = paired(data, CHEAP, TASKS5, s5, a, b)
        print(f"| {a} vs {b} | {p1[0]} / {p1[1]} / {p1[3]:.3g} ({p1[2]} pairs) | {p2[0]} / {p2[1]} / {p2[3]:.3g} ({p2[2]} pairs) |")

    new = []
    for name in RUNS[1:]:
        rows = load(args.runs, name)
        new.append((name, len(rows), sum(float(x.get("cost_usd") or 0) for x in rows), sum(bool(x.get("error")) for x in rows)))
    print("\n### Episodes and cost of the form-of-force runs")
    for name, n, c, e in new:
        print(f"- {name}: {n} episodes, US${c:.2f}, errors {e}")
    print(f"- total US${sum(x[2] for x in new):.2f}")


if __name__ == "__main__":
    main()
