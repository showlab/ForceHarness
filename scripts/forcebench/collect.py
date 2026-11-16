#!/usr/bin/env python3
"""Merge the records of both sides of one ForceBench run by seed: the evaluator's eval/seed_*/summary.json (ForceBench's own
success criteria; authoritative) and the policy server's results.jsonl (model calls, tokens, cost, the agent's verdict and
claimed success, primitive results). Writes merged.jsonl and a Markdown table summary.md into the run directory.
run.sh calls it at the end of every run.

  python3 scripts/forcebench/collect.py --run $FB_RUNS/<run name>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--eval-subdirs", nargs="*", default=None, help="default: every eval*/ directory of the run")
    args = ap.parse_args()
    run = args.run
    evals = [run / d for d in args.eval_subdirs] if args.eval_subdirs else sorted(p for p in run.glob("eval*") if p.is_dir())
    fb = {}
    for ed in evals:
        for s in sorted(ed.glob("seed_*/summary.json")):
            j = json.loads(s.read_text())
            fb[int(j["seed"])] = dict(fb_success=bool(j["success"]), fb_termination=j.get("termination_reason"),
                                     fb_first_success_step=j.get("first_success_step"), fb_steps=j.get("steps"),
                                     fb_clamped=j.get("clamped_action_frames"), fb_task=j.get("task"),
                                     fb_criteria={k: v for k, v in (j.get("final_success_report") or {}).items()
                                                  if isinstance(v, (bool, int, float, str))},
                                     eval_dir=ed.name)
    harness = {}
    rp = run / "results.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                harness[int(r["seed"])] = r
    rows = []
    for seed in sorted(set(fb) | set(harness)):
        h = harness.get(seed, {})
        rows.append(dict(seed=seed, **fb.get(seed, {}),
                         verdict=h.get("verdict"), claimed_success=h.get("claimed_success"), ended_by=h.get("ended_by"),
                         stop_reason=h.get("stop_reason"), n_model_calls=h.get("n_model_calls"), n_tool_calls=h.get("n_tool_calls"),
                         tokens_in=h.get("tokens_in"), tokens_out=h.get("tokens_out"), cost_usd=h.get("cost_usd"),
                         peak_wrench_N=h.get("peak_wrench_N"), n_force_cap=h.get("n_force_cap"), error=h.get("error")))
    (run / "merged.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows), encoding="utf-8")
    k = sum(1 for r in rows if r.get("fb_success"))
    n = sum(1 for r in rows if "fb_success" in r)
    lo, hi = wilson(k, n)
    cost = sum(float(r.get("cost_usd") or 0) for r in rows)
    lines = [f"# {run.name}", "", f"Success by ForceBench's criteria: **{k}/{n}** (Wilson 95% {lo:.2f}–{hi:.2f}); cost US${cost:.2f}", "",
             "| seed | success | termination | first success step | verdict | claimed success | model calls | tool calls | peak force N | stop reason |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| {seed} | {s} | {t} | {f} | {v} | {c} | {m} | {tc} | {pk} | {st} |".format(
            seed=r["seed"], s="✅" if r.get("fb_success") else "❌", t=r.get("fb_termination"), f=r.get("fb_first_success_step"),
            v=r.get("verdict") or "", c=r.get("claimed_success"), m=r.get("n_model_calls"), tc=r.get("n_tool_calls"),
            pk=r.get("peak_wrench_N"), st=r.get("stop_reason")))
    (run / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
