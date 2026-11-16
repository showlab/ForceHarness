"""Aggregation per condition: success rate with a Wilson 95% interval, over-claimed success, verdict accuracy,
mean calls / tokens / cost / time / peak force, and the accuracy of the contact question."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def wilson(k: int, n: int, z: float = 1.96):
    """Success rate k/n with its Wilson score interval; returns (p, lo, hi)."""
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (p, max(0.0, c - h), min(1.0, c + h))


def _mean(xs: List[float]) -> float:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else float("nan")


def aggregate(results_path: str) -> List[Dict[str, Any]]:
    """Read results.jsonl and return one summary row per condition (sorted by condition name)."""
    rows = [json.loads(l) for l in Path(results_path).read_text().splitlines() if l.strip()]
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    out = []
    for cond, rs in sorted(groups.items()):
        n = len(rs)
        k = sum(bool(r.get("success")) for r in rs)
        p, lo, hi = wilson(k, n)
        claimed = [r for r in rs if r.get("claimed_success") is not None]
        over = sum(bool(r.get("claimed_success")) and not r.get("success") for r in claimed)
        vc = [r["verdict_correct"] for r in rs if r.get("verdict_correct") is not None]
        qa_asked = sum(r.get("qa_asked", 0) or 0 for r in rs)
        qa_correct = sum(r.get("qa_correct", 0) or 0 for r in rs)
        out.append(dict(
            condition=cond, n=n, success=k, rate=round(p, 3), ci_lo=round(lo, 3), ci_hi=round(hi, 3),
            overclaim_rate=round(over / len(claimed), 3) if claimed else None,
            verdict_acc=round(sum(vc) / len(vc), 3) if vc else None,
            calls=round(_mean([r.get("n_model_calls") for r in rs]), 1),
            tool_calls=round(_mean([r.get("n_tool_calls") for r in rs]), 1),
            tokens_in=round(_mean([r.get("tokens_in") for r in rs])),
            tokens_out=round(_mean([r.get("tokens_out") for r in rs])),
            cost_usd=round(_mean([r.get("cost_usd") for r in rs]), 3),
            sim_time_s=round(_mean([r.get("sim_time_s") for r in rs]), 1),
            wall_s=round(_mean([r.get("wall_time_s") for r in rs]), 0),
            peak_wrench_N=round(_mean([r.get("peak_wrench_N") for r in rs]), 1),
            peak_obj_force_N=round(_mean([r.get("peak_object_contact_force_N") for r in rs]), 1),
            force_caps=round(_mean([r.get("n_force_cap") for r in rs]), 2),
            qa_acc=round(qa_correct / qa_asked, 3) if qa_asked else None,
            errors=sum(bool(r.get("error")) for r in rs),
        ))
    return out


def markdown(table: List[Dict[str, Any]]) -> str:
    """Render aggregate() rows as a Markdown table."""
    if not table:
        return "(no results)"
    cols = ["condition", "n", "rate", "ci_lo", "ci_hi", "overclaim_rate", "verdict_acc", "calls", "cost_usd", "sim_time_s",
            "peak_wrench_N", "peak_obj_force_N", "force_caps", "qa_acc", "errors"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in table:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r.get(c)) for c in cols) + " |")
    return "\n".join(lines)
