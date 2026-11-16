"""Forms of force: different renderings of the same ContactReport (layer 3 and its ablations).

The form of force is one of the independent variables of the experiments (a row of the evaluation matrix).

| name | what the model receives |
|---|---|
| none | only the kinematic result (where the tool ended, how far it moved, whether it was blocked); no force numbers |
| raw | the current 6-D wrench and the raw numbers of the last few ticks |
| stats | per-axis mean / std / min / max, time of the peak, trend |
| events | one sentence per detected event: contact start / end, stall, slip, resistance drop, ... |
| signals | events + primitive-specific derived signals (payload estimate, mean normal force, angle of peak torque, ...) |
| plot | force / torque versus time rendered as a PNG for a multimodal model |
| phys1-phys4 | the physics translation layer at levels L1-L4 (``translation.quantities``); phys2 is the ForceHarness default |

Forms compose with "+", e.g. "events+plot"; an argument follows a colon, e.g. "raw:16" or "phys:2".
"""
from __future__ import annotations

from typing import Any, Dict, List

from forceharness.primitives.report import ContactReport

Part = Dict[str, Any]   # {"type": "text", "text": str} | {"type": "image", "png": bytes, "label": str}

FORMATS = ["none", "raw", "stats", "events", "signals", "plot", "phys1", "phys2", "phys3", "phys4"]


def _fmt3(v) -> str:
    return "[" + ", ".join(f"{float(x):+.3f}" for x in v) + "]"


def _short_of_target(rep: ContactReport):
    """How far (m) a move_to / move_by / lift that did not finish stopped short of its commanded target.  Computed from
    params, not from commanded_disp, which the lead clamp inflates."""
    import numpy as np
    p = rep.params or {}
    s, e = np.asarray(rep.start_pos, dtype=float), np.asarray(rep.end_pos, dtype=float)
    try:
        if rep.primitive == "move_to" and all(k in p for k in ("x", "y", "z")):
            return float(np.linalg.norm(np.array([p["x"], p["y"], p["z"]], dtype=float) - e))
        if rep.primitive == "move_by" and all(k in p for k in ("dx", "dy", "dz")):
            return float(np.linalg.norm(s + np.array([p["dx"], p["dy"], p["dz"]], dtype=float) - e))
        if rep.primitive == "lift":
            return float(abs(s[2] + float(p.get("height", 0.08)) - e[2]))
    except Exception:
        return None
    return None


def kinematic_summary(rep: ContactReport, hide_force: bool, proprio: str = "full") -> str:
    """The kinematic part shared by every force form.  hide_force=True rewrites force-related termination reasons without
    numbers.  proprio="minimal" (the EpisodeConfig default) omits the duration and the gripper opening while carrying:
    both change with the load, so without force information they would be a side channel (a proprioceptive leak)."""
    term = rep.termination
    if hide_force:
        term = {"contact": "stopped early (motion blocked)", "force_cap": "stopped early (motion blocked, safety stop)",
                "torque_cap": "stopped early (rotation blocked, safety stop)", "stall": "stopped early (not moving)",
                "resistance_drop": "stopped early", "no_contact_within_max_dist": "completed (nothing blocked the motion)"}.get(term, term)
    lines = [f"primitive={rep.primitive} params={rep.params}",
             f"result: {term}" + (f"; duration {rep.duration:.2f} s" if proprio == "full" else ""),
             f"end position {_fmt3(rep.end_pos)} m; commanded displacement {_fmt3(rep.commanded_disp)}, actual {_fmt3(rep.actual_disp)}"]
    if rep.gripper_opening is not None and proprio == "full":
        lines.append(f"gripper opening {rep.gripper_opening*1000:.1f} mm")
    short = _short_of_target(rep) if rep.termination != "completed" else None
    if short is not None and short > 0.005:
        lines.append(f"NOTE: the target was NOT reached: stopped {short:.3f} m short of it (see end position)")
    if rep.yaw_deg:
        lines.append(f"tool yaw changed by {rep.yaw_deg:+.1f} deg")
    if rep.error:
        lines.append(f"error: {rep.error}")
    # derived values that involve no force (e.g. whether the gripper closed on something) are kinematic too
    for k in ("opening_m", "closed_on_something", "fully_closed", "contact_found", "contact_distance_m", "contact_pos", "rotated_deg",
              "wrist_limit_clipped", "requested_deg", "executed_deg", "note", "arg_notes"):
        if k == "opening_m" and proprio != "full" and rep.primitive not in ("grasp", "release"):
            continue
        if k in rep.derived and not (hide_force and k in ("contact_found", "contact_distance_m", "contact_pos")):
            lines.append(f"{k}={rep.derived[k]}")
    return "\n".join(lines)


class Injector:
    """Base class of a force form.  Subclasses implement ``render_force(rep, state)`` (the force part of a tool
    result) and ``describe()``; ``Composite`` prepends the shared kinematic summary."""
    name = "base"
    hide_force = False

    def render(self, rep: ContactReport, state: Dict[str, Any]) -> List[Part]:
        raise NotImplementedError

    def describe(self) -> str:
        """One sentence for the system prompt: how the model will receive force information."""
        return ""


class Composite(Injector):
    """The kinematic summary followed by the force parts of one or more forms (spec "a+b")."""
    def __init__(self, parts: List[Injector]):
        self.parts = parts
        self.name = "+".join(p.name for p in parts)
        self.hide_force = all(p.hide_force for p in parts)

    def render(self, rep, state):
        out: List[Part] = [{"type": "text", "text": kinematic_summary(rep, self.hide_force, (state or {}).get("proprio", "full"))}]
        for p in self.parts:
            out.extend(p.render_force(rep, state))
        return out

    def describe(self):
        return " ".join(p.describe() for p in self.parts if p.describe())


def make_injector(spec: str, **kwargs) -> Injector:
    """Build the injector for a spec such as "phys2", "events+plot" or "raw:16" (see the module docstring)."""
    from forceharness.translation.text import NoneInjector, RawInjector, StatsInjector, EventsInjector, SignalsInjector
    from forceharness.translation.plot import PlotInjector
    from forceharness.translation.quantities import PhysicsInjector

    reg = {"none": NoneInjector, "raw": RawInjector, "stats": StatsInjector, "events": EventsInjector,
           "signals": SignalsInjector, "plot": PlotInjector, "phys": PhysicsInjector,
           "phys1": lambda *a, **k: PhysicsInjector("1", **k), "phys2": lambda *a, **k: PhysicsInjector("2", **k),
           "phys3": lambda *a, **k: PhysicsInjector("3", **k), "phys4": lambda *a, **k: PhysicsInjector("4", **k)}
    parts = []
    for tok in spec.split("+"):
        tok = tok.strip()
        name, _, arg = tok.partition(":")
        if name not in reg:
            raise KeyError(f"unknown force form {name}; choose from {list(reg)}")
        parts.append(reg[name](arg, **kwargs) if arg else reg[name](**kwargs))
    return Composite(parts)
