"""phys1-phys4: the four rendering levels of the physics translation layer (``translation.estimators``).

L1 gives each quantity with its value and unit; L2 (phys2, the default ForceHarness tool result) adds the uncertainty
(+- sigma), the window and the flags; L3 adds references (the noise floor, earlier measurements in the same episode);
L4 adds everyday anchors (``anchors.yaml`` next to this file).  The event lines and the end-of-primitive contact state
are the same as in ``signals``, so phys differs from signals only in how the derived quantities are written.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.primitives.report import ContactReport
from forceharness.translation.base import Injector, Part
from forceharness.translation.text import events_text

ANCHORS_PATH = Path(__file__).resolve().parent / "anchors.yaml"

LABELS = {
    "payload_mass": "payload mass", "com_offset": "centre-of-mass offset", "contact_height": "contact height (tool z)",
    "contact_point": "contact point", "contact_normal": "contact normal", "impact_peak": "impact peak force",
    "stiffness": "surface stiffness", "normal_force": "mean normal force", "friction_coefficient": "friction coefficient",
    "insertion_depth": "insertion advance",
    "pull_force": "mean pulling force", "path_turn": "path turned by", "path_radius": "path radius (pivot)",
    "free_direction": "free direction at end",
}


def _load_anchors() -> Dict[str, Any]:
    try:
        import yaml
        return yaml.safe_load(ANCHORS_PATH.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _nearest(table, value: float):
    if not table or value is None or value <= 0:
        return None
    return min(table, key=lambda kv: abs(math.log(max(value, 1e-6)) - math.log(max(float(kv[0]), 1e-6))))


def _class(table, value: float):
    """Band from a threshold table: the description of the first threshold >= value; above the last threshold, the
    last description."""
    if not table or value is None:
        return None
    for th, desc in table:
        if value <= float(th):
            return desc
    return table[-1][1]


def _dir_text(ex: Dict[str, Any]) -> str:
    a = ex.get("direction_deg")
    if a is None:
        return ""
    a = float(a) % 360
    names = [(0, "+x (away from the base)"), (90, "+y (robot's left)"), (180, "-x (toward the base)"), (270, "-y (robot's right)"), (360, "+x (away from the base)")]
    best = min(names, key=lambda kv: abs(kv[0] - a))
    d = abs(best[0] - a)
    return ("mostly " if d <= 30 else "roughly ") + best[1]


class PhysicsInjector(Injector):
    """Renders ContactReport.quantities at level 1-4 (spec "phys1" ... "phys4", or "phys:<level>")."""
    name = "phys"

    def __init__(self, arg: str = "", level: int = 3, **k):
        lvl = int(str(arg).strip()) if str(arg).strip() else int(level)
        self.level = min(4, max(1, lvl))
        self.name = f"phys{self.level}"
        self.anchors = _load_anchors() if self.level >= 4 else {}

    # ------------------------------------------------------------------ one quantity
    def _value_text(self, q: Dict[str, Any]) -> str:
        v, unit = q.get("value"), q.get("unit", "") or ""
        if v is None:
            return "n/a"
        if self.level >= 2 and q.get("sigma") is not None:
            return f"{v:g} ± {q['sigma']:g} {unit}".strip()
        return f"{v:g} {unit}".strip()

    def _references(self, q: Dict[str, Any], log: List[Dict[str, Any]]) -> List[str]:
        name, ex = q.get("name"), q.get("extra") or {}
        out: List[str] = []
        if name == "payload_mass" and ex.get("floor_kg") is not None:
            out.append(f"an empty gripper would read below {ex['floor_kg']:g} kg at this noise level")
        if name == "com_offset" and ex.get("balanced_floor_cm") is not None:
            out.append(f"a load balanced under the wrist would read below {ex['balanced_floor_cm']:g} cm at this noise level")
        if name == "normal_force" and ex.get("target_N") is not None:
            out.append(f"commanded target was {ex['target_N']} N")
        prev = [m for m in log if m.get("name") == name and m.get("value") is not None]
        if prev:
            last = prev[-1]
            out.append(f"earlier in this episode ({last.get('primitive')}): {last['value']:g} {last.get('unit') or ''}".rstrip())
        return out

    def _anchor(self, q: Dict[str, Any]) -> Optional[str]:
        name, v = q.get("name"), q.get("value")
        if v is None:
            return None
        if name == "payload_mass":
            a = _nearest(self.anchors.get("mass_kg"), float(v))
            return f"about the weight of {a[1]} ({a[0]:g} kg)" if a else None
        if name == "friction_coefficient":
            c = _class(self.anchors.get("friction"), float(v))
            return f"{c}" if c else None
        if name == "stiffness":
            c = _class(self.anchors.get("stiffness_N_per_mm"), float(v))
            return f"{c}" if c else None
        if name == "impact_peak":
            c = _class(self.anchors.get("impact_N"), float(v))
            return f"like {c}" if c else None
        return None

    # ------------------------------------------------------------------ the whole tool result
    def render_force(self, rep: ContactReport, state) -> List[Part]:
        lines = [events_text(rep)]
        fin = np.asarray((rep.final_wrench_net or rep.final_wrench)[:3], dtype=float)
        fmag = float(np.linalg.norm(fin))
        if fmag >= 1.5:
            i = int(np.argmax(np.abs(fin)))
            lines.append(f"contact state at end: IN CONTACT, |F|={fmag:.1f} N, mainly along {'+' if fin[i] > 0 else '-'}{'xyz'[i]}")
        else:
            lines.append("contact state at end: free (no contact)")
        qs = [q for q in (rep.quantities or []) if q.get("name") != "estimator_error"]
        if not qs:
            return [{"type": "text", "text": "\n".join(lines)}]
        log = (state or {}).get("log", []) or []
        lines.append(f"measurements ({rep.primitive}):")
        for q in qs:
            name = q.get("name", "")
            label = LABELS.get(name, name.replace("_", " "))
            ex = q.get("extra") or {}
            main = f"  {label:<26s} {self._value_text(q)}"
            if name == "com_offset":
                main += f"  toward {_dir_text(ex)}"
            if name == "free_direction" and ex.get("vector"):
                main += f"  (unit vector {ex['vector']})"
            if name == "path_radius" and ex.get("pivot"):
                main += f"  about a vertical axis at x,y = {ex['pivot']}"
            if name == "path_turn" and ex.get("path_length_m") is not None:
                main += f" over {ex['path_length_m']} m of travel"
            if name == "pull_force" and ex.get("lateral_force_mean_N") is not None:
                main += f"; sideways force {ex['lateral_force_mean_N']} N"
            if name == "contact_normal" and ex.get("vector"):
                main += f"  (unit vector {ex['vector']})"
            if name == "insertion_depth":
                main += f"; lateral force at end {ex.get('lateral_force_N')} N"
            if name in ("contact_height", "contact_point") and ex.get("point"):
                main += f"  at {ex['point']}"
            lines.append(main)
            if self.level >= 2:
                meta = []
                if q.get("window"):
                    meta.append(str(q["window"]))
                if q.get("flags"):
                    meta.append("flags: " + ", ".join(str(f) for f in q["flags"]))
                if meta:
                    lines.append(f"      [{'; '.join(meta)}]")
            if self.level >= 3:
                for r in self._references(q, log):
                    lines.append(f"      ref: {r}")
            if self.level >= 4:
                a = self._anchor(q)
                if a:
                    lines.append(f"      ~ {a}")
        return [{"type": "text", "text": "\n".join(lines)}]

    def describe(self) -> str:
        s = ("After each primitive you receive detected contact EVENTS plus physical MEASUREMENTS computed from the wrist force/torque "
             "(payload mass, centre-of-mass offset of the held load, contact height/point, contact normal, impact peak, surface stiffness, "
             "normal force, friction coefficient, insertion advance; for pull: pulling force, how much the path turned, its pivot, "
             "the free direction), each with units.")
        if self.level >= 2:
            s += " Each measurement carries an uncertainty (±) and quality flags."
        if self.level >= 3:
            s += " Reference lines say what the same measurement would read if nothing were there (noise floor) and what earlier measurements in this episode gave."
        if self.level >= 4:
            s += " An everyday comparison (e.g. 'about the weight of an apple') is added."
        return s
