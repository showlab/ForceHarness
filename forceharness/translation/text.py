"""Text force forms (ablations of the physics translation layer): none / raw / stats / events / signals."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from forceharness.primitives.report import ContactReport
from forceharness.translation.base import Injector, Part

AX = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]


class NoneInjector(Injector):
    name = "none"
    hide_force = True

    def __init__(self, *a, **k):
        pass

    def render_force(self, rep, state) -> List[Part]:
        return []

    def describe(self):
        return "You receive NO force or torque information; only motion results."


class RawInjector(Injector):
    """The current 6-D wrench plus the raw numbers of the last n ticks.  N / Nm, world frame, empty-hand bias removed;
    positive = the direction in which the environment pushes the gripper."""
    name = "raw"

    def __init__(self, arg: str = "", n_recent: int = 8, **k):
        self.n_recent = int(arg) if arg else n_recent

    def render_force(self, rep: ContactReport, state) -> List[Part]:
        arr = rep.wrench_array()
        lines = ["wrist force/torque (world frame, N and Nm, bias removed; positive = environment pushing the gripper that way):",
                 f"  current: " + " ".join(f"{a}={v:+.2f}" for a, v in zip(AX, rep.final_wrench))]
        if len(arr):
            k = min(self.n_recent, len(arr))
            lines.append(f"  last {k} samples (t_rel, Fx, Fy, Fz, Tx, Ty, Tz):")
            for row in arr[-k:]:
                lines.append("    " + ", ".join(f"{v:+.2f}" for v in row))
        return [{"type": "text", "text": "\n".join(lines)}]

    def describe(self):
        return "After each primitive you receive the raw wrist force/torque numbers (current value and recent samples)."


class StatsInjector(Injector):
    name = "stats"

    def __init__(self, *a, **k):
        pass

    def render_force(self, rep: ContactReport, state) -> List[Part]:
        arr = rep.wrench_array()
        if len(arr) == 0:
            return [{"type": "text", "text": "wrist force: no samples"}]
        W = arr[:, 1:7]
        fmag = np.linalg.norm(W[:, :3], axis=1)
        q = max(1, len(arr) // 4)
        trend = fmag[-q:].mean() - fmag[:q].mean()
        ipk = int(np.argmax(fmag))
        lines = ["wrist force/torque statistics over this primitive (world frame, bias removed):",
                 "  axis   mean    std    min    max"]
        for i, a in enumerate(AX):
            lines.append(f"  {a:<4} {W[:, i].mean():+7.2f} {W[:, i].std():6.2f} {W[:, i].min():+7.2f} {W[:, i].max():+7.2f}")
        lines.append(f"  |F| peak {fmag[ipk]:.2f} N at t={arr[ipk, 0]:.2f} s; final |F| {fmag[-1]:.2f} N; "
                     f"trend {'rising' if trend > 0.5 else 'falling' if trend < -0.5 else 'flat'} ({trend:+.2f} N first→last quarter)")
        lines.append(f"  mean |F| while above contact threshold: {rep.mean_force_in_contact:.2f} N")
        return [{"type": "text", "text": "\n".join(lines)}]

    def describe(self):
        return "After each primitive you receive summary statistics of the wrist force/torque (mean/std/min/max per axis, peak, trend)."


def events_text(rep: ContactReport) -> str:
    if not rep.events:
        return "contact events: none (no contact detected during this primitive)"
    lines = ["contact events (t relative to primitive start):"]
    for e in rep.events:
        lines.append(f"  t={e.t:.2f}s {e.kind}: {e.detail} (|F|={e.force:.1f} N)")
    return "\n".join(lines)


class EventsInjector(Injector):
    name = "events"

    def __init__(self, *a, **k):
        pass

    def render_force(self, rep: ContactReport, state) -> List[Part]:
        fin = np.asarray((rep.final_wrench_net or rep.final_wrench)[:3])
        in_contact = float(np.linalg.norm(fin)) >= 1.5
        txt = events_text(rep) + f"\ncontact state at end: {'IN CONTACT' if in_contact else 'free (no contact)'}"
        return [{"type": "text", "text": txt}]

    def describe(self):
        return ("After each primitive you receive a list of detected contact EVENTS (contact start/end, safety cap, stall, slip, "
                "resistance drop) and whether the tool is in contact at the end.")


class SignalsInjector(Injector):
    name = "signals"

    def __init__(self, *a, **k):
        pass

    def render_force(self, rep: ContactReport, state) -> List[Part]:
        fin = np.asarray((rep.final_wrench_net or rep.final_wrench)[:3])    # judge contact with the held object's weight removed
        fmag = float(np.linalg.norm(fin))
        lines = [events_text(rep)]
        if fmag >= 1.5:
            i = int(np.argmax(np.abs(fin)))
            lines.append(f"contact state at end: IN CONTACT, |F|={fmag:.1f} N, mainly along {'+' if fin[i] > 0 else '-'}{'xyz'[i]}")
        else:
            lines.append("contact state at end: free (no contact)")
        lines.append(f"peak |F| during primitive: {rep.peak_force:.1f} N; peak |T|: {rep.peak_torque:.2f} Nm")
        derived = {k: v for k, v in rep.derived.items() if k not in ("opening_m", "closed_on_something", "fully_closed", "derive_error")}
        if derived:
            lines.append("derived signals: " + ", ".join(f"{k}={v}" for k, v in derived.items()))
        return [{"type": "text", "text": "\n".join(lines)}]

    def describe(self):
        return ("After each primitive you receive detected contact EVENTS plus derived signals (peak force, contact direction, "
                "payload estimate when lifting, mean normal force when pushing/sliding, torque peak when rotating).")
