"""Streaming contact-event detection (layer 2, used inside every primitive).

Feed one force reading per control tick; ``update`` returns the events produced on that tick.  The rules are
thresholds with hysteresis and are deliberately not learned.

Event semantics (the key signals the model sees):
- contact_start / contact_end: |F| crosses the contact threshold (with hysteresis, for k consecutive ticks)
- force_cap / torque_cap: a safety limit is crossed (the primitive stops at once)
- stall: in contact, the commanded displacement is much larger than the actual one (stuck / pushing against something)
- slip: while holding something, the force drops suddenly (the object slipped)
- resistance_drop: the resistance rises to a peak and then clearly falls (a latch giving way, a valve passing its
  peak, a part dropping into a hole)
"""
from __future__ import annotations

from collections import deque
from typing import List, Optional

import numpy as np

from forceharness.primitives.report import ContactEvent

AXES = ("x", "y", "z")


def main_axis(v: np.ndarray) -> str:
    i = int(np.argmax(np.abs(v)))
    return f"{'+' if v[i] >= 0 else '-'}{AXES[i]}"


class EventDetector:
    """Threshold-and-hysteresis event detector.  A Primitives object owns one and resets it at the start of every
    primitive; all thresholds are in newtons (forces) or newton-metres (torques)."""

    def __init__(
        self,
        f_contact: float = 2.0,
        f_release: float = 1.0,
        f_cap: float = 25.0,
        t_cap: float = 3.0,
        min_samples: int = 2,
        stall_window: int = 6,
        stall_ratio: float = 0.35,
        slip_window: int = 6,
        slip_drop: float = 0.5,
        breakaway_drop: float = 0.35,
        breakaway_min_peak: float = 4.0,
        filter_n: int = 4,
    ):
        self.filter_n = filter_n
        self.f_contact, self.f_release, self.f_cap, self.t_cap = f_contact, f_release, f_cap, t_cap
        self.min_samples = min_samples
        self.stall_window, self.stall_ratio = stall_window, stall_ratio
        self.slip_window, self.slip_drop = slip_window, slip_drop
        self.breakaway_drop, self.breakaway_min_peak = breakaway_drop, breakaway_min_peak
        self.reset()

    def reset(self) -> None:
        self.in_contact = False
        self._above = 0
        self._below = 0
        self._cmd = deque(maxlen=self.stall_window)
        self._act = deque(maxlen=self.stall_window)
        self._fhist = deque(maxlen=self.slip_window)
        self._ffilt = deque(maxlen=self.filter_n)
        self._stall_armed = True
        self._peak = 0.0
        self._breakaway_armed = True
        self.t0: Optional[float] = None

    def update(self, t: float, reaction: np.ndarray, torque: np.ndarray, pos: np.ndarray,
               commanded_step: np.ndarray, actual_step: np.ndarray, grip_closed: bool,
               suppress_contact: bool = False) -> List[ContactEvent]:
        """Process one tick and return the new events.

        suppress_contact=True: do not report contact_start on this tick even above the threshold (lifting a held
        object makes the load grow gradually; that is not a collision)."""
        if self.t0 is None:
            self.t0 = t
        tr = t - self.t0
        f_raw = float(np.linalg.norm(reaction))
        self._ffilt.append(f_raw)
        f = float(np.mean(self._ffilt))          # moving-average |F|, used by every threshold test (a 4-tick mean turns 0.5 N noise into 0.25 N)
        tq = float(np.linalg.norm(torque))
        ev: List[ContactEvent] = []
        p = [round(float(x), 4) for x in pos]

        # contact start / end (hysteresis + consecutive count)
        if f >= self.f_contact:
            self._above += 1; self._below = 0
        elif f < self.f_release:
            self._below += 1; self._above = 0
        if not self.in_contact and self._above >= self.min_samples and not suppress_contact:
            self.in_contact = True
            self._peak = f
            self._breakaway_armed = True
            self._stall_armed = True
            ev.append(ContactEvent(tr, "contact_start", f, f"reaction force mainly along {main_axis(reaction)}", p))
        elif self.in_contact and self._below >= self.min_samples:
            self.in_contact = False
            ev.append(ContactEvent(tr, "contact_end", f, "force dropped to ~0", p))

        # safety limits (on the raw value, so they react at once)
        if f_raw >= self.f_cap:
            ev.append(ContactEvent(tr, "force_cap", f, f"|F| >= {self.f_cap:g} N, along {main_axis(reaction)}", p))
        if tq >= self.t_cap:
            ev.append(ContactEvent(tr, "torque_cap", f, f"|T| >= {self.t_cap:g} Nm", p))

        # stall: in contact, the command moves but the hand barely does
        self._cmd.append(float(np.linalg.norm(commanded_step)))
        self._act.append(float(np.linalg.norm(actual_step)))
        if self.in_contact and len(self._cmd) == self.stall_window:
            c, a = sum(self._cmd), sum(self._act)
            if c > 2e-3 and a < self.stall_ratio * c:
                if self._stall_armed:
                    self._stall_armed = False
                    ev.append(ContactEvent(tr, "stall", f, f"commanded {c*1000:.0f} mm but moved {a*1000:.0f} mm; reaction along {main_axis(reaction)}", p))
            else:
                self._stall_armed = True

        # slip: a sudden force drop while holding something
        self._fhist.append(f)
        if grip_closed and len(self._fhist) == self.slip_window:
            fmax = max(self._fhist)
            if fmax > self.f_contact and f < (1 - self.slip_drop) * fmax and self._fhist[-2] >= (1 - self.slip_drop) * fmax:
                ev.append(ContactEvent(tr, "slip", f, f"force fell from {fmax:.1f} N to {f:.1f} N within {self.slip_window} samples", p))

        # resistance passes a peak and then falls
        if self.in_contact:
            if f > self._peak:
                self._peak = f
                self._breakaway_armed = True
            elif self._breakaway_armed and self._peak >= self.breakaway_min_peak and f < (1 - self.breakaway_drop) * self._peak \
                    and float(np.linalg.norm(actual_step)) > 1e-4:
                self._breakaway_armed = False
                ev.append(ContactEvent(tr, "resistance_drop", f, f"resistance peaked at {self._peak:.1f} N then fell to {f:.1f} N while still moving", p))
        return ev
