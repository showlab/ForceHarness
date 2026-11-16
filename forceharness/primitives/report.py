"""ContactReport: what every primitive returns (output of layer 2, input of layer 3).

The report is the only data channel from the backend to the model: every force form in ``translation``, the
physics translation layer included, is a different rendering of the same report.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional

import numpy as np


@dataclasses.dataclass
class ContactEvent:
    """One event from ``primitives.events.EventDetector``."""
    t: float                 # time (s) since the primitive started
    kind: str                # contact_start / contact_end / force_cap / torque_cap / stall / slip / resistance_drop / target_reached / timeout
    force: float             # |net F| (N) at the event
    detail: str = ""         # one-line description (main axis, etc.)
    pos: Optional[List[float]] = None

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class ContactReport:
    """Result of one primitive: termination, motion, events, the wrench trace, derived signals and quantities.

    ``truth`` holds simulator ground truth for evaluation only; no force form ever renders it."""
    primitive: str
    params: Dict[str, Any]
    termination: str                      # completed / contact / force_cap / torque_cap / stall / timeout / resistance_drop / error, or a primitive-specific result (e.g. dropped, seated, jammed, stalled)
    t_start: float
    t_end: float
    start_pos: List[float]
    end_pos: List[float]
    commanded_disp: List[float]           # accumulated commanded displacement (world frame, m)
    actual_disp: List[float]              # actual displacement
    events: List[ContactEvent]
    wrench_rows: List[List[float]]        # [t_rel, Fx,Fy,Fz,Tx,Ty,Tz] (reaction convention, world frame, bias removed); <= max_rows
    final_wrench: List[float]             # 6-D wrench of the last tick (including any held load)
    peak_force: float
    peak_torque: float
    mean_force_in_contact: float
    n_steps: int
    gripper_opening: Optional[float] = None
    yaw_deg: Optional[float] = None
    final_wrench_net: List[float] = dataclasses.field(default_factory=list)   # the same with carry_bias (the weight of the held object) removed
    derived: Dict[str, Any] = dataclasses.field(default_factory=dict)   # primitive-specific derived signals (payload estimate, stiffness estimate, angle of peak resistance, ...)
    quantities: List[Dict[str, Any]] = dataclasses.field(default_factory=list)   # output of the physics translation layer (translation.estimators.estimate); task-agnostic
    pos_rows: List[List[float]] = dataclasses.field(default_factory=list)       # tool position [x, y, z] aligned with wrench_rows
    truth: Dict[str, Any] = dataclasses.field(default_factory=dict)     # ground truth for evaluation; NEVER shown to the model
    error: Optional[str] = None

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return d

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ContactReport":
        d = dict(d)
        d["events"] = [ContactEvent(**e) for e in d.get("events", [])]
        return ContactReport(**d)

    def wrench_array(self) -> np.ndarray:
        return np.asarray(self.wrench_rows, dtype=float).reshape(-1, 7) if self.wrench_rows else np.zeros((0, 7))
