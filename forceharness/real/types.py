"""Hardware-neutral records of the real backend (layer 1: the wrench sample the primitives consume)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class BackendError(RuntimeError):
    """An observation or action cannot be trusted; no automatic retry."""


@dataclass(frozen=True)
class WrenchSample:
    """Same numeric attributes as the simulation record, with explicit real provenance.

    All vectors use BASE axes. Torque is about the FLANGE origin, as in the
    raw reader. eef_pos is TCP: these reference points must not be confused.
    Raw libfranka signs are preserved; only reaction_force() applies force_sign.
    net_* subtract the local base-frame bias, if one has been explicitly measured.
    A supplied bias outside its measured pose/width scope is marked invalid.
    """
    t: float                # workstation monotonic time since connect
    force: np.ndarray
    torque: np.ndarray
    force_world: np.ndarray
    torque_world: np.ndarray
    net_force: np.ndarray
    net_torque: np.ndarray
    net_force_world: np.ndarray
    net_torque_world: np.ndarray
    robot_timestamp_ns: int
    received_at: float
    bias_valid: bool
    reference_point: str = "flange"
    axes: str = "base"

    @property
    def fmag(self) -> float:
        return float(np.linalg.norm(self.net_force_world))

    @property
    def tmag(self) -> float:
        return float(np.linalg.norm(self.net_torque_world))

    def to_row(self) -> list[float]:
        return [self.t, *self.net_force_world.tolist(), *self.net_torque_world.tolist()]
