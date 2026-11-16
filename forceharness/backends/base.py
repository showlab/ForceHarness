"""The contract between ForceHarness and a robot or simulator.

Every backend (MuJoCo, ForceBench, the real Franka FR3, the offline fake) exposes the same small set of members,
and the primitives in ``forceharness.primitives`` are written only against them.  To run ForceHarness on a new
arm or simulator, implement :class:`Backend` and check it with ``python -m forceharness.doctor``.

Conventions
-----------
* World frame = robot base frame, metres, right-handed, z up.  Quaternions are ``[x, y, z, w]``.
* One call to :meth:`Backend.step` or :meth:`Backend.hold` is one control tick of ``dt`` seconds.
* The primitives command motion as a *target displacement* per tick.  The backend's stock impedance or
  operational-space controller turns the lead of the target over the measured pose into force
  (paper, Sec. 3.3, "Force through the impedance controller").  The backend never needs a force controller of its own.
* :meth:`Backend.reaction_force` returns the force that the environment applies to the hand, in the world frame,
  after taring: pressing straight down on a table gives a positive z component.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclasses.dataclass
class WrenchSample:
    """One control tick of force/torque readings.

    ``force`` / ``torque`` are in the sensor frame; the ``*_world`` fields are rotated into the world frame.
    ``net_*`` fields have the current zero point (bias) removed.
    """

    t: float
    force: np.ndarray
    torque: np.ndarray
    force_world: np.ndarray
    torque_world: np.ndarray
    net_force: np.ndarray
    net_torque: np.ndarray
    net_force_world: np.ndarray
    net_torque_world: np.ndarray

    @property
    def fmag(self) -> float:
        return float(np.linalg.norm(self.net_force))

    @property
    def tmag(self) -> float:
        return float(np.linalg.norm(self.net_torque))

    def to_row(self) -> List[float]:
        return [self.t, *self.net_force_world.tolist(), *self.net_torque_world.tolist()]


@runtime_checkable
class Backend(Protocol):
    """Members every backend must provide.  Optional members are listed in :data:`OPTIONAL_MEMBERS`."""

    #: control period in seconds (e.g. 0.05 for 20 Hz)
    dt: float
    #: size of the action vector of the underlying controller (7 with a gripper, 6 without)
    action_dim: int

    @property
    def eef_pos(self) -> np.ndarray:
        """Tool centre point position in the world frame, metres, shape (3,)."""

    @property
    def eef_quat(self) -> np.ndarray:
        """Tool orientation as a quaternion ``[x, y, z, w]``."""

    @property
    def gripper_opening(self) -> Optional[float]:
        """Finger opening in metres, or ``None`` if the tool has no gripper."""

    @property
    def sim_time(self) -> float:
        """Seconds since the episode started (simulated or wall clock)."""

    def step(self, dpos: np.ndarray, drot: Optional[np.ndarray] = None, grip: Optional[float] = None) -> WrenchSample:
        """Run one control tick that moves the target pose by ``dpos`` (m) and ``drot`` (axis-angle, rad).

        ``grip`` is -1 (open) ... +1 (closed) or ``None`` to leave the gripper alone.  Returns the reading after the tick.
        """

    def hold(self, n: int = 1, grip: Optional[float] = None) -> WrenchSample:
        """Keep the current target for ``n`` ticks and return the last reading."""

    def wrench(self) -> WrenchSample:
        """Current reading without moving."""

    def reaction_force(self, w: WrenchSample) -> np.ndarray:
        """Force of the environment on the hand, world frame, newtons, after taring (see module docstring)."""


#: Members that primitives use when present, with what they enable.
OPTIONAL_MEMBERS: Dict[str, str] = {
    "rezero_bias": "rezero_bias(steps): re-take the zero point at rest (the 'tare at rest' rule).",
    "set_kp": "set_kp(kp): change controller stiffness (scalar or 6-vector); primitives soften the arm near contact.",
    "kp": "default translational stiffness (N/m); used to convert lead into force.",
    "step_lead": "step_lead(...): real-robot only; ask the guard for a larger lead once contact is established.",
    "raw_wrench": "raw_wrench(): untared reading, used with bias_force to estimate a held load.",
    "bias_force": "current zero point (sensor frame), used with raw_wrench.",
    "wrist_joint_rad": "wrist joint angle (rad), so rotations can respect the joint range.",
    "set_held_weld": "simulation only: attach the held object rigidly to the hand.",
    "force_noise_std": "noise level (N) the estimators assume when they report an uncertainty.",
    "force_bias_std": "residual bias (N) the estimators assume when they report an uncertainty.",
    "task": "TaskSpec of the running task (simulation), for success checks and ground truth.",
    "task_state": "dict of task-specific state (e.g. latch released, damage).",
    "images": "images(): dict of camera name -> RGB array for the model.",
    "scene": "scene(): list of object dicts (name, position, size ...) for the model.",
    "success": "success(): simulation success check on the final state.",
    "gripper_contact_state": "simulation only: ground-truth contact state, for analysis.",
    "peak_contact_force": "simulation only: ground-truth peak contact force, for damage accounting.",
    "close": "close(): release resources.",
}


def _member_names(obj: Any) -> set:
    """Names an object or class provides, including attributes a class only sets in its methods (``self.x = ...``)."""
    import inspect
    import re

    names = set(dir(obj))
    classes = obj.__mro__ if inspect.isclass(obj) else type(obj).__mro__
    for cls in classes:
        if cls in (object, Protocol):
            continue
        try:
            src = inspect.getsource(cls)
        except (OSError, TypeError):
            continue
        names |= set(re.findall(r"self\.([A-Za-z_]\w*)\s*(?::[^=\n]+)?=(?!=)", src))
    return names


def describe_backend(obj: Any) -> Dict[str, Any]:
    """Report which required and optional members a backend object or class provides (used by the doctor)."""
    required = ["dt", "action_dim", "eef_pos", "eef_quat", "gripper_opening", "sim_time", "step", "hold", "wrench",
                "reaction_force"]
    names = _member_names(obj)
    return dict(missing_required=[m for m in required if m not in names],
                optional_present=[m for m in OPTIONAL_MEMBERS if m in names])
