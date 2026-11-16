"""Pure coordinate operations of the real backend, in metres and xyzw quaternions."""
import numpy as np
from scipy.spatial.transform import Rotation


def vector(value, size: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"{name} must contain {size} finite values")
    return result.copy()


def quaternion(value) -> np.ndarray:
    q = vector(value, 4, "quaternion")
    if abs(np.linalg.norm(q) - 1) > .01:
        raise ValueError("Measured quaternion is not unit length")
    return q / np.linalg.norm(q)


def angular_distance(a, b) -> float:
    return float((Rotation.from_quat(quaternion(a)).inv() * Rotation.from_quat(quaternion(b))).magnitude())


def flange_to_tcp(position, quat, config):
    rot = Rotation.from_quat(quaternion(quat))
    return (vector(position, 3, "position") + rot.apply(config.tcp_translation),
            (rot * Rotation.from_quat(config.tcp_quaternion)).as_quat())


def torque_at_tcp(force_base, torque_at_flange_base, flange_quat, config):
    """Explicit optional shift: tau_TCP = tau_flange - r_flange_to_TCP x F.

    The backend's WrenchSample stays at the flange; this helper never silently
    changes the reference point of a report or a torque limit.
    """
    r = Rotation.from_quat(quaternion(flange_quat)).apply(config.tcp_translation)
    return vector(torque_at_flange_base, 3, "torque") - np.cross(r, vector(force_base, 3, "force"))


def check_flange_workspace(position):
    x, y, z = vector(position, 3, "flange position")
    # Same box as the robot-side guard. With the hand pointing down the fingertips sit about 0.10 m below
    # the flange, so the 0.12 m floor keeps about 2 cm of clearance above a table at z ~ 0.005-0.009;
    # the 8 N collision stop still applies.
    if not (.20 < x < .65 and abs(y) < .30 and .12 < z < .55):
        raise ValueError("Flange outside executor workspace (guard box)")
