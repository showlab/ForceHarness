"""Offline FR3 kinematics for the workstation's feasibility checks; never sends commands.

Craig DH parameters from the Franka documentation; forward kinematics reproduces logged joint -> flange
poses of the arm to 1e-8 m. Joint limits are the robot-side intersection of the Panda and FR3 bounds
(robot_side/franka_polymetis/guard.py). The validation walks a planned flange path with ik() and looks at
the joint margins before any motion is requested, so a motion the arm cannot follow is refused with a
reason instead of faulting on the robot.
"""
import numpy as np
from scipy.spatial.transform import Rotation as R

DH = [(0, .333, 0), (0, 0, -np.pi/2), (0, .316, np.pi/2), (.0825, 0, np.pi/2),
      (-.0825, .384, -np.pi/2), (0, 0, np.pi/2), (.088, 0, np.pi/2)]
FLANGE_D = .107
LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8763, .4398, -2.8973])
HIGH = np.array([2.8973, 1.7628, 2.8973, -.1169, 2.8763, 3.7525, 2.8973])
NUC_JOINT_MARGIN = .03   # the robot-side guard's joint margin (rad)


def _link(a, d, alpha, theta):
    ca, sa, ct, st = np.cos(alpha), np.sin(alpha), np.cos(theta), np.sin(theta)
    return np.array([[ct, -st, 0, a], [st*ca, ct*ca, -sa, -d*sa], [st*sa, ct*sa, ca, d*ca], [0, 0, 0, 1]])


def frames(q):
    T, out = np.eye(4), []
    for (a, d, alpha), theta in zip(DH, q):
        T = T@_link(a, d, alpha, theta)
        out.append(T.copy())
    out.append(T@_link(0, FLANGE_D, 0, 0))
    return out


def fk(q):
    """Base -> flange (panda_link8) transform."""
    return frames(q)[-1]


def jacobian(q):
    fr = frames(q)
    p = fr[-1][:3, 3]
    J = np.zeros((6, 7))
    for i in range(7):
        z, o = fr[i][:3, 2], fr[i][:3, 3]
        J[:3, i], J[3:, i] = np.cross(z, p-o), z
    return J


def ik(q0, target, iters=100, damping=.02, tol=1e-6):
    """Damped least squares from q0 to a 4x4 flange target; returns (q, position error)."""
    q = np.array(q0, dtype=float)
    for _ in range(iters):
        T = fk(q)
        dp = target[:3, 3]-T[:3, 3]
        dr = (R.from_matrix(target[:3, :3])*R.from_matrix(T[:3, :3]).inv()).as_rotvec()
        if np.linalg.norm(dp) < tol and np.linalg.norm(dr) < tol:
            break
        J = jacobian(q)
        q = q+J.T@np.linalg.solve(J@J.T+damping**2*np.eye(6), np.r_[dp, dr])
    return q, float(np.linalg.norm(target[:3, 3]-fk(q)[:3, 3]))


def margin(q):
    """Smallest distance to a joint limit (rad) and the 1-based joint index."""
    m = np.minimum(np.asarray(q)-LOW, HIGH-np.asarray(q))
    return float(m.min()), int(m.argmin())+1


def track(q0, targets):
    """Follow a dense list of flange targets; returns per-target (q, margin, joint, error)."""
    q, out = np.array(q0, dtype=float), []
    for target in targets:
        q, err = ik(q, target)
        out.append((q.copy(), *margin(q), err))
    return out
