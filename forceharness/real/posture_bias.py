"""Posture-dependent zero of the joint-torque force estimate: offline model and runtime use (layer 1).

The robot side reports w = (J^T)^+ tau_ext at the flange (world-aligned). With the hand empty and
still tau_ext should be zero, but it is not: per-joint torque offsets and gravity-model
errors (link and hand masses, the wrist camera, cables) leave

    tau_err(q) = d_tau + Y(q) beta

and (J^T)^+ turns that into a wrench that drifts by about 1 N when the wrist turns.
Two parts, kept apart on purpose:

- beta: first-moment / mass errors per link, fitted once from many empty-hand still poses
  and stored (configs/real/posture_bias_fr3.json) until the hardware changes;
- d_tau: seven joint offsets that move between power-ons; solved again from ONE still
  empty-hand pose at the start of every episode (no motion, no model call).

On the FR3 the posture effect (about 0.3-0.4 N) was smaller than the dependence on how a pose was reached
(about 2 N), and the model did not beat a constant zero on held-out poses, so the shipped file has
use_model false and episodes rely on relaxing the controller plus a per-episode zero reading.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from forceharness.real import kinematics as fr3

G = 9.81
N_FRAMES = 8                 # 7 link frames + flange (hand, camera, cables)
N_PARAMS = 4*N_FRAMES        # per frame: mass at the frame origin + first moment (3)
DEFAULT_FILE = Path(__file__).resolve().parents[2]/'configs/real/posture_bias_fr3.json'


def _potential_terms(q):
    """Potential energy of a unit of each parameter (8 frames x [m, sx, sy, sz])."""
    out = np.zeros(N_PARAMS)
    for k, T in enumerate(fr3.frames(q)):
        out[4*k] = G*T[2, 3]                  # unit mass at the frame origin
        out[4*k+1:4*k+4] = G*T[2, :3]         # z row of R: unit first moment along x, y, z
    return out


def regressor(q, h=1e-6):
    """Y(q): gravity torque (7) of unit parameter errors, by central differences of U."""
    q = np.asarray(q, dtype=float)
    Y = np.zeros((7, N_PARAMS))
    for i in range(7):
        dq = np.zeros(7)
        dq[i] = h
        Y[i] = (_potential_terms(q+dq)-_potential_terms(q-dq))/(2*h)
    return Y


def wrench_from_tau(q, tau):
    """Same mapping as the robot-side reader: w = (J^T)^+ tau, flange origin, world-aligned axes."""
    return np.linalg.pinv(fr3.jacobian(np.asarray(q, dtype=float)).T)@np.asarray(tau, dtype=float)


def fit(samples, ridge=1e-3, session_key='session'):
    """beta (shared) and d_tau per session from still empty-hand samples {q, tau, session}."""
    sessions = sorted({s[session_key] for s in samples})
    index = {name: i for i, name in enumerate(sessions)}
    rows, rhs = [], []
    for s in samples:
        A = np.zeros((7, N_PARAMS+7*len(sessions)))
        A[:, :N_PARAMS] = regressor(s['q'])
        A[:, N_PARAMS+7*index[s[session_key]]:N_PARAMS+7*index[s[session_key]]+7] = np.eye(7)
        rows.append(A)
        rhs.append(np.asarray(s['tau'], dtype=float))
    A, b = np.vstack(rows), np.concatenate(rhs)
    # Ridge only on beta: unidentifiable combinations (e.g. link 1 under gravity) stay at 0.
    reg = np.zeros(A.shape[1])
    reg[:N_PARAMS] = ridge
    x = np.linalg.solve(A.T@A+np.diag(reg), A.T@b)
    beta = x[:N_PARAMS]
    offsets = {name: x[N_PARAMS+7*i:N_PARAMS+7*i+7] for name, i in index.items()}
    residual = (A@x-b).reshape(-1, 7)
    return beta, offsets, residual


def session_offset(q0, tau0, beta):
    """d_tau from one still empty-hand pose (seven equations, seven unknowns)."""
    return np.asarray(tau0, dtype=float)-regressor(q0)@beta


def predicted_wrench(q, beta, d_tau):
    """Wrench the robot side reports for an empty, still hand at q (to subtract from readings)."""
    return wrench_from_tau(q, d_tau+regressor(q)@beta)


class PostureBias:
    """Runtime use: load the stored beta, take d_tau from the episode's first still pose."""

    def __init__(self, beta, meta=None):
        self.beta = np.asarray(beta, dtype=float)
        self.meta = meta or {}
        self.d_tau = None
        self.zero = None

    @classmethod
    def load(cls, path=DEFAULT_FILE):
        doc = json.loads(Path(path).read_text())
        if doc.get('version') != 'posture_bias/1' or len(doc['beta']) != N_PARAMS:
            raise ValueError('Not a posture_bias/1 file: '+str(path))
        return cls(doc['beta'], doc)

    def zero_check(self, q0, tau0):
        self.d_tau = session_offset(q0, tau0, self.beta)
        self.zero = dict(q=np.asarray(q0, dtype=float).tolist(), d_tau=self.d_tau.tolist())
        return self.d_tau

    def wrench(self, q):
        if self.d_tau is None:
            raise RuntimeError('posture bias needs a zero check first')
        return predicted_wrench(q, self.beta, self.d_tau)
