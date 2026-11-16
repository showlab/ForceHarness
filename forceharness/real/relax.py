"""'Relax, then read' (layer 1): unwind the controller lead before a still force reading.

Across 999 still empty-hand windows of robot logs, consecutive still readings differed by 1.07 N (median)
and 3.56 N (p90) when the controller's joint target was >= 0.002 rad ahead of the joints, even at the same
orientation; with the target back on the joints, by 0.08 N and 0.87 N. The lead is the controller pushing
against joint friction, which the torque-based estimate reports as an external force (it also produced
false 8 N contacts). Zero-delta steps let the robot side's desired pose ramp back to the measured pose.
"""
from __future__ import annotations

import numpy as np

STEP_S = .05


def relax_and_read(backend, relax_s=.8, read_s=.5):
    """One short motion of zero steps; returns the mean still reading after the lead is gone.

    Orientation is not held to an old target either (drot = 0), so a twisted hand settles
    too. The caller should reset its own orientation target to backend.eef_quat afterwards.
    """
    forces, wrenches, joints, taus = [], [], [], []
    n_relax, n_read = int(round(relax_s/STEP_S)), max(1, int(round(read_s/STEP_S)))
    with backend.motion(relax_s+read_s+2.):
        for i in range(n_relax+n_read):
            backend.step(np.zeros(3), np.zeros(3))
            if i >= n_relax:
                s = backend._state
                wrenches.append(np.asarray(s['wrench'], dtype=float))
                joints.append(np.asarray(s['joints'], dtype=float))
                if s.get('motor_torques_external') is not None and not isinstance(s['motor_torques_external'], str):
                    taus.append(np.asarray(s['motor_torques_external'], dtype=float))
        backend.motion_result('completed')
    w = np.mean(wrenches, axis=0)
    return dict(raw_wrench=w.tolist(), raw_force=w[:3].tolist(), q=np.mean(joints, axis=0).tolist(),
                tau=np.mean(taus, axis=0).tolist() if len(taus) == len(wrenches) else None,
                force_std=float(np.linalg.norm(np.std(wrenches, axis=0)[:3])), n=len(wrenches),
                tcp=np.asarray(backend.eef_pos, dtype=float).round(4).tolist(), t=float(backend.sim_time))
