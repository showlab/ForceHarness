"""Read-only force checks of a newly connected arm (layer 1): noise, drift, sign, and a known mass.

The robot-side worker is started without --enable-motion, so it never starts a controller and refuses every
motion request; Polymetis keeps its own default controller holding the arm, and the worker answers `state`
requests with the same reading the harness uses: w = (J^T)^+ tau_ext at the flange, solved on the robot PC from
the external joint torques, plus the residual rho and the smallest singular value of J, after the same guard
checks (a raw reading at or above the caps ends the worker). This module samples that stream and computes the
statistics; scripts/real/check_force.py is the command line around it. All forces are reaction forces (the force
of the environment on the hand, force_sign applied), so pressing the fingertips down on a table gives +z.
"""
import time

import numpy as np

G = 9.81
# Values measured on the authors' FR3, for comparison (they are not pass/fail thresholds).
REFERENCE = dict(
    static_std_N='0.02-0.05 N per axis with the arm still (std of 0.02-0.03 N at 50 Hz over 120-600 s)',
    relaxed_consecutive_median_N='0.08 N median difference between consecutive relaxed still readings',
    approach_direction_bias_N='1.5-2 N: the same pose reached from different directions',
    motion_bias_N='2-5 N during free motion (joint friction), up to 8-12 N at large wrist angles',
    absolute_weight_error_g='up to +-400 g for an absolute weight; differences of identical motions to tens of grams',
)


def sample(port, seconds, hz=20., *, clock=time.monotonic, sleep=time.sleep):
    """`state` replies for `seconds` at up to `hz`; each row: t, robot timestamp, wrench, residual, smin, width."""
    rows, begun = [], clock()
    period = 1./hz
    while True:
        t = clock()-begun
        if t >= seconds:
            break
        state = port.request('state')['state']
        rows.append(dict(t=float(t), timestamp_ns=int(state['timestamp_ns']),
                         wrench=[float(v) for v in state['wrench']],
                         residual=float(state.get('residual', float('nan'))),
                         smin=float(state.get('smin', float('nan'))),
                         gripper_width=float(state.get('gripper_width', float('nan'))),
                         joints=list(state.get('joints', [])), flange_pos=list(state.get('flange_pos', [])),
                         flange_quat=list(state.get('flange_quat', []))))
        sleep(max(0., period-((clock()-begun)-t)))
    return rows


def _vec(a):
    return [round(float(v), 5) for v in a]


def window_mean(rows, key_fn, lo, hi):
    picked = [key_fn(r) for r in rows if lo <= r['t'] < hi]
    return np.mean(picked, axis=0) if picked else None


def stream_stats(rows, force_sign, *, zero_n=10, drift_window_s=5.):
    """Statistics of a still stream. The zero is the mean of the first zero_n readings (as the harness re-zeroes
    from 10 still readings before an empty-hand primitive); force is tared and in reaction sign, torque is tared,
    about the flange origin, in the raw sign. Drift = mean of the last window minus the first window (each
    drift_window_s long, or a third of the stream if it is shorter than three windows)."""
    if len(rows) < zero_n+2:
        raise ValueError(f'need at least {zero_n+2} readings, got {len(rows)}')
    s = float(force_sign)
    W = np.array([r['wrench'] for r in rows], dtype=float)
    zero = W[:zero_n].mean(axis=0)
    F = s*(W[:, :3]-zero[:3])
    M = W[:, 3:]-zero[3:]
    t = np.array([r['t'] for r in rows])
    duration = float(t[-1]-t[0]) if len(t) > 1 else 0.
    window = drift_window_s if duration >= 3*drift_window_s else max(duration/3., 1e-9)
    first = F[t < t[0]+window].mean(axis=0)
    last = F[t >= t[-1]-window].mean(axis=0)
    stamps = [r['timestamp_ns'] for r in rows]
    residual = np.array([r['residual'] for r in rows], dtype=float)
    smin = np.array([r['smin'] for r in rows], dtype=float)
    out = dict(samples=len(rows), duration_s=round(duration, 3),
               request_rate_hz=round((len(rows)-1)/duration, 2) if duration > 0 else None,
               unique_robot_timestamps=len(set(stamps)),
               robot_rate_hz=(round((len(set(stamps))-1)/((max(stamps)-min(stamps))*1e-9), 2)
                              if max(stamps) > min(stamps) else None),
               zero_raw_wrench=_vec(zero),
               force_mean_N=_vec(F.mean(axis=0)), force_std_N=_vec(F.std(axis=0)),
               torque_mean_Nm=_vec(M.mean(axis=0)), torque_std_Nm=_vec(M.std(axis=0)),
               force_norm_std_N=round(float(np.linalg.norm(F, axis=1).std()), 5),
               drift_N=_vec(last-first), drift_norm_N=round(float(np.linalg.norm(last-first)), 5),
               drift_window_s=round(window, 3),
               raw_force_mean_N=_vec(W[:, :3].mean(axis=0)), raw_torque_mean_Nm=_vec(W[:, 3:].mean(axis=0)))
    positions = [r['flange_pos'] for r in rows if len(r.get('flange_pos') or []) == 3]
    if positions:
        out.update(flange_pos_m=_vec(np.mean(positions, axis=0)))
    quats = [r['flange_quat'] for r in rows if len(r.get('flange_quat') or []) == 4]
    if quats:
        out.update(flange_quat=_vec(quats[-1]))
    if np.isfinite(residual).any():
        out.update(residual_Nm=dict(mean=round(float(np.nanmean(residual)), 5), max=round(float(np.nanmax(residual)), 5)))
    if np.isfinite(smin).any():
        out.update(sigma_min=dict(mean=round(float(np.nanmean(smin)), 5), min=round(float(np.nanmin(smin)), 5)))
    return out


def reaction(rows, force_sign):
    """Mean reaction force (force_sign applied, untared) of a list of readings."""
    W = np.array([r['wrench'] for r in rows], dtype=float)
    return float(force_sign)*W[:, :3].mean(axis=0)


def press_result(rest_rows, press_rows, force_sign, press='up'):
    """One sign trial: the change of the reaction force from rest to the press (mean over the middle of the press
    window, where the hand is pressing steadily). press='up': the operator pushes the fingertips upward from below,
    as a table does when the hand presses on it, so the reaction z should be positive; 'down': pushed down from
    above, expected negative."""
    if press not in ('up', 'down'):
        raise ValueError("press must be 'up' or 'down'")
    t = np.array([r['t'] for r in press_rows])
    lo, hi = t[0]+.25*(t[-1]-t[0]), t[0]+.75*(t[-1]-t[0])
    middle = [r for r in press_rows if lo <= r['t'] <= hi] or press_rows
    delta = reaction(middle, force_sign)-reaction(rest_rows, force_sign)
    expected = 1. if press == 'up' else -1.
    return dict(press=press, delta_N=_vec(delta), delta_z_N=round(float(delta[2]), 4),
                magnitude_N=round(float(np.linalg.norm(delta)), 4),
                sign_z=int(np.sign(delta[2])), expected_sign_z=int(expected),
                matches=bool(np.sign(delta[2]) == expected and abs(delta[2]) > .5))


def mass_result(empty_rows, loaded_rows, force_sign, known_grams=None, closure_correction=None, closed=False):
    """Weight from the change of the still reading: grams = -dF_z / g (reaction sign), as in the harness's weight
    line. closure_correction: the calibration's closed-gripper wrench correction, given when the gripper closed
    between the two readings; then a corrected value is reported as well. closed (the gripper closed between the
    readings) with known_grams: the change that the known weight does not explain is reported in the raw sign of
    the calibration's gripper_delta_wrench, as an estimate of that correction for this grasp width."""
    dF = reaction(loaded_rows, force_sign)-reaction(empty_rows, force_sign)
    grams = -dF[2]/G*1000.
    out = dict(delta_N=_vec(dF), estimated_g=round(float(grams), 1))
    if closure_correction is not None:
        c = np.asarray(closure_correction, dtype=float)[:3]
        corrected = dF-float(force_sign)*c
        out.update(estimated_corrected_g=round(float(-corrected[2]/G*1000.), 1),
                   closure_correction_N=_vec(c))
    if known_grams is not None:
        out.update(known_g=float(known_grams), error_g=round(float(grams-known_grams), 1),
                   ratio=round(float(grams/known_grams), 3) if known_grams else None)
        if 'estimated_corrected_g' in out:
            out['error_corrected_g'] = round(out['estimated_corrected_g']-float(known_grams), 1)
        if closed:
            weight = np.array([0., 0., -float(known_grams)/1000.*G])
            out['closure_correction_estimate_N'] = _vec(float(force_sign)*(dF-weight))
    return out
