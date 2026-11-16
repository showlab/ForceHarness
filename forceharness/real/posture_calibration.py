"""One-time empty-hand posture calibration of the force zero, and the reuse rule (layer 1, optional).

Once per hardware setup (Franka Hand, wrist camera and its mount, pads, cable routing): the
empty hand visits 6 stations x 11 wrist orientations (vertical, about y +-30/60 deg, about x
+-30 deg), the controller relaxed before every reading. Every station returns to vertical
three times, so the readings also show how much history (friction hysteresis) is left after
relaxing. The result goes to configs/real/posture_bias_fr3.json; the runner reuses it and
only takes a short zero reading per episode. If the file is missing, the runner calls
`run_calibration` first (about 20 minutes), unless --no-auto-calibration is given.
The model is used only if it beats a constant zero (fit_and_save decides; see posture_bias.py).
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from forceharness.real import kinematics as fr3
from forceharness.real import posture_bias as pb
from forceharness.real.relax import relax_and_read
from forceharness.real.geometry import check_flange_workspace

HOME = (.46, 0., .30)
STATION_OFFSETS = [(0, 0, 0), (-.06, -.10, 0), (-.06, .10, 0), (.06, -.10, 0), (.06, .10, 0), (0, 0, -.05)]
# cumulative wrist tilt (deg about +y, deg about +x); consecutive steps <= 30 deg (turn limit)
TILTS = [(0, 0), (30, 0), (60, 0), (30, 0), (0, 0), (-30, 0), (0, 0), (0, 30), (0, 0), (0, -30), (0, 0)]
LEG_M = .09
USE_MODEL_GAIN = .7          # the model must cut the median error by >= 30 %
USE_MODEL_FLOOR_N = .3       # and the raw posture effect must be worth correcting
HARDWARE = ('Franka Hand + finger pads + wrist D405 on its mount; '
            'recalibrate after any change to hand, camera mount, pads or cable routing')


def load(path=pb.DEFAULT_FILE):
    """PostureBias from the stored file, or None when the calibration was never run."""
    path = Path(path)
    return pb.PostureBias.load(path) if path.is_file() else None


def _feasible_turn(backend, axis, angle_deg, margin=.05):
    """Whether the arm can follow this turn from the live joints (the runner's kinematic walk; skipped in demo)."""
    s = backend._state
    q0 = np.asarray(s['joints'], dtype=float)
    T0 = np.eye(4)
    T0[:3, :3] = R.from_quat(s['flange_quat']).as_matrix()
    T0[:3, 3] = s['flange_pos']
    if np.linalg.norm(fr3.fk(q0)[:3, 3]-T0[:3, 3]) > .001:
        return True
    pivot = np.asarray(backend.eef_pos, dtype=float)
    q = q0
    for deg in np.linspace(0, angle_deg, int(math.ceil(abs(angle_deg)))+1)[1:]:
        turn = R.from_rotvec(np.asarray(axis, dtype=float)*math.radians(deg))
        T = np.eye(4)
        T[:3, :3] = (turn*R.from_matrix(T0[:3, :3])).as_matrix()
        T[:3, 3] = pivot+turn.apply(T0[:3, 3]-pivot)
        q, err = fr3.ik(q, T)
        try:
            check_flange_workspace(T[:3, 3])
        except ValueError:
            return False
        if err > 1e-4 or fr3.margin(q)[0] < margin or np.abs(q-q0).max() > .85:
            return False
    return True


def _move(prims, goal):
    b = prims.sim
    goal = np.asarray(goal, dtype=float)
    while np.linalg.norm(goal-b.eef_pos) > .003:
        delta = goal-b.eef_pos
        step = delta*min(1., LEG_M/np.linalg.norm(delta))
        target = b.eef_pos+step
        prims.call('move_to', x=float(target[0]), y=float(target[1]), z=float(target[2]))
        if b.fault:
            raise RuntimeError('backend fault during calibration move: '+str(b.fault))
        if np.linalg.norm(b.eef_pos-target) > .01:        # stopped early: do not insist
            break


def run_calibration(prims, path=pb.DEFAULT_FILE, out_dir=None, home=HOME, log=print):
    """Drive the empty hand through the poses, fit, and write the reusable calibration file."""
    b = prims.sim
    if b._closed is not False or b.gripper_opening < .075:
        raise ValueError('posture calibration needs the gripper open and empty')
    start = np.asarray(b.eef_pos, dtype=float).copy()
    samples = []

    def read(station, tilt):
        st = relax_and_read(b)
        prims.target_quat = np.asarray(b.eef_quat, dtype=float).copy()
        approach = R.from_quat(b.eef_quat).apply([0, 0, 1])
        st.update(station=station, tilt=list(tilt), hand_tilt_deg=float(np.degrees(np.arccos(np.clip(-approach[2], -1, 1)))))
        samples.append(st)
        if out_dir is not None:     # keep every reading even if the session dies later
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir)/'posture_samples.json').write_text(json.dumps(samples)+'\n')
        log(f'  station {station} tilt {tilt}: raw force {np.round(st["raw_force"], 2).tolist()} N')

    try:
        for k, offset in enumerate(STATION_OFFSETS):
            _move(prims, np.add(home, offset))
            current = (0, 0)
            for tilt in TILTS:
                for index, axis in ((0, [0., 1., 0.]), (1, [1., 0., 0.])):
                    step = tilt[index]-current[index]
                    if not step:
                        continue
                    if not _feasible_turn(b, axis, step):
                        log(f'  station {k}: tilt {tilt} skipped (joint or workspace limit)')
                        break
                    rep = prims.call('turn', axis=axis, pivot=np.asarray(b.eef_pos, dtype=float).tolist(), angle_deg=float(step))
                    if b.fault:
                        raise RuntimeError('backend fault during calibration turn: '+str(b.fault))
                    if rep.termination != 'completed':
                        log(f'  station {k}: turn toward {tilt} ended {rep.termination}; reading where the hand is')
                        break
                    current = (tilt[0], current[1]) if index == 0 else (current[0], tilt[1])
                read(k, current)
        _move(prims, start)
    except BaseException:
        if len(samples) >= 30:      # a late fault still leaves a usable calibration
            fit_and_save(samples, path, out_dir, partial=True)
        raise
    return fit_and_save(samples, path, out_dir)


def fit_and_save(samples, path=pb.DEFAULT_FILE, out_dir=None, source='', partial=False):
    """Decide whether a posture model is worth using; always write the file (so it is reused)."""
    F = np.array([s['raw_force'] for s in samples])
    raw = np.linalg.norm(F-F[0], axis=1)[1:]
    vertical = [s for s in samples if s['tilt'] == [0, 0]]
    by_station = {}
    for s in vertical:
        by_station.setdefault(s['station'], []).append(np.asarray(s['raw_force']))
    history = [float(np.linalg.norm(np.ptp(np.array(v), axis=0))) for v in by_station.values() if len(v) > 1]
    beta, use_model, corrected = np.zeros(pb.N_PARAMS), False, None
    usable = [s for s in samples if s.get('tau') is not None]
    if len(usable) >= 20 and len(usable) == len(samples):
        beta, _, _ = pb.fit([dict(q=s['q'], tau=s['tau'], session='calibration') for s in usable], ridge=1e-2)
        d0 = pb.session_offset(usable[0]['q'], usable[0]['tau'], beta)
        pred = np.array([pb.predicted_wrench(s['q'], beta, d0)[:3] for s in usable])
        corrected = np.linalg.norm(F-pred, axis=1)[1:]
        use_model = bool(np.median(raw) > USE_MODEL_FLOOR_N and np.median(corrected) < USE_MODEL_GAIN*np.median(raw))
    stats = dict(readings=len(samples), partial=bool(partial),
                 raw_change_from_first_N=dict(p50=float(np.median(raw)), p90=float(np.percentile(raw, 90)), max=float(raw.max())),
                 corrected_N=(None if corrected is None else
                              dict(p50=float(np.median(corrected)), p90=float(np.percentile(corrected, 90)), max=float(corrected.max()))),
                 same_pose_history_spread_N=dict(p50=float(np.median(history)) if history else None,
                                                 max=float(max(history)) if history else None))
    doc = dict(version='posture_bias/1', created=datetime.now().astimezone().isoformat(),
               use_model=use_model, beta=beta.tolist(), stats=stats, hardware=HARDWARE,
               source=str(out_dir or source), method='empty hand, controller relaxed before every reading; '
               'tau_err(q) = d_tau + Y(q) beta, d_tau re-read per episode')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2)+'\n')
    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir)/'posture_samples.json').write_text(json.dumps(samples)+'\n')
    return doc
