"""Physics translation layer, estimators (layer 3).

Turns the force time series of a ContactReport into task-agnostic physical quantities, each with a unit, an
uncertainty, the window it was computed over, and flags.

The estimators are fixed formulas, least squares and error propagation: nothing is learned and no task threshold is
known here (a threshold can only come from the task instruction the model reads, or from the model itself).  The list
of Quantity objects is attached to ContactReport.quantities; the force forms phys1-phys4 are four rendering levels
of it (``translation.quantities``).

| quantity | primitives | method |
|---|---|---|
| payload_mass | lift / hold (gripper closed) | |mean Fz| / g over the still window; sigma = sqrt(std^2/N + bias^2) / g |
| com_offset | the same, with a load | horizontal lever arm r = tau / (m g): magnitude, direction, and the 2-sigma floor a balanced load would read |
| contact_height / contact_point | any primitive with a contact_start | event position moved back along the approach direction by the detector lag |
| contact_normal | the same | direction of the mean reaction force over the first ticks after contact |
| impact_peak | the same | peak |F| within 0.3 s after contact; gentle / moderate / hard |
| stiffness | the same, with position rows | least-squares slope of normal force against penetration (N/mm); soft / firm / rigid |
| normal_force / friction_coefficient | slide / push | over the steady in-contact window: mean |F_t| / mean F_n; stick-slip index |
| insertion_depth | search / insert | advance along the normal + lateral force at the end + outcome flag |
| pull_force / path_turn / path_radius / free_direction | pull | mean pulling force; turn of the path; radius and pivot of a circle fitted to the xy path; final direction of motion |
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.primitives.report import ContactReport

G = 9.81
# Sign of the centre-of-mass lever: r_x = TORQUE_SIGN * Ty / (m g), r_y = -TORQUE_SIGN * Tx / (m g).
# The torque columns keep the sensor's own convention (net_torque_world, not multiplied by force_sign); the sign was
# calibrated against the simulator's ground-truth lever arm of the load.
TORQUE_SIGN = -1.0


@dataclasses.dataclass
class Quantity:
    """One physical quantity: value and unit, sigma (None where no uncertainty is defined), the number of samples n,
    the window it was computed over, the method, flags, and extra fields used by the renderers."""
    name: str
    value: float
    unit: str
    sigma: Optional[float] = None
    n: int = 0
    window: str = ""
    method: str = ""
    flags: List[str] = dataclasses.field(default_factory=list)
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class EstimatorContext:
    """What the estimators need to know about the primitive library: the tared load, the force noise model, the tick
    length, the contact threshold, the approach speed and the length of the detector's moving average."""
    carry_bias: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(3))
    grip_closed: bool = False
    noise_std: float = 0.5      # per-tick force noise sigma (N)
    bias_std: float = 0.3       # residual bias sigma (N, constant within an episode)
    dt: float = 0.05
    f_contact: float = 3.0
    speed_near: float = 0.02
    filter_n: int = 4


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _pos_arr(rep: ContactReport) -> np.ndarray:
    return np.asarray(rep.pos_rows, dtype=float).reshape(-1, 3) if rep.pos_rows else np.zeros((0, 3))


# ----------------------------------------------------------------------------- payload and centre of mass
def est_payload_and_com(rep: ContactReport, ctx: EstimatorContext) -> List[Quantity]:
    if rep.primitive not in ("lift", "hold") or not ctx.grip_closed:
        return []
    arr = rep.wrench_array()
    if len(arr) < 6:
        return []
    p = rep.params if isinstance(rep.params, dict) else {}
    if rep.primitive == "lift":
        n = int(float(p.get("hold_time", 1.0)) / ctx.dt)
    else:
        n = len(arr) // 2
    n = int(min(len(arr), max(5, n)))
    tail = arr[-n:]
    fz = tail[:, 3]
    fz_mean = float(fz.mean())
    m = abs(fz_mean) / G
    sig_f = math.sqrt((float(fz.std()) / math.sqrt(n)) ** 2 + ctx.bias_std ** 2)
    sigma = sig_f / G
    flags: List[str] = []
    pos = _pos_arr(rep)
    if len(pos) >= n and float(pos[-n:, 2].max() - pos[-n:, 2].min()) > 0.003:
        flags.append("not_settled")
    if float(fz.std()) > 3 * ctx.noise_std + 0.5:
        flags.append("unsteady")
    no_load = m < 2 * sigma
    if no_load:
        flags.append("no_load")
    elif fz_mean > 0:
        flags.append("pushing_up_not_hanging")      # reaction points up: the object pushes on the gripper (still on the table), it is not hanging
    window = f"last {n} samples, {n * ctx.dt:.1f} s"
    out = [Quantity("payload_mass", round(m, 3), "kg", round(sigma, 3), n, window,
                    "|mean Fz| / g over the static hold window (gravity-compensated wrist force)", flags,
                    extra=dict(hold_force_z_N=round(fz_mean, 2), floor_kg=round(2 * sigma, 3)))]
    if no_load or "pushing_up_not_hanging" in flags or m < 0.03:
        return out
    tx, ty = float(tail[:, 4].mean()), float(tail[:, 5].mean())
    stx, sty = float(tail[:, 4].std()) / math.sqrt(n), float(tail[:, 5].std()) / math.sqrt(n)
    rx = TORQUE_SIGN * ty / (m * G)
    ry = -TORQUE_SIGN * tx / (m * G)
    r = math.hypot(rx, ry)
    s_tau = math.hypot(stx, sty) + 0.01          # noise of the mean torque + 0.01 N m residual bias
    sr = math.sqrt((s_tau / (m * G)) ** 2 + (r * sigma / m) ** 2)
    floor = 2 * sr
    flags2 = ["within_noise"] if r < floor else []
    out.append(Quantity("com_offset", round(r * 100, 2), "cm", round(sr * 100, 2), n, window,
                        "horizontal lever arm of the load's centre of mass from the wrist = hold torque / (m g)", flags2,
                        extra=dict(x_cm=round(rx * 100, 2), y_cm=round(ry * 100, 2),
                                   direction_deg=round(math.degrees(math.atan2(ry, rx)), 1),
                                   balanced_floor_cm=round(floor * 100, 2), torque_xy_Nm=[round(tx, 3), round(ty, 3)])))
    return out


# ----------------------------------------------------------------------------- contact: position, normal, impact, stiffness
def est_contact(rep: ContactReport, ctx: EstimatorContext) -> List[Quantity]:
    ev = next((e for e in rep.events if e.kind == "contact_start"), None)
    if ev is None or rep.primitive in ("grasp", "release", "pull"):
        return []
    arr = rep.wrench_array()
    if len(arr) == 0:
        return []
    # ignore a brief, weak touch (contact_end within 0.15 s, or a peak below the threshold); otherwise empty
    # quantities such as "impact 0 N / stiffness -0" would appear
    ends = [e.t for e in rep.events if e.kind == "contact_end" and e.t >= ev.t]
    brief = bool(ends) and (ends[0] - ev.t) < 0.15
    win0 = arr[(arr[:, 0] >= ev.t) & (arr[:, 0] <= ev.t + 0.3)]
    pk0 = float(np.linalg.norm(win0[:, 1:4] - ctx.carry_bias, axis=1).max()) if len(win0) else 0.0
    if brief or pk0 < ctx.f_contact * 0.8:
        return []
    pos = _pos_arr(rep)
    p = rep.params if isinstance(rep.params, dict) else {}
    d = p.get("direction")
    u = _unit(d) if d is not None else _unit(np.asarray(rep.end_pos, dtype=float) - np.asarray(rep.start_pos, dtype=float))
    if not np.any(u):
        u = np.array([0.0, 0.0, -1.0])
    i0 = int(np.searchsorted(arr[:, 0], ev.t))
    after = arr[i0:i0 + 6] if i0 < len(arr) else arr[-3:]
    F = after[:, 1:4] - ctx.carry_bias
    fmean = F.mean(axis=0)
    fn = float(np.linalg.norm(fmean))
    out: List[Quantity] = []
    lag = ctx.filter_n * ctx.dt * ctx.speed_near          # position lag of the moving average + consecutive count (about 4 mm at 2 cm/s)
    p_ev = np.asarray(ev.pos, dtype=float) if ev.pos else np.asarray(rep.end_pos, dtype=float)
    p_contact = p_ev - u * lag
    sig_pos = lag / 2 + ctx.speed_near * ctx.dt
    pt = [round(float(x), 4) for x in p_contact]
    if abs(float(u[2])) > 0.7:
        out.append(Quantity("contact_height", round(float(p_contact[2]), 4), "m", round(sig_pos, 4), len(after), f"t={ev.t:.2f} s",
                            "tool z at first contact, corrected for the detector's lag along the approach direction", [],
                            extra=dict(point=pt, approach=[round(float(x), 2) for x in u])))
    else:
        out.append(Quantity("contact_point", round(float(np.dot(p_contact - np.asarray(rep.start_pos, dtype=float), u)), 4), "m along approach from start", round(sig_pos, 4), len(after),
                            f"t={ev.t:.2f} s", "tool position at first contact, corrected for detector lag", [],
                            extra=dict(point=pt, approach=[round(float(x), 2) for x in u])))
    if fn > 1e-6:
        nrm = fmean / fn
        ang = math.degrees(math.acos(min(1.0, abs(float(nrm[2])))))
        out.append(Quantity("contact_normal", round(ang, 1), "deg from vertical", None, len(after), "first samples after contact",
                            "direction of the mean reaction force right after contact", [],
                            extra=dict(vector=[round(float(x), 2) for x in nrm], opposes_motion=bool(float(np.dot(nrm, u)) < -0.5))))
    win = arr[(arr[:, 0] >= ev.t) & (arr[:, 0] <= ev.t + 0.3)]
    if len(win):
        pk = float(np.linalg.norm(win[:, 1:4] - ctx.carry_bias, axis=1).max())
        cls = "gentle" if pk < 5 else "moderate" if pk < 15 else "hard"
        out.append(Quantity("impact_peak", round(pk, 1), "N", None, len(win), "0.3 s after contact", "peak |F| right after contact", [cls]))
    if len(pos) == len(arr) and i0 < len(arr) - 3:
        seg = slice(i0, min(len(arr), i0 + 12))
        pen = (pos[seg] - p_ev) @ u
        fnorm = -((arr[seg, 1:4] - ctx.carry_bias) @ u)
        ok = pen > 0.0005
        if int(ok.sum()) >= 4 and float(pen[ok].max() - pen[ok].min()) > 0.0005:
            A = np.vstack([pen[ok], np.ones(int(ok.sum()))]).T
            coef, res, *_ = np.linalg.lstsq(A, fnorm[ok], rcond=None)
            k = float(coef[0]) / 1000.0                       # N/mm
            dof = max(1, int(ok.sum()) - 2)
            resid = float(np.sqrt(res[0] / dof)) if len(res) else 0.0
            sxx = float(np.sqrt(((pen[ok] - pen[ok].mean()) ** 2).sum()))
            sk = resid / max(sxx, 1e-6) / 1000.0
            if k <= 0.05:
                return out
            cls = "soft" if k < 1 else "firm" if k < 10 else "rigid"
            out.append(Quantity("stiffness", round(k, 2), "N/mm", round(sk, 2), int(ok.sum()), "samples after contact",
                                "least-squares slope of normal force vs penetration", [cls]))
    return out


# ----------------------------------------------------------------------------- normal force and friction
def est_friction(rep: ContactReport, ctx: EstimatorContext) -> List[Quantity]:
    if rep.primitive not in ("slide", "push"):
        return []
    arr = rep.wrench_array()
    if len(arr) < 8:
        return []
    p = rep.params if isinstance(rep.params, dict) else {}
    F = arr[:, 1:4] - ctx.carry_bias
    i0 = len(arr) // 5
    if rep.primitive == "push":
        nrm = _unit(p.get("direction", [0, 0, -1]))
        fn = -(F @ nrm)
        seg = fn[i0:]
        ok = seg > ctx.f_contact * 0.5
        if int(ok.sum()) < 3:
            return [Quantity("normal_force", round(float(seg.mean()), 2), "N", None, len(seg), "last 80%", "mean reaction force along the push direction", ["lost_contact"])]
        n = int(ok.sum())
        return [Quantity("normal_force", round(float(seg[ok].mean()), 2), "N", round(float(seg[ok].std()) / math.sqrt(n), 2), n,
                         "steady part (last 80%, in contact)", "mean reaction force along the push direction", [],
                         extra=dict(target_N=p.get("target_force")))]
    u = _unit(p.get("direction", [1, 0, 0]))
    nrm = _unit(p.get("normal", [0, 0, -1]))
    fn = -(F @ nrm)
    ft = F @ u
    seg_fn, seg_ft = fn[i0:], ft[i0:]
    ok = seg_fn > ctx.f_contact * 0.5
    if int(ok.sum()) < 3:
        return [Quantity("normal_force", round(float(seg_fn.mean()), 2), "N", None, len(seg_fn), "last 80%", "mean normal force", ["lost_contact"],
                         extra=dict(target_N=p.get("normal_force")))]
    fn_ok, ft_ok = seg_fn[ok], seg_ft[ok]
    n = int(ok.sum())
    mfn, mft = float(fn_ok.mean()), float(np.abs(ft_ok).mean())
    mu = mft / max(mfn, 1e-3)
    rel = math.sqrt((float(ft_ok.std()) / math.sqrt(n) / max(mft, 1e-3)) ** 2 + (float(fn_ok.std()) / math.sqrt(n) / max(mfn, 1e-3)) ** 2)
    osc = float(np.abs(ft_ok).std()) / max(mft, 1e-3)
    flags = ["stick_slip"] if osc > 0.5 else []
    return [Quantity("normal_force", round(mfn, 2), "N", round(float(fn_ok.std()) / math.sqrt(n), 2), n, "steady part (last 80%, in contact)",
                     "mean reaction force along the surface normal", [],
                     extra=dict(target_N=p.get("normal_force"), lost_contact_fraction=round(float((~ok).mean()), 2))),
            Quantity("friction_coefficient", round(mu, 2), "", round(mu * rel, 2), n, "steady part (last 80%, in contact)",
                     "mean |tangential force| / mean normal force while sliding", flags,
                     extra=dict(tangential_N=round(mft, 2), oscillation_index=round(osc, 2)))]


# ----------------------------------------------------------------------------- insertion state
def est_insertion(rep: ContactReport, ctx: EstimatorContext) -> List[Quantity]:
    if rep.primitive not in ("search", "insert"):
        return []
    p = rep.params if isinstance(rep.params, dict) else {}
    nrm = _unit(p.get("normal", [0, 0, -1]))
    s, e = np.asarray(rep.start_pos, dtype=float), np.asarray(rep.end_pos, dtype=float)
    depth = float((e - s) @ nrm)
    arr = rep.wrench_array()
    tail = (arr[-5:, 1:4] - ctx.carry_bias) if len(arr) >= 5 else np.zeros((1, 3))
    lat = tail - np.outer(tail @ nrm, nrm)
    lat_mean = lat.mean(axis=0)
    lat_mag = float(np.linalg.norm(lat_mean))
    if rep.primitive == "insert":
        state = {"seated": "seated", "jammed": "jammed", "stalled": "stalled_no_progress", "completed": "not_seated", "timeout": "not_seated", "force_cap": "jammed_hard"}.get(rep.termination, rep.termination)
    else:
        state = {"resistance_drop": "dropped_in", "dropped": "dropped_in", "completed": "no_hole_found", "stall": "jammed",
                 "force_cap": "jammed_hard"}.get(rep.termination, rep.termination)
    flags = [state] + (["lateral_load"] if lat_mag > ctx.f_contact else [])
    return [Quantity("insertion_depth", round(depth, 4), "m along normal", None, len(arr), "whole primitive",
                     "advance of the tool along the insertion normal (start to end)", flags,
                     extra=dict(lateral_force_N=round(lat_mag, 2),
                                lateral_direction=[round(float(x), 2) for x in (lat_mean / lat_mag if lat_mag > 1e-6 else lat_mean)]))]


# ----------------------------------------------------------------------------- pulling along a constraint: the path of the mechanism
def est_pull(rep: ContactReport, ctx: EstimatorContext) -> List[Quantity]:
    if rep.primitive != "pull":
        return []
    dv = rep.derived or {}
    out: List[Quantity] = []
    stopped = ["mechanism_stopped"] if rep.termination == "stalled" else []
    if dv.get("pull_force_mean_N") is not None:
        n = max(1, len(rep.wrench_rows) * 4 // 5)
        out.append(Quantity("pull_force", float(dv["pull_force_mean_N"]), "N", round(float(dv.get("pull_force_std_N", 0.0)) / math.sqrt(n) + ctx.bias_std, 2),
                            n, "last 80%", "mean force along the (adapting) pulling direction", stopped,
                            extra=dict(lateral_force_mean_N=dv.get("lateral_force_mean_N"), target_N=(rep.params or {}).get("force"))))
    travel = float(dv.get("path_length_m") or 0.0)
    turn = dv.get("turned_deg") if travel >= 0.03 else None      # with almost no travel the turn / free direction carry no information: not reported (avoids false conclusions such as "straight line")
    if turn is not None:
        out.append(Quantity("path_turn", float(turn), "deg", 2.0, 0, "whole primitive",
                            "how much the direction of motion turned about the vertical axis (positive = counter-clockwise seen from above)",
                            ["straight_line"] if abs(float(turn)) < 3.0 else [],
                            extra=dict(path_length_m=dv.get("path_length_m"))))
    if dv.get("radius_m") is not None:
        out.append(Quantity("path_radius", float(dv["radius_m"]), "m", float(dv.get("radius_fit_rms_m", 0.0)) + 0.005, 0, "whole primitive",
                            "radius of the circle fitted to the hand's path; its centre is the pivot (vertical axis) the mechanism turns about", [],
                            extra=dict(pivot=dv.get("pivot_xy"))))
    fd = dv.get("final_direction") if travel >= 0.01 else None
    if fd is not None:
        out.append(Quantity("free_direction", round(math.degrees(math.atan2(fd[1], fd[0])), 1), "deg heading", None, 0, "end of primitive",
                            "direction the mechanism was letting the hand move at the end (pull along it next)", [],
                            extra=dict(vector=fd)))
    return out


ESTIMATORS = (est_payload_and_com, est_contact, est_friction, est_insertion, est_pull)


def estimate(rep: ContactReport, ctx: EstimatorContext) -> List[Quantity]:
    """Run every estimator on one report.  An estimator that raises yields an ``estimator_error`` entry instead of
    failing the others."""
    out: List[Quantity] = []
    for fn in ESTIMATORS:
        try:
            out.extend(fn(rep, ctx))
        except Exception as e:  # one failing estimator does not affect the others
            out.append(Quantity("estimator_error", 0.0, "", None, 0, "", fn.__name__, [str(e)[:80]]))
    return out
