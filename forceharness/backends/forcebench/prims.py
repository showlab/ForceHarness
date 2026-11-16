"""ForceBench-only primitives and tool schemas (primitive layer; the base library primitives/library.py is unchanged).

- `press_turn` (force interface): press down on a knob or a nut and turn it while holding the press force.
- `move_turn` (no-force control; also offered with force on Nut Thread): the same motion with a commanded height and no
  force reading.
- `turn` (ForceBench version): the base rigid turn with a cap on how far the attitude target may lead the lever.
- `_turn_quantities`: the resisting torque of press_turn / turn as translation-layer quantities.
- Schemas of the perception tools `locate` and `locate_object`, which FBEpisode executes.

Why press_turn: the Stove Press knob turns only while it is pressed with at least 7 N, and threading a nut needs a light
press while turning so that the nut follows the thread down. Every base primitive starts by moving its target onto the measured
pose (`_sync_target`), so the press left by a push is gone as soon as the next rotate_z starts. press_turn uses the admittance
law of push: along -z it brings the press force to press_force (lead cap 3 cm) while it turns by angle_deg about the vertical
axis through the TCP. This is the idea of the real-robot screw primitive (press first, then turn while pressing), with
parameters for the ForceBench arm (about 1.2 N/mm)."""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.translation.estimators import Quantity
from forceharness.primitives.library import Primitives
from forceharness.primitives.report import ContactReport

# Reading used to pick the easier direction on Stove Press: the median of the samples taken while moving away from the start
# angle, 8-25 deg out on each side. Replayed offline on 56 recorded episodes it picks the easier direction in 53, against 47 for
# the mean over the second half of a turn: the spike in the last 0.3 s of a turn pulls that mean down, and a turn back toward
# the start still reads the side it came from.
SIDE_LO_DEG, SIDE_HI_DEG = 8.0, 25.0


def _turn_quantities(rep: ContactReport) -> List[Dict[str, Any]]:
    """The resisting torque measured by press_turn / turn as translation-layer quantities. The phys formats render only
    quantities; while these numbers were kept only in `derived`, the tool result carried a note ("compare directions") and no
    number at all."""
    d = rep.derived or {}
    out: List[Quantity] = []
    if rep.primitive == "press_turn":
        sides = "resistance_ccw_side_n" in d
        if d.get("resisting_torque_mean_Nm") is not None and not sides:
            out.append(Quantity("resisting_torque", d["resisting_torque_mean_Nm"], "N*m", d.get("resisting_torque_se_Nm"),
                                int(d.get("resisting_torque_n") or 0), "second half of this turn; positive opposes the turn"))
            out.append(Quantity("resisting_torque_peak", d["resisting_torque_peak_Nm"], "N*m", window="whole turn; single-sample spikes are common"))
        if d.get("press_force_mean_N") is not None:
            out.append(Quantity("press_force", d["press_force_mean_N"], "N", window="while turning"))
        if d.get("press_depth_m") is not None:
            out.append(Quantity("press_depth", round(1000.0 * float(d["press_depth_m"]), 1), "mm", window="tool went down, this primitive"))
        if sides:
            for nm in ("ccw", "cw"):
                n = int(d.get(f"resistance_{nm}_side_n") or 0)
                v = d.get(f"resistance_{nm}_side_Nm")
                win = (f"median while moving away from the start, {SIDE_LO_DEG:g}-{SIDE_HI_DEG:g} deg out on that side; {n} samples so far"
                       if v is not None else f"not measured yet: turn past {SIDE_HI_DEG:g} deg on that side of the start")
                out.append(Quantity(f"resistance_{nm}_side", v, "N*m", d.get(f"resistance_{nm}_side_se_Nm"), n, win))
    elif rep.primitive == "turn" and d.get("resisting_moment_mean_Nm") is not None:
        out.append(Quantity("resisting_moment", d["resisting_moment_mean_Nm"], "N*m", window="second half of this turn; positive opposes the turn"))
        out.append(Quantity("resisting_moment_peak", d["resisting_moment_peak_Nm"], "N*m", window=f"at {d.get('resisting_moment_peak_at_deg')} deg of this turn"))
        if d.get("resisting_moment_end_Nm") is not None:
            out.append(Quantity("resisting_moment_end", d["resisting_moment_end_Nm"], "N*m", window="end of this turn"))
    return [q.to_dict() for q in out]


PRESS_TURN_SCHEMA: Dict[str, Any] = dict(
    name="press_turn",
    description=("Press down on what the gripper is holding (or touching) and turn it about the vertical axis through the tool centre point: "
                 "first the downward force is brought to press_force newtons (admittance control), then the tool rotates by angle_deg "
                 "(positive = counter-clockwise seen from above) while the press force is held. Use it for a knob that must be pushed in "
                 "to turn, or to screw a nut down a thread. stop_depth (metres, optional) stops as soon as the tool has gone down that "
                 "far since the start. Reports the press depth (how far the tool went down), the press force and the resisting torque about "
                 "the vertical axis (positive = opposing the turn)."),
    parameters={"type": "object", "properties": {
        "angle_deg": {"type": "number"},
        "press_force": {"type": "number", "description": "N, default 8"},
        "rate_deg_s": {"type": "number", "description": "deg/s, default 6 (faster turns make the fingers stick-slip on the knob)"},
        "press_time": {"type": "number", "description": "s spent pressing before the turn starts, default 1.5"},
        "stop_depth": {"type": "number", "description": "m; stop once the tool has gone down this far (optional)"}},
        "required": ["angle_deg"]},
)


# For the no-force control ("none|pose"). press_turn closes a loop on the force reading, so the control cannot have it. But every
# primitive starts by moving its target onto the measured pose, so with move_by + rotate_z the press is gone once rotate_z
# starts, and Stove Press and Nut Thread become impossible as motions: the control would measure a missing motion, not missing
# force. This tool follows a commanded height and reads no force, which is what an absolute pose command of a learned policy
# (height and yaw together) can do.
MOVE_TURN_SCHEMA: Dict[str, Any] = dict(
    name="move_turn",
    description=("Position-controlled press-and-turn, with no force measurement. First the commanded height of the tool moves by press_dz "
                 "metres over press_time seconds (negative = down), then the tool rotates by angle_deg about the vertical axis through the "
                 "tool centre point (positive = counter-clockwise seen from above) while the commanded height stays there and moves a "
                 "further turn_dz metres in proportion to the rotation. Unlike move_by followed by rotate_z, the command stays below where "
                 "the hand can actually go, so a knob stays pressed in or a nut stays seated during the turn: the arm acts like a spring of "
                 "about 1 N per mm of command below the hand's actual height. Use it for a knob that must be pushed in to turn, or to screw "
                 "a nut down a thread. Reports the end position and the yaw change."),
    parameters={"type": "object", "properties": {
        "angle_deg": {"type": "number"},
        "press_dz": {"type": "number", "description": "m moved before the turn starts, negative = down (at most 3 cm)"},
        "turn_dz": {"type": "number", "description": "m moved during the turn, negative = down (at most 3 cm)"},
        "rate_deg_s": {"type": "number", "description": "deg/s, default 6"},
        "press_time": {"type": "number", "description": "s spent moving by press_dz before the turn starts, default 1.5"}},
        "required": ["angle_deg"]},
)


# locate needs an exact click and the right height from the model. With the oblique cameras, clicking the middle of the bolt and
# intersecting at the height of its top puts the answer more than 1 cm off in x, while Nut Thread needs centring within 2.5 mm.
# With this tool the model clicks roughly and names the object, and the harness fits the object's known shape. It uses only the
# current image, the camera calibration and the dimensions, never ground truth. The enum of `object` is filled per task.
LOCATE_OBJECT_SCHEMA: Dict[str, Any] = dict(
    name="locate_object",
    description=("Precise position of a known object from a camera image. Click roughly on the object in the latest image of that camera: "
                 "the harness separates the object from its background around the click and fits the object's known 3D shape using the "
                 "camera calibration, which corrects for the oblique view, so the click only has to land on or next to the object. Returns "
                 "the world x, y of the object's vertical axis and a fit score from 0 to 1. Far more accurate than locate for tall objects: "
                 "typically 1-2 mm with the wrist camera with the hand 8-15 cm above the object. A score below about 0.6 means the fit is "
                 "unreliable (object cut off by the image edge, or the click landed on something else): move closer or click again."),
    parameters={"type": "object", "properties": {
        "camera": {"type": "string", "enum": ["main", "wrist"]},
        "u": {"type": "number"}, "v": {"type": "number"},
        "object": {"type": "string"}},
        "required": ["camera", "u", "v", "object"]},
)


class FBPrimitives(Primitives):
    turn_sides = False        # True: press_turn records the resisting torque separately on each side of the start angle (task.extra["turn_sides"]; Stove Press)
    # False: the result of turn carries no resisting-moment numbers. The paper's Faucet Turn and Faucet Direction runs used this
    # (configs/forcebench/paper_settings.yaml sets it through --prim-overrides). True (default) adds the numbers; that version
    # was not evaluated with models.
    turn_quantities = True
    _turn_abs_deg = 0.0
    _side_res: Optional[Dict[int, List[float]]] = None

    def grasp(self, steps: int = 16) -> ContactReport:
        self._turn_abs_deg, self._side_res = 0.0, {1: [], -1: []}       # new grasp: the turned angle and the per-side readings restart
        return super().grasp(steps)

    ARG_ALIASES = {**Primitives.ARG_ALIASES,
                   "press_turn": {"angle": "angle_deg", "degrees": "angle_deg", "force": "press_force", "target_force": "press_force",
                                  "rate": "rate_deg_s", "depth": "stop_depth", "stop_on_click": "stop_on_resistance_drop"},
                   "move_turn": {"angle": "angle_deg", "degrees": "angle_deg", "dz": "press_dz", "rate": "rate_deg_s"}}

    turn_lead_cap_deg = 2.0      # first 10 deg of a turn: with 5 deg the lever jammed at about 25 deg; with no lead, small probe turns could not tell the directions apart
    turn_lead_cap_far_deg = 0.5  # after 10 deg: with 2 deg the lever still stopped at 27-28 deg; the smaller the cap, the further it turns

    def turn(self, axis, pivot, angle_deg: float, rate_deg_s: float = 10.0, torque_cap: Optional[float] = None,
             stop_on_resistance_drop: bool = False) -> ContactReport:
        """ForceBench version of turn. Like the base class it turns the hand rigidly with the lever about the hinge and measures
        the resisting moment the same way, with one addition: the attitude target may lead the lever's measured angle (the angle
        the TCP has swept on its arc) by at most turn_lead_cap_deg (2 deg), the same idea as the lead clamp on position.

        Why: the ForceBench arm is soft. Once the lever resists, the base class keeps twisting the attitude toward the commanded
        angle (30 deg of pitch with the lever at 20 deg) and jams the lever between the fingers: on Faucet Direction every turn in
        the easier direction stuck at about 20 deg with 14 N at the wrist. Letting the attitude fully follow the measured angle
        (as ForceBench's expert does) removes the jam at large angles, but then small probe turns cannot tell the two directions
        apart (the readings overlap or even swap sign), while small rigid probe turns agree with the task specification (toward
        the robot 0.17 vs. away 0.08 N*m, specified 0.16 vs. 0.07). So the rigid-turn reading is kept, and the attitude lead is
        limited only when the lever lags."""
        if self.interface != "force":
            torque_cap, stop_on_resistance_drop = None, False
        from scipy.spatial.transform import Rotation as _R

        from forceharness.primitives.library import _unit

        ax = _unit(axis)
        pv = np.asarray(pivot, dtype=float).reshape(3)
        total = math.radians(float(angle_deg))
        sgn = 1.0 if total >= 0 else -1.0
        rate = math.radians(float(rate_deg_s))
        p0 = np.asarray(self.sim.eef_pos, dtype=float)
        q0 = _R.from_quat(self.target_quat)
        v0 = p0 - pv
        v0 = v0 - (v0 @ ax) * ax
        acc = [0.0]
        old_tcap = self.detector.t_cap
        if torque_cap:
            self.detector.t_cap = float(torque_cap)
        mom: Dict[str, Any] = dict(buf=[], peak=0.0, peak_deg=0.0, below=0, click=False, click_deg=None, hist=[])

        def live_angle(pos) -> float:
            v = np.asarray(pos, dtype=float) - pv
            v = v - (v @ ax) * ax
            if float(np.linalg.norm(v0)) < 1e-4 or float(np.linalg.norm(v)) < 1e-4:
                return 0.0
            return math.atan2(float(ax @ np.cross(v0, v)), float(v0 @ v))

        def ctrl(s):
            if abs(acc[0]) < abs(total) - 1e-9:
                acc[0] += sgn * min(rate * self.sim.dt, abs(total) - abs(acc[0]))
            goal = pv + _R.from_rotvec(ax * acc[0]).apply(p0 - pv)
            cap = math.radians(self.turn_lead_cap_deg if abs(acc[0]) < math.radians(10.0) else self.turn_lead_cap_far_deg)
            live = live_angle(s["pos"])
            self.target_quat = (_R.from_rotvec(ax * float(np.clip(acc[0], live - cap, live + cap))) * q0).as_quat()
            # resisting moment defined as in the base class (calibrated on the door_latch task): positive = opposing the turn
            m_res = sgn * (float(ax @ np.asarray(s["torque"], dtype=float))
                           - float(ax @ np.cross(np.asarray(s["pos"], dtype=float) - pv, np.asarray(s["reaction"], dtype=float))))
            mom["buf"] = (mom["buf"] + [m_res])[-4:]
            m_s = float(np.mean(mom["buf"]))
            ang = math.degrees(abs(acc[0]))
            mom["hist"].append((round(ang, 1), round(m_s, 3)))
            if s["t"] < 0.6:
                pass
            elif m_s > mom["peak"]:
                mom["peak"], mom["peak_deg"], mom["below"] = m_s, ang, 0
            elif mom["peak"] > 0.03 and (mom["peak"] - m_s) >= max(0.03, 0.3 * mom["peak"]):
                mom["below"] += 1
                if mom["below"] >= 2 and not mom["click"]:
                    mom["click"], mom["click_deg"] = True, ang
            else:
                mom["below"] = 0
            if stop_on_resistance_drop and mom["click"]:
                return np.zeros(3), 0.0, True
            settled = float(np.linalg.norm(goal - s["pos"])) < 0.003 or s["t"] > abs(total) / rate + 1.0
            return goal - s["target"], 0.0, abs(acc[0]) >= abs(total) - 1e-9 and settled

        def derive(rep):
            v1 = np.asarray(rep.end_pos, dtype=float) - pv
            v1 = v1 - (v1 @ ax) * ax
            turned = math.degrees(math.atan2(float(ax @ np.cross(v0, v1)), float(v0 @ v1))) if np.linalg.norm(v0) > 1e-4 else 0.0
            out: Dict[str, Any] = dict(turned_deg=round(turned, 1), commanded_deg=round(math.degrees(acc[0]), 1),
                                       radius_m=round(float(np.linalg.norm(v0)), 3))
            if self.interface == "force":
                h = mom["hist"]
                tail = [m for _, m in h[len(h) // 2:]] or [0.0]
                out.update(resisting_moment_mean_Nm=round(float(np.mean(tail)), 3),
                           resisting_moment_peak_Nm=round(mom["peak"], 3), resisting_moment_peak_at_deg=round(mom["peak_deg"], 1),
                           resisting_moment_end_Nm=(h[-1][1] if h else None), click_detected=mom["click"], click_at_deg=mom["click_deg"],
                           note=("resisting moment about the turn axis (positive = opposing the turn); mean is over the second half "
                                 "of the turn; a peak followed by a clear drop is a 'click'"))
            return out

        try:
            rep = self._run("turn", dict(axis=np.round(ax, 3).tolist(), pivot=np.round(pv, 4).tolist(), angle_deg=float(angle_deg),
                                         torque_cap=torque_cap, stop_on_resistance_drop=stop_on_resistance_drop),
                            ctrl, abs(total) / rate + 2.0, stop_on_contact=False,
                            grip=1.0 if self.grip_closed else None, derive=derive, suppress_contact=self.grip_closed)
        finally:
            self.detector.t_cap = old_tcap
        if stop_on_resistance_drop and mom["click"]:
            rep.termination = "resistance_drop"
        if rep.termination != "completed":
            self.target_quat = np.asarray(self.sim.eef_quat, dtype=float).copy()
        return rep

    def _estimate(self, rep: ContactReport) -> None:
        super()._estimate(rep)
        if self.interface == "force" and (rep.primitive != "turn" or self.turn_quantities):
            rep.quantities = list(rep.quantities or []) + _turn_quantities(rep)

    def move_turn(self, angle_deg: float, press_dz: float = 0.0, turn_dz: float = 0.0, rate_deg_s: float = 6.0,
                  press_time: float = 1.5) -> ContactReport:
        """Position-controlled press-and-turn: follows a commanded height and reads no force in its control loop (see the
        comment above MOVE_TURN_SCHEMA). Lead cap 3 cm as for press_turn; the 25 N force limit still applies.
        In strict mode the model fills in every optional field, so 0 for rate_deg_s / press_time means the default."""
        sim = self.sim
        total = math.radians(float(angle_deg))
        sgn = 1.0 if total >= 0 else -1.0
        rate = math.radians(float(rate_deg_s) if float(rate_deg_s) > 0 else 6.0)
        t_press = float(press_time) if float(press_time) > 0 else 1.5
        dz_press = float(np.clip(press_dz, -0.03, 0.03))
        dz_turn = float(np.clip(turn_dz, -0.03, 0.03))
        step_press = dz_press * sim.dt / t_press
        st: Dict[str, Any] = dict(acc=0.0, pressed=0.0, turn_t0=None)

        def ctrl(s):
            if abs(st["pressed"]) < abs(dz_press) - 1e-9:
                d = float(np.clip(step_press, -(abs(dz_press) - abs(st["pressed"])), abs(dz_press) - abs(st["pressed"])))
                st["pressed"] += d
                return np.array([0.0, 0.0, d]), 0.0, False
            if st["turn_t0"] is None:
                st["turn_t0"] = s["t"]
            if abs(st["acc"]) < abs(total) - 1e-9:
                d = min(rate * sim.dt, abs(total) - abs(st["acc"]))
                st["acc"] += sgn * d
                return np.array([0.0, 0.0, dz_turn * d / abs(total)]), sgn * d, False
            return np.zeros(3), 0.0, (s["t"] - st["turn_t0"]) >= abs(total) / rate + 0.3

        return self._run("move_turn", dict(angle_deg=float(angle_deg), press_dz=dz_press, turn_dz=dz_turn, rate_deg_s=math.degrees(rate),
                                           press_time=t_press),
                         ctrl, t_press + abs(total) / rate + 2.0, stop_on_contact=False, compliant=True, soft_dir=np.array([0.0, 0.0, 1.0]),
                         max_lead=0.03, grip=1.0 if self.grip_closed else None, suppress_contact=True)

    def press_turn(self, angle_deg: float, press_force: float = 8.0, rate_deg_s: float = 6.0, press_time: float = 1.5,
                   stop_depth: Optional[float] = None, stop_on_resistance_drop: bool = False) -> ContactReport:
        sim = self.sim
        press_force = float(min(max(press_force, 0.5), self.f_cap * 0.8))
        total = math.radians(float(angle_deg))
        sgn = 1.0 if total >= 0 else -1.0
        rate = math.radians(max(float(rate_deg_s), 0.5))
        if stop_depth is not None and float(stop_depth) <= 0:      # strict mode sends 0 for an unused field; read literally, "stop after 0 m" would end the primitive before it moves
            stop_depth = None
        kf = self.kf * 3.0
        p0 = sim.eef_pos
        st: Dict[str, Any] = dict(acc=0.0, turn_t0=None, tz=[], fz=[], ang=[], click=False, peak=0.0, below=0, stop="")
        if self._side_res is None:
            self._side_res = {1: [], -1: []}
        side_res = self._side_res

        def ctrl(s):
            f_n = float(s["reaction"][2])                       # while pressing, the environment pushes the hand toward +z
            v = float(np.clip(kf * (press_force - f_n), -self.speed_near, self.speed_near))
            dpos = np.array([0.0, 0.0, -v * sim.dt])
            dyaw = 0.0
            depth = float(p0[2] - s["pos"][2])
            if stop_depth is not None and depth >= float(stop_depth):
                st["stop"] = "depth_reached"
                return np.zeros(3), 0.0, True
            if s["t"] >= press_time:
                if st["turn_t0"] is None:
                    st["turn_t0"] = s["t"]
                if abs(st["acc"]) < abs(total) - 1e-9:
                    d = min(rate * sim.dt, abs(total) - abs(st["acc"]))
                    st["acc"] += sgn * d
                    dyaw = sgn * d
                tz_res = sgn * float(s["torque"][2])            # raw reading = torque of the arm on the hand: against a resisting load it has the sign of the turn
                st["tz"].append(tz_res)
                if self.turn_sides and dyaw != 0.0:
                    ang_abs = self._turn_abs_deg + math.degrees(st["acc"])
                    if ang_abs * sgn > 0 and SIDE_LO_DEG <= abs(ang_abs) <= SIDE_HI_DEG:      # only moving away from the start (turning back still reads the side it came from)
                        side_res[1 if sgn > 0 else -1].append(tz_res)
                st["fz"].append(f_n)
                st["ang"].append(math.degrees(abs(st["acc"])))
                m = float(np.mean(st["tz"][-4:]))
                if st["ang"][-1] > 2.0:
                    if m > st["peak"]:
                        st["peak"], st["below"] = m, 0
                    elif st["peak"] > 0.02 and (st["peak"] - m) >= max(0.015, 0.4 * st["peak"]):
                        st["below"] += 1
                        if st["below"] >= 2:
                            st["click"] = True
                    else:
                        st["below"] = 0
                if stop_on_resistance_drop and st["click"]:
                    st["stop"] = "resistance_drop"
                    return dpos, 0.0, True
                done = abs(st["acc"]) >= abs(total) - 1e-9 and (s["t"] - st["turn_t0"]) >= abs(total) / rate + 0.3
                return dpos, dyaw, done
            return dpos, 0.0, False

        def derive(rep):
            out: Dict[str, Any] = dict(turned_deg=round(math.degrees(st["acc"]), 1),
                                       press_depth_m=round(float(p0[2] - sim.eef_pos[2]), 4))
            if self.interface == "force" and st["tz"]:
                n = len(st["tz"])
                half = st["tz"][n // 2:] or st["tz"]
                out.update(press_force_mean_N=round(float(np.mean(st["fz"])), 2),
                           resisting_torque_mean_Nm=round(float(np.mean(half)), 4),
                           resisting_torque_se_Nm=round(float(np.std(half)) / math.sqrt(len(half)), 4), resisting_torque_n=len(half),
                           resisting_torque_peak_Nm=round(float(np.max(st["tz"])), 4),
                           click_detected=bool(st["click"]))
                if self.turn_sides:
                    for key, nm in ((1, "ccw"), (-1, "cw")):
                        v = side_res[key]
                        out[f"resistance_{nm}_side_n"] = len(v)
                        if len(v) >= 4:
                            out[f"resistance_{nm}_side_Nm"] = round(float(np.median(v)), 3)
                            out[f"resistance_{nm}_side_se_Nm"] = round(1.253 * float(np.std(v)) / math.sqrt(len(v)), 3)
            if st["stop"]:
                out["stopped"] = st["stop"]
            return out

        rep = self._run("press_turn", dict(angle_deg=float(angle_deg), press_force=press_force, rate_deg_s=float(rate_deg_s),
                                           press_time=float(press_time), stop_depth=stop_depth,
                                           stop_on_resistance_drop=bool(stop_on_resistance_drop)),
                        ctrl, press_time + abs(total) / rate + 2.0, stop_on_contact=False, derive=derive,
                        compliant=True, soft_dir=np.array([0.0, 0.0, 1.0]), max_lead=0.03,
                        grip=1.0 if self.grip_closed else None, suppress_contact=True)
        self._turn_abs_deg += math.degrees(st["acc"])
        if st["stop"] == "resistance_drop":
            rep.termination = "resistance_drop"
        elif st["stop"] == "depth_reached":
            rep.termination = "depth_reached"
        return rep


LOCATE_SCHEMA: Dict[str, Any] = dict(
    name="locate",
    description=("Turn a pixel of the latest camera image into a world position. camera is 'main' or 'wrist'; u, v are pixel coordinates "
                 "in the original 224 x 224 image (u to the right, v downward; the grid drawn on the images is in these units). There is "
                 "no depth camera: the result is where that pixel's line of sight meets a plane you name. Usually the pixel lies on a "
                 "horizontal surface: plane = 'z' and z = the height (metres) of that surface, e.g. 0 for the table top or the top face "
                 "of an object; a wrong z shifts the answer sideways. For a camera looking sideways at a vertical surface use plane = "
                 "'x' or 'y' and value = that coordinate (the vertical plane x = value or y = value); z is then ignored. Reports the "
                 "position and about how many millimetres one pixel covers there."),
    parameters={"type": "object", "properties": {
        "camera": {"type": "string", "enum": ["main", "wrist"]},
        "u": {"type": "number"}, "v": {"type": "number"},
        "z": {"type": "number", "description": "for plane 'z': height of the horizontal surface under the pixel, m (0 = table top)"},
        "plane": {"type": "string", "enum": ["z", "x", "y"],
                  "description": "'z' (default): horizontal plane at height z; 'x' / 'y': vertical plane x = value / y = value"},
        "value": {"type": "number", "description": "for plane 'x' or 'y': that coordinate of the vertical plane, m (ignored for 'z')"}},
        "required": ["camera", "u", "v"]},
)
