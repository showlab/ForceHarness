"""Expert force controller: one hand-written state machine per task that drives the same tool interface as the model.

The expert reads only tool results and the noisy scene, never privileged simulator state. It is the "Expert force
controller" reference row of HiddenPhys for weight_sort, com_sort, wipe_hard, stack_gentle and peg_in_hole_hard, and
it runs the shifted variants with unchanged constants (door_latch and lock_or_stiff have their own expert scripts in
``scripts/hiddenphys/``). Its constants are set by a designer who knows the tasks, so it measures how far a hand-tuned
expert gets, not a task-free method.

Run under the pose interface (``interface="pose"``), the same script cannot weigh, measure a moment or feel contact: it
guesses, or uses nominal heights from the scene. That is the "Position only" row for weight_sort, com_sort, wipe_hard
and stack_gentle (peg_in_hole_hard uses ``baselines/classical.py::PositionOnlyPeg``).

The expert also runs the whole pipeline offline (no API cost) and records the demonstrations that the learned policies
are trained on. It calls the force primitives (layer 2) and reads layer-3 derived signals from the tool results, e.g.
``payload_estimate_kg`` and ``hold_torque_xy_Nm``.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.agent.llm import LLMResponse, ToolCall


YAW_LO, YAW_HI = -110.0, 140.0     # allowed cumulative yaw relative to the start pose (Panda joint 7 starts at 0.8 rad, range +/-2.9 rad; +yaw moves joint 7 toward 0)


def _wrap_yaw(a: float, period: float = 90.0) -> float:
    """Fold an alignment angle into [0, period)."""
    return float(a % period)


def _pick_yaw(a: float, period: float, yaw_acc: float) -> float:
    """Among the equivalent angles a + k * period, pick the one that keeps the cumulative yaw in range and turns least."""
    best, best_cost = None, None
    for k in range(-4, 5):
        cand = a + k * period
        tot = yaw_acc + cand
        cost = abs(cand) + (0 if YAW_LO <= tot <= YAW_HI else 1000 + abs(tot))
        if best_cost is None or cost < best_cost:
            best, best_cost = cand, cost
    return float(best)


def _find(scene: List[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    for o in scene:
        if o.get("name") == name:
            return o
    return None


class ScriptedAgent:
    kind = "scripted"

    def __init__(self, task_name: str, interface: str = "force", light_kg: float = 0.15, heavy_kg: float = 0.60):
        self.task_name = task_name
        self.interface = interface
        self.mid_kg = 0.5 * (light_kg + heavy_kg)
        self.stage = 0
        self.mem: Dict[str, Any] = {}
        self._n = 0
        self.policy = {"weight_sort": self._weight_sort, "wipe": self._wipe, "door": self._door,
                       "stack_gentle": self._stack, "peg_in_hole": self._peg, "com_sort": self._com,
                       "wipe_hard": self._wipe, "peg_in_hole_hard": self._peg,
                       # shifted variants: the same scripts with unchanged constants (run outside the distribution they were designed for)
                       "weight_sort_ood": self._weight_sort, "com_sort_ood": self._com, "wipe_hard_ood": self._wipe,
                       "peg_in_hole_hard_ood": self._peg}[task_name]
        try:
            from forceharness.hiddenphys.tasks import get_task
            self.task_extra = dict(get_task(task_name).extra)
        except Exception:
            self.task_extra = {}

    def _tc(self, name: str, **args) -> ToolCall:
        self._n += 1
        if name == "rotate_z":
            self.mem["yaw_acc"] = self.mem.get("yaw_acc", 0.0) + float(args.get("angle_deg", 0.0))
        return ToolCall(f"s{self._n}", name, args)

    def _rot(self, a: float, period: float) -> ToolCall:
        return self._tc("rotate_z", angle_deg=_pick_yaw(a, period, self.mem.get("yaw_acc", 0.0)))

    def decide(self, ctx: Dict[str, Any]) -> LLMResponse:
        calls = self.policy(ctx)
        return LLMResponse(text="", tool_calls=calls, usage_in=0, usage_out=0, latency_s=0.0)

    @property
    def force(self) -> bool:
        return self.interface == "force"

    def _descend(self, z_target: float, max_dist: float = 0.2, thr: float = 3.0) -> ToolCall:
        """Force interface: descend until contact. Other interfaces: go straight to the height expected from the scene."""
        if self.force:
            return self._tc("move_until_contact", direction=[0, 0, -1], max_dist=max_dist, f_threshold=thr)
        return self._tc("move_to", x=self.mem["xy"][0], y=self.mem["xy"][1], z=z_target)

    # ------------------------------------------------------------------ weight_sort
    def _weight_sort(self, ctx) -> List[ToolCall]:
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        scene = ctx["last_scene"]
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            cube, table = _find(scene, "cube"), _find(scene, "table")
            self.mem.update(cube=cube, top=table["top_z"], xy=cube["pos"][:2], retries=0,
                            zones={o["name"]: o for o in scene if o.get("kind") == "zone"})
            self.stage = 2
            return [self._rot(cube.get("yaw_deg", 0.0), 90.0),
                    self._tc("move_to", x=cube["pos"][0], y=cube["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 2:
            self.stage = 3
            return [self._descend(self.mem["top"] + 0.02)]
        if self.stage == 3:
            if self.force and last is not None and last.termination == "contact" and last.end_pos[2] > self.mem["top"] + 0.03 \
                    and self.mem["retries"] < 2:
                self.mem["retries"] += 1   # the fingers landed on top of the cube: lift a little, turn 45 deg, descend again
                self.stage = 2
                return [self._tc("move_by", dx=0, dy=0, dz=0.03), self._rot(45.0, 90.0)]
            self.stage = 4
            calls = []
            if self.force:
                calls.append(self._tc("move_by", dx=0, dy=0, dz=0.018))
            calls.append(self._tc("grasp"))
            return calls
        if self.stage == 4:
            self.stage = 5
            return [self._tc("lift", height=0.10, hold_time=1.0)]
        if self.stage == 5:
            payload = (last.derived.get("payload_estimate_kg") if last is not None else None)
            verdict = "heavy" if (payload is not None and payload > self.mid_kg) else "light"
            self.mem["verdict"] = verdict
            zone = self.mem["zones"][f"{verdict}_zone"]
            self.mem["xy"] = zone["pos"][:2]
            self.stage = 6
            return [self._tc("move_to", x=zone["pos"][0], y=zone["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 6:
            self.stage = 7
            return [self._descend(self.mem["top"] + 0.025)]
        if self.stage == 7:
            self.stage = 8
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.10),
                    self._tc("done", success=True, verdict=self.mem["verdict"], note="scripted")]
        return [self._tc("done", success=False, verdict=self.mem.get("verdict", ""))]

    # ------------------------------------------------------------------ wipe
    def _wipe(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        if self.stage == 0:
            self.stage = 1
            self.mem["passes"] = 0
            return [self._tc("scene")]
        if self.stage == 1:
            m = _find(scene, "dirt_markers")
            if m is None or m["count"] == 0 or self.mem["passes"] >= 4:
                return [self._tc("done", success=(m is None or m["count"] == 0), note="scripted")]
            pts = np.array(m["xy_list"])
            c = pts.mean(0)
            u, s, vt = np.linalg.svd(pts - c)
            d = vt[0]
            proj = (pts - c) @ d
            start, end = c + d * (proj.min() - 0.03), c + d * (proj.max() + 0.03)
            if self.mem["passes"] % 2 == 1:
                start, end = end, start
            self.mem.update(start=start, end=end, dir=(end - start) / np.linalg.norm(end - start), dist=float(np.linalg.norm(end - start)))
            self.mem["xy"] = start.tolist()
            self.stage = 2
            tbl = _find(scene, "table")
            self.mem["z_nominal"] = float(tbl.get("top_z_nominal", 0.90)) if tbl else 0.90
            return [self._tc("move_to", x=float(start[0]), y=float(start[1]), z=self.mem["z_nominal"] + 0.06)]
        if self.stage == 2:
            self.stage = 3
            if self.force:
                return [self._tc("move_until_contact", direction=[0, 0, -1], max_dist=0.10, f_threshold=3.0)]
            return [self._tc("move_to", x=self.mem["xy"][0], y=self.mem["xy"][1], z=self.mem["z_nominal"] + 0.005)]   # guess the nominal height
        if self.stage == 3:
            self.stage = 1
            self.mem["passes"] += 1
            d = self.mem["dir"]
            if self.force:
                return [self._tc("slide", direction=[float(d[0]), float(d[1]), 0.0], distance=self.mem["dist"],
                                 normal_force=float(self.task_extra.get("wipe_force", 6.0))),
                        self._tc("move_by", dx=0, dy=0, dz=0.05), self._tc("scene")]
            e = self.mem["end"]
            return [self._tc("move_to", x=float(e[0]), y=float(e[1]), z=self.mem["z_nominal"] + 0.005), self._tc("move_by", dx=0, dy=0, dz=0.05), self._tc("scene")]
        return [self._tc("done", success=False)]

    # ------------------------------------------------------------------ door
    def _door(self, ctx) -> List[ToolCall]:
        """The handle is a bar close to the panel (about 7 cm in front of it). Grasping from above makes the 21 cm wide
        palm hit the panel while the fingertips stay 7 cm short, so the expert grasps from the side: 12 cm in front of
        the handle and 10 cm above it, level the tool with point_tool (fingers toward the door, opening up and down),
        descend to the handle height, push in horizontally and close on the bar. Then pull 2 cm at a time along the
        tangent around the hinge and turn the wrist (rotate_z) by the arc of that step to follow the handle."""
        scene = ctx["last_scene"]
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            self.mem["pulls"] = 0
            return [self._tc("scene")]
        if self.stage == 1:
            h, hinge = np.array(_find(scene, "door_handle")["pos"]), np.array(_find(scene, "door_hinge")["pos"])
            r = (h - hinge)[:2]
            r /= np.linalg.norm(r)
            n = np.array([-r[1], r[0]])                    # panel normal; when the door opens (counter-clockwise seen from above) the handle moves along it
            if float(n @ (-h[:2])) < 0:                    # take the side that faces the robot (origin)
                n = -n
            self.mem.update(h=h, hinge=hinge, n=n)
            pre = h.copy(); pre[:2] += 0.12 * n
            self.mem["pre"] = pre
            self.stage = 2
            return [self._tc("move_to", x=float(pre[0]), y=float(pre[1]), z=float(pre[2] + 0.10)),
                    self._tc("point_tool", direction=[float(-n[0]), float(-n[1]), 0.0])]
        if self.stage == 2:
            pre, h, n = self.mem["pre"], self.mem["h"], self.mem["n"]
            goal = h.copy(); goal[:2] -= 0.01 * n                  # fingertips 1 cm past the centre line of the bar
            self.stage = 3
            return [self._tc("move_to", x=float(pre[0]), y=float(pre[1]), z=float(pre[2])),
                    self._tc("move_to", x=float(goal[0]), y=float(goal[1]), z=float(goal[2]))]
        if self.stage == 3:
            self.stage = 4
            return [self._tc("grasp")]
        if self.stage == 4:
            if last is not None and last.primitive == "grasp" and not last.derived.get("closed_on_something", True):
                self.stage = 9
                return [self._tc("release"), self._tc("done", success=False, note="scripted: grasp missed the handle")]
            ang = _find(scene, "door_angle_deg")
            if ang is not None and ang["value"] > 20.0:
                self.stage = 9
                return [self._tc("release"), self._tc("move_by", dx=float(self.mem["n"][0] * 0.08), dy=float(self.mem["n"][1] * 0.08), dz=0.0),
                        self._tc("done", success=True, note="scripted")]
            if self.mem["pulls"] >= 14:
                self.stage = 9
                return [self._tc("release"), self._tc("done", success=False, note="scripted: pull budget")]
            self.mem["pulls"] += 1
            p = np.array(last.end_pos) if last is not None else self.mem["h"]
            r = p[:2] - self.mem["hinge"][:2]
            rad = float(np.linalg.norm(r))
            tangent = np.array([-r[1], r[0]]) / (rad + 1e-9)        # counter-clockwise about +z = opening direction
            step = 0.02
            return [self._tc("move_by", dx=float(tangent[0] * step), dy=float(tangent[1] * step), dz=0.0, stop_on_contact=False),
                    self._tc("rotate_z", angle_deg=float(math.degrees(step / rad))),
                    self._tc("scene")]
        return [self._tc("done", success=False)]

    # ------------------------------------------------------------------ stack_gentle
    def _stack(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            a, b, table = _find(scene, "red_cube"), _find(scene, "green_cube"), _find(scene, "table")
            self.mem.update(a=np.array(a["pos"]), b=np.array(b["pos"]), top=table["top_z"], xy=a["pos"][:2], retries=0)
            self.stage = 2
            return [self._rot(a.get("yaw_deg", 0.0), 90.0),
                    self._tc("move_to", x=a["pos"][0], y=a["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 2:
            self.stage = 3
            return [self._descend(self.mem["top"] + 0.02)]
        if self.stage == 3:
            if self.force and last is not None and last.termination == "contact" and last.end_pos[2] > self.mem["top"] + 0.03 \
                    and self.mem["retries"] < 2:
                self.mem["retries"] += 1
                self.stage = 2
                return [self._tc("move_by", dx=0, dy=0, dz=0.03), self._rot(45.0, 90.0)]
            self.stage = 4
            calls = [self._tc("move_by", dx=0, dy=0, dz=0.018)] if self.force else []
            calls += [self._tc("grasp"), self._tc("lift", height=0.10, hold_time=0.5)]
            return calls
        if self.stage == 4:
            b = self.mem["b"]
            self.mem["xy"] = b[:2].tolist()
            self.stage = 5
            return [self._tc("move_to", x=float(b[0]), y=float(b[1]), z=float(b[2]) + 0.025 + 0.02 + 0.06)]
        if self.stage == 5:
            self.stage = 6
            if self.force:
                return [self._tc("move_until_contact", direction=[0, 0, -1], max_dist=0.10, f_threshold=2.5)]
            return [self._tc("move_to", x=self.mem["xy"][0], y=self.mem["xy"][1], z=float(self.mem["b"][2]) + 0.025 + 0.02)]
        if self.stage == 6:
            self.stage = 7
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.10), self._tc("done", success=True, note="scripted")]
        return [self._tc("done", success=False)]

    # ------------------------------------------------------------------ peg_in_hole (the peg starts in the gripper)
    def _peg(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            hole, peg = _find(scene, "hole"), _find(scene, "peg")
            h = np.array(hole["pos"])
            self.mem.update(h=h, tip_offset=0.04, xy=h[:2].tolist())
            self.stage = 2
            return [self._tc("move_to", x=float(h[0]), y=float(h[1]), z=float(h[2]) + 0.04 + 0.03)]   # peg tip 3 cm above the hole opening
        if self.stage == 2:
            self.stage = 3
            return [self._tc("move_until_contact", direction=[0, 0, -1], max_dist=0.08, f_threshold=3.0)]
        if self.stage == 3:
            z_in = float(self.mem["h"][2]) + 0.04 - 0.02          # tool height when the peg is 2 cm in
            # Spiral search sliding on the face of the hole block while pressing (over 5x faster than probing point by
            # point); once the peg drops in, a compliant insert seats it; if it jams, back off 1 cm and try once more.
            if last is not None and last.primitive == "move_until_contact" and last.termination == "contact" and last.end_pos[2] > z_in + 0.005 and not self.mem.get("searched"):
                self.mem["searched"] = True
                return [self._tc("search", normal=[0, 0, -1], radius_max=float(self.task_extra.get("search_radius", 0.02)), pitch=0.004,
                                 drop=0.012, mode="slide", normal_force=4.0, speed=0.02)]
            if last is not None and last.primitive == "search" and last.termination in ("dropped", "stuck"):
                return [self._tc("insert", depth=0.02, max_force=12.0)]
            if last is not None and last.primitive == "insert" and last.termination == "jammed" and not self.mem.get("retried"):
                self.mem["retried"] = True
                return [self._tc("move_by", dx=0, dy=0, dz=0.01), self._tc("insert", depth=0.025, max_force=15.0)]
            if last is not None and last.primitive == "insert" and last.termination not in ("seated", "stalled") and last.end_pos[2] > z_in + 0.003 and not self.mem.get("retried2"):
                self.mem["retried2"] = True
                return [self._tc("insert", depth=0.02, max_force=14.0)]
            self.stage = 4
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.08), self._tc("done", success=True, note="scripted")]
        return [self._tc("done", success=False)]

    # ------------------------------------------------------------------ com_sort
    def _com(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            bar, table = _find(scene, "bar"), _find(scene, "table")
            self.mem.update(bar=bar, top=table["top_z"], xy=bar["pos"][:2], retries=0,
                            zones={o["name"]: o for o in scene if o.get("kind") == "zone"})
            self.stage = 2
            # The finger axis must be perpendicular to the long axis. The long axis points along yaw_deg and the finger
            # axis lies along y at yaw = 0, so the tool yaw equals the long-axis direction.
            return [self._rot(bar.get("yaw_deg", 0.0), 180.0),
                    self._tc("move_to", x=bar["pos"][0], y=bar["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 2:
            self.stage = 3
            return [self._descend(self.mem["top"] + 0.02)]
        if self.stage == 3:
            self.stage = 4
            calls = [self._tc("move_by", dx=0, dy=0, dz=0.018)] if self.force else []
            calls += [self._tc("grasp"), self._tc("lift", height=0.10, hold_time=1.0)]
            return calls
        if self.stage == 4:
            tq = last.derived.get("hold_torque_xy_Nm") if last is not None else None
            mag = float(np.hypot(*tq)) if tq else 0.0
            verdict = "offset" if mag > 0.12 else "balanced"
            self.mem["verdict"] = verdict
            zone = self.mem["zones"][f"{verdict}_zone"]
            self.mem["xy"] = zone["pos"][:2]
            self.stage = 5
            return [self._tc("move_to", x=zone["pos"][0], y=zone["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 5:
            self.stage = 6
            return [self._descend(self.mem["top"] + 0.025)]
        if self.stage == 6:
            self.stage = 7
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.10),
                    self._tc("done", success=True, verdict=self.mem["verdict"], note="scripted")]
        return [self._tc("done", success=False, verdict=self.mem.get("verdict", ""))]
