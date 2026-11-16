"""Scripted agent for the eight ForceBench tasks (a reference baseline, not ForceHarness itself): the ForceBench counterpart of
the expert force controller of HiddenPhys (`forceharness/baselines/expert.py`). One hand-written state machine per task that
calls the same tools as the model, with no API calls.

Decision constants follow from the task instructions (weight: midpoint of the two mass classes; centre of mass: half of
m*g*offset). The agent never reads privileged truth (class, mass, centre of mass). It reads only what the model can also see:
the scene (privileged mode) or the camera images through classical vision (RGB-only mode, see rgb_detect.py), and the
primitive results."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.agent.llm import LLMResponse, ToolCall
from forceharness.baselines.expert import _find, _pick_yaw
from forceharness.backends.forcebench.tasks import RS_MASS, RS_OFFSET, WS_HEAVY, WS_LIGHT

G = 9.81


class FBScriptedAgent:
    kind = "scripted"

    def __init__(self, task_name: str, interface: str = "force", grasp_dz: Optional[float] = None, descend_thr: float = 3.0,
                 lift_height: float = 0.10, hold_time: Optional[float] = None, perception: str = "privileged", sim=None):
        self.task_name = task_name
        self.interface = interface
        # rise after the fingertips touch the table, then close: grasp near mid-height (weight_sort box 30 mm -> 18 mm; rattle_sort parcel 23 mm -> 12 mm, close to the expert's TCP z 0.014)
        self.grasp_dz = grasp_dz if grasp_dz is not None else {"weight_sort": 0.018, "rattle_sort": 0.012, "nut_thread": 0.015}.get(task_name, 0.018)
        self.descend_thr = descend_thr
        self.lift_height = lift_height
        self.hold_time = hold_time if hold_time is not None else {"weight_sort": 1.0, "rattle_sort": 2.0}.get(task_name, 1.0)   # CoM: hold still for 2 s (as ForceBench's expert does)
        self.stage = 0
        self.mem: Dict[str, Any] = {}
        self._n = 0
        self.mid_kg = 0.5 * (WS_LIGHT + WS_HEAVY)
        self.com_thr = 0.5 * RS_MASS * G * RS_OFFSET          # 0.0177 N*m
        self.perception = perception
        self.sim = sim                    # RGB only: current images and camera calibration come from FBSim (colour detection = classical vision)
        self.loc_log: List[Dict[str, Any]] = []
        self.pstage = 0                   # RGB only: perception stage (0 = main view not yet seen; then one step per viewpoint)
        self.est: Dict[str, Any] = {}     # perception estimates (world frame)
        self.rgb_scene: Optional[List[Dict[str, Any]]] = None
        self.dbg_imgs: List[Dict[str, Any]] = []   # images + calibration seen at each perception step (saved for offline checks)
        self.policy = {"weight_sort": self._sort_rgb if perception == "rgb" else self._sort, "rattle_sort": self._sort, "stove_turn_on": self._stove,
                       "stove_press_turn": self._stove_press, "faucet_turn": self._faucet, "faucet_force_turn": self._faucet_force,
                       "radmanso_drawer_sort_table": self._drawer, "nut_thread": self._nut}[task_name]

    @property
    def force(self) -> bool:
        return self.interface == "force"

    def _tc(self, name: str, **args) -> ToolCall:
        self._n += 1
        if name == "rotate_z":
            self.mem["yaw_acc"] = self.mem.get("yaw_acc", 0.0) + float(args.get("angle_deg", 0.0))
        return ToolCall(f"s{self._n}", name, args)

    def decide(self, ctx: Dict[str, Any]) -> LLMResponse:
        if self.perception == "rgb" and self.task_name != "weight_sort":
            calls = self._rgb_perceive()
            if calls is not None:
                return LLMResponse(text="", tool_calls=calls, usage_in=0, usage_out=0, latency_s=0.0)
            ctx = dict(ctx, last_scene=self.rgb_scene)
        return LLMResponse(text="", tool_calls=self.policy(ctx), usage_in=0, usage_out=0, latency_s=0.0)

    def _descend(self, z_target: float, max_dist: float = 0.2) -> ToolCall:
        if self.force:
            return self._tc("move_until_contact", direction=[0, 0, -1], max_dist=max_dist, f_threshold=self.descend_thr)
        return self._tc("move_to", x=self.mem["xy"][0], y=self.mem["xy"][1], z=z_target)

    def _verdict(self, rep) -> str:
        d = rep.derived if rep is not None else {}
        if self.task_name == "weight_sort":
            m = d.get("payload_estimate_kg")
            return "heavy" if (m is not None and m > self.mid_kg) else "light"
        tq = d.get("hold_torque_xy_Nm")
        mag = float(np.hypot(*tq)) if tq else 0.0
        return "imbalanced" if mag > self.com_thr else "uniform"

    def _dest(self, scene, verdict: str) -> Dict[str, Any]:
        name = {"light": "black_container", "heavy": "blue_container", "imbalanced": "black_basket", "uniform": "conveyor"}[verdict]
        return _find(scene, name)

    def _sort(self, ctx) -> List[ToolCall]:
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        scene = ctx["last_scene"]
        obj_name = "box" if self.task_name == "weight_sort" else "parcel"
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            obj, table = _find(scene, obj_name), _find(scene, "table")
            self.mem.update(obj=obj, top=table["top_z"], xy=obj["pos"][:2], retries=0, scene=scene)
            self.stage = 2
            # at the initial yaw the fingers close along world y; the long side has yaw_deg ~ 0 -> align the tool with it (period 180 deg)
            return [self._tc("rotate_z", angle_deg=_pick_yaw(obj.get("yaw_deg", 0.0), 180.0, self.mem.get("yaw_acc", 0.0))),
                    self._tc("move_to", x=obj["pos"][0], y=obj["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 2:
            self.stage = 3
            return [self._descend(self.mem["top"] + 0.02)]
        if self.stage == 3:
            top = self.mem["top"]
            if self.force and last is not None and last.termination == "contact" and last.end_pos[2] > top + 0.02 \
                    and self.mem["retries"] < 2:
                self.mem["retries"] += 1          # fingers landed on top of the object: lift a little and descend again
                self.stage = 2
                return [self._tc("move_by", dx=0, dy=0, dz=0.03)]
            self.stage = 4
            calls = []
            if self.force:
                calls.append(self._tc("move_by", dx=0, dy=0, dz=self.grasp_dz))
            calls += [self._tc("grasp"), self._tc("lift", height=self.lift_height, hold_time=self.hold_time)]
            return calls
        if self.stage == 4:
            verdict = self._verdict(last)
            self.mem["verdict"] = verdict
            dest = self._dest(self.mem["scene"], verdict)
            self.mem["xy"] = dest["pos"][:2]
            self.mem["dest_z"] = float(dest["pos"][2])
            self.stage = 5
            return [self._tc("move_to", x=dest["pos"][0], y=dest["pos"][1], z=self.mem["top"] + 0.15)]
        if self.stage == 5:
            self.stage = 6
            return [self._descend(self.mem["dest_z"] + 0.03)]
        if self.stage == 6:
            self.stage = 7
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.10),
                    self._tc("done", success=True, verdict=self.mem["verdict"], note="scripted")]
        return [self._tc("done", success=False, verdict=self.mem.get("verdict", ""))]

    # ------------------------------------------------------------------ mechanisms: grasp first, then turn / press-turn / pull
    def _grasp_at(self, g, above: float) -> List[ToolCall]:
        """Descend from `above` metres straight above to grasp_point (no contact stop on the last segment: the fingers slide
        down along the knob / rod), then close the gripper."""
        return [self._tc("move_to", x=g[0], y=g[1], z=g[2] + above),
                self._tc("move_to", x=g[0], y=g[1], z=g[2], speed=0.03, stop_on_contact=False),
                self._tc("grasp")]

    def _stove(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            self.stage = 2
            return self._grasp_at(_find(scene, "stove_knob")["grasp_point"], 0.12)
        if self.stage == 2:
            self.stage = 3        # ignition holds on the first tick past detent + 0.5 deg; turn to 100 deg (limit 120 deg)
            return [self._tc("rotate_z", angle_deg=100.0)]
        return [self._tc("done", success=True, verdict="", note="scripted")]

    def _stove_press(self, ctx) -> List[ToolCall]:
        scene, reps = ctx["last_scene"], ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            self.stage = 2
            return self._grasp_at(_find(scene, "stove_knob")["grasp_point"], 0.12)
        if self.stage == 2 and not self.force:
            self.stage = 9        # no force: the easier side cannot be measured, so guess counter-clockwise; command 15 mm down (8 mm travel + ~7 mm arm give at 7 N)
            self.mem["verdict"] = "ccw"
            return [self._tc("move_turn", angle_deg=95.0, press_dz=-0.015)]
        if self.stage == 2:
            self.stage = 3        # probe counter-clockwise (+15 deg)
            return [self._tc("press_turn", angle_deg=15.0, press_force=8.0)]
        if self.stage == 3:
            self.mem["m_ccw"] = float((last.derived or {}).get("resisting_torque_mean_Nm") or 0.0)
            self.stage = 4        # turn back by -30 deg to -15 deg (the second half of this turn is on the clockwise side)
            return [self._tc("press_turn", angle_deg=-30.0, press_force=8.0)]
        if self.stage == 4:
            self.mem["m_cw"] = float((last.derived or {}).get("resisting_torque_mean_Nm") or 0.0)
            easy_ccw = self.mem["m_ccw"] <= self.mem["m_cw"]
            self.mem["verdict"] = "ccw" if easy_ccw else "cw"
            self.stage = 5        # now at -15 deg; success once past detent + 0.5 deg (at most 90.5 deg)
            return [self._tc("press_turn", angle_deg=(15.0 + 95.0) if easy_ccw else -(95.0 - 15.0), press_force=8.0)]
        return [self._tc("done", success=True, verdict=self.mem.get("verdict", ""), note="scripted")]

    def _faucet(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            lever = _find(scene, "faucet_lever")
            self.mem.update(pivot=lever["pivot"])
            self.stage = 2
            return self._grasp_at(lever["grasp_point"], 0.09)
        if self.stage == 2:
            self.stage = 3        # rod top toward the robot = negative turn about +y; success once past detent + 0.95 deg (at most 31 deg)
            return [self._tc("turn", axis=[0, 1, 0], pivot=self.mem["pivot"], angle_deg=-34.0, rate_deg_s=8.0)]
        return [self._tc("done", success=True, verdict="", note="scripted")]

    def _faucet_force(self, ctx) -> List[ToolCall]:
        scene, reps = ctx["last_scene"], ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            lever = _find(scene, "faucet_lever")
            self.mem.update(pivot=lever["pivot"])
            self.stage = 2
            return self._grasp_at(lever["grasp_point"], 0.09)
        if self.stage == 2:
            self.stage = 3        # probe 8 deg toward the robot (both probes start from 0, so both compare their 4-8 deg half)
            return [self._tc("turn", axis=[0, 1, 0], pivot=self.mem["pivot"], angle_deg=-8.0, rate_deg_s=6.0)]
        if self.stage == 3:
            self.mem["m_toward"] = float((last.derived or {}).get("resisting_moment_mean_Nm") or 0.0)
            self.mem["at"] = float((last.derived or {}).get("turned_deg") or -8.0)
            self.stage = 35       # back to 0
            return [self._tc("turn", axis=[0, 1, 0], pivot=self.mem["pivot"], angle_deg=-self.mem["at"], rate_deg_s=6.0)]
        if self.stage == 35:
            self.mem["at"] += float((last.derived or {}).get("turned_deg") or 0.0)
            self.stage = 4        # probe 8 deg away from the robot
            return [self._tc("turn", axis=[0, 1, 0], pivot=self.mem["pivot"], angle_deg=8.0, rate_deg_s=6.0)]
        if self.stage == 4:
            self.mem["m_away"] = float((last.derived or {}).get("resisting_moment_mean_Nm") or 0.0)
            self.mem["at"] += float((last.derived or {}).get("turned_deg") or 8.0)
            toward = self.mem["m_toward"] <= self.mem["m_away"]
            self.mem["verdict"] = "toward_robot" if toward else "away_from_robot"
            self.stage = 5        # past the detent (at most 30 deg) to 35 deg
            goal = -35.0 if toward else 35.0
            return [self._tc("turn", axis=[0, 1, 0], pivot=self.mem["pivot"], angle_deg=goal - self.mem["at"], rate_deg_s=8.0)]
        if self.stage == 5:
            self.stage = 6        # success needs a release, TCP z >= 0.108 and a still valve
            return [self._tc("release"), self._tc("move_by", dx=0.0, dy=0.0, dz=0.13), self._tc("hold", seconds=1.0)]
        return [self._tc("done", success=True, verdict=self.mem.get("verdict", ""), note="scripted")]

    def _drawer(self, ctx) -> List[ToolCall]:
        scene = ctx["last_scene"]
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            h = _find(scene, "drawer_handle")["pos"]
            self.mem["h"] = h
            self.stage = 2
            return [self._tc("point_tool", direction=[0, -1, 0]), self._tc("move_to", x=h[0], y=h[1] + 0.10, z=h[2])]
        if self.stage == 2:
            h = self.mem["h"]
            self.stage = 3
            return [self._tc("move_to", x=h[0], y=h[1], z=h[2], speed=0.03, stop_on_contact=False), self._tc("grasp")]
        if self.stage == 3:
            self.stage = 4        # pull toward +y with a constant 2.4 N: free travel needs ~0.35 N; the hand stops where the rising resistance balances 2.4 N, the target
            return [self._tc("push", direction=[0, 1, 0], target_force=2.4, duration=18.0, max_dist=0.16)]
        return [self._tc("hold", seconds=1.0), self._tc("done", success=True, verdict="", note="scripted")]

    def _nut_pose(self, ctx) -> List[ToolCall]:
        """No force reading: position only. Grasp the nut at mid-height (TCP z 0.018), carry it straight above the bolt, and
        from where the nut bottom meets the shank top command 3 mm further down while turning 1.5 turns."""
        scene = ctx["last_scene"]
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            nut, bolt = _find(scene, "nut"), _find(scene, "bolt")
            self.mem.update(bolt=bolt)
            self.stage = 2
            x, y = nut["pos"][:2]
            return [self._tc("move_to", x=x, y=y, z=0.12), self._tc("move_to", x=x, y=y, z=0.018, speed=0.03, stop_on_contact=False),
                    self._tc("grasp"), self._tc("move_by", dx=0, dy=0, dz=0.12)]
        if self.stage == 2:
            b = self.mem["bolt"]
            self.stage = 3
            x, y, top = b["pos"][0], b["pos"][1], b["shank_top_z"]
            return [self._tc("move_to", x=x, y=y, z=top + 0.008 + 0.015),
                    self._tc("move_to", x=x, y=y, z=top + 0.008, speed=0.01, stop_on_contact=False),
                    self._tc("move_turn", angle_deg=-540.0, press_dz=-0.001, turn_dz=-0.003, rate_deg_s=30.0, press_time=0.5)]
        if self.stage == 3:
            self.stage = 4
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.08)]
        return [self._tc("done", success=True, verdict="", note="scripted pose")]

    def _nut(self, ctx) -> List[ToolCall]:
        if not self.force:
            return self._nut_pose(ctx)
        scene, reps = ctx["last_scene"], ctx["reports"]
        last = reps[-1] if reps else None
        if self.stage == 0:
            self.stage = 1
            return [self._tc("scene")]
        if self.stage == 1:
            nut, bolt = _find(scene, "nut"), _find(scene, "bolt")
            self.mem.update(nut=nut, bolt=bolt, xy=nut["pos"][:2])
            self.stage = 2
            return [self._tc("move_to", x=nut["pos"][0], y=nut["pos"][1], z=0.12), self._descend(0.0, max_dist=0.15)]
        if self.stage == 2:
            self.stage = 3        # after touching the table, rise to the nut's mid-height and close (expert TCP z 0.018), then lift
            return [self._tc("move_by", dx=0, dy=0, dz=self.grasp_dz), self._tc("grasp"), self._tc("lift", height=0.12, hold_time=0.5)]
        if self.stage == 3:
            b = self.mem["bolt"]
            self.mem["tcp_over_nut"] = 0.008
            self.stage = 4        # nut bottom 1.5 cm above the shank top, then descend until contact
            return [self._tc("move_to", x=b["pos"][0], y=b["pos"][1], z=b["shank_top_z"] + 0.008 + 0.015),
                    self._tc("move_until_contact", direction=[0, 0, -1], max_dist=0.04, f_threshold=2.0, speed=0.008)]
        if self.stage == 4:
            b = self.mem["bolt"]
            on_tip = last is not None and last.end_pos[2] > b["shank_top_z"] + 0.008 - 0.004
            self.stage = 5
            if on_tip:            # nut rests on the shank top: not aligned. Its bore is smaller than the bolt, so even aligned it sinks only 2-3 mm (onto the chamfer) -> probe point by point
                return [self._tc("search", normal=[0, 0, -1], radius_max=0.006, pitch=0.002, drop=0.0015, mode="hop", hop=0.003, speed=0.02)]
            return [self._tc("hold", seconds=0.3)]
        if self.stage == 5:
            self.mem["z_found"] = float(last.end_pos[2]) if last is not None else None
            self.stage = 6        # check: half a turn with a light press; engaged, the nut sinks ~1 mm per half turn; much more = it slid off the shank edge
            return [self._tc("press_turn", angle_deg=-180.0, press_force=1.5, rate_deg_s=30.0, press_time=0.5)]
        if self.stage == 6:
            b = self.mem["bolt"]
            sank = float((last.derived or {}).get("press_depth_m") or 0.0) if last is not None else 0.0
            if sank > 0.0025 and self.mem.get("retries", 0) < 1:
                self.mem["retries"] = self.mem.get("retries", 0) + 1
                self.stage = 4    # slid off: go back above the bolt and search once more
                return [self._tc("move_to", x=b["pos"][0], y=b["pos"][1], z=b["shank_top_z"] + 0.008 + 0.015),
                        self._tc("move_until_contact", direction=[0, 0, -1], max_dist=0.04, f_threshold=2.0, speed=0.008)]
            z_goal = b["target_nut_bottom_z"] + self.mem["tcp_over_nut"] + 0.0005
            depth = max(0.0, float(last.end_pos[2]) - z_goal) if last is not None else 0.0
            self.stage = 7        # keep turning clockwise; stop when the nut bottom reaches the goal height
            return [self._tc("press_turn", angle_deg=-720.0, press_force=1.5, rate_deg_s=30.0, press_time=0.3, stop_depth=depth)]
        if self.stage == 7:
            self.stage = 8
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.08)]
        return [self._tc("done", success=True, verdict="", note="scripted")]

    # ------------------------------------------------------------------ RGB-only Weight Sort: colour detection + line of sight on a plane of known height
    def _see(self, what: str):
        """Find `what` (box_main, box_wrist, black_container or blue_container) in the current images; return its world
        position or None."""
        from forceharness.backends.forcebench.vision import find_blob, find_rect

        sim = self.sim
        main = sim.images_raw.get("main")
        if what == "box_main":
            b = find_blob(main, "tan", (150, 3000))
            cam, z = "main", 0.018        # the oblique main view includes the side facing the camera: intersect at half the box height (top face is +16 mm off in x)
        elif what == "box_wrist":
            b = find_blob(sim.images_raw.get("wrist"), "tan", (300, 40000))
            cam, z = "wrist", 0.0331
        else:
            blue = find_blob(main, "blue", (1000, 6000))
            if blue is None:
                return None
            b = blue if what == "blue_container" else find_rect(main, "black", (blue["bw"], blue["bh"]))
            cam, z = "main", 0.04          # half the container height: the visible part is rim + inner walls + floor
        if b is None:
            return None
        p = sim.locate_on_plane(cam, b["u"], b["v"], z)
        truth = (getattr(sim, "diag", {}) or {}).get("entities", {})
        self.loc_log.append(dict(step=sim.step_index, what=what, camera=cam, u=b["u"], v=b["v"], z=z,
                                 p=None if p is None else np.round(p, 4).tolist(),
                                 truth_box=(truth.get("box") or {}).get("pos")))
        return p

    def _sort_rgb(self, ctx) -> List[ToolCall]:
        reps = ctx["reports"]
        last = reps[-1] if reps else None
        top = 0.0
        if self.stage == 0:
            box = self._see("box_main")
            cont = {"black_container": self._see("black_container"), "blue_container": self._see("blue_container")}
            if box is None or any(v is None for v in cont.values()):
                return [self._tc("done", success=False, verdict="", note="scripted: could not find box / containers in the main image")]
            self.mem.update(box=box, cont=cont, retries=0, top=top)
            self.stage = 1        # go 15 cm above the box as seen in the main view, then refine with the wrist camera
            return [self._tc("move_to", x=float(box[0]), y=float(box[1]), z=top + 0.15)]
        if self.stage == 1:
            fine = self._see("box_wrist")
            if fine is not None:
                self.mem["box"] = fine
            b = self.mem["box"]
            self.mem["xy"] = [float(b[0]), float(b[1])]
            self.stage = 2
            return [self._tc("move_to", x=float(b[0]), y=float(b[1]), z=top + 0.15), self._descend(top + 0.02)]
        if self.stage == 2:
            if self.force and last is not None and last.termination == "contact" and last.end_pos[2] > top + 0.02 \
                    and self.mem["retries"] < 2:
                self.mem["retries"] += 1      # fingers landed on top of the box: lift a little and descend again
                self.stage = 1
                return [self._tc("move_by", dx=0, dy=0, dz=0.03)]
            self.stage = 3
            calls = [self._tc("move_by", dx=0, dy=0, dz=self.grasp_dz)] if self.force else []
            return calls + [self._tc("grasp"), self._tc("lift", height=self.lift_height, hold_time=self.hold_time)]
        if self.stage == 3:
            verdict = self._verdict(last)
            self.mem["verdict"] = verdict
            c = self.mem["cont"]["black_container" if verdict == "light" else "blue_container"]
            self.mem["xy"] = [float(c[0]), float(c[1])]
            self.stage = 4
            return [self._tc("move_to", x=float(c[0]), y=float(c[1]), z=top + 0.15)]
        if self.stage == 4:
            self.stage = 5
            return [self._descend(top + 0.03)]
        if self.stage == 5:
            self.stage = 6
            return [self._tc("release"), self._tc("move_by", dx=0, dy=0, dz=0.10),
                    self._tc("done", success=True, verdict=self.mem["verdict"], note="scripted rgb")]
        return [self._tc("done", success=False, verdict=self.mem.get("verdict", ""))]


    # ------------------------------------------------------------------ RGB-only versions of the other seven tasks: look first (coarse
    # position from the main view, then a finer one from the wrist camera at a viewpoint), assemble the estimates into a scene in the
    # privileged format, and reuse the action sequences above (_sort / _faucet / _stove / _drawer / _nut). Inputs are only the two
    # RGB images, the camera calibration and the dimensions / heights from the instruction; colour thresholds and rules such as
    # "the leftmost part" or "the middle one" are hand-written classical vision.
    def _log(self, what: str, p, how: str) -> None:
        truth = (getattr(self.sim, "diag", {}) or {}).get("entities", {})
        self.loc_log.append(dict(step=self.sim.step_index, what=what, how=how, p=None if p is None else np.round(np.asarray(p, float), 4).tolist(),
                                 truth={k: (v or {}).get("pos") for k, v in truth.items()}))

    def _rgb_perceive(self) -> Optional[List[ToolCall]]:
        """Return the next batch of tool calls; when perception is finished (or has failed) return None with self.rgb_scene /
        self.stage set."""
        if self.rgb_scene is not None:
            return None
        from forceharness.backends.forcebench import rgb_detect as D

        fn = getattr(D, "perceive_" + self.task_name)
        self.dbg_imgs.append(dict(k=self.pstage, step=self.sim.step_index, state=self.sim.state.tolist(), calib=self.sim.calib,
                                  images={c: np.asarray(v, dtype=np.uint8) for c, v in self.sim.images_raw.items()}))
        calls, scene = fn(self, self.sim, self.pstage)
        self.pstage += 1
        if scene is not None:
            self.rgb_scene = scene
            self.stage = 1               # skip the scene call of stage 0 of the privileged sequence
            return None
        return calls
