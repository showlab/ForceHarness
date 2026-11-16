"""Offline fake backend without robosuite: a point-like tool, a table and one graspable cube.

It exists to test the episode loop, contact events and the translation layer on any machine (no MuJoCo, no GPU). The
physics is three rules: (1) the tool follows the command, and a target below the table top is clamped to the surface with
a reaction of k * (depth of the target below the surface), k = 300 N/m; (2) closing the gripper while the cube is between
the fingers grasps it, and the cube then moves with the tool; (3) while the grasped cube is off the table, its weight
-m g is added to the z reaction. That is enough for a full weight_sort episode; the other tasks need the MuJoCo backend.
"""
from __future__ import annotations

import types
from typing import Any, Dict, List, Optional

import numpy as np

from forceharness.backends.base import WrenchSample

G = 9.81


class FakeSim:
    dt = 0.05

    def __init__(self, task, seed: int = 0, table_top: float = 0.8, scene_noise_pos: float = 0.0,
                 scene_noise_yaw_deg: float = 0.0, force_sign: float = 1.0, **_):
        self.task = task
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.table_top = table_top
        self.env = types.SimpleNamespace(table_offset=np.array([0.0, 0.0, table_top]), _check_success=lambda: False)
        self.action_dim = 7
        self.scene_noise_pos, self.scene_noise_yaw_deg = scene_noise_pos, scene_noise_yaw_deg
        self.force_sign = force_sign
        self._scene_offsets: Dict[str, np.ndarray] = {}
        self.task_state: Dict[str, Any] = {}
        self.camera_names: List[str] = []
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.t = 0.0
        self.pos = np.array([0.0, 0.0, self.table_top + 0.25])
        self.target = self.pos.copy()
        self.yaw = 0.0
        self.opening = 0.08
        self.grip_cmd = -1.0
        self._pen = 0.0
        self.objects = {"cube": dict(pos=np.array([0.02, -0.01, self.table_top + 0.02]), half=0.02, mass=0.1, grasped=False)}
        self._scene_offsets = {}
        self.task_state = {}
        self.peak_contact_force = 0.0
        self.peak_wrench = 0.0
        self.task.reset_hook(self, self.rng)
        self.home_pos = self.pos.copy()
        return {}

    @property
    def eef_pos(self):
        return self.pos.copy()

    @property
    def eef_quat(self):
        return np.array([0.0, 0.0, np.sin(self.yaw / 2), np.cos(self.yaw / 2)])

    @property
    def gripper_opening(self) -> Optional[float]:
        return float(self.opening)

    @property
    def sim_time(self) -> float:
        return self.t

    def body_pos(self, name: str):
        key = name.replace("_main", "")
        return self.objects[key]["pos"].copy()

    def body_quat_wxyz(self, name: str):
        return np.array([1.0, 0, 0, 0])

    def set_body_mass(self, name: str, mass: float):
        self.objects[name.replace("_main", "")]["mass"] = mass

    def noised(self, key, pos, yaw_deg=0.0):
        if key not in self._scene_offsets:
            dp = self.rng.normal(0, self.scene_noise_pos, 3) if self.scene_noise_pos > 0 else np.zeros(3)
            dp[2] = 0
            self._scene_offsets[key] = np.array([*dp, 0.0])
        o = self._scene_offsets[key]
        return np.asarray(pos) + o[:3], yaw_deg + o[3]

    # ------------------------------------------------------------------ physics
    def _floor(self) -> float:
        return self.table_top

    def _over_cube(self) -> bool:
        c = self.objects["cube"]
        return bool(np.all(np.abs(self.pos[:2] - c["pos"][:2]) < c["half"] + 0.005))

    def _reaction(self) -> np.ndarray:
        f = np.zeros(3)
        pen = getattr(self, "_pen", 0.0)
        if pen > 0:
            f[2] += 300.0 * pen
        c = self.objects["cube"]
        if c["grasped"] and c["pos"][2] - c["half"] > self.table_top + 1e-3:
            f[2] += -c["mass"] * G
        return f

    def step(self, dpos, drot=None, grip=None) -> WrenchSample:
        dpos = np.clip(np.asarray(dpos, dtype=float), -0.05, 0.05)   # lead of at most 5 cm per tick (like robosuite's output_max)
        self.target = self.pos + dpos
        floor = self._floor()
        new = self.target.copy()
        self._pen = 0.0
        if new[2] < floor:
            self._pen = floor - new[2]                     # depth of the target below the surface -> spring force k * pen (k = 300 N/m, 5 cm -> 15 N)
            new[2] = floor
        self.pos = new
        if drot is not None:
            self.yaw += float(np.clip(drot[2], -0.5, 0.5))
        if grip is not None:
            self.grip_cmd = float(grip)
        c = self.objects["cube"]
        if self.grip_cmd > 0:
            can_grasp = self._over_cube() and self.pos[2] < self.table_top + 0.035 and not c["grasped"]
            min_open = 2 * c["half"] if (can_grasp or c["grasped"]) else 0.0
            self.opening = max(min_open, self.opening - 0.01)
            if can_grasp and self.opening <= 2 * c["half"] + 1e-6:
                c["grasped"] = True
                c["offset"] = c["pos"] - self.pos
        else:
            self.opening = min(0.08, self.opening + 0.01)
            if c["grasped"] and self.opening > 2 * c["half"] + 0.01:
                c["grasped"] = False
                c["pos"][2] = self.table_top + c["half"]
        if c["grasped"]:
            c["pos"] = self.pos + c["offset"]
        self.t += self.dt
        f = self._reaction()
        w = WrenchSample(t=self.t, force=f, torque=np.zeros(3), force_world=f, torque_world=np.zeros(3),
                         net_force=f, net_torque=np.zeros(3), net_force_world=f, net_torque_world=np.zeros(3))
        self.peak_wrench = max(self.peak_wrench, w.fmag)
        if self._pen > 0:
            self.peak_contact_force = max(self.peak_contact_force, 300.0 * self._pen)
        return w

    def hold(self, n=1, grip=None):
        w = None
        for _ in range(n):
            w = self.step(np.zeros(3), None, grip)
        return w

    def wrench(self) -> WrenchSample:
        f = self._reaction()
        return WrenchSample(t=self.t, force=f, torque=np.zeros(3), force_world=f, torque_world=np.zeros(3),
                            net_force=f, net_torque=np.zeros(3), net_force_world=f, net_torque_world=np.zeros(3))

    def reaction_force(self, w: WrenchSample):
        return self.force_sign * w.net_force_world

    def gripper_contact_state(self):
        touching = []
        if getattr(self, "_pen", 0.0) > 0:
            touching.append("table")
        if self.objects["cube"]["grasped"]:
            touching.append("cube_main")
        return dict(touching=touching, total_force=float(np.linalg.norm(self._reaction())), in_contact=bool(touching))

    def object_contact_force(self, substrs):
        return 0.0

    def images(self):
        return {}

    def scene(self):
        return self.task.describe_scene(self)

    def success(self) -> bool:
        return bool(self.task.success(self))

    def close(self):
        pass
