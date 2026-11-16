"""Demonstration recorder: records every control tick of an episode as a demonstration for the learned policies
(Diffusion Policy, pi0.5, ForceVLA) of the HiddenPhys reference rows.

It hooks robosuite's ``env.step``, which ForceSim.step / hold / rezero_bias all go through, so the recorded actions are
exactly the normalised OSC deltas the environment received ([-1, 1]; times output_max_pos metres / output_max_rot
radians; on tasks with a gripper the 7th dimension is the gripper, -1 open ... +1 closed). There is one more observation
than actions: frame 0 is recorded when recording starts, then each tick records the action and then the observation after
that tick, so observation i is what was seen before action i. Replaying the actions in order from the starting state
(together with the two side channels below) reproduces the trajectory; ``scripts/hiddenphys/replay_demos.py`` checks this.

Two things besides the action change behaviour and are recorded on every tick:
  - the OSC stiffness kp (primitives lower the stiffness along the contact direction from 150 to 40 when seeking contact
    or pushing, and set it per axis when inserting); without it, a policy under the same controller cannot learn the
    expert's contact behaviour;
  - whether the hand weld is active (the peg starts welded in the hand, and only the release primitive removes the weld).
Each tick also records whether it came through ForceSim.step (if so, the task's step_hook runs: latch logic, damage).

Force: the recorder never calls sim.wrench(), which draws noise random numbers and would change the rest of the episode's
noise sequence. It records the noise-free raw sensor reading and the sensor orientation R, plus the episode constants (the
zero point tared at reset and the residual per-episode bias after taring); conversion to training data adds noise with the
same model as ForceSim:
  F_world = force_sign * R (f_raw + bias_ep + N(0, sigma) - b0)
Scene: the recorder never calls sim.scene() itself (the first call draws scene-noise random numbers); it captures the scene
the agent saw when the agent called it (noisy poses, the same offsets for the same seed).
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

_CUR_PRIMITIVE = {"name": ""}          # updated by install_primitive_tracker: the primitive running at each tick (analysis only, not a training input)


def install_primitive_tracker() -> None:
    """Wrap Primitives.call in this process to record the name of the current primitive. Each episode runs in its own
    process (as in the runner), so the class-level patch affects only this process."""
    from forceharness.primitives.library import Primitives

    if getattr(Primitives.call, "_fh_tracked", False):
        return
    orig = Primitives.call

    def call(self, name, **kw):
        prev = _CUR_PRIMITIVE["name"]
        _CUR_PRIMITIVE["name"] = name
        try:
            return orig(self, name, **kw)
        finally:
            _CUR_PRIMITIVE["name"] = prev

    call._fh_tracked = True
    Primitives.call = call


class DemoRecorder:
    def __init__(self, sim, cams: Sequence[str] = ("agentview", "robot0_eye_in_hand"), jpeg_quality: int = 90):
        import cv2

        self._cv2 = cv2
        self.sim = sim
        self.cams = list(cams) if sim.render_enabled else []
        self.q = int(jpeg_quality)
        self.active = False
        self._in_fs_step = False
        self.weld = False
        self.scene0: Optional[List[Dict[str, Any]]] = None
        self._reset_buffers()
        # wrap: env.step (recording point), ForceSim.step (marks whether step_hook runs), reset (starts recording),
        # set_held_weld (weld side channel), scene (captures the scene the agent sees)
        self._env_step = sim.env.step
        self._fs_step = sim.step
        self._reset = sim.reset
        self._set_weld = sim.set_held_weld
        self._scene = sim.scene
        sim.env.step = self._wrap_env_step
        sim.step = self._wrap_fs_step
        sim.reset = self._wrap_reset
        sim.set_held_weld = self._wrap_set_weld
        sim.scene = self._wrap_scene

    def _reset_buffers(self) -> None:
        self.rows: Dict[str, List[Any]] = {k: [] for k in ("t", "action", "eef_pos", "eef_quat", "gripper_qpos", "joint_pos",
                                                           "force_raw", "torque_raw", "R_ft", "kp", "weld", "via_step", "prim")}
        self.jpegs: Dict[str, List[bytes]] = {c: [] for c in self.cams}
        self.prim_names: List[str] = []

    # ------------------------------------------------------------------ start recording
    def start(self) -> None:
        """Start recording from the current state. ForceSim.__init__ already ends with a reset, so call this after building
        the sim and before building the Episode / Primitives: on replay, a new ForceSim with the same seed returns to the
        same starting point."""
        s = self.sim
        self._reset_buffers()
        self.scene0 = None
        self.b0_force = np.asarray(s.bias_force, float).copy()
        self.b0_torque = np.asarray(s.bias_torque, float).copy()
        self.bias_ep_force = np.asarray(s._force_bias_episode, float).copy()
        self.bias_ep_torque = np.asarray(s._torque_bias_episode, float).copy()
        self.home_pos = np.asarray(s.home_pos, float).copy()
        self.weld = self._weld_active()
        self._record_obs(s.obs)              # frame 0: the observation before the first action (observation i pairs with action i)
        self.active = True

    # ------------------------------------------------------------------ wrappers
    def _wrap_reset(self, *a, **k):
        self.active = False
        out = self._reset(*a, **k)
        self.start()
        return out

    def _wrap_fs_step(self, *a, **k):
        self._in_fs_step = True
        try:
            return self._fs_step(*a, **k)
        finally:
            self._in_fs_step = False

    def _wrap_set_weld(self, active: bool):
        self._set_weld(active)
        self.weld = bool(active) and self._weld_present()

    def _wrap_scene(self, *a, **k):
        sc = self._scene(*a, **k)
        if self.active and self.scene0 is None:
            self.scene0 = json.loads(json.dumps(sc, default=_jsonable))
        return sc

    def _wrap_env_step(self, action):
        out = self._env_step(action)
        if self.active:
            self._record_step(np.asarray(action, dtype=float))
            self._record_obs(out[0])
        return out

    # ------------------------------------------------------------------ record one tick
    def _weld_present(self) -> bool:
        mj, m = self.sim.mujoco, self.sim.m
        return any((mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i) or "").startswith("held_weld") for i in range(m.neq))

    def _weld_active(self) -> bool:
        mj, m, d = self.sim.mujoco, self.sim.m, self.sim.d
        act = getattr(d, "eq_active", None)
        if act is None:
            act = m.eq_active
        return any((mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i) or "").startswith("held_weld") and bool(act[i]) for i in range(m.neq))

    def _record_step(self, a: np.ndarray) -> None:
        """The action sent on this tick and its side channels (stiffness, weld, whether it came through ForceSim.step,
        current primitive); one entry per action."""
        s, r = self.sim, self.rows
        a7 = np.full(7, np.nan)
        a7[:len(a)] = a
        r["action"].append(a7)
        kp = getattr(s, "kp_vec", None)
        r["kp"].append(np.full(6, float(s.kp)) if kp is None else np.asarray(kp, float).reshape(6))
        r["weld"].append(bool(self.weld))
        r["via_step"].append(bool(self._in_fs_step))
        name = _CUR_PRIMITIVE["name"]
        if name not in self.prim_names:
            self.prim_names.append(name)
        r["prim"].append(self.prim_names.index(name))

    def _record_obs(self, obs: Dict[str, Any]) -> None:
        """One observation frame; there is one more frame than actions (a frame is recorded when recording starts)."""
        s, r = self.sim, self.rows
        f, t, R = s.raw_wrench()
        grip = obs.get("robot0_gripper_qpos")
        r["t"].append(float(s.d.time))
        r["eef_pos"].append(np.asarray(obs["robot0_eef_pos"], float))
        r["eef_quat"].append(np.asarray(obs["robot0_eef_quat"], float))
        g = np.zeros(2) if grip is None or np.asarray(grip).size < 2 else np.asarray(grip, float)[:2]     # the wiping tool has no fingers: robosuite returns an empty array
        r["gripper_qpos"].append(g)
        r["joint_pos"].append(np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), float)[:7])
        r["force_raw"].append(np.asarray(f, float))
        r["torque_raw"].append(np.asarray(t, float))
        r["R_ft"].append(np.asarray(R, float).reshape(9))
        for c in self.cams:
            img = obs[f"{c}_image"]                         # off-screen rendering is upside down (OpenGL convention); stored as is, flipped at conversion
            ok, buf = self._cv2.imencode(".jpg", np.ascontiguousarray(img[..., ::-1]), [self._cv2.IMWRITE_JPEG_QUALITY, self.q])
            assert ok
            self.jpegs[c].append(buf.tobytes())

    # ------------------------------------------------------------------ finish
    def stop(self) -> None:
        self.active = False

    def arrays(self) -> Dict[str, np.ndarray]:
        r = self.rows
        out: Dict[str, np.ndarray] = {
            "t": np.asarray(r["t"], np.float64),
            "action": np.asarray(r["action"], np.float64).reshape(-1, 7),     # float64: float32 rounding grows in sliding contact with friction (a peg replay ended 19 mm off)
            "eef_pos": np.asarray(r["eef_pos"], np.float32).reshape(-1, 3),
            "eef_quat": np.asarray(r["eef_quat"], np.float32).reshape(-1, 4),
            "gripper_qpos": np.asarray(r["gripper_qpos"], np.float32).reshape(-1, 2),
            "joint_pos": np.asarray(r["joint_pos"], np.float32).reshape(-1, 7),
            "force_raw": np.asarray(r["force_raw"], np.float32).reshape(-1, 3),
            "torque_raw": np.asarray(r["torque_raw"], np.float32).reshape(-1, 3),
            "R_ft": np.asarray(r["R_ft"], np.float32).reshape(-1, 9),
            "kp": np.asarray(r["kp"], np.float64).reshape(-1, 6),
            "weld": np.asarray(r["weld"], bool),
            "via_step": np.asarray(r["via_step"], bool),
            "prim": np.asarray(r["prim"], np.int16),
            "b0_force": self.b0_force, "b0_torque": self.b0_torque,
            "bias_ep_force": self.bias_ep_force, "bias_ep_torque": self.bias_ep_torque,
            "home_pos": self.home_pos,
        }
        for c in self.cams:
            lens = np.fromiter((len(b) for b in self.jpegs[c]), dtype=np.int64, count=len(self.jpegs[c]))
            out[f"img_{c}_jpeg"] = np.frombuffer(b"".join(self.jpegs[c]), dtype=np.uint8)
            out[f"img_{c}_off"] = np.concatenate([[0], np.cumsum(lens)]).astype(np.int64)
        return out

    def save(self, path: str, meta: Dict[str, Any]) -> int:
        """Save as one npz (uncompressed: the JPEG frames are already compressed). meta holds task, seed, success, ground
        truth, cameras, noise parameters, scene0 and the table of primitive names. Returns the file size in bytes."""
        arr = self.arrays()
        m = dict(meta, cams=self.cams, prim_names=self.prim_names, scene0=self.scene0, n_steps=int(len(arr["action"])),
                 output_max_pos=float(self.sim.output_max_pos), output_max_rot=float(self.sim.output_max_rot),
                 force_sign=float(self.sim.force_sign), force_noise_std=float(self.sim.force_noise_std),
                 force_bias_std=float(self.sim.force_bias_std), control_freq=int(self.sim.control_freq),
                 image_size=int(self.sim.image_size), images_upside_down=True, jpeg_channel_order="BGR (cv2)")
        arr["meta_json"] = np.frombuffer(json.dumps(m, ensure_ascii=False, default=_jsonable).encode("utf-8"), dtype=np.uint8)
        np.savez(path, **arr)
        import os

        return os.path.getsize(path if path.endswith(".npz") else path + ".npz")


from forceharness.data.demos import decode_frame, load_demo  # noqa: E402,F401  (loading lives in forceharness.data, which does not need robosuite, so training environments can use it)


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
