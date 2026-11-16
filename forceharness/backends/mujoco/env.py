"""robosuite (MuJoCo) backend: wrist force/torque, contact ground truth, end-effector pose, camera frames and the
privileged scene.

``ForceSim`` implements the backend contract of ``forceharness/backends/base.py`` for the HiddenPhys tasks
(``forceharness/hiddenphys/tasks.py``); the force primitives (layer 2) run on it unchanged.

Design notes
- Force is read directly from MuJoCo's force / torque sensors (``force_ee`` / ``torque_ee`` in robosuite's gripper XML),
  not through observables, so robosuite 1.4 and 1.5 both work. Sensors are looked up by substring, and a missing sensor
  is an error, never silent zeros: a force channel that silently reads zero looks exactly like "no contact".
- The reading is the reaction between wrist and hand in the sensor frame and **includes the gripper's own weight**. After
  reset the arm rests for a few ticks to take a bias, and ``net = raw - bias``; the raw value is kept as well (ForceBench
  gives its policies the uncompensated raw value).
- Contact ground truth: iterate over ``d.contact`` with ``mj_contactForce`` to get the contact forces on the gripper and
  the task objects. Used only for evaluation (peak contact force, accuracy of contact-state judgements), **never shown to
  the model**.
- Action: OSC_POSE delta (6 dims + 1 gripper dim; 6 dims for Wipe, whose tool has no fingers). The stiffness kp is
  adjustable (low kp = compliant).
- Cameras: off-screen EGL rendering; robosuite images are upside down and are flipped back here.
- Noise: optional per-tick force noise (``force_noise_std``) and a per-episode residual bias after taring
  (``force_bias_std``) bring the clean simulated sensor closer to a real arm's force estimate; object poses given to the
  model carry a per-episode fixed offset (``scene_noise_pos``, ``scene_noise_yaw_deg``).
"""
from __future__ import annotations

import os
import sys
import dataclasses
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Off-screen rendering backend: EGL on Linux (headless GPUs), CGL on macOS; set MUJOCO_GL yourself to override (e.g. osmesa).
os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")

from forceharness.backends.base import WrenchSample  # noqa: E402  (re-exported for backwards compatibility)


def make_controller_config(kp: float = 150.0, output_max_pos: float = 0.05, output_max_rot: float = 0.5) -> Dict[str, Any]:
    """Return a robosuite controller config: OSC_POSE deltas, stiffness kp, critical damping. Only existing keys are
    changed, so robosuite 1.4 and 1.5 both work."""
    out_max = [output_max_pos] * 3 + [output_max_rot] * 3
    out_min = [-v for v in out_max]

    def _patch(c: Dict[str, Any]) -> None:
        for k, v in dict(kp=kp, damping_ratio=1.0, impedance_mode="fixed", output_max=out_max, output_min=out_min,
                         input_max=1.0, input_min=-1.0).items():
            if k in c:
                c[k] = v

    try:  # robosuite >= 1.5
        from robosuite.controllers import load_composite_controller_config

        cfg = load_composite_controller_config(controller="BASIC", robot="Panda")
        for part, pc in cfg.get("body_parts", {}).items():
            if isinstance(pc, dict) and str(pc.get("type", "")).upper().startswith("OSC"):
                _patch(pc)
        return cfg
    except ImportError:  # robosuite 1.4
        from robosuite import load_controller_config

        cfg = load_controller_config(default_controller="OSC_POSE")
        _patch(cfg)
        return cfg


class ForceSim:
    """Simulation session of one task. All coordinates are in the world frame (metres); quaternions are xyzw (robosuite
    convention)."""

    def __init__(
        self,
        task,
        seed: int = 0,
        kp: float = 150.0,
        control_freq: int = 20,
        render: bool = True,
        camera_names: Sequence[str] = ("agentview", "robot0_eye_in_hand"),
        image_size: int = 256,
        bias_steps: int = 10,
        gpu_id: Optional[int] = None,
        scene_noise_pos: float = 0.0,
        scene_noise_yaw_deg: float = 0.0,
        force_sign: float = -1.0,
        force_noise_std: float = 0.0,
        force_bias_std: float = 0.0,
        grip_force_scale: float = 1.0,
    ):
        import mujoco
        import robosuite as suite

        self.mujoco = mujoco
        self.task = task
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.kp = kp
        self.control_freq = control_freq
        self.dt = 1.0 / control_freq
        self.render_enabled = render
        self.camera_names = list(camera_names)
        self.image_size = image_size
        self.bias_steps = bias_steps
        # The model's "perception" is the ground-truth pose plus a fixed offset per object and episode (like the 3-10 mm
        # error of real-robot vision).
        self.scene_noise_pos = scene_noise_pos
        self.scene_noise_yaw_deg = scene_noise_yaw_deg
        self._scene_offsets: Dict[str, np.ndarray] = {}
        # Sign convention: reaction = force_sign * net_force_world, so that pressing on the table gives a reaction along +z.
        # robosuite's force_ee reads the force of the arm on the gripper (+5.1 N at rest, holding the gripper up), and the net
        # value turns negative when pressing on the table, hence the default -1.
        self.force_sign = force_sign
        # A force signal closer to a real arm: a real external-force estimate has a bias that differs per episode and noise on
        # every tick, while the simulated sensor is very clean; these two terms make simulated results transfer better.
        self.force_noise_std = force_noise_std
        self.force_bias_std = force_bias_std
        self._force_bias_episode = np.zeros(3)
        self._torque_bias_episode = np.zeros(3)
        self.task_state: Dict[str, Any] = {}
        self.grip_cmd = -1.0          # the gripper command is sticky: without a new one the last one holds; open after reset
        self.output_max_pos = 0.05
        self.output_max_rot = 0.5

        kwargs = dict(
            env_name=task.env_name,
            robots="Panda",
            controller_configs=make_controller_config(kp, self.output_max_pos, self.output_max_rot),
            has_renderer=False,
            has_offscreen_renderer=render,
            use_camera_obs=render,
            use_object_obs=True,
            control_freq=control_freq,
            horizon=10**9,
            ignore_done=True,
            hard_reset=False,
            reward_shaping=False,
        )
        if render:
            kwargs.update(camera_names=self.camera_names, camera_heights=image_size, camera_widths=image_size)
            if gpu_id is not None:
                kwargs["render_gpu_device_id"] = gpu_id
        kwargs.update(task.env_kwargs)
        np.random.seed(seed)              # robosuite's object placement / table-height randomisation uses the global np.random; fix it by seed
        self._n_resets = 0
        self.env = suite.make(**kwargs)
        self.m = self.env.sim.model._model
        self.d = self.env.sim.data._data
        self._resolve_sensors()
        if grip_force_scale != 1.0:           # a Franka Hand holds up to 70 N continuously; robosuite's default forcerange is small and thin parts turn between the fingers
            for i in range(self.m.nu):
                name = self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_ACTUATOR, i) or ""
                if "gripper" in name:
                    self.m.actuator_forcerange[i] *= grip_force_scale
        self.action_dim = int(self.env.action_dim)
        self.obs: Dict[str, Any] = {}
        self.bias_force = np.zeros(3)
        self.bias_torque = np.zeros(3)
        self.n_steps = 0
        self.peak_contact_force = 0.0       # peak contact force on the task objects (for evaluation)
        self.peak_wrench = 0.0              # peak wrist force |net F|
        self.robot_body_prefixes = ("robot0", "gripper0")
        self.reset()

    # ------------------------------------------------------------------ sensors
    def _find_sensor(self, substr: str):
        mj = self.mujoco
        for i in range(self.m.nsensor):
            name = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_SENSOR, i)
            if name and substr in name:
                return i, name
        raise RuntimeError(f"no MuJoCo sensor whose name contains '{substr}'; sensors: {self.sensor_names()}")

    def sensor_names(self) -> List[str]:
        mj = self.mujoco
        return [mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_SENSOR, i) for i in range(self.m.nsensor)]

    def _resolve_sensors(self) -> None:
        fi, self.force_sensor_name = self._find_sensor("force_ee")
        ti, self.torque_sensor_name = self._find_sensor("torque_ee")
        self._f_adr = int(self.m.sensor_adr[fi])
        self._t_adr = int(self.m.sensor_adr[ti])
        self._ft_site = int(self.m.sensor_objid[fi])
        self.ft_site_name = self.mujoco.mj_id2name(self.m, self.mujoco.mjtObj.mjOBJ_SITE, self._ft_site)

    def raw_wrench(self):
        f = self.d.sensordata[self._f_adr:self._f_adr + 3].copy()
        t = self.d.sensordata[self._t_adr:self._t_adr + 3].copy()
        R = self.d.site_xmat[self._ft_site].reshape(3, 3)
        return f, t, R

    def wrench(self) -> WrenchSample:
        f, t, R = self.raw_wrench()
        if self.force_noise_std > 0:
            f = f + self.rng.normal(0.0, self.force_noise_std, 3)
            t = t + self.rng.normal(0.0, self.force_noise_std * 0.1, 3)
        f = f + self._force_bias_episode
        t = t + self._torque_bias_episode
        nf, nt = f - self.bias_force, t - self.bias_torque
        return WrenchSample(
            t=float(self.d.time), force=f, torque=t, force_world=R @ f, torque_world=R @ t,
            net_force=nf, net_torque=nt, net_force_world=R @ nf, net_torque_world=R @ nt,
        )

    # ------------------------------------------------------------------ state
    @property
    def eef_pos(self) -> np.ndarray:
        return np.asarray(self.obs["robot0_eef_pos"], dtype=float).copy()

    @property
    def eef_quat(self) -> np.ndarray:
        return np.asarray(self.obs["robot0_eef_quat"], dtype=float).copy()

    @property
    def gripper_qpos(self) -> Optional[np.ndarray]:
        q = self.obs.get("robot0_gripper_qpos")
        return None if q is None else np.asarray(q, dtype=float).copy()

    @property
    def gripper_opening(self) -> Optional[float]:
        """Proxy for the finger gap (Panda: qpos[0] - qpos[1], about 0.08 open and 0 closed). None for a gripper without
        fingers."""
        q = self.gripper_qpos
        if q is None or len(q) < 2:
            return None
        return float(q[0] - q[1])

    @property
    def sim_time(self) -> float:
        return float(self.d.time)

    def body_pos(self, name: str) -> np.ndarray:
        bid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(name)
        return self.d.xpos[bid].copy()

    def body_quat_wxyz(self, name: str) -> np.ndarray:
        bid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, name)
        return self.d.xquat[bid].copy()

    def site_pos(self, name: str) -> np.ndarray:
        sid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_SITE, name)
        if sid < 0:
            raise KeyError(name)
        return self.d.site_xpos[sid].copy()

    def find_names(self, kind: str, substr: str) -> List[str]:
        mj = self.mujoco
        objtype = {"body": mj.mjtObj.mjOBJ_BODY, "site": mj.mjtObj.mjOBJ_SITE, "joint": mj.mjtObj.mjOBJ_JOINT,
                   "geom": mj.mjtObj.mjOBJ_GEOM}[kind]
        n = {"body": self.m.nbody, "site": self.m.nsite, "joint": self.m.njnt, "geom": self.m.ngeom}[kind]
        out = []
        for i in range(n):
            name = mj.mj_id2name(self.m, objtype, i)
            if name and substr.lower() in name.lower():
                out.append(name)
        return out

    def set_body_mass(self, name: str, mass: float) -> None:
        """Set the mass of a body (inertia scaled proportionally), used to randomise physical properties such as in
        weight_sort."""
        bid = self.mujoco.mj_name2id(self.m, self.mujoco.mjtObj.mjOBJ_BODY, name)
        old = float(self.m.body_mass[bid])
        if old > 0:
            self.m.body_inertia[bid] *= mass / old
        self.m.body_mass[bid] = mass
        # mj_setConst cannot be used (it resets d.qpos to the XML qpos0, sending the robot back to its zero pose); update the
        # subtree masses by hand
        delta = mass - old
        b = bid
        while b >= 0:
            self.m.body_subtreemass[b] += delta
            if b == 0:
                break
            b = int(self.m.body_parentid[b])

    # ------------------------------------------------------------------ contact ground truth
    def contacts(self) -> List[Dict[str, Any]]:
        mj = self.mujoco
        out = []
        f6 = np.zeros(6)
        for i in range(self.d.ncon):
            c = self.d.contact[i]
            mj.mj_contactForce(self.m, self.d, i, f6)
            b1 = int(self.m.geom_bodyid[c.geom1])
            b2 = int(self.m.geom_bodyid[c.geom2])
            n1 = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, b1) or f"body{b1}"
            n2 = mj.mj_id2name(self.m, mj.mjtObj.mjOBJ_BODY, b2) or f"body{b2}"
            out.append(dict(body1=n1, body2=n2, fn=float(f6[0]), f=float(np.linalg.norm(f6[:3])), pos=c.pos.copy()))
        return out

    def gripper_contact_state(self) -> Dict[str, Any]:
        """Ground truth: what the gripper / arm touches now and the total contact force."""
        touching, total = set(), 0.0
        for c in self.contacts():
            a, b = c["body1"], c["body2"]
            ra, rb = a.startswith(self.robot_body_prefixes), b.startswith(self.robot_body_prefixes)
            if ra and not rb:
                touching.add(b); total += c["f"]
            elif rb and not ra:
                touching.add(a); total += c["f"]
        return dict(touching=sorted(touching), total_force=total, in_contact=bool(touching))

    def object_contact_force(self, body_substrs: Sequence[str]) -> float:
        tot = 0.0
        for c in self.contacts():
            if any(s in c["body1"] or s in c["body2"] for s in body_substrs):
                tot += c["f"]
        return tot

    # ------------------------------------------------------------------ stepping
    def reset(self) -> Dict[str, Any]:
        np.random.seed(self.seed * 1000 + self._n_resets)
        self._n_resets += 1
        self.obs = self.env.reset()
        self._scene_offsets = {}
        self.task_state = {}
        # one uncompensated bias per episode (taring at rest after reset removes only part of it: this models the drift left
        # after taring)
        if self.force_bias_std > 0:
            self._force_bias_episode = self.rng.normal(0.0, self.force_bias_std, 3)
            self._torque_bias_episode = self.rng.normal(0.0, self.force_bias_std * 0.1, 3)
        self.grip_cmd = -1.0                 # open by default; a task's reset_hook may change it (e.g. the peg starts in the hand -> +1)
        self.task.reset_hook(self, self.rng)
        self.mujoco.mj_forward(self.m, self.d)
        self.n_steps = 0
        self.peak_contact_force = 0.0
        self.peak_wrench = 0.0
        # rest for a few ticks to take the bias (gripper weight + controller steady state + weight of anything held)
        fs, ts = [], []
        for _ in range(self.bias_steps):
            a = np.zeros(self.action_dim)
            if self.action_dim > 6:
                a[6] = self.grip_cmd
            self.obs = self.env.step(a)[0]
            f, t, _ = self.raw_wrench()
            fs.append(f); ts.append(t)
        self.bias_force = np.mean(fs, axis=0)
        self.bias_torque = np.mean(ts, axis=0)
        self.home_pos = self.eef_pos
        self.home_quat = self.eef_quat
        if self.force_bias_std > 0:          # taring absorbed the earlier bias; draw a new drift to model the residual after taring
            self._force_bias_episode = self.rng.normal(0.0, self.force_bias_std, 3)
            self._torque_bias_episode = self.rng.normal(0.0, self.force_bias_std * 0.1, 3)
        return self.obs

    def set_held_weld(self, active: bool) -> None:
        """Activate / deactivate the "held in the hand" weld constraints (names starting with held_weld). On activation the
        current relative pose is written into eq_data, so that the part is not pulled elsewhere."""
        mj, m, d = self.mujoco, self.m, self.d
        for i in range(m.neq):
            name = mj.mj_id2name(m, mj.mjtObj.mjOBJ_EQUALITY, i) or ""
            if not name.startswith("held_weld"):
                continue
            if active:
                b1, b2 = int(m.eq_obj1id[i]), int(m.eq_obj2id[i])
                p1, q1 = d.xpos[b1].copy(), d.xquat[b1].copy()
                p2, q2 = d.xpos[b2].copy(), d.xquat[b2].copy()
                q1n = np.zeros(4); mj.mju_negQuat(q1n, q1)
                p_rel = np.zeros(3); mj.mju_rotVecQuat(p_rel, p2 - p1, q1n)
                q_rel = np.zeros(4); mj.mju_mulQuat(q_rel, q1n, q2)
                m.eq_data[i, 0:3] = 0.0
                m.eq_data[i, 3:6] = p_rel
                m.eq_data[i, 6:10] = q_rel
            if hasattr(d, "eq_active"):
                d.eq_active[i] = 1 if active else 0
            else:
                m.eq_active[i] = 1 if active else 0
        mj.mj_forward(m, d)

    def rezero_bias(self, steps: int = 10) -> None:
        """Rest for ``steps`` ticks in the current pose and re-zero. The reset zero is taken at the initial pitch of about
        9 deg; once the primitives level the tool, the lever arm of the gripper's weight changes and leaves about 0.1 N*m,
        a systematic error of about 1.3 cm in centre-of-mass estimates, so Primitives levels the tool first and then calls
        this. The random residual drift after taring is kept unchanged."""
        fs, ts = [], []
        for _ in range(steps):
            a = np.zeros(self.action_dim)
            if self.action_dim > 6:
                a[6] = self.grip_cmd
            self.obs = self.env.step(a)[0]
            f, t, _ = self.raw_wrench()
            fs.append(f); ts.append(t)
        self.bias_force = np.mean(fs, axis=0)
        self.bias_torque = np.mean(ts, axis=0)

    def step(self, dpos: np.ndarray, drot: Optional[np.ndarray] = None, grip: Optional[float] = None) -> WrenchSample:
        """One control tick: dpos (metres, world-frame delta, clipped at output_max), drot (axis-angle delta in radians),
        grip (-1 open ... +1 closed)."""
        a = np.zeros(self.action_dim)
        a[:3] = np.clip(np.asarray(dpos, dtype=float) / self.output_max_pos, -1, 1)
        if drot is not None:
            a[3:6] = np.clip(np.asarray(drot, dtype=float) / self.output_max_rot, -1, 1)
        if grip is not None:
            self.grip_cmd = float(np.clip(grip, -1, 1))
        if self.action_dim > 6:
            a[6] = self.grip_cmd
        self.obs = self.env.step(a)[0]
        self.n_steps += 1
        w = self.wrench()
        self.peak_wrench = max(self.peak_wrench, w.fmag)
        hook = getattr(self.task, "step_hook", None)
        if hook is not None:
            hook(self, w)
        if self.task.objects:
            self.peak_contact_force = max(self.peak_contact_force, self.object_contact_force(self.task.object_body_substrs()))
        return w

    def hold(self, n: int = 1, grip: Optional[float] = None) -> WrenchSample:
        w = None
        for _ in range(n):
            w = self.step(np.zeros(3), None, grip)
        return w

    # ------------------------------------------------------------------ stiffness
    def _osc(self):
        r = self.env.robots[0]
        cc = getattr(r, "composite_controller", None)
        if cc is not None and hasattr(cc, "part_controllers"):      # robosuite 1.5
            for part, c in cc.part_controllers.items():
                if hasattr(c, "kp") and hasattr(c, "kd"):
                    return c
        return getattr(r, "controller", None)                         # robosuite 1.4

    def set_kp(self, kp) -> None:
        """Change the OSC stiffness at run time (a scalar or 6 values: xyz translation + three rotations); kd follows for
        critical damping. Compliance goes only on the axis that seeks contact or regulates force and the other axes keep
        full stiffness: at low stiffness the null-space term would drag the end effector 1-2 cm sideways."""
        c = self._osc()
        if c is None:
            return
        kp_arr = np.ones(6) * float(kp) if np.isscalar(kp) else np.asarray(kp, dtype=float).reshape(6)
        c.kp = kp_arr
        c.kd = 2.0 * np.sqrt(kp_arr) * 1.0
        self.kp_vec = kp_arr

    @property
    def wrist_joint_rad(self) -> Optional[float]:
        q = self.obs.get("robot0_joint_pos")
        return None if q is None or len(q) < 7 else float(q[6])

    # ------------------------------------------------------------------ cameras and scene
    def images(self) -> Dict[str, np.ndarray]:
        if not self.render_enabled:
            return {}
        out = {}
        for cam in self.camera_names:
            img = self.obs.get(f"{cam}_image")
            if img is not None:
                out[cam] = np.asarray(img)[::-1].copy()
        return out

    def noised(self, key: str, pos: np.ndarray, yaw_deg: float = 0.0):
        """Add this episode's fixed offset to a ground-truth pose (same key -> same offset, so repeated scene() calls show a
        consistent "perception")."""
        if key not in self._scene_offsets:
            dp = self.rng.normal(0.0, self.scene_noise_pos, size=3) if self.scene_noise_pos > 0 else np.zeros(3)
            dp[2] = 0.0  # height uncertainty is up to each task (e.g. table-height randomisation); only x and y are perturbed here
            dyaw = self.rng.normal(0.0, self.scene_noise_yaw_deg) if self.scene_noise_yaw_deg > 0 else 0.0
            self._scene_offsets[key] = np.array([*dp, dyaw])
        o = self._scene_offsets[key]
        return (np.asarray(pos, dtype=float) + o[:3]), float(yaw_deg + o[3])

    def reaction_force(self, w: WrenchSample) -> np.ndarray:
        """Net force of the environment on the gripper (world frame), with the sign convention applied."""
        return self.force_sign * w.net_force_world

    def scene(self) -> List[Dict[str, Any]]:
        """Privileged scene description (ground truth plus noise: the benchmark isolates force from visual perception)."""
        return self.task.describe_scene(self)

    def success(self) -> bool:
        return bool(self.task.success(self))

    def close(self) -> None:
        try:
            self.env.close()
        except Exception:
            pass
