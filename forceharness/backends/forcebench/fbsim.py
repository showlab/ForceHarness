"""FBSim: the ForceBench backend adapter. On top of the tick-by-tick observations of ForceBench's evaluator it gives the
primitive layer the same interface as the MuJoCo backend's ForceSim (`backends/mujoco/env.py`).

The primitives, event detection, physics translation layer, force injection and agent loop are unchanged; only the arm differs:
- Observation: ForceBench's `policy_state` = midpoint between the fingertips (world frame) + XYZ Euler angles (Isaac Lab
  convention R = Rz*Ry*Rx) + gripper value 2*opening-1; the wrist wrench `force_sensor_joint_raw` is the 6-D parent-to-child
  reaction of the panda_hand joint (no gravity compensation, no filtering).
- Action: absolute world-frame TCP target (xyz + rpy) + gripper (+1 open / -1 closed) at 15 Hz. ForceBench's controller is
  DLS IK + quintic joint interpolation + joint PD (stiffness 80 N*m/rad, damping 12; `forcebench/control.py`). It is softer
  than the OSC controller of the MuJoCo backend, and its stiffness cannot be changed at run time.
- The OSC position increment `dpos` becomes the absolute target `x_meas + g*dpos`, where g = desired stiffness / measured
  ForceBench stiffness (`stiffness_gain`, per axis, looked up by the value given to `set_kp`), so that the primitives' design
  rule "lead x stiffness = force" still holds on this arm. g defaults to 1 (no scaling); measure the arm before changing it.
- Perception: in privileged mode the evaluator copy (`scripts/forcebench/eval_forcebench.py --privileged_geometry`) adds the
  true object geometry under "privileged", and noise that is fixed per episode by the seed is added here, as ForceSim.noised
  does. In RGB-only mode the request carries the camera calibration instead (see `pixel_ray` and `locate_on_plane`).
Success is decided by ForceBench's own criteria (the evaluator writes summary.json); success() here always returns False.
"""
from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
from scipy.spatial.transform import Rotation as R

from forceharness.backends.base import WrenchSample
from forceharness.backends.forcebench.session import Session

FB_DT = 1.0 / 15.0
FINGER_MAX_OPEN_M = 0.08          # ForceBench: opening = mean of the two finger joints / 0.04; fully open, the fingers are about 8 cm apart


def rpy_to_quat_xyzw(rpy: Sequence[float]) -> np.ndarray:
    """Convention of Isaac Lab's `quat_from_euler_xyz(roll, pitch, yaw)`: R = Rz(yaw)*Ry(pitch)*Rx(roll) (extrinsic XYZ)."""
    return R.from_euler("xyz", np.asarray(rpy, dtype=float)).as_quat()


def quat_xyzw_to_rpy(q: Sequence[float], ref_rpy: Optional[Sequence[float]] = None) -> np.ndarray:
    rpy = R.from_quat(np.asarray(q, dtype=float)).as_euler("xyz")
    if ref_rpy is not None:          # unwrap around the current angles, as ForceBench's canonical_action does, to avoid jumps at +-pi
        ref = np.asarray(ref_rpy, dtype=float)
        rpy = ref + (rpy - ref + np.pi) % (2 * np.pi) - np.pi
    return rpy


@dataclasses.dataclass
class FBTask:
    """Harness-side description of a ForceBench task (the counterpart of the MuJoCo TaskSpec, with only the fields that the
    agent loop and the primitives use)."""

    name: str
    instruction: str
    describe_scene: Callable[["FBSim"], List[Dict[str, Any]]]
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)
    has_gripper: bool = True
    truth_fn: Optional[Callable] = None
    pregrasp_hook: Optional[Callable] = None
    objects: List[Dict[str, str]] = dataclasses.field(default_factory=list)


class FBSim:
    def __init__(
        self,
        session: Session,
        task: FBTask,
        seed: int,
        kp: float = 150.0,
        force_sign: float = -1.0,
        wrench_frame: str = "hand",          # "hand": readings in the hand (TCP) frame, rotated to world with the measured attitude; "world": already world
        stiffness_gain: Optional[Dict[str, float]] = None,   # {"150": g_free, "40": g_contact}: MuJoCo OSC kp -> target scaling factor
        max_gain: float = 2.5,
        scene_noise_pos: float = 0.004,
        scene_noise_yaw_deg: float = 2.0,
        force_noise_std: float = 0.0,
        force_bias_std: float = 0.0,
        max_steps: int = 1500,
    ):
        self.session = session
        self.task = task
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.kp = float(kp)
        self.kp_vec = np.ones(6) * self.kp
        self.control_freq = 15
        self.dt = FB_DT
        self.action_dim = 7
        self.force_sign = float(force_sign)
        self.wrench_frame = wrench_frame
        self.stiffness_gain = {float(k): float(v) for k, v in (stiffness_gain or {}).items()}
        self.max_gain = float(max_gain)
        self.scene_noise_pos = scene_noise_pos
        self.scene_noise_yaw_deg = scene_noise_yaw_deg
        self.force_noise_std = force_noise_std
        self.force_bias_std = force_bias_std
        self._force_bias_episode = self.rng.normal(0.0, force_bias_std, 3) if force_bias_std > 0 else np.zeros(3)
        self._torque_bias_episode = self.rng.normal(0.0, force_bias_std * 0.1, 3) if force_bias_std > 0 else np.zeros(3)
        self.max_steps = int(max_steps)
        self.task_state: Dict[str, Any] = {}
        self._scene_offsets: Dict[str, np.ndarray] = {}
        self.grip_cmd = -1.0
        self.n_steps = 0
        self.peak_wrench = 0.0
        self.peak_contact_force = 0.0
        self.bias_force = np.zeros(3)
        self.bias_torque = np.zeros(3)
        self.trace: List[List[float]] = []       # per tick: step, target xyz rpy gripper, measured xyz rpy gripper, raw wrench (6), sag compensation z
        # Load-sag compensation. The ForceBench arm has no integral term and does not compensate the weight of what it holds:
        # a 0.18 kg box at x = 0.68 leaves the hand about 14 mm below its target, more than the 12 mm lead cap of guarded moves,
        # so the lead clamp pulls the target down every tick (a ratchet) and a carried box sinks all the way (to about 7 cm).
        # The z error is integrated only while the reading looks like a hanging load (pointing down, small horizontal part) and
        # the gripper is closed; after contact, putting down or releasing, the compensation decays every tick. Contact forces
        # keep their meaning.
        self.sag_comp = 0.0
        self.sag_ki = 0.1
        self.sag_max = 0.04
        self.sag_tau = 0.2                       # free-space tracking time constant (a 2 cm step reaches 63% in 2-3 ticks); the lag v*tau is removed before integrating
        self._prev_req_z: Optional[float] = None
        self.priv_trace: List[List[float]] = []  # diagnostics: true position + rpy of the first scene entity per tick (logs only, never shown to the model)
        self._set_obs(session.first_obs)
        self.home_pos = self.eef_pos
        self.home_quat = self.eef_quat

    # ---------------------------------------------------------------- observation
    def _set_obs(self, req: Dict[str, Any]) -> None:
        self.req = req
        o = req["observation"]
        self.state = np.asarray(o["state"], dtype=float).reshape(7)
        self.raw6 = np.asarray(o["eef_wrench"], dtype=float).reshape(6)
        self.step_index = int(req.get("step", 0))
        self.images_raw = {k: np.asarray(v) for k, v in dict(o.get("images", {})).items()}
        self.privileged = dict(req.get("privileged") or {})
        self.calib = dict(req.get("camera_calibration") or {})       # camera calibration for RGB-only perception (sensor parameters)
        self.diag = dict(req.get("diag_truth") or {})                # logged only, to measure localisation error; never read by the task code or the model
        self.prompt = str(o.get("prompt", ""))

    @property
    def eef_pos(self) -> np.ndarray:
        return self.state[:3].copy()

    @property
    def eef_rpy(self) -> np.ndarray:
        return self.state[3:6].copy()

    @property
    def eef_quat(self) -> np.ndarray:
        return rpy_to_quat_xyzw(self.state[3:6])

    @property
    def gripper_opening(self) -> float:
        return float((self.state[6] + 1.0) * 0.5 * FINGER_MAX_OPEN_M)

    @property
    def sim_time(self) -> float:
        return self.step_index * FB_DT

    @property
    def steps_left(self) -> int:
        return self.max_steps - self.step_index

    # ---------------------------------------------------------------- force
    def _sensor_R(self) -> np.ndarray:
        return R.from_euler("xyz", self.state[3:6]).as_matrix() if self.wrench_frame == "hand" else np.eye(3)

    def raw_wrench(self):
        return self.raw6[:3].copy(), self.raw6[3:].copy(), self._sensor_R()

    def wrench(self) -> WrenchSample:
        f, t, Rs = self.raw_wrench()
        if self.force_noise_std > 0:
            f = f + self.rng.normal(0.0, self.force_noise_std, 3)
            t = t + self.rng.normal(0.0, self.force_noise_std * 0.1, 3)
        f = f + self._force_bias_episode
        t = t + self._torque_bias_episode
        nf, nt = f - self.bias_force, t - self.bias_torque
        return WrenchSample(t=self.sim_time, force=f, torque=t, force_world=Rs @ f, torque_world=Rs @ t,
                            net_force=nf, net_torque=nt, net_force_world=Rs @ nf, net_torque_world=Rs @ nt)

    def reaction_force(self, w: WrenchSample) -> np.ndarray:
        return self.force_sign * w.net_force_world

    def rezero_bias(self, steps: int = 10) -> None:
        fs, ts = [], []
        for _ in range(steps):
            self.step(np.zeros(3), None, None)
            f, t, _ = self.raw_wrench()
            fs.append(f)
            ts.append(t)
        self.bias_force = np.mean(fs, axis=0)
        self.bias_torque = np.mean(ts, axis=0)

    # ---------------------------------------------------------------- action
    def set_kp(self, kp) -> None:
        self.kp_vec = np.ones(6) * float(kp) if np.isscalar(kp) else np.asarray(kp, dtype=float).reshape(6)

    def _gain(self) -> np.ndarray:
        if not self.stiffness_gain:
            return np.ones(3)
        keys = np.array(sorted(self.stiffness_gain))
        vals = np.array([self.stiffness_gain[k] for k in keys])
        g = np.interp(self.kp_vec[:3], keys, vals)
        return np.clip(g, 0.0, self.max_gain)

    def step(self, dpos: np.ndarray, drot: Optional[np.ndarray] = None, grip: Optional[float] = None) -> WrenchSample:
        """One tick with the MuJoCo backend's semantics: dpos = target increment relative to the measured position (m, world frame),
        drot = attitude increment (axis-angle, world frame), grip -1 open ... +1 closed."""
        if grip is not None:
            self.grip_cmd = float(np.clip(grip, -1.0, 1.0))
            self.session.grip_cmd = self.grip_cmd
        p_req = self.eef_pos + self._gain() * np.asarray(dpos, dtype=float).reshape(3)
        self._update_sag(p_req)
        p = p_req + np.array([0.0, 0.0, self.sag_comp])
        if drot is not None and float(np.linalg.norm(drot)) > 0:
            q = (R.from_rotvec(np.asarray(drot, dtype=float)) * R.from_quat(self.eef_quat)).as_quat()
            rpy = quat_xyzw_to_rpy(q, self.eef_rpy)
        else:
            rpy = self.eef_rpy
        action = np.concatenate([p, rpy, [Session.fb_gripper(self.grip_cmd)]]).astype(np.float32)
        before = self.state.copy()
        nxt = self.session.submit(action)
        self._set_obs(nxt)
        self.n_steps += 1
        self.trace.append([self.step_index, *action.tolist(), *before.tolist(), *self.raw6.tolist(), self.sag_comp])
        ents = (self.privileged or self.diag or {}).get("entities") or {}
        if ents:
            e0 = next(iter(ents.values()))
            self.priv_trace.append([self.step_index, *e0.get("pos", [np.nan] * 3), *e0.get("rpy", [np.nan] * 3)])
        w = self.wrench()
        self.peak_wrench = max(self.peak_wrench, float(np.linalg.norm(w.net_force)))
        return w

    def _update_sag(self, p_req: np.ndarray) -> None:
        """Integral part of a PI loop on z, integrated only while holding a hanging load: error = target z requested by the
        primitive - measured z."""
        f, _, Rs = self.raw_wrench()                       # not wrench(): its optional noise would consume random numbers
        r = self.force_sign * (Rs @ (f + self._force_bias_episode - self.bias_force))
        load_like = bool(r[2] < -0.2 and float(np.linalg.norm(r[:2])) <= max(0.5, 0.5 * abs(float(r[2]))))
        v_req = 0.0 if self._prev_req_z is None else (float(p_req[2]) - self._prev_req_z) / self.dt
        self._prev_req_z = float(p_req[2])
        if load_like and self.grip_cmd > 0:
            err = float(p_req[2] - self.eef_pos[2]) - self.sag_tau * v_req
            self.sag_comp = float(np.clip(self.sag_comp + self.sag_ki * err, 0.0, self.sag_max))
        else:
            self.sag_comp *= 0.7

    def hold(self, n: int = 1, grip: Optional[float] = None) -> WrenchSample:
        w = None
        for _ in range(n):
            w = self.step(np.zeros(3), None, grip)
        return w

    def set_held_weld(self, active: bool) -> None:
        """The MuJoCo backend welds held objects because its fingers are too weak; ForceBench grasps through real contact, so
        this does nothing."""
        return None

    # ---------------------------------------------------------------- camera calibration: pixel <-> world (RGB only, no depth)
    def camera(self, name: str):
        c = self.calib.get(name)
        if not c:
            raise ValueError(f"no calibration for camera {name!r} (evaluator must run with --camera_calibration)")
        return (np.asarray(c["K"], dtype=float), np.asarray(c["pos"], dtype=float), np.asarray(c["R_world_cam"], dtype=float),
                int(c["width"]), int(c["height"]))

    def pixel_ray(self, name: str, u: float, v: float):
        """Line of sight through the pixel centre (u+0.5, v+0.5): returns (camera centre, unit direction in the world frame).
        ROS optical frame: x right, y down, z forward."""
        K, pos, Rw, w, h = self.camera(name)
        d = np.linalg.solve(K, np.array([float(u) + 0.5, float(v) + 0.5, 1.0]))
        d = Rw @ d
        return pos, d / np.linalg.norm(d)

    def locate_on_plane(self, name: str, u: float, v: float, z: float, axis: int = 2) -> Optional[np.ndarray]:
        """Intersection of the line of sight with the plane "world coordinate number `axis` = z" (axis 2: horizontal plane
        z = const; 0 / 1: vertical plane x / y = const). Returns None if the line of sight is parallel to the plane or meets it
        behind the camera."""
        o, d = self.pixel_ray(name, u, v)
        if abs(d[axis]) < 1e-6:
            return None
        t = (float(z) - o[axis]) / d[axis]
        return None if t <= 0 else o + t * d

    def project(self, name: str, p_world) -> Optional[np.ndarray]:
        K, pos, Rw, w, h = self.camera(name)
        pc = Rw.T @ (np.asarray(p_world, dtype=float) - pos)
        if pc[2] <= 1e-6:
            return None
        uv = K @ (pc / pc[2])
        return uv[:2] - 0.5

    def metres_per_pixel(self, name: str, p_world) -> float:
        K, pos, Rw, w, h = self.camera(name)
        dist = float(np.linalg.norm(np.asarray(p_world, dtype=float) - pos))
        return dist / float(K[0, 0])

    # ---------------------------------------------------------------- scene and cameras
    def images(self) -> Dict[str, np.ndarray]:
        return {("agentview" if k == "main" else "robot0_eye_in_hand" if k == "wrist" else k): v for k, v in self.images_raw.items()}

    def noised(self, key: str, pos: Sequence[float], yaw_deg: float = 0.0):
        if key not in self._scene_offsets:
            dp = self.rng.normal(0.0, self.scene_noise_pos, size=3) if self.scene_noise_pos > 0 else np.zeros(3)
            dp[2] = 0.0
            dyaw = self.rng.normal(0.0, self.scene_noise_yaw_deg) if self.scene_noise_yaw_deg > 0 else 0.0
            self._scene_offsets[key] = np.array([*dp, dyaw])
        o = self._scene_offsets[key]
        return np.asarray(pos, dtype=float) + o[:3], float(yaw_deg + o[3])

    def scene(self) -> List[Dict[str, Any]]:
        return self.task.describe_scene(self)

    def gripper_contact_state(self) -> Dict[str, Any]:
        return {"in_contact": False, "touching": []}        # no ground-truth contact here (only affects statistics in report.truth)

    def success(self) -> bool:
        return False                                         # ForceBench's summary.json is authoritative

    def close(self) -> None:
        return None
