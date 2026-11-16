"""Demonstrations for the learned baselines: load the episodes recorded by scripts/hiddenphys/record_demos.py (one npz
per episode) and turn them into training observations and actions.  Depends only on numpy + cv2.

Observation (per tick):
  images: agentview and robot0_eye_in_hand (stored as 128x128 upside-down JPEG; flipped upright here);
  proprioception, 11-D: end-effector position 3 + 6-D orientation 6 (first two columns of the rotation matrix) +
      two finger joints 2;
  wrench, 6-D (optional): world frame, zeroed at the start of the episode, with noise added by the same noise model
      as ForceSim:
      F = force_sign * R (f_raw + bias_ep + N(0, sigma) - b0), torque likewise (with sigma x 0.1).
      At evaluation time force_sign * net_force_world from sim.wrench() is the same quantity (evaluation calls no
      primitive, so the zero point stays the one set at reset);
  scene vector (optional): the noisy object poses the agent sees on its first scene call (same seed, same offsets):
      position and (sin, cos) of yaw per object, in order of object name, constant over the episode.
Action (per tick): the normalised OSC delta the environment actually received (7-D with a gripper, 6-D for the
wiping task) + the OSC stiffness kp at that tick (6-D).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def load_demo(path: str) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files if not k.startswith("img_") and k != "meta_json"}
    d["meta"] = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    d["jpeg"] = {c: (z[f"img_{c}_jpeg"], z[f"img_{c}_off"]) for c in d["meta"].get("cams", [])}
    return d


def decode_frame(demo: Dict[str, Any], cam: str, i: int, flip: bool = True) -> np.ndarray:
    """Image of tick i as RGB uint8; flip=True turns it upright (robosuite renders offscreen images upside down)."""
    import cv2

    buf, off = demo["jpeg"][cam]
    img = cv2.imdecode(np.asarray(buf[off[i]:off[i + 1]]), cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img[::-1]) if flip else np.ascontiguousarray(img)


def prep_jpeg(data, size: int, crop: int, train: bool = False, rng=None) -> np.ndarray:
    """Image preprocessing shared by training and evaluation: JPEG bytes (the recorder and the evaluation client both
    encode BGR, quality 90, upside down) -> upright -> RGB -> resize to size -> crop to crop (random for training,
    centred for evaluation) -> 3 x crop x crop uint8."""
    import cv2

    img = cv2.imdecode(np.frombuffer(data, np.uint8) if isinstance(data, (bytes, bytearray)) else np.asarray(data), cv2.IMREAD_COLOR)
    img = np.ascontiguousarray(img[::-1, :, ::-1])
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    if train:
        y, x = (rng or np.random.default_rng()).integers(0, size - crop + 1, 2)
    else:
        y = x = (size - crop) // 2
    return np.ascontiguousarray(img[y:y + crop, x:x + crop].transpose(2, 0, 1))


def encode_frame(img_rgb_upside_down: np.ndarray, quality: int = 90) -> bytes:
    """At evaluation time, encode a raw robosuite frame exactly as the recorder does (BGR, quality 90, not flipped), so
    that training and evaluation see the same JPEG artefacts."""
    import cv2

    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img_rgb_upside_down[..., ::-1]), [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    assert ok
    return buf.tobytes()


def pack(a: np.ndarray):
    """Pack an array for transfer between Python environments (their numpy major versions may differ, so ndarrays are
    not pickled directly)."""
    a = np.ascontiguousarray(a)
    return (a.dtype.str, tuple(a.shape), a.tobytes())


def unpack(p) -> np.ndarray:
    dt, shape, b = p
    return np.frombuffer(b, dtype=np.dtype(dt)).reshape(shape).copy()


def quat_to_rot6d(q: np.ndarray) -> np.ndarray:
    """Quaternion (xyzw, as in robosuite) -> first two columns of the rotation matrix (6-D).  The formula is quadratic
    in the quaternion components, so q and -q give the same value; unlike a raw quaternion it does not flip sign around
    w ~ 0, which is exactly the case with the gripper pointing down."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = x * x + y * y + z * z + w * w
    s = np.where(n > 0, 2.0 / np.maximum(n, 1e-12), 0.0)
    c0 = np.stack([1 - s * (y * y + z * z), s * (x * y + z * w), s * (x * z - y * w)], -1)
    c1 = np.stack([s * (x * y - z * w), 1 - s * (x * x + z * z), s * (y * z + x * w)], -1)
    return np.concatenate([c0, c1], -1).astype(np.float32)


def proprio(eef_pos: np.ndarray, eef_quat: np.ndarray, gripper_qpos) -> np.ndarray:
    """11-D proprioception: end-effector position 3 + 6-D orientation 6 + two finger joints 2.  The wiping tool has no
    gripper (robosuite returns an empty array), so the finger part is zero."""
    p = np.asarray(eef_pos, np.float32)
    g = np.asarray(gripper_qpos if gripper_qpos is not None else [], np.float32)
    want = p.shape[:-1] + (2,)
    if g.size == 0 or g.shape[:-1] != p.shape[:-1]:
        g = np.zeros(want, np.float32)
    return np.concatenate([p, quat_to_rot6d(eef_quat), g[..., :2]], -1)


def wrench_obs(d: Dict[str, Any], seed: int) -> np.ndarray:
    """6-D wrist wrench observation of a demonstration (see the module docstring).  The noise comes from a seeded
    generator, giving one fixed sequence per episode."""
    m = d["meta"]
    T = len(d["t"])
    rng = np.random.default_rng(int(seed) * 7919 + 17)
    sig = float(m.get("force_noise_std", 0.0))
    nf = rng.normal(0.0, sig, (T, 3)) if sig > 0 else np.zeros((T, 3))
    nt = rng.normal(0.0, sig * 0.1, (T, 3)) if sig > 0 else np.zeros((T, 3))
    R = d["R_ft"].reshape(T, 3, 3).astype(np.float64)
    f = d["force_raw"] + d["bias_ep_force"] + nf - d["b0_force"]
    t = d["torque_raw"] + d["bias_ep_torque"] + nt - d["b0_torque"]
    s = float(m.get("force_sign", -1.0))
    fw = s * np.einsum("tij,tj->ti", R, f)
    tw = s * np.einsum("tij,tj->ti", R, t)
    return np.concatenate([fw, tw], -1).astype(np.float32)


def scene_vector(scene0: Optional[List[Dict[str, Any]]], keys: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """Concatenate the noisy object poses into a vector.  When keys is given its order is used (evaluation must use the
    training order); missing objects are filled with zeros."""
    objs = {o["name"]: o for o in (scene0 or []) if isinstance(o, dict) and "name" in o and "pos" in o}
    if keys is None:
        keys = sorted(objs)
    out = []
    for k in keys:
        o = objs.get(k)
        p = list(o["pos"])[:3] if o is not None else [0.0, 0.0, 0.0]
        p += [0.0] * (3 - len(p))
        yaw = float(o.get("yaw_deg", 0.0)) if o is not None else 0.0
        out += [*p, np.sin(np.radians(yaw)), np.cos(np.radians(yaw))]
    return np.asarray(out, np.float32), keys


def action_matrix(d: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    """[normalised OSC action (6 or 7) | kp (6)].  Returns (T x A array, number of OSC dimensions in the action)."""
    a = d["action"]
    adim = 7 if not np.isnan(a[:, 6]).any() else 6
    return np.concatenate([a[:, :adim], d["kp"]], -1).astype(np.float32), adim


def list_demos(task_dir: str, n: Optional[int] = None) -> List[str]:
    fs = sorted(Path(task_dir).glob("s*.npz"), key=lambda p: int(p.stem[1:]))
    return [str(p) for p in (fs[:n] if n else fs)]
