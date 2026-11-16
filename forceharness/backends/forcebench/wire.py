"""Encoding and decoding of ForceBench's policy wire format (backend layer, below the primitives).

Attribution: this module re-implements the message format of ForceBench (https://github.com/forcebench-platform/ForceBench,
commit 31acf73; MIT License, Copyright (c) 2026 ForceBench contributors): the `forcebench_policy` contract of
`forcebench/policy_api.py` and the msgpack NumPy encoding of `forcebench/policy_transport.py`. It is written to be
byte-compatible with them so that the harness can serve ForceBench's evaluator without importing ForceBench (whose
simulator environment it does not share).

Whenever the evaluator needs an action it sends one msgpack request (`PolicyObservation.canonical_request()`):
  {"contract": "forcebench_policy", "step", "rollout_id", "control_rate_hz": 15.0,
   "observation": {"state"[7], "images": {"main", "wrist"}, "eef_wrench"[6], "prompt", ...}}
Our evaluator copy (scripts/forcebench/eval_forcebench.py) may add "camera_calibration", "diag_truth" (logging only) and, in
privileged mode, "privileged" (object geometry; the harness adds the noise).
Reply: {"contract", "actions"[H, 7], "action_space", "control_rate_hz"}. NumPy arrays use ForceBench's `__ndarray__` dict encoding.
"""
from __future__ import annotations

import io
from typing import Any, Mapping

import msgpack
import numpy as np

POLICY_CONTRACT = "forcebench_policy"
ACTION_SPACE = "world_tcp_xyz_rpy_gripper_open_minus1_plus1"
CONTROL_RATE_HZ = 15.0


def pack_numpy(value: Any):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported NumPy dtype: {value.dtype}")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    raise TypeError(f"cannot serialize {type(value)!r}")


def unpack_numpy(value: dict):
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    if b"__jpeg_rgb__" in value:          # the evaluator sends JPEG when FORCEBENCH_IMAGE_TRANSPORT=jpeg
        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(value[b"data"])).convert("RGB"))
    return value


def dumps(obj: Mapping[str, Any]) -> bytes:
    return msgpack.packb(dict(obj), default=pack_numpy)


def loads(raw: bytes) -> Any:
    return msgpack.unpackb(raw, object_hook=unpack_numpy)


def action_reply(actions: np.ndarray) -> bytes:
    a = np.asarray(actions, dtype=np.float32).reshape(-1, 7)
    return dumps({"contract": POLICY_CONTRACT, "actions": a, "action_space": ACTION_SPACE,
                  "control_rate_hz": CONTROL_RATE_HZ})
