"""Offline test of scripts/data/export_hf.py on a synthetic episode in the recorder's format."""
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_exporter():
    spec = importlib.util.spec_from_file_location("export_hf", ROOT / "scripts" / "data" / "export_hf.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_episode(path: Path, seed: int, meta_extra=None, T: int = 4):
    Image = pytest.importorskip("PIL.Image")
    bufs = []
    for i in range(T + 1):
        b = io.BytesIO()
        Image.fromarray(np.full((8, 8, 3), i * 10, np.uint8)).save(b, format="JPEG")
        bufs.append(b.getvalue())
    arr = dict(t=np.arange(T + 1) * .05, action=np.zeros((T, 7)), eef_pos=np.zeros((T + 1, 3), np.float32),
               eef_quat=np.zeros((T + 1, 4), np.float32), gripper_qpos=np.zeros((T + 1, 2), np.float32),
               joint_pos=np.zeros((T + 1, 7), np.float32), force_raw=np.zeros((T + 1, 3), np.float32),
               torque_raw=np.zeros((T + 1, 3), np.float32), R_ft=np.tile(np.eye(3).reshape(9), (T + 1, 1)).astype(np.float32),
               kp=np.full((T, 6), 150.), weld=np.zeros(T, bool), via_step=np.ones(T, bool), prim=np.zeros(T, np.int16),
               b0_force=np.zeros(3), b0_torque=np.zeros(3), bias_ep_force=np.zeros(3), bias_ep_torque=np.zeros(3), home_pos=np.zeros(3),
               img_agentview_jpeg=np.frombuffer(b"".join(bufs), np.uint8),
               img_agentview_off=np.concatenate([[0], np.cumsum([len(x) for x in bufs])]).astype(np.int64))
    meta = dict(task="weight_sort", seed=seed, success=True, verdict="heavy", truth=dict(mass=0.6), n_steps=T, cams=["agentview"],
                prim_names=["lift"], scene0=[], worker=3, egl=1, recorded_at="2026-01-01 00:00:00",
                episode_cfg=dict(kp=150.0, out_dir="/home/someone/runs", force_noise_std=0.5))
    meta.update(meta_extra or {})
    arr["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arr)


def test_raw_export_cleans_metadata_and_counts(tmp_path):
    ex = _load_exporter()
    demos = tmp_path / "demos"
    _fake_episode(demos / "weight_sort" / "s20000.npz", 20000)
    _fake_episode(demos / "weight_sort" / "s20002.npz", 20002)
    (demos / "weight_sort" / "index.jsonl").write_text("".join(json.dumps(dict(seed=s, success=s != 20001)) + "\n"
                                                              for s in (20000, 20001, 20002)))
    counts = ex.export_raw(demos, tmp_path / "out", ["weight_sort"], None)
    assert counts["weight_sort"]["attempted"] == 3 and counts["weight_sort"]["released"] == 2
    z = np.load(tmp_path / "out" / "raw" / "weight_sort" / "episode_20000.npz")
    meta = json.loads(bytes(z["meta_json"]).decode())
    assert "egl" not in meta and "recorded_at" not in meta and meta["worker"] == 3      # only the listed bookkeeping is dropped
    assert meta["episode_cfg"] == {"kp": 150.0, "out_dir": "runs", "force_noise_std": 0.5} and meta["truth"] == {"mass": 0.6}
    lines = (tmp_path / "out" / "raw" / "weight_sort" / "index.jsonl").read_text().splitlines()
    assert [json.loads(x)["seed"] for x in lines] == [20000, 20002]


def test_export_refuses_suspicious_metadata(tmp_path):
    ex = _load_exporter()
    assert ex.SUSPICIOUS.search("table within \u00b12 cm") is None and ex.SUSPICIOUS.search("\u7ecf\u5178") is not None
    _fake_episode(tmp_path / "demos" / "weight_sort" / "s20000.npz", 20000, meta_extra=dict(verdict="see /workspace/x"))
    with pytest.raises(ValueError):
        ex.export_raw(tmp_path / "demos", tmp_path / "out", ["weight_sort"], None)


def test_export_checks_frame_counts(tmp_path):
    ex = _load_exporter()
    p = tmp_path / "demos" / "weight_sort" / "s20000.npz"
    _fake_episode(p, 20000)
    z = dict(np.load(p))
    z["eef_pos"] = z["eef_pos"][:-1]
    np.savez(p, **z)
    with pytest.raises(ValueError):
        ex.export_raw(tmp_path / "demos", tmp_path / "out", ["weight_sort"], None)
