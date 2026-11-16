"""Convert the recorded HiddenPhys demonstrations (one npz per episode, written by scripts/hiddenphys/record_demos*.py) into
ONE multi-task LeRobot dataset (format v2.1, the lerobot version pinned by openpi) for the pi0.5 and ForceVLA baselines.
Modeled on openpi's examples/libero/convert_libero_data_to_lerobot.py.

Per-frame fields (observation frame i is paired with action tick i; the npz holds one more observation than actions, so only
the first T frames are used):
  image        agentview, upright RGB, 128x128 (recorded at this size, not resized)
  wrist_image  robot0_eye_in_hand, same format
  state        proprioception 11 (end-effector position 3 + rotation 6D 6 + two finger joints 2) + noisy scene vector (5 per
               object, objects sorted by name), zero-padded to the longest scene vector among the selected tasks
  wrench       wrist wrench 6 (world frame, zeroed at reset, with noise added by the ForceSim noise model; = demos.wrench_obs)
  actions      13 = normalized OSC delta 7 (6 on wipe_hard, padded with a 0 at index 6) + OSC stiffness kp 6
  task         one short English instruction per task (PROMPTS below)
The script also writes meta_fh.json at the dataset root: per task the scene_keys, scene-vector length before / after padding,
max_T, osc_dim, prompt, and the per-dimension min / max of the actions (the inference server returns them as act_norm).
pi05_fh.py compute_norm_stats copies it into the assets, and training copies it into every checkpoint.

Usage (openpi environment, from the repository root; the dataset goes to $HF_LEROBOT_HOME/<repo_id>):
  HF_LEROBOT_HOME=demos/lerobot python scripts/baselines/pi05/convert_to_lerobot.py \
      --demos demos/hiddenphys [--n-per-task 200] [--tasks weight_sort,com_sort] [--overwrite]
Needs only numpy + cv2 (to read the npz) and lerobot (to write the dataset); does not import robosuite or jax.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[3]          # repository root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forceharness.data.demos import action_matrix, decode_frame, list_demos, load_demo, proprio, scene_vector, wrench_obs  # noqa: E402

REPO_ID = "forceharness/hiddenphys_demos"      # local dataset name; nothing is pushed to the Hub
FPS = 20                                             # robosuite control_freq (demos are recorded at 20 Hz)
TASKS = ["weight_sort", "com_sort", "wipe_hard", "stack_gentle", "peg_in_hole_hard", "door_latch", "lock_or_stiff"]
# One short instruction per task (the pi0.5 prompt also carries the discretized state tokens, so a short instruction saves
# tokens; the full task descriptions are in forceharness/hiddenphys/tasks.py). These strings are model inputs: keep them as is.
PROMPTS = {
    "weight_sort": "pick up the cube, judge its weight, and put it in the heavy or light zone",
    "com_sort": "pick up the bar at its center, judge its balance, and put it in the balanced or offset zone",
    "wipe_hard": "wipe all the dirt spots off the table, pressing firmly but not too hard",
    "stack_gentle": "gently stack the red cube on top of the green cube",
    "peg_in_hole_hard": "feel for the hole and insert the held peg into it",
    "door_latch": "press the lever handle down to release the latch and pull the door open",
    "lock_or_stiff": "pull the door open if it is stiff, and stop if it is locked",
}
CAMS = {"image": "agentview", "wrist_image": "robot0_eye_in_hand"}   # dataset key -> camera name in recording / evaluation
PROPRIO_DIM, WRENCH_DIM, OSC_DIM_MAX, KP_DIM = 11, 6, 7, 6
ACTION_DIM = OSC_DIM_MAX + KP_DIM                    # 13
IMAGE_SIZE = 128


def pad_scene(sv: np.ndarray, scene_pad: int) -> np.ndarray:
    """Zero-pad the scene vector to scene_pad (tasks have different object counts; the prompt tells tasks apart, so the padding
    zeros are never confused with another task's objects)."""
    sv = np.asarray(sv, np.float32).reshape(-1)
    if sv.size > scene_pad:
        raise ValueError(f"scene vector has {sv.size} dims > scene_pad {scene_pad}")
    return np.concatenate([sv, np.zeros(scene_pad - sv.size, np.float32)])


def build_state(prop: np.ndarray, sv: np.ndarray, scene_pad: int) -> np.ndarray:
    """Dataset state layout: proprioception 11 + zero-padded scene vector. Shared by the conversion (here) and the inference
    server (pi05_server.py)."""
    return np.concatenate([np.asarray(prop, np.float32).reshape(-1), pad_scene(sv, scene_pad)]).astype(np.float32)


def actions13(act: np.ndarray, osc_dim: int) -> np.ndarray:
    """[OSC (6 or 7) | kp 6] -> 13 dims: wipe_hard (osc_dim = 6, no gripper) gets a 0 at index 6."""
    act = np.asarray(act, np.float32)
    if osc_dim == OSC_DIM_MAX:
        return act
    return np.concatenate([act[:, :osc_dim], np.zeros((len(act), OSC_DIM_MAX - osc_dim), np.float32), act[:, osc_dim:]], -1)


def episode_arrays(d: Dict[str, Any], scene_keys: List[str], scene_pad: int) -> Dict[str, Any]:
    """One demo -> per-frame state / wrench / actions (length T = number of action ticks) and osc_dim. Images are decoded
    separately, frame by frame (decode_frame)."""
    act, osc_dim = action_matrix(d)
    T = len(act)
    assert len(d["t"]) == T + 1, "expected one more observation frame than actions (re-record demos in the old format)"
    seed = int(d["meta"]["seed"])
    prop = proprio(d["eef_pos"], d["eef_quat"], d["gripper_qpos"])[:T]
    sv, _ = scene_vector(d["meta"].get("scene0"), scene_keys)
    state = np.stack([build_state(prop[i], sv, scene_pad) for i in range(T)])
    wrench = wrench_obs(d, seed)[:T]
    return dict(state=state, wrench=wrench.astype(np.float32), actions=actions13(act, osc_dim), osc_dim=int(osc_dim), T=T,
                seed=seed, scene_dim=int(sv.size))


def scan(demos_root: Path, tasks: List[str], n_per_task: Optional[int]) -> Dict[str, Dict[str, Any]]:
    """First pass: per task the files, the scene_keys (object names of the first demo, sorted; later demos use this order, as
    train_dp.py does) and the unpadded scene-vector length."""
    out = {}
    for t in tasks:
        files = list_demos(str(demos_root / t), n_per_task)
        if not files:
            raise FileNotFoundError(f"no s*.npz under {demos_root / t}")
        d0 = load_demo(files[0])
        sv, keys = scene_vector(d0["meta"].get("scene0"))
        out[t] = dict(files=files, scene_keys=keys, scene_dim=int(sv.size))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", required=True, help="demonstration root containing <task>/s*.npz (e.g. demos/hiddenphys)")
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--tasks", default=",".join(TASKS), help="comma-separated; default: the 7 HiddenPhys tasks")
    ap.add_argument("--n-per-task", type=int, default=None, help="at most this many episodes per task (the first n by seed); default: all")
    ap.add_argument("--scene-pad", type=int, default=None, help="pad the scene vector to this length; default: the longest among the selected tasks")
    ap.add_argument("--overwrite", action="store_true", help="delete and rebuild $HF_LEROBOT_HOME/<repo_id> if it exists")
    ap.add_argument("--writer-threads", type=int, default=8, help="threads of the lerobot asynchronous PNG writer")
    ap.add_argument("--writer-procs", type=int, default=0, help="processes of the lerobot asynchronous PNG writer (0 = threads only)")
    args = ap.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    bad = [t for t in tasks if t not in PROMPTS]
    if bad:
        raise SystemExit(f"unknown task(s) {bad}; known: {list(PROMPTS)}")
    info = scan(Path(args.demos), tasks, args.n_per_task)
    scene_pad = args.scene_pad if args.scene_pad is not None else max(v["scene_dim"] for v in info.values())
    state_dim = PROPRIO_DIM + scene_pad
    print(f"[convert] tasks {tasks}; demos/task {[len(v['files']) for v in info.values()]}; scene dims "
          f"{ {t: v['scene_dim'] for t, v in info.items()} } → pad {scene_pad}; state {state_dim}, wrench {WRENCH_DIM}, "
          f"actions {ACTION_DIM}", flush=True)

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset   # reads the HF_LEROBOT_HOME env var

    out_dir = HF_LEROBOT_HOME / args.repo_id
    if out_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{out_dir} exists; pass --overwrite to rebuild")
        shutil.rmtree(out_dir)
    img_ft = {"dtype": "image", "shape": (IMAGE_SIZE, IMAGE_SIZE, 3), "names": ["height", "width", "channel"]}
    ds = LeRobotDataset.create(
        repo_id=args.repo_id, robot_type="panda", fps=FPS,
        features={
            "image": dict(img_ft), "wrist_image": dict(img_ft),
            "state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
            "wrench": {"dtype": "float32", "shape": (WRENCH_DIM,), "names": ["wrench"]},
            "actions": {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=args.writer_threads, image_writer_processes=args.writer_procs,
    )

    t0 = time.time()
    meta_tasks: Dict[str, Any] = {}
    all_act, all_state, all_wrench = [], [], []
    for t in tasks:
        v = info[t]
        acts, Ts, seeds, osc_dims, n_frames = [], [], [], set(), 0
        for f in v["files"]:
            d = load_demo(f)
            sv, keys = scene_vector(d["meta"].get("scene0"))
            if keys != v["scene_keys"]:
                print(f"  [warn] {f}: scene keys {keys} != {v['scene_keys']} (first demo); using the first demo's order, missing → 0")
            e = episode_arrays(d, v["scene_keys"], scene_pad)
            for i in range(e["T"]):
                frame = {k: decode_frame(d, cam, i) for k, cam in CAMS.items()}
                if frame["image"].shape != (IMAGE_SIZE, IMAGE_SIZE, 3):
                    raise ValueError(f"{f}: image {frame['image'].shape}, expected {(IMAGE_SIZE, IMAGE_SIZE, 3)}")
                frame.update(state=e["state"][i], wrench=e["wrench"][i], actions=e["actions"][i], task=PROMPTS[t])
                ds.add_frame(frame)
            ds.save_episode()
            acts.append(e["actions"]); all_state.append(e["state"]); all_wrench.append(e["wrench"])
            Ts.append(e["T"]); seeds.append(e["seed"]); osc_dims.add(e["osc_dim"]); n_frames += e["T"]
        if len(osc_dims) != 1:
            raise ValueError(f"{t}: inconsistent osc_dim across demos: {osc_dims}")
        a = np.concatenate(acts)
        all_act.append(a)
        meta_tasks[t] = dict(prompt=PROMPTS[t], scene_keys=v["scene_keys"], scene_dim=v["scene_dim"], scene_pad=scene_pad,
                             max_T=int(max(Ts)), osc_dim=int(osc_dims.pop()), n_episodes=len(Ts), n_frames=int(n_frames),
                             seeds=seeds, actions_min=a.min(0).tolist(), actions_max=a.max(0).tolist())
        print(f"  {t}: {len(Ts)} episodes, {n_frames} frames, max_T {max(Ts)}, osc_dim {meta_tasks[t]['osc_dim']}, "
              f"scene {v['scene_keys']} ({v['scene_dim']}) — {time.time() - t0:.0f}s", flush=True)
    ds.stop_image_writer()

    A, S, W = np.concatenate(all_act), np.concatenate(all_state), np.concatenate(all_wrench)
    meta = dict(
        repo_id=args.repo_id, fps=FPS, created=time.strftime("%Y-%m-%d %H:%M:%S"), demos_root=str(Path(args.demos).resolve()),
        n_per_task=args.n_per_task, cams=CAMS, image_size=[IMAGE_SIZE, IMAGE_SIZE],
        image_note="upright RGB (the recorder / evaluation JPEGs are BGR, upside down, q90; decode_frame and pi05_server flip them first)",
        proprio_dim=PROPRIO_DIM, scene_pad=scene_pad, state_dim=state_dim, wrench_dim=WRENCH_DIM, action_dim=ACTION_DIM,
        state_layout=f"proprio {PROPRIO_DIM} (eef_pos 3 + rot6d 6 + gripper 2) + scene vector zero-padded to {scene_pad} "
                     "(5 per object: pos 3 + sin/cos yaw, objects sorted by name = scene_keys)",
        action_layout="normalized OSC delta 7 (wipe_hard: 6 + a zero pad at index 6) + OSC stiffness kp 6",
        actions_min=A.min(0).tolist(), actions_max=A.max(0).tolist(),
        state_min=S.min(0).tolist(), state_max=S.max(0).tolist(), wrench_min=W.min(0).tolist(), wrench_max=W.max(0).tolist(),
        n_frames=int(len(A)), tasks=meta_tasks,
    )
    (out_dir / "meta_fh.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[convert] done: {sum(m['n_episodes'] for m in meta_tasks.values())} episodes, {len(A)} frames in "
          f"{(time.time() - t0) / 60:.1f} min → {out_dir} (+ meta_fh.json)", flush=True)


if __name__ == "__main__":
    main()
