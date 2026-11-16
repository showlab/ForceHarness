---
license: mit
task_categories:
- robotics
tags:
- force
- wrist-force
- contact-rich-manipulation
- mujoco
- robosuite
- franka-panda
- demonstrations
- imitation-learning
- lerobot
pretty_name: HiddenPhys Expert Demonstrations
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files: data/*/*.parquet
---

# HiddenPhys expert demonstrations

[Paper](https://arxiv.org/abs/ARXIV_ID) · [Project page](https://showlab.github.io/ForceHarness) · [Code](https://github.com/showlab/ForceHarness)

Expert demonstrations for the seven tasks of **HiddenPhys**, the MuJoCo benchmark released with
*ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models*.
Each HiddenPhys task hides one physical variable that decides success (a mass, a centre of mass, a table height, a hole
position, an unlock angle, whether a door is locked or only stiff) that position alone does not reveal.
The demonstrations were recorded from the benchmark's expert force controller and used to train the learned baselines of the
paper (Diffusion Policy, π0.5 and ForceVLA, each with and without force input).

Code, task definitions and the recorder: the ForceHarness repository (`forceharness/hiddenphys/tasks.py`,
`forceharness/backends/mujoco/recorder.py`, `scripts/hiddenphys/record_demos*.py`, `scripts/hiddenphys/replay_demos.py`).

## Content

| Task | Released episodes | Attempted | Frames (control steps) | Raw size |
|---|---|---|---|---|
<!-- COUNTS_TABLE -->

| Task | Hidden variable | What force must do |
|---|---|---|
| `weight_sort` | cube mass, 0.15 or 0.60 kg | weigh the held cube, place it in the heavy or light zone |
| `com_sort` | bar's centre of mass, centred or 4 cm off | measure the held moment, place it in the balanced or offset zone |
| `wipe_hard` | table height, up to about 6 cm low | press 5 to 15 N on 12 dirt spots |
| `stack_gentle` | lower cube's top, known to ±1 cm | feel contact and place gently |
| `peg_in_hole_hard` | hole position, 15 mm error per axis | search by touch until the peg drops, then insert |
| `door_latch` | unlock angle, 15 to 35° | press the lever to release the latch, then pull |
| `lock_or_stiff` | locked, or stiff and freed at 6 to 11 N | pull below the 15 N damage threshold; stop if locked |

Simulator: robosuite 1.5.1 on MuJoCo 3.2.7, Franka Panda with the `OSC_POSE` controller at 20 Hz, two cameras
(`agentview`, `robot0_eye_in_hand`) at 128×128. Only successful episodes are released; episode seeds start at 20000 and never
overlap the benchmark's development (9000-9019) or test (9100-9149) seeds.

## Two forms in one repository

### `raw/`: the recorded episodes (exact replay)

`raw/<task>/episode_<seed>.npz`, one file per episode, plus `raw/<task>/index.jsonl` (seed, length, hidden variable, verdict)
and `raw/attempts.json`. Observations have one more frame than actions: frame `i` is what the controller saw before action `i`.

| Key | Shape | Meaning |
|---|---|---|
| `action` | (T, 7) float64 | normalised OSC delta actually sent to the environment, in [-1, 1]: position × `output_max_pos` metres, rotation × `output_max_rot` rad, gripper (-1 open … +1 closed); `wipe_hard` has no fingers, so its 7th entry is NaN |
| `kp` | (T, 6) float64 | OSC stiffness during the step (the primitives soften the arm near contact) |
| `weld` | (T,) bool | whether the held object was welded to the hand (pegs start held) |
| `via_step` | (T,) bool | whether the step went through the harness step (task hooks such as latches and damage ran) |
| `prim` | (T,) int16 | index into `meta.prim_names`: the primitive running at that step (analysis only) |
| `t` | (T+1,) | simulation time, s |
| `eef_pos`, `eef_quat` | (T+1, 3), (T+1, 4) | end-effector position (m) and quaternion `[x, y, z, w]`, world frame |
| `gripper_qpos`, `joint_pos` | (T+1, 2), (T+1, 7) | finger and joint positions; `gripper_qpos` is empty in `wipe_hard`, whose wiping tool has no fingers (the loader reads it as zeros) |
| `force_raw`, `torque_raw` | (T+1, 3) | wrist sensor reading in the sensor frame, **noise-free and not tared** |
| `R_ft` | (T+1, 9) | sensor-to-world rotation (row-major) |
| `b0_force`, `b0_torque`, `bias_ep_force`, `bias_ep_torque` | (3,) | the zero taken at reset and the per-episode residual bias |
| `img_<cam>_jpeg`, `img_<cam>_off` | bytes, offsets | JPEG frames (quality 90, BGR channel order, **upside down** as robosuite renders them); frame `i` is `jpeg[off[i]:off[i+1]]` |
| `meta_json` | bytes | JSON: task, seed, hidden-variable ground truth (`truth`; `label` / `theta_u` in the two mechanism tasks), verdict, noise settings, cameras, the noisy scene the controller was given (`scene0`), primitive names, and the episode configuration replay needs (`episode_cfg`; `start_after_pregrasp` and `prims_args` for `lock_or_stiff`) |

The force the harness uses is `F = force_sign · R (f_raw + bias_ep + n − b0)` with `n ~ N(0, σ)` drawn per step
(`σ = meta.force_noise_std`, torques 0.1 σ). Recording the noise-free reading keeps the episode's random stream untouched and
lets a training pipeline add noise with the same model; `forceharness/data/demos.py` (`wrench_obs`) does exactly that.
Replaying `action`, `kp` and `weld` from the episode's seed reproduces the trajectory (`scripts/hiddenphys/replay_demos.py`;
the end-effector end point matches to 0.00 mm in our checks).

```python
from forceharness.data.demos import load_demo, decode_frame
d = load_demo("raw/weight_sort/episode_20000.npz")
img = decode_frame(d, "agentview", 0)          # upright RGB uint8
print(d["meta"]["truth"], d["action"].shape, d["force_raw"].shape)
```

### Root: LeRobot v2.1 (training)

The same episodes as one multi-task LeRobot v2.1 dataset (`meta/`, `data/`), as used to train the π0.5 and ForceVLA baselines:

| Feature | Shape | Meaning |
|---|---|---|
| `image`, `wrist_image` | 128×128×3 | `agentview` and wrist camera, upright RGB |
| `state` | (26,) | end-effector position 3 + orientation 6D 6 + fingers 2, then the noisy object poses the controller was given (5 numbers per object: position, sin/cos yaw), zero-padded to 15 |
| `wrench` | (6,) | wrist force and torque in the world frame, tared at the start, with the harness noise model applied |
| `actions` | (13,) | the 7 OSC action entries (6 for `wipe_hard`, padded with 0) followed by the 6 stiffness values |
| `task` | text | a short instruction per task |

```python
# any lerobot release that reads codebase version v2.1, e.g. the one pinned by openpi
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset("<this repository id>")
```

## How the data was recorded

The expert is a hand-written state machine per task that calls the same self-terminating force primitives the model uses in
ForceHarness and reads the force they report (five tasks through `forceharness/baselines/expert.py`, the two mechanism tasks
through `scripts/hiddenphys/door_latch_expert.py` and `lock_or_stiff_expert.py`). Its constants were chosen with knowledge of the
task design (for example, the weight threshold is the midpoint of the two masses), so it is an upper reference, not a
task-agnostic method. On `lock_or_stiff`, recording starts after the scripted pre-grasp of the handle (the environment performs
it for every method) and uses the soft arm (stiffness 150). Episode budgets follow the benchmark protocol (40 turns,
80 tool calls, 300 s).

## Intended use and limitations

- Training and evaluating learned policies on HiddenPhys, studying force as an input, and replaying expert behaviour.
- Simulation only. The expert's behaviour is narrow; policies trained on it inherit its strategy.
- Information differs from the harness setting: learned policies see camera images (the wrist camera can see the true hole
  position), whereas the harness receives object poses as noisy text.
- Learned policies have no safety stop of their own; report peak force and damage (handle force above 15 N in the mechanism
  tasks) together with success, as the benchmark does.

## License and citation

MIT. The simulated scenes are built from robosuite (MIT) assets. If you use this dataset, please cite the paper:

```bibtex
@article{sun2026forceharness,
  title   = {ForceHarness: Turning Force into Action and Physics for Frozen Frontier Models},
  author  = {Sun, Ke and Ma, Kevin Yuchen and Zhang, Yike and Chen, Yanzhe and Shou, Mike Zheng},
  journal = {arXiv preprint arXiv:ARXIV_ID},
  year    = {2026}
}
```
