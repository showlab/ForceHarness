# Learned-policy baselines on HiddenPhys

The HiddenPhys comparison (Table 5 of the paper and its supplement) has learned rows next to the classical controllers and
ForceHarness: Diffusion Policy, pi0.5 and ForceVLA, each trained on about 200 demonstrations per task recorded from the
expert force controller, with and without wrist force. This document covers what each baseline is, the data, the training
recipes, the compute used, the evaluation protocol, how to reproduce every row, and the caveats needed to read the numbers.
None of this code is part of the three ForceHarness layers; the baselines drive the same simulator controller directly.

All baselines are **controlled variants**: they reuse published architectures and recipes, adapted to our tasks, inputs and
action space. Nothing here reproduces a published system in its original setting.

## 1. Rows, variants and code

| Paper row | Variant / config | Model | Force input |
|---|---|---|---|
| DP | `dp` | Diffusion Policy (CNN U-Net), one model per task | none |
| DP + force | `dp_force` | same | 6-D wrist wrench concatenated to the low-dim observation (as in FoAR / ForceVLA's "DP + force" control) |
| DP, images only | `dpnoscene` | same, without the noisy scene vector (ForceBench-style input) | none |
| DP, images only + force | `dpnoscene_force` | same | concatenated wrench |
| DP, 50 demonstrations (supplement) | `dp50`, `dp50_force` | same, first 50 demos per task | none / concatenated wrench |
| pi0.5 | config `pi05_fh_f0`, results name `pi05_f0` | pi0.5, one multi-task model, task selected by the prompt | none (ForceBench F0) |
| pi0.5 + force, Prompt Fusion | config `pi05_fh_f2`, results name `pi05_f2` | same | wrench appended to the state, normalized, discretized into 256 levels and written into the prompt (ForceBench Prompt Fusion) |
| ForceVLA | config `forcevla_fh`, results name `forcevla` | pi0 + force token + LIMoE (ForceVLA's model code), one multi-task model | wrench projected to a token that enters a mixture-of-experts block with the VLM output |

| Path | What it is |
|---|---|
| `forceharness/baselines/diffusion_policy.py` | the Diffusion Policy model (encoders, conditional 1D U-Net, schedulers, min-max normalization) |
| `forceharness/data/demos.py` | loading recorded demos and building the observations / actions shared by all learned rows |
| `scripts/baselines/dp/train_dp.py`, `dp_server.py` | DP training and inference server (torch environment) |
| `scripts/baselines/dp/eval_policy.py` | the evaluation client for **all** learned rows (simulator environment) |
| `scripts/baselines/dp/dp_pipeline.sh`, `dp_shift_eval.sh` | train -> serve -> evaluate one DP variant on one task; distribution-shift evaluation |
| `scripts/baselines/pi05/convert_to_lerobot.py` | demos -> one multi-task LeRobot v2.1 dataset + `meta_fh.json` (also used by ForceVLA) |
| `scripts/baselines/pi05/pi05_fh.py` | pi0.5 data transforms and TrainConfigs registered into openpi; normalization fix; bf16 EMA-only checkpoints |
| `scripts/baselines/pi05/pi05_server.py` | inference server for pi0.5 and ForceVLA (openpi environment), with a model-free self-test |
| `scripts/baselines/pi05/eval_multitask.sh` | one server + parallel evaluation of the 7 tasks and 4 shifted variants |
| `scripts/baselines/pi05/README.md` | pi0.5 details: settings table, sizes, pitfalls, checks |
| `scripts/baselines/forcevla/forcevla_fh.py` | ForceVLA TrainConfig, assets, model / batch / policy checks |
| `scripts/baselines/forcevla/pi0_force.py`, `limoe_simple.py` | ForceVLA model code (Apache-2.0, from ForceBench's packaged ForceVLA runtime; see §9) |

Every learned row runs as a **policy server plus evaluation client**. The server (torch or openpi environment) holds the model;
the client (`eval_policy.py`, in the ForceHarness simulator environment) runs MuJoCo, sends the latest observation frames over
a local socket (`multiprocessing.connection` on `127.0.0.1:<port>`), receives an action chunk, and executes it. Arrays travel as
`(dtype, shape, bytes)`, so the environments may have different numpy versions. The port is bound to localhost; the fixed
authkey is not a security boundary, so do not expose it.

## 2. Data

### 2.1 Demonstrations

The demonstrations come from the **expert force controller** row (one hand-written state machine per task over the ForceHarness
primitives). Only successful episodes are saved, one `s<seed>.npz` per episode under `<demo root>/<task>/`, plus an
`index.jsonl` per task that also lists failed attempts. Demonstration seeds start at 20000 (20000-20209, up to 20239 on Peg);
they never overlap the development seeds (9000-9019) or the test seeds (9100-9149).

```bash
# 5 tasks driven by the expert agent (episode settings from the expert matrix, the same as its comparison row)
for t in weight_sort com_sort wipe_hard stack_gentle; do
  python scripts/hiddenphys/record_demos.py --task $t --seed-start 20000 --n 210 --matrix configs/hiddenphys/expert.yaml \
      --out demos/hiddenphys --egl 0 --workers 8
done
python scripts/hiddenphys/record_demos.py --task peg_in_hole_hard --seed-start 20000 --n 240 \
    --matrix configs/hiddenphys/expert.yaml --out demos/hiddenphys --egl 0 --workers 8
# the two mechanism tasks use the expert's task-specific force scripts
for t in door_latch lock_or_stiff; do
  python scripts/hiddenphys/record_demos_mechanisms.py --task $t --seed-start 20000 --n 210 --out demos/hiddenphys --egl 0 --workers 16
done
```

| Task | Attempts | Saved | Frames (action ticks) | Longest demo | Evaluation horizon (1.5 x longest) |
|---|---|---|---|---|---|
| weight_sort | 210 | 210 | 144,312 | 723 | 1084 |
| com_sort | 210 | 208 | 140,607 | 709 | 1063 |
| wipe_hard | 210 | 207 | 68,960 | 453 | 679 |
| stack_gentle | 210 | 206 | 121,311 | 793 | 1189 |
| peg_in_hole_hard | 240 | 185 | 122,194 | 2310 | 3465 |
| door_latch | 210 | 197 | 111,329 | 673 | 1009 |
| lock_or_stiff | 210 | 210 | 18,154 | 254 | 381 |
| total | 1500 | 1423 | 726,867 | | |

What an episode file holds (`forceharness/backends/mujoco/recorder.py` documents the format):

- **Per frame** (one more frame than actions; frame i is what was seen before action i): two 128x128 JPEG images (quality
  90, robosuite's upside-down orientation, BGR) from `agentview` and `robot0_eye_in_hand`, end-effector position and quaternion,
  gripper finger joints, joint positions, the noise-free raw wrist force / torque and the sensor orientation.
- **Per tick**: the normalized OSC action the environment actually received (float64), the OSC stiffness kp (6; the
  primitives switch between 150 and 40 and set it per axis when inserting), whether the grasp weld is active, whether the tick
  went through `ForceSim.step` (which runs the task's step hook), and the name of the running primitive (analysis only).
- **Per episode**: the force zero point tared at reset, the residual per-episode bias, the noise settings, the scene the expert
  saw at its first `scene` call (noisy object poses), seed and metadata.

The recorder never draws random numbers, so recording does not change an episode. Force noise is added when the data is
loaded, with the same model as the simulator: `F_world = force_sign * R (f_raw + bias_ep + N(0, sigma) - b0)`, sigma 0.5 N
for force and 0.05 N m for torque, per-episode bias sigma 0.3 N, with a generator seeded by the episode seed
(`forceharness/data/demos.py: wrench_obs`). On lock_or_stiff the soft arm (kp 150) is used and recording starts after the
environment's pre-grasp, which the evaluation repeats. `scripts/hiddenphys/replay_demos.py` replays an episode from its seed;
9 episodes across com_sort, peg_in_hole_hard, door_latch and lock_or_stiff matched in success with 0.00 mm final
end-effector error (weight_sort, wipe_hard and stack_gentle were not replay-checked).

### 2.2 Observations and actions

All learned rows use the same quantities (`forceharness/data/demos.py`):

- **Images**: both cameras, flipped upright. DP resizes 128 -> 96 and crops 88 (random in training, center at test); openpi
  resizes to 224.
- **Proprioception (11)**: end-effector position (3), rotation as the first two columns of the rotation matrix (6; unlike a
  quaternion it does not flip sign near w = 0, which is where a downward-facing gripper sits), the two finger joints (2;
  zeros on wipe_hard, whose tool has no gripper).
- **Wrist wrench (6)**: world frame, zeroed when recording starts (at reset; after the pre-grasp on lock_or_stiff), noisy as
  above. At evaluation the client reads the same quantity from the simulator (`force_sign * net_force_world`); the policy
  runs no primitive, so the zero stays the one in effect when it starts, as in the demos.
- **Scene vector**: the noisy object poses of the first `scene` call (per-episode fixed offsets, x / y position sigma 4 mm,
  15 mm on Peg, yaw sigma 2 degrees), 5 numbers per object (position 3, sin and cos of yaw), objects sorted by name, constant
  for the whole episode. Lengths are 15 / 15 / 0 / 10 / 5 / 10 / 10 for weight_sort, com_sort, wipe_hard, stack_gentle,
  peg_in_hole_hard, door_latch, lock_or_stiff. It holds no door or lever angle. On wipe_hard it is empty, so "images only"
  equals the full input there.
- **Actions**: the normalized OSC delta (7 with the gripper, 6 on wipe_hard) and the OSC stiffness kp (6). Learning the
  stiffness is necessary: the expert lowers stiffness along the contact direction, and a fixed-stiffness policy under the same
  controller cannot reproduce its contact behaviour.

### 2.3 LeRobot dataset for pi0.5 and ForceVLA

`scripts/baselines/pi05/convert_to_lerobot.py` builds **one** multi-task LeRobot dataset (format v2.1, fps 20, local name
`forceharness/hiddenphys_demos` under `$HF_LEROBOT_HOME`; nothing is pushed) from all saved episodes:
`image`, `wrist_image` (upright RGB 128x128, PNG), `state` (26 = proprioception 11 + scene vector zero-padded to 15), `wrench`
(6), `actions` (13 = OSC 7 with a zero at index 6 on wipe_hard + kp 6) and `task` (one short instruction per task, defined in
`PROMPTS`; the instructions contain no numbers). It also writes `meta_fh.json` (per task: scene keys, scene length, longest
demo, osc_dim, prompt, action min / max), which the training copies into every checkpoint and the server reads. The dataset is
about 16 GB of PNG and conversion takes about 30 minutes (both extrapolated from a 21-episode test). If you start from a
released LeRobot export instead of the raw episodes, check that it has exactly this layout and a `meta_fh.json`; otherwise
convert the raw episodes with this script.

## 3. Environments and dependencies

Three Python environments are involved; they talk only through the local socket.

| Environment | Runs | Install |
|---|---|---|
| simulator | `eval_policy.py` (all rows), demo recording and replay | `pip install -e ".[sim]"` (robosuite 1.5.1, mujoco 3.2.7) |
| torch | DP: `train_dp.py`, `dp_server.py` | `pip install -e ".[learn]"`: torch, torchvision, diffusers, opencv. It may be the same environment as the simulator one. The exact versions of the paper's DP runs were not recorded; the code needs torch >= 2.1, torchvision >= 0.13 (`resnet18(weights=None)`), diffusers with `EMAModel(use_ema_warmup, inv_gamma, power)` and `DDIMScheduler` (>= 0.27 is safe) |
| openpi | pi0.5 and ForceVLA: conversion, statistics, training, `pi05_server.py` | openpi's own environment, below |

**openpi.** The scripts call openpi's unmodified code. They were written and run against upstream openpi commit
`215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-25, "docs(droid): fix config search instruction (#1023)"): the development
checkout sits at this commit with comment-only local edits (its code is identical to upstream), and an earlier pi0.5
fine-tuning with the same approach recorded the same commit. The copy on the training machine was set up separately and
its commit was not logged, so treat this pin as the best available record rather than a verified one. The
scripts depend on APIs of this version: `Pi0Config(pi05=True, discrete_state_input=...)`, `compute_norm_stats.main(config,
max_frames=...)`, `checkpoints.save_state(manager, state, data_loader, step)` and `gemma.Module.init(use_adarms=...)`.
openpi pins Python 3.11, jax[cuda12] 0.5.3, flax 0.10.2, orbax-checkpoint 0.11.13, chex 0.1.90, transformers 4.53.2,
torch 2.7.1, numpy < 2 and lerobot from git revision `0cf864870cf29f4738d3ade893e6fd13fbd7cdb5` (dataset format v2.1).

```bash
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git third_party/openpi
git -C third_party/openpi checkout 215abfb217dbac7d5f1273282331b9b1866c0479
(cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync && GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .)
export OPENPI_ROOT=$PWD/third_party/openpi                # default of the scripts; any location works
export FH_OPENPI_PY=$OPENPI_ROOT/.venv/bin/python
export OPENPI_DATA_HOME=/path/to/openpi_cache            # pi05_base, pi0_base and the PaliGemma tokenizer are downloaded here
```

The openpi environment needs nothing else from ForceHarness: the scripts import `forceharness.data.demos` from the repository
root, which uses only numpy and OpenCV.

**flaxformer (ForceVLA only).** `limoe_simple.py` imports `flaxformer.architectures.moe.routing` (TokensChooseScatterRouter,
RouterWeights and the router base classes), `flaxformer.architectures.moe.scatter_utils` (`scatter_nd`),
`flaxformer.components.dense` (`MlpBlock`, the experts) and `flaxformer.types`; transitively `flaxformer.activation_partitioning`
(needs `absl-py`, present in the openpi environment). That is 5 modules of about 1,500 lines out of a 27k-line package.
The ForceVLA code ships flaxformer as source. ForceBench's copy (`baselines/forcevla/runtime_base/flaxformer`, setup.py
version 0.8.8, 2024 license headers) has Python modules byte-identical to the ones the paper's ForceVLA model ran with. One
known change from upstream is in `components/dense.py`: the three AQT imports are wrapped in
`try / except ModuleNotFoundError`, because AQT's legacy modules import TensorFlow even when quantization is off, and ForceVLA
never quantizes. Neither ForceBench nor ForceVLA records the upstream flaxformer commit. Install one of:

```bash
# (a) recommended, identical to what produced the paper's numbers: ForceBench's bundled copy, without its dependencies
#     (its install_requires would pull aqtp, which the guarded import makes unnecessary)
FORCEBENCH_ROOT=${FORCEBENCH_ROOT:-$HOME/forcebench/ForceBench}                         # as in scripts/forcebench/env.sh
git clone https://github.com/forcebench-platform/ForceBench.git "$FORCEBENCH_ROOT"      # skip if already cloned
git -C "$FORCEBENCH_ROOT" checkout 31acf7353d2f723985e6576a5708f67a81707d61
(cd "$OPENPI_ROOT" && uv pip install --no-deps "$FORCEBENCH_ROOT/baselines/forcevla/runtime_base/flaxformer")

# (b) upstream flaxformer, then apply the same guard to flaxformer/components/dense.py before installing with --no-deps:
#       try:
#         from aqt.jax_legacy.jax import flax_layers as aqt_flax_layers
#         from aqt.jax_legacy.jax import quant_config as aqt_config
#         from aqt.jax_legacy.jax import quantization as aqt
#       except ModuleNotFoundError:
#         aqt_flax_layers = aqt_config = aqt = None
#     Without the guard you need aqtp and TensorFlow importable in the openpi environment (untested).
```

`forcevla_fh.py` also puts `scripts/baselines/forcevla/third_party/flaxformer_pkg` on `sys.path`, so placing a flaxformer
package directory there (`.../flaxformer_pkg/flaxformer/`) works without installing it. To confirm an installation matches
the paper's code, compare SHA-256 of the five modules:

| Module | SHA-256 |
|---|---|
| `flaxformer/components/dense.py` | `ba10194586416cd35c8e29757434084000e64f0678fa8e7844062ceffbc4043b` |
| `flaxformer/architectures/moe/routing.py` | `2ff28e4cd4cdfcf98844692f0108ddcae0fbec4fec348d4fd960e1a5d063fd6b` |
| `flaxformer/architectures/moe/scatter_utils.py` | `38622707a4ee0a354e7e5ea8efebbbf5262ffac3435a873408e43ed58e2d203d` |
| `flaxformer/types.py` | `979d48e859e6f5e70cd4ece634b1e956ab919f21c43b7247d6bd8f66b3d99a1e` |
| `flaxformer/activation_partitioning.py` | `c2dd8525d553253dae958eef1b3ad9a0c65bec2fddd4df53c6b08471cb6a97fe` |

## 4. Training recipes

Every model is trained once (one training seed) and evaluated at its **last** checkpoint, with no selection on any seed.

### 4.1 Diffusion Policy (`train_dp.py`)

A compact rewrite of the CNN Diffusion Policy of Chi et al. (`diffusion_unet_hybrid_image_policy`).

| Item | Value |
|---|---|
| Vision | one ResNet18 per camera, not pretrained, BatchNorm replaced by GroupNorm; spatial softmax with 32 keypoints; Linear to 64 |
| Images | 128 -> 96 -> random crop 88 (original: 84 -> 76) |
| Low-dim input | proprioception 11 (+ wrench 6 for `*force*`) (+ scene vector unless `*noscene*`), min-max normalized to [-1, 1] |
| Horizons | 2 observed frames, 16 predicted actions, 8 executed |
| Denoiser | conditional 1D U-Net, down dims 256 / 512 / 1024, kernel 5, FiLM conditioning, diffusion-step embedding 128 |
| Diffusion | DDPM with 100 steps for training (squaredcos_cap_v2, clip_sample, epsilon prediction); **DDIM with 16 steps** at inference (original: 100 DDPM steps; about 6x faster) |
| Action | OSC delta 7 (6 on wipe_hard) + stiffness kp 6, min-max normalized; the client clamps kp to the training range |
| Data | per task the first 200 saved episodes by seed (all 185 on Peg, all 197 on Latch); 50 for `dp50*`; one sample per action tick |
| Optimization | 40,000 steps, batch 64, AdamW lr 1e-4, betas (0.95, 0.999), weight decay 1e-6, cosine schedule with 500 warmup steps, EMA (power 0.75, max decay 0.9999), TF32 matmuls |
| Checkpoint | EMA weights, `ema_last.pt` (about 359 MB, overwritten every 10,000 steps); `config.json` records every setting and the normalization ranges |

### 4.2 pi0.5 (`pi05_fh.py`)

Full fine-tuning of `pi05_base` with unmodified openpi; one model for all 7 tasks, each task with one fixed prompt.

| Item | Value |
|---|---|
| Inputs | base camera -> `base_0_rgb`, wrist camera -> `left_wrist_0_rgb`, `right_wrist_0_rgb` zero-filled and masked; state = proprioception 11 + scene 15 (f0), + wrench 6 (f2), **discretized into 256 levels and written into the prompt** (`discrete_state_input=True`; openpi's `pi05_libero` uses False, under which the model would not see proprioception or force); `max_token_len` 200 (prompts are 106-143 tokens) |
| Actions | 13 dims, no DeltaActions (already deltas; kp is an absolute level); action horizon 10 (0.5 s), 5 executed before replanning |
| Normalization | openpi quantiles (q01 / q99), then `fix_norm_stats`: kp dims use the data [min, max] (discrete levels 150 / 40 / 10 otherwise collapse a quantile range to a point and normalize to about +-1e8); OSC dims keep the quantile center and widen until every training value has \|z\| <= 5 (a rare large lever-turning action would otherwise reach +-40); almost constant state dims use [min, max]. Raw statistics are kept in `norm_stats.raw.json`. State values outside the range become the edge tokens (-1 / 255), as in openpi; this affects about 1-2% of the f2 wrench values |
| Optimization | openpi `pi05_libero` defaults except the batch: batch 32, 30,000 steps, AdamW (b1 0.9, b2 0.95, weight decay 1e-10, gradient clipping 1.0), 10,000 warmup steps to 5e-5 then constant, EMA 0.999, seed 42 |
| Checkpoints | EMA parameters only, cast to bf16 (about 5-6 GB), every 10,000 steps and at step 29,999 (the evaluated one); `--full-state` keeps openpi's resumable state instead |

Batch 32 for 30,000 steps passes over the 726,867 frames only about 1.3 times (`pi05_libero`: about 28 passes over LIBERO at
batch 256), and warmup takes the first 10,000 steps, so both pi0.5 rows may be undertrained.

### 4.3 ForceVLA (`forcevla_fh.py`)

ForceVLA's model code (pi0 backbone; the 6-D wrench projected by `force_in_proj` to a token; a LIMoE block with 4 experts and
top-1 routing over the VLM prefix output and the force token, added to the action expert's output) on the same dataset,
transforms and evaluation as pi0.5 + force. The 32-D state is proprioception 11 + scene 15 (into `state_proj`) + wrench 6
(into `force_in_proj`); the state is a continuous token, not prompt text.

| Same as ForceBench's recorded recipe | Different (controlled variant) |
|---|---|
| start from `pi0_base`, all parameters trained; LIMoE and the force projection randomly initialized | one multi-task model for 30,000 steps (ForceVLA paper, App. B) instead of one model per task for 10,000 steps |
| global batch 64; AdamW with openpi defaults; warmup 1,000 steps to 2.5e-5, cosine decay to 2.5e-6 at 30,000 steps; EMA 0.99 | our state / action dimensions and meaning (ForceBench: 7-D proprioception, 7-D absolute TCP poses converted to deltas), so no DeltaActions |
| z-score normalization; seed 42 | data parallel on 2 GPUs instead of FSDP (same global batch) |
| | action std widened by the pi0.5 rule to \|z\| <= 5 (dims 0, 3, 4, 5); almost constant state dims get std 1 |

Checkpoints keep bf16 EMA parameters only (about 6.6 GB). `prepare-assets` derives the z-score statistics from the raw
statistics of `pi05_fh_f2` (same data, same transform chain), so `pi05_fh.py compute_norm_stats --config pi05_fh_f2` must run
first. Set `FH_PI0_BASE` to a local copy of `pi0_base/params` (openpi caches it as
`$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params`) to let `check-model` compare parameter names and shapes.

## 5. Compute used

| Model | Training | Notes |
|---|---|---|
| DP (each variant, each task) | 40,000 steps in about 1-1.7 h on one NVIDIA RTX A5000 (24 GB) | the 14 DP and DP + force models took about 22 GPU-hours; several runs can share one 24 GB GPU (four ran concurrently) |
| pi0.5 (each config) | about 1.3-1.4 s per step on one NVIDIA H200 (141 GB), about 11-12 h for 30,000 steps | full fine-tuning needs more than 70 GB; with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` the process reserves about 130 GB |
| ForceVLA | about 1.3 s per step on two H200, 11.1 h (about 22 GPU-hours) | plus a short smoke run |

Evaluation runs MuJoCo on CPU with EGL rendering. A pi0.5 server on one RTX A5000 was ready about 36 s after start and answered
in about 0.15 s per call after warm-up; one server shared by all clients evaluated 7 tasks x 20 seeds in about 2.5-2.7 h.
The datasets library's Arrow cache of the LeRobot dataset needs about 16-22 GB of disk per build.

## 6. Evaluation protocol

`scripts/baselines/dp/eval_policy.py` evaluates every learned row, with the same seeds and the same physical success check
(`TaskSpec.success`) as the harness and the classical rows:

- **Seeds**: test seeds 9100-9149. Table 5 uses 9100-9119 for every row (episodes are paired by seed); DP and DP + force were
  also run on all 50. A seed fixes the hidden variable and the layout.
- **Episode settings**: the `episode` section of `configs/hiddenphys/expert.yaml` (stiffness 150, scene noise 4 mm / 2 degrees,
  force noise 0.5 N with bias 0.3 N, force sign; contact threshold 3 N and force cap 25 N, used only by the lock_or_stiff
  pre-grasp). Turn and tool-call budgets do not apply to learned rows.
- **Loop**: every Ta ticks the client sends the latest To frames (DP: To 2, Ta 8; openpi models: To 1, Ta 5), receives a chunk
  and executes Ta actions. Each tick it sets the OSC stiffness from the kp part of the action (clamped to the training range),
  releases the grasp weld when the gripper command turns from closed to open (as the release primitive does), steps the
  environment, reads the wrench, updates the peak and runs the task's step hook (latch logic, damage).
- **Horizon**: a fixed number of ticks, 1.5 x the longest demonstration in the model's training set, after which the final
  state is judged. Episodes never stop early, so a policy that succeeds and then undoes it fails.
- **No safety stop**: learned policies drive the controller directly, without the 25 N stop that every primitive (and so every
  harness, expert and program row) has. The results record the peak wrench (`peak_wrench_N`) and the task's damage flag
  (`damaged`: on door_latch and lock_or_stiff, a wrist reaction above 15 N while the gripper is commanded closed, on
  lock_or_stiff counted from 1 s after closing; a damaged episode is a failure), and the tables report damage and peak force
  for every row.
- **lock_or_stiff**: the environment's pre-grasp (primitives on the soft arm, as in model episodes and demos) grasps the handle
  before the policy starts. Learned rows give no locked / stiff verdict, so a policy that does nothing passes every locked door.
- **Distribution shift**: `weight_sort_ood` (masses 0.30 / 0.45 kg instead of 0.15 / 0.60), `com_sort_ood` (center-of-mass
  offset 2.5 cm instead of 4 cm), `wipe_hard_ood` (table 3 cm lower on average), `peg_in_hole_hard_ood` (hole position noise
  25 mm instead of 15 mm). Models are not retrained, and multi-task models receive the original task's prompt
  (`pi05_server.py` maps `<task>_ood` to `<task>`; `eval_policy.py --policy-task` does the same from the client side).
- **Outputs**: one line per episode in `<run>/results.jsonl` (success, ticks, horizon, policy calls, simulated time, peak
  wrench, damage, hidden-variable truth, checkpoint and step); reruns skip episode ids already present.
  `python scripts/hiddenphys/learned_table.py --run runs/learned` summarizes a results root (Wilson 95% intervals, median peak
  force, episodes above 25 N, damage, lock_or_stiff split by door type).

## 7. Reproducing the rows

Run from the repository root. Paths default to `demos/hiddenphys` (demos), `checkpoints/` (models) and `runs/` (results) and
can be changed with `FH_DEMOS`, `FH_CKPT` and `FH_RUNS`. `FH_TORCH_PY`, `FH_OPENPI_PY` and `FH_PY` select the python of the
torch, openpi and simulator environments. MuJoCo renders with EGL; set `EGL_DEVICE` when the EGL device index of a GPU differs
from its CUDA index. Use a different port for each server that runs at the same time.

**DP and DP + force** (Table 5; the supplement's 50-seed numbers):

```bash
for t in weight_sort com_sort wipe_hard stack_gentle peg_in_hole_hard door_latch lock_or_stiff; do
  for v in dp dp_force; do
    EVAL_SEEDS=9100-9149 bash scripts/baselines/dp/dp_pipeline.sh $t $v 0 47001     # train 40k steps, serve, evaluate
  done
done
```

**DP, images only (+ force)** and **DP, 50 demonstrations (+ force)**: the same loop with variants `dpnoscene`,
`dpnoscene_force`, `dp50`, `dp50_force` (the paper evaluated these on 9100-9119, the default of `EVAL_SEEDS`).

**DP under shift** (50 seeds per task; the shift column of Table 5 counts the first 20 of each of the 4 tasks):

```bash
for t in weight_sort com_sort wipe_hard peg_in_hole_hard; do
  for v in dp dp_force; do bash scripts/baselines/dp/dp_shift_eval.sh $t $v 0 47002; done
done
```

**pi0.5 and pi0.5 + force** (environment variables in `scripts/baselines/pi05/README.md`):

```bash
export HF_LEROBOT_HOME=$PWD/demos/lerobot
$FH_OPENPI_PY scripts/baselines/pi05/convert_to_lerobot.py --demos demos/hiddenphys
for c in pi05_fh_f0 pi05_fh_f2; do
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py compute_norm_stats --config $c --num-workers 10
done
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 CUDA_VISIBLE_DEVICES=0 $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py train \
    --config pi05_fh_f0 --exp-name bs32_30k --steps 30000 --save-interval 10000 --keep-period 10000
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 CUDA_VISIBLE_DEVICES=1 $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py train \
    --config pi05_fh_f2 --exp-name bs32_30k --steps 30000 --save-interval 10000 --keep-period 10000
bash scripts/baselines/pi05/eval_multitask.sh pi05_fh_f0 checkpoints/pi05/checkpoints/pi05_fh_f0/bs32_30k/29999 0 47101 pi05_f0
bash scripts/baselines/pi05/eval_multitask.sh pi05_fh_f2 checkpoints/pi05/checkpoints/pi05_fh_f2/bs32_30k/29999 0 47102 pi05_f2
```

**ForceVLA** (after the pi0.5 statistics above; flaxformer installed, §3):

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/forcevla/forcevla_fh.py prepare-assets
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/forcevla/forcevla_fh.py check-model   # optional
XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 CUDA_VISIBLE_DEVICES=0,1 $FH_OPENPI_PY scripts/baselines/forcevla/forcevla_fh.py train \
    --exp-name bs64_30k --steps 30000 --save-interval 10000 --keep-period 10000
bash scripts/baselines/pi05/eval_multitask.sh forcevla_fh checkpoints/pi05/checkpoints/forcevla_fh/bs64_30k/29999 0 47103 forcevla
```

Results land in `runs/learned/<name>_<task>/` and `runs/learned_shift/<name>_<task>_ood/`. Reference values on seeds
9100-9119 (successes out of 20 per task; damage = damaged handles on Latch and Lock):

| Row | Weight | CoM | Wipe | Stack | Peg | Latch | Lock | Total /140 | Damage | Shift /80 |
|---|---|---|---|---|---|---|---|---|---|---|
| DP | 16 | 17 | 4 | 11 | 15 | 9 | 11 | 83 | 20 | 52 -> 40 |
| DP + force | 18 | 17 | 7 | 6 | 15 | 1 | 10 | 74 | 13 | 57 -> 43 |
| DP, images only | 9 | 17 | 4 | 12 | 14 | 0 | 9 | 65 | 25 | |
| DP, images only + force | 16 | 17 | 5 | 10 | 14 | 3 | 9 | 74 | 16 | |
| DP, 50 demos / + force | | | | | | | | 62 / 53 | | |
| pi0.5 | 9 | 9 | 14 | 12 | 12 | 2 | 11 | 69 | 24 | 44 -> 35 |
| pi0.5 + force | 15 | 10 | 7 | 14 | 10 | 0 | 11 | 67 | 18 | 42 -> 35 |
| ForceVLA | 11 | 10 | 12 | 12 | 15 | 2 | 8 | 70 | 29 | 48 -> 36 |

On all 50 seeds DP is 203/350 and DP + force 204/350. With one training seed per model and 20 episodes per cell, a retrained
model can differ by a few episodes per task; Wilson intervals at n = 20 are about +-20 percentage points.

## 8. Known caveats

- **Information asymmetry in favour of the learned rows.** They see both camera images, and on peg_in_hole_hard the wrist
  camera shows the true hole, while the harness and the expert read the hole position as text with 15 mm noise per axis.
  The noisy scene vector also helped DP without force (83 vs 65 without it, mostly on weight_sort and door_latch) for a reason
  we did not find.
- **Against the learned rows.** Without a safety stop they push through: they can complete Peg by force (DP + force: median
  peak 40 N on all 50 seeds, 28 of 50 episodes above 25 N) and they damage handles on Latch and Lock. Their prompts hold no
  numbers, so under shift they cannot read the new ranges that the model rows read in the instruction.
- **lock_or_stiff** success for learned rows is mostly free: doing nothing passes a locked door, and stiff doors succeed in
  only 3-4 of 25 for DP.
- **Pairing.** Hidden variables match across rows per seed. The learned rows read the scene before the leveling ticks that use
  the same random generator in the harness and expert rows, so on six tasks their scene offsets share the distribution but not
  the values; per-tick force noise is not paired. Learned rows do not level and re-zero the tool (except through the Lock
  pre-grasp).
- **Training budget.** One training seed per model, last checkpoint, no selection; pi0.5 sees the data about 1.3 times and may be
  undertrained.
- **ForceVLA vs pi0.5** differ in more than force: backbone (pi0 vs pi0.5), continuous state vs prompt tokens, a twice larger
  batch (which favours ForceVLA) and normalization. There is no pi0-without-force control, so the gap between these rows
  cannot be credited to force.
- **Controlled variants.** DP: rewritten, DDIM sampling, different image sizes, stiffness in the action, scene vector.
  pi0.5: discrete state in the prompt, batch 32, stiffness, 7 tasks in one model. ForceVLA: the table in §4.3. ForceFlow, the
  other ForceBench fusion designs, GR00T and pi0.5 with 50 demonstrations were not trained.
- **Shared design.** The expert that produced the demonstrations and the ForceHarness primitives share designers, and the
  task instructions are hand-written per task.
- **Environment pins.** The DP environment versions were not recorded; openpi's commit is the best available record (§3).

## 9. Third-party code and licenses

- `scripts/baselines/forcevla/pi0_force.py`: from ForceBench (commit `31acf7353d2f723985e6576a5708f67a81707d61`),
  `baselines/forcevla/runtime_base/src/openpi/models/pi0_force.py`, ForceBench's packaging of ForceVLA
  (https://github.com/ft-robotic/ForceVLA, a fork of openpi). Apache-2.0. Changes, listed in its header, do not affect the
  numerics: `limoe_simple` imported locally, `use_adarms=[False, False]` for this openpi version, an unused `import pdb` removed.
- `scripts/baselines/forcevla/limoe_simple.py`: same source, Apache-2.0 (Copyright 2024 Google LLC header kept); code
  unmodified.
- flaxformer (Apache-2.0) is not redistributed; install it as in §3.
- openpi (Apache-2.0) is used as an external dependency and not modified; `pi05_base` / `pi0_base` weights are downloaded by
  openpi under their own terms.
- `forceharness/baselines/diffusion_policy.py` is our compact reimplementation of the CNN Diffusion Policy of Chi et al.;
  its module docstring lists where it follows and where it departs from the original.
