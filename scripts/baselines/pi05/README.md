# pi0.5 baselines: data conversion, training configs, inference server

Two rows of the HiddenPhys comparison (one multi-task model each, the task is selected by the prompt):

| Row | Config | State (discretized into tokens in the prompt) | Corresponds to |
|---|---|---|---|
| pi0.5 | `pi05_fh_f0` | proprioception 11 + noisy scene vector (zero-padded to 15) = 26 | ForceBench F0 |
| pi0.5 + force | `pi05_fh_f2` | proprioception 11 + scene vector 15 + wrist wrench 6 = 32 | ForceBench Prompt Fusion (F2) |

The 7 tasks are weight_sort, com_sort, wipe_hard, stack_gentle, peg_in_hole_hard, door_latch and lock_or_stiff.
**openpi is not modified**: `pi05_fh.py` registers the data transforms and TrainConfigs and then calls openpi's own
`compute_norm_stats` / `train` scripts. The ForceVLA baseline (`../forcevla/`) reuses the dataset, transforms, assets and
this server. The full recipe, the evaluation protocol and the caveats are in [docs/baselines.md](../../../docs/baselines.md).

| File | What it is |
|---|---|
| `convert_to_lerobot.py` | npz demonstrations -> one multi-task LeRobot v2.1 dataset + `meta_fh.json`; the task prompts `PROMPTS` are defined here |
| `pi05_fh.py` | data transforms + the two TrainConfigs; subcommands `compute_norm_stats` / `train` / `show-config` / `check-batch` |
| `pi05_server.py` | inference server with the protocol of `../dp/dp_server.py`, so `../dp/eval_policy.py` is the client; `--selftest` checks the protocol without a model |
| `eval_multitask.sh` | starts one server and evaluates the 7 tasks and the 4 shifted variants against it |

## 1. Settings

| Item | Value | Why |
|---|---|---|
| Data | `<demo root>/<task>/s*.npz`, about 200 episodes per task (1,423 in total); only the first T observation frames are used (the npz has one more observation than actions; frame i pairs with action tick i) | recording format: `forceharness/backends/mujoco/recorder.py` |
| Dataset | repo_id `forceharness/hiddenphys_demos` (local only, nothing is pushed), fps 20, at `$HF_LEROBOT_HOME/forceharness/hiddenphys_demos` | after openpi's `examples/libero/convert_libero_data_to_lerobot.py` |
| Per-frame fields | `image` (agentview), `wrist_image` (robot0_eye_in_hand): upright RGB 128x128; `state` 26; `wrench` 6; `actions` 13; `task` | images stored as PNG (lerobot `image` type, as for LIBERO); f0 and f2 share one dataset, f2 appends the wrench to the state in its transform |
| Scene vector | 5 per object (position 3 + sin / cos of yaw), objects sorted by name; lengths 15 / 15 / 0 / 10 / 5 / 10 / 10 for the 7 tasks, zero-padded to 15 | same as `train_dp.py`; the prompt tells tasks apart, so padding zeros are unambiguous |
| Actions | normalized OSC delta 7 (6 on wipe_hard, index 6 padded with 0) + OSC stiffness kp 6 = 13; **no DeltaActions** | already deltas; kp is an absolute stiffness level and must not become a delta; as LIBERO's `extra_delta_transform=False` |
| Initialization | `pi05_base`, full fine-tuning | |
| Image slots | image -> base_0_rgb, wrist_image -> left_wrist_0_rgb, right_wrist_0_rgb zero-filled and masked | as for LIBERO |
| State input | `discrete_state_input=True`, `max_token_len` 200 | openpi's `pi05_libero` uses False, with which the model never sees proprioception or force; measured prompts are 106-123 tokens (f0) and 125-143 (f2) |
| Action chunk | action_horizon 10 (0.5 s at 20 Hz); the server reports To = 1, Ta = 5 | the client replans every Ta = 5 ticks |
| Normalization | openpi `compute_norm_stats` (quantiles q01 / q99), then `fix_norm_stats` widens degenerate or too narrow ranges: kp uses the data [min, max]; OSC dims keep their center and widen until every training value has \|z\| <= 5; state fixes only almost constant dims. The raw statistics are kept in `norm_stats.raw.json` | kp takes discrete levels (150 / 40 / 10); a level with < 1% of the samples collapses the quantile range to a point and then normalizes to about +-1e8, which causes loss spikes |
| Hyperparameters | batch 32; otherwise openpi's `pi05_libero`: AdamW (gradient clipping 1.0), peak lr 5e-5, 10k warmup steps, then constant (decay_lr = peak), EMA 0.999, 30k steps, seed 42; all can be changed on the command line | |
| Checkpoints | by default only the EMA parameters cast to bf16 (about 5-6 GB each), every `--save-interval` steps and at the last step; `--full-state` keeps openpi's full state (resumable, about 50 GB each) | inference loads bf16 parameters anyway; `meta_fh.json` is copied with the norm stats into `<ckpt>/assets/forceharness/hiddenphys_demos/`, where the server reads it |

## 2. How to run

All commands run from the repository root. Environment of the openpi side (see docs/baselines.md for installing openpi):

```bash
export OPENPI_ROOT=/path/to/openpi                 # openpi checkout (scripts/compute_norm_stats.py, scripts/train.py)
export FH_OPENPI_PY=$OPENPI_ROOT/.venv/bin/python  # python of the openpi environment
export OPENPI_DATA_HOME=/path/to/openpi_cache      # openpi downloads pi05_base / pi0_base and the PaliGemma tokenizer here
export FH_PI05_WORK=$PWD/checkpoints/pi05          # assets/ and checkpoints/ of these configs (this is the default)
export HF_LEROBOT_HOME=$PWD/demos/lerobot          # where the LeRobot dataset is written and read
export HF_DATASETS_CACHE=/path/with/space/hf_datasets   # the datasets Arrow cache of this dataset is about 20 GB
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90 TF_CPP_MIN_LOG_LEVEL=2
```

Optional overrides: `FH_CKPT` (default `checkpoints`; `FH_PI05_WORK` defaults to `$FH_CKPT/pi05` under the repository
root), `FH_PI05_ASSETS` / `FH_PI05_CKPT` (default `$FH_PI05_WORK/assets` and `$FH_PI05_WORK/checkpoints`), `FH_REPO_ID`
(dataset name, default `forceharness/hiddenphys_demos`), `FH_CKPT_MIN_FREE_GB` (default 9: before each save, wait up to
3 h for this much free disk space), and for ForceVLA `FH_PI0_BASE` (default `gs://openpi-assets/checkpoints/pi0_base/params`).

### 2.1 Data: convert, then statistics (no GPU needed)

```bash
$FH_OPENPI_PY scripts/baselines/pi05/convert_to_lerobot.py --demos demos/hiddenphys
# Statistics must run with the GPU hidden, otherwise JAX pre-allocates 90% of a GPU
for c in pi05_fh_f0 pi05_fh_f2; do
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py compute_norm_stats --config $c --num-workers 10
  CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py check-batch --config $c --num-workers 4   # optional
done
```

`--n-per-task 50` converts only the first 50 episodes per task (data-size ablation); give it another `--repo-id` and
export `FH_REPO_ID` to the same value for training and serving.

### 2.2 Training (one GPU with more than 70 GB per config)

```bash
CUDA_VISIBLE_DEVICES=0 $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py train --config pi05_fh_f0 --exp-name bs32_30k \
    --steps 30000 --save-interval 10000 --keep-period 10000
CUDA_VISIBLE_DEVICES=1 $FH_OPENPI_PY scripts/baselines/pi05/pi05_fh.py train --config pi05_fh_f2 --exp-name bs32_30k \
    --steps 30000 --save-interval 10000 --keep-period 10000
# checkpoints: $FH_PI05_WORK/checkpoints/<config>/bs32_30k/{10000,20000,29999}/
```

The paper's rows use the last step (29999). openpi refuses to start when the checkpoint directory already exists, unless
`--overwrite` (or `--resume` for `--full-state` checkpoints) is given.

### 2.3 Serving and evaluation

The server runs in the openpi environment; the evaluation client runs in the simulator environment of ForceHarness (they can
be on different machines if the port is forwarded, since the client always connects to `127.0.0.1:<port>`).

```bash
FH_PY=/path/to/sim/env/bin/python bash scripts/baselines/pi05/eval_multitask.sh pi05_fh_f2 \
    $FH_PI05_WORK/checkpoints/pi05_fh_f2/bs32_30k/29999 0 47102 pi05_f2
# -> runs/learned/pi05_f2_<task>/results.jsonl and runs/learned_shift/pi05_f2_<task>_ood/results.jsonl
```

Use `pi05_fh_f0` / `pi05_f0` for the row without force. One server handles all clients (one thread per connection,
inference serialized by a lock). The paper's evaluations ran the server on a 24 GB RTX A5000 with a JAX memory fraction
of 0.35 (the script's default; `FH_SERVE_MEM_FRACTION` changes it); the server prints `ready on 127.0.0.1:<port>` after
compiling and two warm-up inferences (about 36 s).

Self-test of the server (no model, no GPU; openpi environment):

```bash
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/pi05/pi05_server.py --selftest \
    --selftest-demos demos/hiddenphys --port 47290
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu $FH_OPENPI_PY scripts/baselines/pi05/pi05_server.py --selftest --transforms \
    --config pi05_fh_f2 --selftest-demos demos/hiddenphys --port 47292
```

## 3. Sizes and compute

| Item | Amount | Basis |
|---|---|---|
| Dataset | 726,867 frames (sum of action ticks of the 1,423 successful demos), about 16 GB of PNG; the datasets Arrow cache adds about 16-22 GB; conversion takes about 30 min | measured: 21 episodes = 10,213 frames = 230 MB in 24 s, extrapolated by frame count |
| Training memory | fits one 141 GB GPU (fsdp 1, batch 32); with `XLA_PYTHON_CLIENT_MEM_FRACTION=0.90` the process reserves about 130 GB | openpi documents > 70 GB for full fine-tuning |
| Training time | about 1.3-1.4 s per step on one H200 -> about 11-12 h per config | measured |
| Checkpoints | about 5-6 GB each (bf16 EMA parameters) | |
| Evaluation | 1.5 x max_T / 5 policy calls per episode (about 75 on lock_or_stiff, up to about 700 on peg_in_hole_hard); about 0.15 s per call after warm-up on an RTX A5000; the 7 tasks x 20 seeds took about 2.5-2.7 h with one server | measured |

Batch 32 for 30k steps passes over the 726,867 frames only about 1.3 times (`pi05_libero`: batch 256 for 30k steps, about
28 passes over LIBERO), and warmup takes the first 10k steps, so both rows may be undertrained. The paper keeps the
`pi05_libero` defaults and states this. Data parallelism over two GPUs (`CUDA_VISIBLE_DEVICES=a,b`, fsdp 1) was not tried
for pi0.5 (it was used for ForceVLA).

## 4. Conventions and pitfalls

- **wipe_hard returns 12 dims**: the client takes OSC from `a[:osc_dim]` and kp from `a[osc_dim:osc_dim+6]`, and slices
  `act_norm` the same way; on wipe_hard osc_dim = 6, so the server drops the padded index 6 (with 13 dims kp would shift by
  one). `act_norm` defaults to the min / max over the whole dataset (kp clamped to [10 or 40, 150]); `--act-norm task` uses
  the task's own range.
- **JPEG orientation**: the recorder and the evaluation client encode robosuite's raw frames (upside down) as BGR, quality 90;
  the conversion and the server both flip them upright and convert to RGB. The self-test checks that the server's decoded image
  equals the dataset frame pixel by pixel.
- **The client's low-dim order differs from the training state**: the client sends proprioception + [wrench] + scene, the
  training state is proprioception + padded scene [+ wrench] (f2); the server reorders. The scene vector follows the
  `scene_keys` order returned by the config call (object names of each task's first demo, sorted).
- The wiping tool has no gripper: `gripper_qpos` in the npz has shape (0, 2) and `proprio` pads zeros.
- State values outside [q01, q99] become the edge tokens -1 / 255 (openpi's behaviour, unchanged); about 1-2% of the f2
  wrench frames fall outside.
- `compute_norm_stats` and `check-batch` initialize JAX: always hide the GPU (`CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu`).
- openpi training writes the JAX compilation cache to `~/.cache/jax`.
- The HF datasets library may rebuild the Arrow cache (about 20 GB) for every new job; delete old caches under
  `$HF_DATASETS_CACHE` when no job is running.

## 5. Checks done without a GPU

With 3 episodes per task:

- Conversion: 21 episodes, 10,213 frames; scene lengths 15 / 15 / 0 / 10 / 5 / 10 / 10 -> padded to 15; state 26,
  wrench 6, actions 13.
- `compute_norm_stats` ran for both configs; the fix widened the OSC yaw range ([-0.006, 0.023] -> [-0.041, 0.057]) and
  4 kp dims.
- `check-batch` (openpi data loader, batch 32): three image slots (32, 224, 224, 3), the third all -1 with mask False;
  state (32, 32); actions (32, 10, 32) with the 19 padding dims all 0; tokenized_prompt (32, 200), which decodes to
  `Task: ..., State: 187 210 ...;\nAction: ` with 26 state numbers for f0 and 32 for f2.
- Server self-test: config / act round trips for f0 / f2 x 7 tasks have the right shapes ((10, 12) on wipe_hard, (10, 13)
  otherwise, float32); with messages built from real demo frames, the state / wrench / both images assembled by the server
  equal the dataset frame exactly; with `--transforms`, against the same frame from the training data loader the tokenized
  prompt is identical token by token and the normalized state and images differ by 0; 4 concurrent clients work.
- `_save_params_only` saved fake parameters on CPU, and openpi's `restore_params` / `load_norm_stats` read them back, with
  `meta_fh.json` in the checkpoint assets.

The GPU path (training, `create_trained_policy`, inference speed, full evaluation episodes) was then exercised by the runs
reported in the paper.
