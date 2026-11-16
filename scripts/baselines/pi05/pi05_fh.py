"""pi0.5 baseline (openpi, JAX): multi-task full fine-tuning on the 7 HiddenPhys tasks, two configs:
  pi05_fh_f0  state = proprioception 11 + scene vector (ForceBench F0, no force)
  pi05_fh_f2  state = proprioception 11 + scene vector + wrist wrench 6 (ForceBench Prompt Fusion / F2: force is appended to
              the state, and pi0.5 discretizes the state into tokens written into the prompt)

openpi itself is not modified: this file registers the data transforms and TrainConfigs in openpi's config table and then calls
openpi's own compute_norm_stats / train scripts (found under $OPENPI_ROOT/scripts). The dataset comes from
convert_to_lerobot.py in this directory.

Usage (openpi environment, from the repository root; environment variables in README.md of this directory):
  python scripts/baselines/pi05/pi05_fh.py compute_norm_stats --config pi05_fh_f2   # quantile stats -> $FH_PI05_WORK/assets/pi05_fh_f2/<repo_id>/
  python scripts/baselines/pi05/pi05_fh.py train --config pi05_fh_f2 --exp-name bs32_30k [--steps N] [--batch-size B] [--overwrite]
  python scripts/baselines/pi05/pi05_fh.py show-config --config pi05_fh_f2
  python scripts/baselines/pi05/pi05_fh.py check-batch --config pi05_fh_f2          # one batch from openpi's data loader, print shapes
                                                                                    # (no model is built; runs on CPU)

Key settings (reasons in README.md):
  - start from pi05_base, full fine-tuning; action_horizon 10 (the evaluation client replans every Ta = 5 ticks); batch 32;
    steps and learning rate default to openpi's pi05_libero (all can be changed on the command line)
  - discrete_state_input=True (openpi's pi05_libero uses False, with which the model never sees proprioception or force);
    max_token_len stays at the pi0.5 default of 200
  - image slots: image -> base_0_rgb, wrist_image -> left_wrist_0_rgb, right_wrist_0_rgb zero-filled and masked (as for LIBERO)
  - actions are already deltas (normalized OSC delta + stiffness kp), so no DeltaActions (as LIBERO's extra_delta_transform=False)
  - normalization: openpi's compute_norm_stats (quantiles), then degenerate or too narrow ranges are widened (see fix_norm_stats)
  - checkpoints keep only the EMA parameters cast to bf16 (about 5 GB each); --full-state keeps openpi's full training state
    (resumable, about 10x larger)
"""
import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OPENPI = pathlib.Path(os.environ.get("OPENPI_ROOT", str(HERE.parents[2] / "third_party" / "openpi")))   # openpi git checkout
sys.path.insert(0, str(OPENPI / "scripts"))           # openpi's own compute_norm_stats.py / train.py
sys.path.insert(0, str(HERE))

import einops  # noqa: E402
from typing_extensions import override  # noqa: E402

import openpi.models.model as _model  # noqa: E402
import openpi.models.pi0_config as pi0_config  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402
import openpi.transforms as _transforms  # noqa: E402

from convert_to_lerobot import ACTION_DIM, OSC_DIM_MAX, REPO_ID as _DEFAULT_REPO_ID  # noqa: E402

REPO_ID = os.environ.get("FH_REPO_ID", _DEFAULT_REPO_ID)
WORK = pathlib.Path(os.environ.get("FH_PI05_WORK", str(HERE.parents[2] / os.environ.get("FH_CKPT", "checkpoints") / "pi05")))
ASSETS_BASE = os.environ.get("FH_PI05_ASSETS", str(WORK / "assets"))
CKPT_BASE = os.environ.get("FH_PI05_CKPT", str(WORK / "checkpoints"))
VARIANTS = {"pi05_fh_f0": False, "pi05_fh_f2": True}      # config name -> whether the wrist wrench is appended to the state
PI05_BASE = "gs://openpi-assets/checkpoints/pi05_base/params"   # openpi caches the download under $OPENPI_DATA_HOME


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class FhInputs(_transforms.DataTransformFn):
    """Dataset / inference-server keys -> the three pi0.5 image slots + state + actions + prompt (after
    libero_policy.LiberoInputs). At inference pi05_server.py sends the same keys: observation/image, observation/wrist_image
    (upright RGB uint8 HWC), observation/state (proprioception 11 + zero-padded scene vector), observation/wrench (6), prompt."""

    model_type: _model.ModelType
    use_force: bool = False

    def __call__(self, data: dict) -> dict:
        base = _parse_image(data["observation/image"])
        wrist = _parse_image(data["observation/wrist_image"])
        state = np.asarray(data["observation/state"], dtype=np.float32)
        if self.use_force:                                   # f2: wrench appended to the state -> normalized -> tokens in the prompt
            state = np.concatenate([state, np.asarray(data["observation/wrench"], dtype=np.float32)], axis=-1)
        inputs = {
            "state": state,
            "image": {"base_0_rgb": base, "left_wrist_0_rgb": wrist, "right_wrist_0_rgb": np.zeros_like(base)},
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            p = data["prompt"]
            inputs["prompt"] = p.decode("utf-8") if isinstance(p, bytes) else p
        return inputs


@dataclasses.dataclass(frozen=True)
class FhOutputs(_transforms.DataTransformFn):
    """The model outputs 32 dims (pi0.5's action dimension); keep the first 13: OSC 7 (index 6 is padding on wipe_hard) + kp 6."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :ACTION_DIM])}


@dataclasses.dataclass(frozen=True)
class FhDataConfig(_config.DataConfigFactory):
    use_force: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> _config.DataConfig:
        repack = _transforms.Group(inputs=[_transforms.RepackTransform({
            "observation/image": "image",
            "observation/wrist_image": "wrist_image",
            "observation/state": "state",
            "observation/wrench": "wrench",
            "actions": "actions",
            "prompt": "prompt",
        })])
        data_transforms = _transforms.Group(
            inputs=[FhInputs(model_type=model_config.model_type, use_force=self.use_force)],
            outputs=[FhOutputs()],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=_config.ModelTransformFactory()(model_config),
        )


def make_config(name: str = "pi05_fh_f2", exp_name: str = "dev", steps: int = 30_000, batch_size: int = 32, fsdp: int = 1,
                action_horizon: int = 10, peak_lr: float = 5e-5, warmup: int = 10_000, decay_steps: int = 1_000_000,
                decay_lr: float = 5e-5, num_workers: int = 8, save_interval: int = 5000, keep_period: int = 5000,
                log_interval: int = 100, seed: int = 42, max_token_len: int = 200) -> _config.TrainConfig:
    """Defaults = openpi's pi05_libero (peak_lr 5e-5, 10k warmup steps, decay to 5e-5, i.e. constant after warmup, 30k steps,
    AdamW with gradient clipping 1.0, EMA 0.999), except batch 32 instead of 256 and discrete_state_input switched on."""
    if name not in VARIANTS:
        raise ValueError(f"unknown config {name}; choose from {list(VARIANTS)}")
    return _config.TrainConfig(
        name=name,
        exp_name=exp_name,
        model=pi0_config.Pi0Config(pi05=True, action_horizon=action_horizon, discrete_state_input=True,
                                   max_token_len=max_token_len),
        data=FhDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True), use_force=VARIANTS[name]),
        weight_loader=weight_loaders.CheckpointWeightLoader(PI05_BASE),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=warmup, peak_lr=peak_lr, decay_steps=decay_steps,
                                                   decay_lr=decay_lr),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        ema_decay=0.999,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=steps,
        log_interval=log_interval,
        save_interval=save_interval,
        keep_period=keep_period,
        fsdp_devices=fsdp,
        seed=seed,
        wandb_enabled=False,
        assets_base_dir=ASSETS_BASE,
        checkpoint_base_dir=CKPT_BASE,
    )


def register(cfg: _config.TrainConfig) -> None:
    _config._CONFIGS_DICT[cfg.name] = cfg


# ----------------------------------------------------------------------------------------------- normalization statistics fix
ACT_Z_MAX = 5.0     # actions: widen each quantile range until every training value normalizes to |z| <= 5
ACT_MIN_HALF = 1e-3  # actions: half-width at least 0.001 (OSC deltas are in normalized [-1, 1] units)
STATE_MIN_RANGE = 1e-4  # state: if q99 - q01 is narrower than this (almost constant), use [min, max] instead


def dataset_meta(repo_id: str = REPO_ID) -> dict:
    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

    p = HF_LEROBOT_HOME / repo_id / "meta_fh.json"
    if not p.exists():
        raise FileNotFoundError(f"{p} not found — run convert_to_lerobot.py first (same HF_LEROBOT_HOME)")
    return json.loads(p.read_text())


def fix_norm_stats(cfg: _config.TrainConfig) -> None:
    """Run after compute_norm_stats: keep the raw statistics as norm_stats.raw.json, then edit q01 / q99 in norm_stats.json
    (pi0.5 does not use mean / std):
      - actions, kp (6 dims): discrete stiffness levels (150 / 40 / 10). Quantiles can collapse to a point (q01 = q99 when a
        level has < 1% of the samples), and the collapsed level then normalizes to about +-1e8, which shows up as loss spikes.
        The kp dims use the data [min, max] instead.
      - actions, OSC dims: keep the center of the quantile range and widen its half-width to
        max(half-width, max|x - center| / ACT_Z_MAX, ACT_MIN_HALF), so every training value has |z| <= 5. Rare large actions
        (e.g. turning a door lever, |x| 0.2 against the usual +-0.01) are then not blown up to +-40.
      - state: only fix almost constant dims (q99 - q01 < 1e-4) -> [min, max]; the others keep openpi's quantiles (values outside
        [q01, q99] become the edge tokens -1 / 255, as in the official pi0.5).
    min / max come from meta_fh.json written by convert_to_lerobot.py (per-frame statistics over the whole dataset). Also copies
    meta_fh.json into the assets (the inference server reads it)."""
    from openpi.shared import normalize as _normalize

    out_dir = pathlib.Path(cfg.assets_dirs) / REPO_ID
    path = out_dir / "norm_stats.json"
    raw = _normalize.load(out_dir)
    shutil.copyfile(path, out_dir / "norm_stats.raw.json")
    meta = dataset_meta()
    stats = {k: dataclasses.replace(v) for k, v in raw.items()}

    a = stats["actions"]
    q01, q99 = np.array(a.q01, np.float64), np.array(a.q99, np.float64)
    amin, amax = np.array(meta["actions_min"], np.float64), np.array(meta["actions_max"], np.float64)
    for d in range(len(q01)):
        lo, hi = q01[d], q99[d]
        if d >= OSC_DIM_MAX:                                     # kp
            nlo, nhi = (amin[d], amax[d]) if amax[d] - amin[d] > 1e-6 else (lo, hi)
        else:
            c, half = (lo + hi) / 2, (hi - lo) / 2
            nhalf = max(half, max(abs(amin[d] - c), abs(amax[d] - c)) / ACT_Z_MAX, ACT_MIN_HALF)
            nlo, nhi = c - nhalf, c + nhalf
        if abs(nlo - lo) > 1e-9 or abs(nhi - hi) > 1e-9:
            print(f"  actions[{d}]: q01..q99 [{lo:.5g}, {hi:.5g}] -> [{nlo:.5g}, {nhi:.5g}]  (data min/max [{amin[d]:.5g}, {amax[d]:.5g}])")
        q01[d], q99[d] = nlo, nhi
    stats["actions"] = dataclasses.replace(a, q01=q01.astype(np.float32), q99=q99.astype(np.float32))

    s = stats["state"]
    q01, q99 = np.array(s.q01, np.float64), np.array(s.q99, np.float64)
    smin = np.array(meta["state_min"] + (meta["wrench_min"] if len(q01) > len(meta["state_min"]) else []), np.float64)
    smax = np.array(meta["state_max"] + (meta["wrench_max"] if len(q01) > len(meta["state_max"]) else []), np.float64)
    assert len(smin) == len(q01), (len(smin), len(q01))
    for d in range(len(q01)):
        if q99[d] - q01[d] < STATE_MIN_RANGE and smax[d] - smin[d] >= STATE_MIN_RANGE:
            print(f"  state[{d}]: q01..q99 [{q01[d]:.5g}, {q99[d]:.5g}] -> data [min, max] [{smin[d]:.5g}, {smax[d]:.5g}]")
            q01[d], q99[d] = smin[d], smax[d]
    stats["state"] = dataclasses.replace(s, q01=q01.astype(np.float32), q99=q99.astype(np.float32))
    _normalize.save(out_dir, stats)
    (out_dir / "meta_fh.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print(f"[norm] fixed stats → {path} (raw: norm_stats.raw.json); meta_fh.json copied")


# ----------------------------------------------------------------------------------------------- training
MIN_FREE_GB = float(os.environ.get("FH_CKPT_MIN_FREE_GB", "9"))   # one bf16 checkpoint is about 6.2 GB


def _wait_for_space(directory, step):
    """Before saving, check the free space on the checkpoint disk (on a quota-managed file system this is the quota). If it
    is below MIN_FREE_GB, log "[fh] LOW DISK" and re-check every 5 minutes for up to 3 hours, so space can be freed meanwhile;
    save anyway afterwards."""
    import logging
    import time

    for i in range(37):
        free = shutil.disk_usage(str(directory)).free / 2**30
        if free >= MIN_FREE_GB:
            return
        logging.warning("[fh] LOW DISK before saving step %d: %.1f GB free in %s (want %.0f), waiting %d/36",
                        step, free, directory, MIN_FREE_GB, i + 1)
        time.sleep(300)


def _save_params_only(checkpoint_manager, state, data_loader, step):
    """Replacement for openpi's save_state: store only the assets (normalization stats + meta_fh.json) and the EMA
    parameters in bf16, without optimizer state. openpi inference loads parameters in bf16 anyway
    (policy_config.create_trained_policy), so serving is unaffected; the cost is that training cannot be resumed."""
    import logging

    import jax
    import jax.numpy as jnp

    import openpi.training.checkpoints as _checkpoints
    from openpi.shared import array_typing as at

    # orbax 0.11 keeps the failure of the previous asynchronous save and raises it only in wait_until_finished() at the start
    # of the next save(). Without this, one failed intermediate save (e.g. a full disk) lets training continue and then crashes
    # the final save, losing the final parameters. Catch and log it here so later saves are unaffected.
    try:
        checkpoint_manager.wait_until_finished()
    except Exception:  # noqa: BLE001
        logging.exception("[fh] previous async checkpoint save failed; continuing with step %d", step)
    _wait_for_space(checkpoint_manager.directory, step)

    def save_assets(directory):
        data_config = data_loader.data_config()
        if data_config.norm_stats is not None and data_config.asset_id is not None:
            _checkpoints._normalize.save(directory / data_config.asset_id, data_config.norm_stats)
            src = pathlib.Path(ASSETS_BASE) / _CURRENT["name"] / data_config.asset_id / "meta_fh.json"
            if src.exists():
                shutil.copyfile(src, pathlib.Path(directory) / data_config.asset_id / "meta_fh.json")

    with at.disable_typechecking():
        params = state.ema_params if state.ema_params is not None else state.params
        params = jax.tree.map(lambda x: x.astype(jnp.bfloat16) if jnp.issubdtype(x.dtype, jnp.floating) else x, params)
    checkpoint_manager.save(step, {"assets": save_assets, "params": {"params": params}})


_CURRENT = {"name": None}


def check_batch(cfg: _config.TrainConfig, batch_size: int) -> None:
    """Take one batch from openpi's data loader (the training transform chain: repack -> FhInputs -> quantile normalization ->
    resize 224 / tokenize / pad), print the shapes, and decode the tokenized prompt back to text to confirm that the discretized
    state tokens are in the prompt. Builds no model and loads no weights."""
    import jax

    import openpi.models.tokenizer as _tokenizer
    import openpi.training.data_loader as _data_loader

    cfg = dataclasses.replace(cfg, batch_size=batch_size)
    loader = _data_loader.create_data_loader(cfg, shuffle=True, num_batches=1)
    obs, actions = next(iter(loader))
    print(f"[check] devices {jax.devices()}  batch {batch_size}")
    for k, v in obs.images.items():
        print(f"  image[{k}]: {v.shape} {v.dtype} range [{float(v.min()):.3f}, {float(v.max()):.3f}]  mask {np.asarray(obs.image_masks[k]).tolist()[:4]}…")
    st = np.asarray(obs.state)
    print(f"  state: {st.shape} {st.dtype}  (first {_state_dim(cfg)} dims are data, the rest is zero padding to 32)  |z| max {np.abs(st).max():.2f}")
    toks, mask = np.asarray(obs.tokenized_prompt), np.asarray(obs.tokenized_prompt_mask)
    print(f"  tokenized_prompt: {toks.shape}  valid tokens min / max {int(mask.sum(1).min())} / {int(mask.sum(1).max())} "
          f"(max_token_len {cfg.model.max_token_len})")
    a = np.asarray(actions)
    print(f"  actions: {a.shape} {a.dtype}  first {ACTION_DIM} dims |z| max {np.abs(a[..., :ACTION_DIM]).max():.2f}, "
          f"padding dims |max| {np.abs(a[..., ACTION_DIM:]).max():.2f}")
    tok = _tokenizer.PaligemmaTokenizer(cfg.model.max_token_len)
    for i in range(min(2, len(toks))):
        text = tok._tokenizer.decode([int(t) for t in toks[i][mask[i]]])
        n_state = len(text.split("State:")[1].split(";")[0].split()) if "State:" in text else 0
        print(f"  prompt[{i}] ({int(mask[i].sum())} tokens, {n_state} state numbers): {text!r}")
        assert n_state == _state_dim(cfg), f"expected {_state_dim(cfg)} state numbers in the prompt, got {n_state}"
    print("[check] ok")


def _state_dim(cfg: _config.TrainConfig) -> int:
    meta = dataset_meta()
    return int(meta["state_dim"]) + (int(meta["wrench_dim"]) if VARIANTS[cfg.name] else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["compute_norm_stats", "train", "show-config", "check-batch"])
    ap.add_argument("--config", required=True, choices=list(VARIANTS))
    ap.add_argument("--exp-name", default="dev")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--fsdp", type=int, default=1, help="FSDP shards over this many devices (1 = no sharding; with several visible GPUs and fsdp=1 training is data parallel)")
    ap.add_argument("--action-horizon", type=int, default=10)
    ap.add_argument("--peak-lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=10_000)
    ap.add_argument("--decay-steps", type=int, default=1_000_000)
    ap.add_argument("--decay-lr", type=float, default=5e-5)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--save-interval", type=int, default=5000)
    ap.add_argument("--keep-period", type=int, default=5000)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-frames", type=int, default=None, help="compute_norm_stats: use only this many randomly sampled frames (default: all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--resume", action="store_true", help="resume training (only from checkpoints saved with --full-state)")
    ap.add_argument("--full-state", action="store_true", help="save openpi's full training state (resumable, about 50 GB per checkpoint)")
    a = ap.parse_args()
    cfg = make_config(a.config, a.exp_name, a.steps, a.batch_size, a.fsdp, a.action_horizon, a.peak_lr, a.warmup, a.decay_steps,
                      a.decay_lr, a.num_workers, a.save_interval, a.keep_period, a.log_interval, a.seed)
    cfg = dataclasses.replace(cfg, overwrite=a.overwrite, resume=a.resume)
    register(cfg)
    _CURRENT["name"] = cfg.name
    if a.cmd == "show-config":
        print(cfg)
    elif a.cmd == "compute_norm_stats":
        import compute_norm_stats

        compute_norm_stats.main(cfg.name, max_frames=a.max_frames)
        fix_norm_stats(cfg)
    elif a.cmd == "check-batch":
        check_batch(cfg, batch_size=a.batch_size)
    else:
        import train as openpi_train

        import openpi.training.checkpoints as _checkpoints

        if not a.full_state:
            _checkpoints.save_state = _save_params_only
            openpi_train._checkpoints.save_state = _save_params_only
        openpi_train.main(cfg)


if __name__ == "__main__":
    main()
