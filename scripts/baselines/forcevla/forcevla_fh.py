"""ForceVLA baseline: pi0 + LIMoE, multi-task full fine-tuning on the 7 HiddenPhys tasks.

The model code is ForceVLA's as packaged in ForceBench (see the header of pi0_force.py). Data, data transforms and evaluation
are shared with pi0.5 + force (pi05_fh_f2):
  state 32 = proprioception 11 + noisy scene vector 15 + wrist wrench 6 (assembled by FhInputs(use_force=True)). In the model
  the first 26 dims go through state_proj and the last 6 through force_in_proj -> LIMoE.
  actions 13 (OSC 7 + kp 6), zero-padded to 32; action_horizon 10 and replanning every 5 ticks, as for pi0.5.

Same as the recipe recorded by ForceBench (recipe_catalog.json): start from pi0_base, full fine-tuning, global batch 64, AdamW
with default settings, warmup over 1000 steps to 2.5e-5 then cosine decay to 2.5e-6 at 30k steps, EMA 0.99, z-score
normalization, seed 42.
Different (a controlled variant, stated as such in the paper):
  - one multi-task model trained for 30k steps (the multi-task setting of the ForceVLA paper, App. B; the decay horizon is
    exactly 30k); ForceBench trains one model per task for 10k steps.
  - state / action dimensions and action meaning are ours (ForceBench: 7-D proprioception and 7-D absolute TCP poses
    converted to deltas), so there is no DeltaActions transform.
  - data parallel on 2 GPUs instead of FSDP: same global batch, mathematically the same, only faster.
  - the action std is widened with the same rule as for pi0.5 until every training value has |z| <= 5 (prepare-assets;
    affects only the 3 rotation dims and dim 0); the raw statistics stay in norm_stats.raw.json.

Usage (openpi environment with flaxformer installed, from the repository root; environment variables as in
scripts/baselines/pi05/README.md):
  python scripts/baselines/forcevla/forcevla_fh.py check-model      # abstract evaluation only (no weights allocated): the model
                                                                    # builds, the loss runs, parameter names match pi0_base
  python scripts/baselines/forcevla/forcevla_fh.py prepare-assets   # norm_stats.json + meta_fh.json of this config, derived
                                                                    # from the raw statistics of pi05_fh_f2
  python scripts/baselines/forcevla/forcevla_fh.py check-batch      # one batch from the data loader, print shapes
  python scripts/baselines/forcevla/forcevla_fh.py train --exp-name bs64_30k
  python scripts/baselines/forcevla/forcevla_fh.py check-policy --ckpt <step dir>   # load a trained checkpoint, run inference on
                                                                                   # dummy observations, time it
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
# this directory (pi0_force, limoe_simple); third_party/flaxformer_pkg, an optional local flaxformer checkout (otherwise the
# installed flaxformer is used, see docs/baselines.md); ../pi05 (pi05_fh, convert_to_lerobot)
sys.path[:0] = [str(HERE), str(HERE / "third_party" / "flaxformer_pkg"), str(HERE.parent / "pi05")]

import pi05_fh  # noqa: E402  # also puts openpi's scripts on sys.path and registers the data transforms

import openpi.models.model as _model  # noqa: E402
import openpi.shared.download as download  # noqa: E402
import openpi.training.config as _config  # noqa: E402
import openpi.training.optimizer as _optimizer  # noqa: E402
import openpi.training.weight_loaders as weight_loaders  # noqa: E402

import pi0_force  # noqa: E402

CONFIG_NAME = "forcevla_fh"
PROPRIO_DIM, FORCE_DIM = 26, 6                  # proprioception 11 + scene 15; wrench 6; 32 in total, exactly action_dim
REPO_ID = pi05_fh.REPO_ID
PI0_BASE = os.environ.get("FH_PI0_BASE", "gs://openpi-assets/checkpoints/pi0_base/params")
ACT_Z_MAX = pi05_fh.ACT_Z_MAX
STATE_STD_FLOOR = 1e-4                          # almost constant state dims (empty scene slots) get std 1, so a tiny
                                                # deviation at inference is not divided by 1e-6


@dataclasses.dataclass(frozen=True)
class Pi0GuidanceWeightLoader(weight_loaders.WeightLoader):
    """As ForceBench's weight_loaders.Pi0GuidanceWeightLoader: pi0_base has no LIMoE and no force projection, so these keep
    their random initialization."""

    params_path: str

    def load(self, params):
        loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        return weight_loaders._merge_params(loaded, params, missing_regex=".*lora.*|.*limoe.*|.*force.*")


def make_config(exp_name: str = "dev", steps: int = 30_000, batch_size: int = 64, fsdp: int = 1, action_horizon: int = 10,
                num_workers: int = 16, save_interval: int = 10_000, keep_period: int = 10_000, log_interval: int = 100,
                seed: int = 42) -> _config.TrainConfig:
    return _config.TrainConfig(
        name=CONFIG_NAME,
        exp_name=exp_name,
        model=pi0_force.Pi0_GuidanceConfig(action_horizon=action_horizon, proprio_dim=PROPRIO_DIM, force_dim=FORCE_DIM),
        data=pi05_fh.FhDataConfig(repo_id=REPO_ID, base_config=_config.DataConfig(prompt_from_task=True), use_force=True),
        weight_loader=Pi0GuidanceWeightLoader(PI0_BASE),
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=1_000, peak_lr=2.5e-5, decay_steps=30_000, decay_lr=2.5e-6),
        optimizer=_optimizer.AdamW(),
        ema_decay=0.99,
        batch_size=batch_size,
        num_workers=num_workers,
        num_train_steps=steps,
        log_interval=log_interval,
        save_interval=save_interval,
        keep_period=keep_period,
        fsdp_devices=fsdp,
        seed=seed,
        wandb_enabled=False,
        assets_base_dir=pi05_fh.ASSETS_BASE,
        checkpoint_base_dir=pi05_fh.CKPT_BASE,
    )


def prepare_assets() -> None:
    """z-score statistics: openpi's compute_norm_stats over the same transform chain (repack -> FhInputs(use_force=True))
    gives exactly pi05_fh_f2's norm_stats.raw.json (same data, same 32-D state, same 13-D actions), so the dataset is not
    scanned again (about 20 min). mean / std are used directly, with the two fixes described in the module docstring."""
    from openpi.shared import normalize as _normalize

    base = pathlib.Path(pi05_fh.ASSETS_BASE)
    src, out = base / "pi05_fh_f2" / REPO_ID, base / CONFIG_NAME / REPO_ID
    raw = json.loads((src / "norm_stats.raw.json").read_text())["norm_stats"]
    meta = json.loads((src / "meta_fh.json").read_text())
    s, a = raw["state"], raw["actions"]
    assert len(s["mean"]) == PROPRIO_DIM + FORCE_DIM and len(a["mean"]) == pi05_fh.ACTION_DIM, (len(s["mean"]), len(a["mean"]))
    amean, astd = np.array(a["mean"], np.float64), np.array(a["std"], np.float64)
    amin, amax = np.array(meta["actions_min"], np.float64), np.array(meta["actions_max"], np.float64)
    for d in range(pi05_fh.OSC_DIM_MAX):
        need = max(abs(amin[d] - amean[d]), abs(amax[d] - amean[d])) / ACT_Z_MAX
        if need > astd[d]:
            print(f"  actions[{d}]: std {astd[d]:.5g} -> {need:.5g} (max training |z| {need * ACT_Z_MAX / (astd[d] + 1e-6):.1f} → {ACT_Z_MAX})")
            astd[d] = need
    smean, sstd = np.array(s["mean"], np.float64), np.array(s["std"], np.float64)
    low = np.where(sstd < STATE_STD_FLOOR)[0]
    print(f"  state: {len(low)} almost constant dims {low.tolist()}, std set to 1")
    sstd[low] = 1.0
    stats = {
        "state": _normalize.NormStats(mean=smean.astype(np.float32), std=sstd.astype(np.float32),
                                      q01=np.array(s["q01"], np.float32), q99=np.array(s["q99"], np.float32)),
        "actions": _normalize.NormStats(mean=amean.astype(np.float32), std=astd.astype(np.float32),
                                        q01=np.array(a["q01"], np.float32), q99=np.array(a["q99"], np.float32)),
    }
    out.mkdir(parents=True, exist_ok=True)
    _normalize.save(out, stats)
    shutil.copyfile(src / "norm_stats.raw.json", out / "norm_stats.raw.json")
    shutil.copyfile(src / "meta_fh.json", out / "meta_fh.json")
    print(f"[assets] {out}: norm_stats.json (z-score), norm_stats.raw.json (from pi05_fh_f2), meta_fh.json")


def check_model(cfg: _config.TrainConfig) -> None:
    """Abstract evaluation only (nnx.eval_shape / jax.eval_shape), no weights allocated; takes tens of seconds on a CPU. Checks
    that the model builds, that compute_loss / sample_actions have the right shapes, and that every parameter except LIMoE and
    the force projection exists in pi0_base with the same shape (needs FH_PI0_BASE pointing to a local directory)."""
    import flax.nnx as nnx
    import flax.traverse_util as tu
    import jax
    import jax.numpy as jnp

    model_cfg = cfg.model
    model = nnx.eval_shape(model_cfg.create, jax.random.key(0))
    params = nnx.state(model, nnx.Param).to_pure_dict()
    flat = tu.flatten_dict(params, sep="/")
    n = sum(int(np.prod(v.shape)) for v in flat.values())
    new = {k: v for k, v in flat.items() if "limoe" in k or "force" in k}
    print(f"[model] params {n / 1e9:.3f} B; new (LIMoE + force projection) {sum(int(np.prod(v.shape)) for v in new.values()) / 1e6:.1f} M, {len(new)} arrays")
    obs_spec, act_spec = model_cfg.inputs_spec(batch_size=2)
    loss = nnx.eval_shape(lambda m, o, a: m.compute_loss(jax.random.key(0), o, a, train=True), model, obs_spec, act_spec)
    print(f"[model] compute_loss shape {loss.shape} (expected (2, {model_cfg.action_horizon}))")
    out = nnx.eval_shape(lambda m, o: m.sample_actions(jax.random.key(0), o, num_steps=2), model, obs_spec)
    print(f"[model] sample_actions shape {out.shape} (expected (2, {model_cfg.action_horizon}, {model_cfg.action_dim}))")
    base = pathlib.Path(PI0_BASE)
    if base.exists():
        import orbax.checkpoint as ocp

        with ocp.PyTreeCheckpointer() as ckptr:
            md = ckptr.metadata(base)
        tree = getattr(md, "item_metadata", md)
        tree = getattr(tree, "tree", tree)
        tree = tree["params"] if "params" in tree else tree
        bflat = tu.flatten_dict(dict(tree), sep="/")
        bflat = {k[:-6] if k.endswith("/value") else k: v for k, v in bflat.items()}
        miss = [k for k in flat if k not in bflat and k not in new]
        extra = [k for k in bflat if k not in flat]
        bad = [k for k in flat if k in bflat and tuple(getattr(bflat[k], "shape", ())) != tuple(flat[k].shape)]
        print(f"[model] pi0_base {len(bflat)} arrays; in the model but not in base (and not new) {len(miss)}: {miss[:5]}; only in base {len(extra)}: {extra[:5]}; shape mismatches {len(bad)}: {bad[:5]}")
        assert not miss and not bad and not extra
    else:
        print(f"[model] skipping the comparison with pi0_base ({PI0_BASE} is not a local directory)")


def check_batch(cfg: _config.TrainConfig, batch_size: int = 4) -> None:
    """Take one batch from openpi's data loader (the training transform chain) and print shapes. pi0.5's check_batch looks for
    discretized state tokens in the prompt, which this model does not have, hence a separate function."""
    import openpi.training.data_loader as _data_loader

    loader = _data_loader.create_data_loader(dataclasses.replace(cfg, batch_size=batch_size), shuffle=True, num_batches=1)
    obs, actions = next(iter(loader))
    for k, v in obs.images.items():
        print(f"  image[{k}]: {v.shape} range [{float(v.min()):.2f}, {float(v.max()):.2f}] mask {np.asarray(obs.image_masks[k]).tolist()}")
    st, ac = np.asarray(obs.state), np.asarray(actions)
    print(f"  state {st.shape}: proprioception + scene |z| max {np.abs(st[:, :PROPRIO_DIM]).max():.2f}, wrench |z| max {np.abs(st[:, PROPRIO_DIM:]).max():.2f}")
    mask = np.asarray(obs.tokenized_prompt_mask)
    print(f"  tokenized_prompt {np.asarray(obs.tokenized_prompt).shape}, valid tokens {mask.sum(1).tolist()} (max_token_len {cfg.model.max_token_len})")
    print(f"  actions {ac.shape}: first {pi05_fh.ACTION_DIM} dims |z| max {np.abs(ac[..., :pi05_fh.ACTION_DIM]).max():.2f}, padding dims max {np.abs(ac[..., pi05_fh.ACTION_DIM:]).max():.2f}")
    print("[check] ok")


def check_policy(cfg: _config.TrainConfig, ckpt: str) -> None:
    import time

    from openpi.policies import policy_config

    policy = policy_config.create_trained_policy(cfg, ckpt)
    obs = {"observation/image": np.zeros((128, 128, 3), np.uint8), "observation/wrist_image": np.zeros((128, 128, 3), np.uint8),
           "observation/state": np.zeros(PROPRIO_DIM, np.float32), "observation/wrench": np.zeros(FORCE_DIM, np.float32),
           "prompt": "pick up the cube, judge its weight, and put it in the heavy or light zone"}
    for i in range(3):
        t0 = time.time()
        a = policy.infer(obs)["actions"]
        print(f"[policy] inference {i + 1}: {time.time() - t0:.3f} s, actions {np.asarray(a).shape}, first kp dims {np.asarray(a)[0, 7:10].round(1).tolist()}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["check-model", "prepare-assets", "check-batch", "train", "check-policy", "show-config"])
    ap.add_argument("--exp-name", default="dev")
    ap.add_argument("--steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--fsdp", type=int, default=1, help="1 = no parameter sharding; with 2 visible GPUs training is data parallel")
    ap.add_argument("--action-horizon", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=16)
    ap.add_argument("--save-interval", type=int, default=10_000)
    ap.add_argument("--keep-period", type=int, default=10_000)
    ap.add_argument("--log-interval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    cfg = make_config(a.exp_name, a.steps, a.batch_size, a.fsdp, a.action_horizon, a.num_workers, a.save_interval, a.keep_period,
                      a.log_interval, a.seed)
    cfg = dataclasses.replace(cfg, overwrite=a.overwrite)
    pi05_fh.register(cfg)
    pi05_fh._CURRENT["name"] = cfg.name
    if a.cmd == "show-config":
        print(cfg)
    elif a.cmd == "check-model":
        check_model(cfg)
    elif a.cmd == "prepare-assets":
        prepare_assets()
    elif a.cmd == "check-batch":
        check_batch(cfg)
    elif a.cmd == "check-policy":
        check_policy(cfg, a.ckpt)
    else:
        import train as openpi_train

        import openpi.training.checkpoints as _checkpoints

        _checkpoints.save_state = pi05_fh._save_params_only        # as for pi0.5: EMA parameters in bf16 only (about 6.6 GB each), not resumable
        openpi_train._checkpoints.save_state = pi05_fh._save_params_only
        openpi_train.main(cfg)


if __name__ == "__main__":
    main()
