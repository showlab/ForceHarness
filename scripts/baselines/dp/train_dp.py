"""Train the Diffusion Policy baseline on HiddenPhys demonstrations, one model per task.

Rows: without --force this is "DP"; --force concatenates the 6-D wrist wrench to the low-dim observation ("DP + force");
--no-scene drops the noisy scene vector ("DP, images only", the ForceBench-style input: images + proprioception [+ force]).
Run it from the repository root in an environment with torch, torchvision and diffusers (pip install -e ".[learn]"):
  CUDA_VISIBLE_DEVICES=0 python scripts/baselines/dp/train_dp.py --demos demos/hiddenphys/com_sort \
      --out checkpoints/dp_force_com_sort --force --steps 40000
Observations: two cameras (128 -> 96 -> random crop 88) + proprioception 11 (+ wrist wrench 6) + noisy scene vector
(constant within an episode; --no-scene removes it).
Actions: normalized OSC delta (7-D, 6-D on wipe_hard) + OSC stiffness kp (6-D).
scripts/baselines/dp/dp_pipeline.sh chains training, serving and evaluation; docs/baselines.md has the full recipe.
"""
import _bootstrap  # noqa: F401
import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from forceharness.data.demos import action_matrix, list_demos, load_demo, prep_jpeg, proprio, scene_vector, wrench_obs
from forceharness.baselines.diffusion_policy import DiffusionPolicy, MinMax

cv2.setNumThreads(1)


def prep_img(jpeg_buf, off, i, size, crop, train, rng=None):
    return prep_jpeg(jpeg_buf[off[i]:off[i + 1]], size, crop, train, rng)


class DemoDataset(torch.utils.data.Dataset):
    def __init__(self, files, To, Tp, use_force, use_scene, size, crop, scene_keys=None):
        self.To, self.Tp, self.size, self.crop = To, Tp, size, crop
        self.eps, self.index = [], []
        self.scene_keys = scene_keys
        for f in files:
            d = load_demo(f)
            seed = int(d["meta"]["seed"])
            low = [proprio(d["eef_pos"], d["eef_quat"], d["gripper_qpos"])]
            if use_force:
                low.append(wrench_obs(d, seed))
            if use_scene:
                sv, self.scene_keys = scene_vector(d["meta"].get("scene0"), self.scene_keys)
                low.append(np.repeat(sv[None], len(d["t"]), 0))
            act, adim = action_matrix(d)
            assert len(d["t"]) == len(act) + 1, f"{f}: expected one more observation frame than actions (re-record demos in the old format)"
            e = dict(low=np.concatenate(low, -1).astype(np.float32), act=act, jpeg=d["jpeg"], T=len(act), seed=seed)
            self.adim = adim
            self.cams = list(d["jpeg"].keys())
            k = len(self.eps)
            self.eps.append(e)
            self.index += [(k, t) for t in range(e["T"])]      # one sample per action tick: action t <- observation frame t (and the To-1 before it)
        lows = np.concatenate([e["low"] for e in self.eps])
        acts = np.concatenate([e["act"] for e in self.eps])
        self.low_norm = MinMax(lows.min(0), lows.max(0))
        self.act_norm = MinMax(acts.min(0), acts.max(0))
        self.max_T = max(e["T"] for e in self.eps)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        k, t = self.index[i]
        e = self.eps[k]
        rng = np.random.default_rng((os.getpid() * 1000003 + i * 7919 + int(time.time() * 1e3)) % (2 ** 32))
        oi = [max(0, t - self.To + 1 + j) for j in range(self.To)]
        ai = [min(e["T"] - 1, t + j) for j in range(self.Tp)]
        imgs = {c: np.stack([prep_img(*e["jpeg"][c], j, self.size, self.crop, True, rng) for j in oi]) for c in self.cams}
        low = self.low_norm.norm(e["low"][oi])
        act = self.act_norm.norm(e["act"][ai])
        return {c: torch.from_numpy(v) for c, v in imgs.items()}, torch.from_numpy(low), torch.from_numpy(act)


def collate(batch):
    cams = batch[0][0].keys()
    imgs = {c: torch.stack([b[0][c] for b in batch]) for c in cams}
    return imgs, torch.stack([b[1] for b in batch]), torch.stack([b[2] for b in batch])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demos", required=True, help="demonstration directory of one task (s*.npz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="DP + force: add the 6-D wrist wrench to the low-dim observation")
    ap.add_argument("--no-scene", action="store_true", help="drop the noisy scene vector (ForceBench-style input: images + proprioception + force)")
    ap.add_argument("--n-demos", type=int, default=200, help="use the first n successful demos (sorted by seed); 50 for the data-size ablation")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--crop", type=int, default=88)
    ap.add_argument("--To", type=int, default=2)
    ap.add_argument("--Tp", type=int, default=16)
    ap.add_argument("--Ta", type=int, default=8)
    ap.add_argument("--save-every", type=int, default=10000)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = list_demos(args.demos, args.n_demos)
    ds = DemoDataset(files, args.To, args.Tp, args.force, not args.no_scene, args.size, args.crop)
    print(f"[dp] {len(files)} demos, {len(ds)} samples, low {ds.eps[0]['low'].shape[1]}, act {ds.eps[0]['act'].shape[1]} "
          f"(osc {ds.adim} + kp 6), cams {ds.cams}, max_T {ds.max_T}, device {dev}", flush=True)
    dl = torch.utils.data.DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers, drop_last=True,
                                     collate_fn=collate, persistent_workers=True, pin_memory=True)
    pol = DiffusionPolicy(ds.cams, ds.eps[0]["low"].shape[1], ds.eps[0]["act"].shape[1], args.To, args.Tp, args.crop).to(dev)
    from diffusers.optimization import get_scheduler
    from diffusers.training_utils import EMAModel

    opt = torch.optim.AdamW(pol.parameters(), lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    sched = get_scheduler("cosine", optimizer=opt, num_warmup_steps=500, num_training_steps=args.steps)
    ema = EMAModel(pol.parameters(), decay=0.9999, use_ema_warmup=True, inv_gamma=1.0, power=0.75)
    cfg = dict(cams=ds.cams, low_dim=int(ds.eps[0]["low"].shape[1]), act_dim=int(ds.eps[0]["act"].shape[1]), osc_dim=int(ds.adim),
               To=args.To, Tp=args.Tp, Ta=args.Ta, size=args.size, crop=args.crop, force=args.force, scene=not args.no_scene,
               scene_keys=ds.scene_keys, low_norm=ds.low_norm.state(), act_norm=ds.act_norm.state(), max_T=int(ds.max_T),
               n_demos=len(files), demo_seeds=[e["seed"] for e in ds.eps], demos_dir=args.demos, steps=args.steps, batch=args.batch,
               infer="DDIM 16 steps", train="DDPM 100 steps squaredcos_cap_v2 epsilon")
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1))
    log = (out / "train_log.jsonl").open("a")
    step, t0, run_loss = 0, time.time(), []
    pol.train()
    while step < args.steps:
        for imgs, low, act in dl:
            imgs = {c: v.to(dev, non_blocking=True).float() / 255.0 for c, v in imgs.items()}
            loss = pol.loss(imgs, low.to(dev, non_blocking=True), act.to(dev, non_blocking=True))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            ema.step(pol.parameters())
            step += 1
            run_loss.append(loss.item())
            if step % 200 == 0:
                rec = dict(step=step, loss=float(np.mean(run_loss)), lr=sched.get_last_lr()[0], it_s=round(step / (time.time() - t0), 2))
                log.write(json.dumps(rec) + "\n"); log.flush()
                print(f"  step {step}: loss {rec['loss']:.4f}  {rec['it_s']} it/s", flush=True)
                run_loss = []
            if step % args.save_every == 0 or step == args.steps:
                ema.store(pol.parameters())
                ema.copy_to(pol.parameters())
                torch.save(dict(state_dict=pol.state_dict(), config=cfg, step=step), out / "ema_last.pt")    # about 359 MB each; only the latest is kept
                ema.restore(pol.parameters())
            if step >= args.steps:
                break
    print(f"[dp] done {step} steps in {(time.time() - t0) / 3600:.2f} h → {out}", flush=True)


if __name__ == "__main__":
    main()
