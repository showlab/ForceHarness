"""Diffusion Policy baseline: a compact rewrite of the CNN variant of Chi et al., RSS 2023.

A learned-policy baseline for the HiddenPhys benchmark, outside the three ForceHarness layers: it maps camera images,
proprioception, optionally the 6-D wrist wrench and optionally the noisy scene vector to OSC deltas plus OSC stiffness.
Training is scripts/baselines/dp/train_dp.py, serving scripts/baselines/dp/dp_server.py; see docs/baselines.md.

Matches the original (diffusion_unet_hybrid_image_policy in real-stanford/diffusion_policy):
  - one ResNet18 per camera (not pretrained, BatchNorm replaced by GroupNorm) -> SpatialSoftmax (32 keypoints) -> Linear 64;
  - observation window To=2, prediction horizon Tp=16, executed actions Ta=8; conditional 1D U-Net (down_dims 256/512/1024,
    kernel 5, FiLM conditioning);
  - DDPM training with 100 steps (squaredcos_cap_v2, clip_sample, epsilon prediction); low-dim observations and actions are
    mapped linearly to [-1, 1] by their dataset min / max;
  - EMA (power 0.75, max decay 0.9999), AdamW 1e-4, cosine schedule with 500 warmup steps.
Differs from the original (all recorded in the checkpoint config): inference uses DDIM with 16 steps (the original samples
with 100 DDPM steps; about 6x faster); images are resized 128 -> 96 and randomly cropped to 88 (original 84 -> 76); the
"+ force" variant concatenates the 6-D wrist wrench to the low-dim observation (the "DP + force concatenation" control used
in the FoAR and ForceVLA papers).
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------- normalization
class MinMax:
    """Linear map to [-1, 1]; a dimension with min == max (constant) maps to 0 and denormalizes back to the constant."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray):
        self.lo = np.asarray(lo, np.float32)
        self.hi = np.asarray(hi, np.float32)
        span = self.hi - self.lo
        self.const = span < 1e-6
        self.scale = np.where(self.const, 1.0, 2.0 / np.maximum(span, 1e-6)).astype(np.float32)
        self.offset = np.where(self.const, -self.lo, -1.0 - self.lo * self.scale).astype(np.float32)

    def norm(self, x):
        if isinstance(x, torch.Tensor):
            return x * torch.as_tensor(self.scale, device=x.device) + torch.as_tensor(self.offset, device=x.device)
        return x * self.scale + self.offset

    def denorm(self, y):
        if isinstance(y, torch.Tensor):
            return (y - torch.as_tensor(self.offset, device=y.device)) / torch.as_tensor(self.scale, device=y.device)
        return (y - self.offset) / self.scale

    def state(self) -> Dict[str, list]:
        return dict(lo=self.lo.tolist(), hi=self.hi.tolist())

    @classmethod
    def from_state(cls, s):
        return cls(np.asarray(s["lo"]), np.asarray(s["hi"]))


# ---------------------------------------------------------------------- image encoder
class SpatialSoftmax(nn.Module):
    def __init__(self, c: int, h: int, w: int, num_kp: int = 32):
        super().__init__()
        self.kp = nn.Conv2d(c, num_kp, 1)
        self.num_kp, self.h, self.w = num_kp, h, w
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
        self.register_buffer("pos_x", xs.reshape(1, 1, h * w))
        self.register_buffer("pos_y", ys.reshape(1, 1, h * w))

    def forward(self, x):
        f = self.kp(x).reshape(x.shape[0], self.num_kp, self.h * self.w)
        a = F.softmax(f, dim=-1)
        return torch.cat([(a * self.pos_x).sum(-1), (a * self.pos_y).sum(-1)], -1)        # B x 2K


def _bn_to_gn(m: nn.Module) -> nn.Module:
    for name, ch in m.named_children():
        if isinstance(ch, nn.BatchNorm2d):
            setattr(m, name, nn.GroupNorm(max(1, ch.num_features // 16), ch.num_features))
        else:
            _bn_to_gn(ch)
    return m


class ImageEncoder(nn.Module):
    def __init__(self, crop: int = 88, feat: int = 64, num_kp: int = 32):
        super().__init__()
        import torchvision

        r = _bn_to_gn(torchvision.models.resnet18(weights=None))
        self.trunk = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4)
        with torch.no_grad():
            c, h, w = self.trunk(torch.zeros(1, 3, crop, crop)).shape[1:]
        self.pool = SpatialSoftmax(c, h, w, num_kp)
        self.proj = nn.Linear(2 * num_kp, feat)

    def forward(self, x):                      # B x 3 x H x W, values in [0, 1]
        return self.proj(self.pool(self.trunk(x)))


# ---------------------------------------------------------------------- conditional 1D U-Net
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        e = math.log(10000) / (half - 1)
        e = torch.exp(torch.arange(half, device=t.device) * -e)
        e = t.float()[:, None] * e[None, :]
        return torch.cat([e.sin(), e.cos()], -1)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k, groups=8):
        super().__init__()
        self.block = nn.Sequential(nn.Conv1d(cin, cout, k, padding=k // 2), nn.GroupNorm(groups, cout), nn.Mish())

    def forward(self, x):
        return self.block(x)


class CondResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim, k=5, groups=8):
        super().__init__()
        self.b1, self.b2 = Conv1dBlock(cin, cout, k, groups), Conv1dBlock(cout, cout, k, groups)
        self.cout = cout
        self.film = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, 2 * cout))
        self.res = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        out = self.b1(x)
        e = self.film(cond).unsqueeze(-1)
        out = e[:, :self.cout] * out + e[:, self.cout:]
        return self.b2(out) + self.res(x)


class ConditionalUnet1D(nn.Module):
    def __init__(self, input_dim: int, global_cond_dim: int, step_dim: int = 128, down_dims: Sequence[int] = (256, 512, 1024),
                 k: int = 5, groups: int = 8):
        super().__init__()
        dims = [input_dim, *down_dims]
        self.step_enc = nn.Sequential(SinusoidalPosEmb(step_dim), nn.Linear(step_dim, 4 * step_dim), nn.Mish(), nn.Linear(4 * step_dim, step_dim))
        cd = step_dim + global_cond_dim
        io = list(zip(dims[:-1], dims[1:]))
        self.down = nn.ModuleList([nn.ModuleList([CondResBlock1D(a, b, cd, k, groups), CondResBlock1D(b, b, cd, k, groups),
                                                  nn.Conv1d(b, b, 3, 2, 1) if i < len(io) - 1 else nn.Identity()])
                                   for i, (a, b) in enumerate(io)])
        mid = dims[-1]
        self.mid = nn.ModuleList([CondResBlock1D(mid, mid, cd, k, groups), CondResBlock1D(mid, mid, cd, k, groups)])
        self.up = nn.ModuleList([nn.ModuleList([CondResBlock1D(2 * b, a, cd, k, groups), CondResBlock1D(a, a, cd, k, groups),
                                                nn.ConvTranspose1d(a, a, 4, 2, 1)])
                                 for a, b in reversed(io[1:])])
        self.final = nn.Sequential(Conv1dBlock(down_dims[0], down_dims[0], k, groups), nn.Conv1d(down_dims[0], input_dim, 1))

    def forward(self, x, t, cond):              # x: B x Tp x A
        x = x.transpose(1, 2)
        g = torch.cat([self.step_enc(t), cond], -1)
        hs = []
        for r1, r2, dn in self.down:
            x = r2(r1(x, g), g)
            hs.append(x)
            x = dn(x)
        for m in self.mid:
            x = m(x, g)
        for r1, r2, upm in self.up:
            x = upm(r2(r1(torch.cat([x, hs.pop()], 1), g), g))
        return self.final(x).transpose(1, 2)


# ---------------------------------------------------------------------- policy
class DiffusionPolicy(nn.Module):
    def __init__(self, cams: List[str], low_dim: int, act_dim: int, To: int = 2, Tp: int = 16, crop: int = 88,
                 train_steps: int = 100, infer_steps: int = 16):
        super().__init__()
        from diffusers import DDIMScheduler, DDPMScheduler

        self.cams, self.low_dim, self.act_dim, self.To, self.Tp = list(cams), low_dim, act_dim, To, Tp
        self.enc = nn.ModuleDict({c: ImageEncoder(crop) for c in self.cams})
        cond = To * (64 * len(self.cams) + low_dim)
        self.unet = ConditionalUnet1D(act_dim, cond)
        self.ddpm = DDPMScheduler(num_train_timesteps=train_steps, beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                  prediction_type="epsilon")
        self.ddim = DDIMScheduler(num_train_timesteps=train_steps, beta_schedule="squaredcos_cap_v2", clip_sample=True,
                                  prediction_type="epsilon", set_alpha_to_one=True, steps_offset=0)
        self.infer_steps = infer_steps

    def cond(self, imgs: Dict[str, torch.Tensor], low: torch.Tensor) -> torch.Tensor:
        """imgs[cam]: B x To x 3 x H x W; low: B x To x D (normalized) -> B x cond."""
        B = low.shape[0]
        feats = [self.enc[c](imgs[c].flatten(0, 1)).reshape(B, self.To, -1) for c in self.cams]
        return torch.cat([*feats, low], -1).reshape(B, -1)

    def loss(self, imgs, low, act) -> torch.Tensor:
        g = self.cond(imgs, low)
        noise = torch.randn_like(act)
        t = torch.randint(0, self.ddpm.config.num_train_timesteps, (act.shape[0],), device=act.device).long()
        noisy = self.ddpm.add_noise(act, noise, t)
        return F.mse_loss(self.unet(noisy, t, g), noise)

    @torch.no_grad()
    def sample(self, imgs, low) -> torch.Tensor:
        g = self.cond(imgs, low)
        x = torch.randn(low.shape[0], self.Tp, self.act_dim, device=low.device)
        self.ddim.set_timesteps(self.infer_steps)
        for t in self.ddim.timesteps:
            eps = self.unet(x, t.expand(x.shape[0]).to(x.device), g)
            x = self.ddim.step(eps, t, x).prev_sample
        return x
