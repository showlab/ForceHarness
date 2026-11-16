"""Inference server for the Diffusion Policy baseline.

Runs in the torch environment. The evaluation client (scripts/baselines/dp/eval_policy.py) runs in the simulator environment;
the two talk over a local socket (multiprocessing.connection). The client sends the latest To frames (JPEG bytes + raw low-dim
values); the server applies the same preprocessing as training, samples with DDIM, denormalizes and returns a Tp x A action
chunk. Arrays travel as (dtype, shape, bytes), so the two environments need not share a numpy version.
Usage (from the repository root):
  CUDA_VISIBLE_DEVICES=0 python scripts/baselines/dp/dp_server.py --ckpt checkpoints/dp_force_com_sort/ema_last.pt --port 47001
"""
import _bootstrap  # noqa: F401
import argparse
import json
import threading
from multiprocessing.connection import Listener

import numpy as np
import torch

from forceharness.data.demos import pack, prep_jpeg, unpack
from forceharness.baselines.diffusion_policy import DiffusionPolicy, MinMax

AUTH = b"fh-dp"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--infer-steps", type=int, default=16)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    pol = DiffusionPolicy(cfg["cams"], cfg["low_dim"], cfg["act_dim"], cfg["To"], cfg["Tp"], cfg["crop"], infer_steps=args.infer_steps)
    pol.load_state_dict(ck["state_dict"])
    pol.to(dev).eval()
    low_norm, act_norm = MinMax.from_state(cfg["low_norm"]), MinMax.from_state(cfg["act_norm"])
    lock = threading.Lock()
    info = dict(cfg, ckpt=args.ckpt, ckpt_step=ck.get("step"), infer_steps=args.infer_steps)

    def serve(conn):
        while True:
            try:
                msg = conn.recv()
            except (EOFError, ConnectionResetError, OSError):
                break
            if msg.get("cmd") == "config":
                conn.send(json.dumps(info))
                continue
            imgs = {c: torch.from_numpy(np.stack([prep_jpeg(b, cfg["size"], cfg["crop"], False) for b in msg["jpeg"][c]]))[None]
                    .to(dev).float() / 255.0 for c in cfg["cams"]}
            low = torch.from_numpy(low_norm.norm(unpack(msg["low"]).astype(np.float32)))[None].to(dev)
            with lock, torch.no_grad():
                a = pol.sample(imgs, low)[0].float().cpu().numpy()
            conn.send(pack(act_norm.denorm(a).astype(np.float32)))
        conn.close()

    lst = Listener(("127.0.0.1", args.port), authkey=AUTH)
    print(f"[dp_server] {args.ckpt} (step {ck.get('step')}) on 127.0.0.1:{args.port}, device {dev}", flush=True)
    while True:
        threading.Thread(target=serve, args=(lst.accept(),), daemon=True).start()


if __name__ == "__main__":
    main()
