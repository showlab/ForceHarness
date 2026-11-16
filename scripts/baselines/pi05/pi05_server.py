"""Inference server for the pi0.5 and ForceVLA baselines (one multi-task model serves all 7 tasks).

Runs in the openpi environment. The evaluation client (scripts/baselines/dp/eval_policy.py) runs in the simulator environment;
the two talk over a local socket (multiprocessing.connection, authkey b"fh-dp") with the same protocol as
scripts/baselines/dp/dp_server.py, plus a task name in every message:
  {"cmd": "config", "task": t}  -> JSON string: cams, scene (True), scene_keys, force (True for f2 / ForceVLA), osc_dim, To (1),
                                   Ta (5), act_norm {lo, hi} (the client only uses [osc_dim:osc_dim+6] to clamp kp), max_T,
                                   ckpt, ckpt_step, plus the prompt and other details
  {"cmd": "act", "task": t, "jpeg": {cam: [To JPEG frames]}, "low": pack(To x D)}  -> pack(Tp x A) actions in physical units
    low = proprioception 11 + [wrench 6 if force] + [scene vector, 5 * len(scene_keys)]; the server splits it and rebuilds the
    training state layout (proprioception + zero-padded scene vector [+ wrench]), adds the task's prompt and calls
    policy.infer. openpi's output transforms already denormalize, so the 13-D action chunk is returned as is, except that
    on wipe_hard (osc_dim 6) the padded index 6 is dropped and 12 dims are returned: the client takes OSC from a[:osc_dim] and
    kp from a[osc_dim:osc_dim+6], and act_norm is cut the same way.
JPEGs are decoded with the recorder / evaluation-client convention (BGR, upside down, q90 -> upright RGB), the same as
decode_frame in the conversion.

Usage (the checkpoint is one step directory saved by pi05_fh.py train, e.g. $FH_PI05_WORK/checkpoints/pi05_fh_f2/bs32_30k/29999):
  CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35 python scripts/baselines/pi05/pi05_server.py --config pi05_fh_f2 \
      --ckpt <step dir> --port 47102
  python scripts/baselines/pi05/pi05_server.py --selftest [--selftest-demos <demo root>] [--meta <meta_fh.json>] [--transforms --config ...]
      Loads no model: starts the server with a fake policy, sends config / act messages the way eval_policy.py does with
      multiprocessing.connection.Client, and checks shapes and the state / image layout. With --transforms the fake policy runs
      openpi's real input / output transforms (normalization, tokenization) and is compared with the same frame from the
      training data loader.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import threading
import time
import traceback
from multiprocessing.connection import Client, Listener
from typing import Any, Dict, Optional

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[2]                                 # repository root
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from forceharness.data.demos import pack, unpack  # noqa: E402
from convert_to_lerobot import (ACTION_DIM, CAMS, KP_DIM, OSC_DIM_MAX, PROPRIO_DIM, REPO_ID, WRENCH_DIM,  # noqa: E402
                                build_state)

AUTH = b"fh-dp"
TO, TA = 1, 5           # pi0.5 sees only the current frame; chunks of 10 ticks, replanned after 5 (the client executes Ta)


def decode_upright(jpeg) -> np.ndarray:
    """JPEG bytes (BGR, upside down: the recorder / encode_frame convention) -> upright RGB uint8 HWC, as demos.decode_frame."""
    import cv2

    buf = np.frombuffer(jpeg, np.uint8) if isinstance(jpeg, (bytes, bytearray)) else np.asarray(jpeg, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)[..., ::-1]
    return np.ascontiguousarray(img[::-1])


def ckpt_step_of(ckpt: Optional[str]) -> Optional[int]:
    try:
        return int(pathlib.Path(str(ckpt)).resolve().name)
    except (TypeError, ValueError):
        return None


class PolicyServer:
    """policy only needs infer(obs) -> {"actions": (H, 13)}: openpi's Policy when serving, a fake policy in the self-test."""

    def __init__(self, policy, meta: Dict[str, Any], force: bool, config_name: str, ckpt: Optional[str] = None,
                 act_norm_scope: str = "global", horizon: int = 10):
        self.policy, self.meta, self.force = policy, meta, bool(force)
        self.config_name, self.ckpt, self.horizon = config_name, ckpt, int(horizon)
        self.act_norm_scope = act_norm_scope
        self.scene_pad = int(meta["scene_pad"])
        self.lock = threading.Lock()
        self.n_calls = 0

    # ------------------------------------------------------------------ protocol
    def base_task(self, task: str) -> str:
        """A shifted variant <task>_ood (end of forceharness/hiddenphys/tasks.py) changes only the distribution of the hidden
        variable; objects, scene format and success check are unchanged. Use the original task's metadata and training prompt
        without retraining, as the Diffusion Policy shift test (dp_shift_eval.sh, original task's model) does."""
        if task not in self.meta["tasks"] and task.endswith("_ood") and task[:-4] in self.meta["tasks"]:
            return task[:-4]
        return task

    def keep_index(self, task: str):
        """Indices of the 13-D action returned to the client: the first osc_dim OSC dims + kp 6 (wipe_hard drops the padded index 6)."""
        osc = int(self.meta["tasks"][task]["osc_dim"])
        return list(range(osc)) + list(range(OSC_DIM_MAX, ACTION_DIM))

    def task_config(self, task: str) -> Dict[str, Any]:
        if task not in self.meta["tasks"]:
            raise KeyError(f"task {task!r} not in this model's training set {list(self.meta['tasks'])}")
        m = self.meta["tasks"][task]
        src = m if self.act_norm_scope == "task" else self.meta
        keep = self.keep_index(task)
        lo, hi = np.asarray(src["actions_min"], np.float64)[keep], np.asarray(src["actions_max"], np.float64)[keep]
        return dict(cams=list(CAMS.values()), scene=True, scene_keys=list(m["scene_keys"]), force=self.force,
                    osc_dim=int(m["osc_dim"]), act_dim=len(keep), To=TO, Ta=TA, Tp=self.horizon,
                    act_norm={"lo": lo.tolist(), "hi": hi.tolist(), "scope": self.act_norm_scope},
                    max_T=int(m["max_T"]), ckpt=self.ckpt, ckpt_step=ckpt_step_of(self.ckpt), policy="pi05",
                    config=self.config_name, prompt=m["prompt"], scene_dim=int(m["scene_dim"]), scene_pad=self.scene_pad,
                    low_layout=f"proprio {PROPRIO_DIM}" + (f" + wrench {WRENCH_DIM}" if self.force else "")
                               + f" + scene {m['scene_dim']}")

    def make_obs(self, task: str, msg: Dict[str, Any]) -> Dict[str, Any]:
        """Client message -> openpi observation dict (same keys as pi05_fh.FhInputs)."""
        m = self.meta["tasks"][task]
        sd = int(m["scene_dim"])
        low = unpack(msg["low"]).astype(np.float32)
        row = low[-1] if low.ndim == 2 else low.reshape(-1)
        want = PROPRIO_DIM + (WRENCH_DIM if self.force else 0) + sd
        if row.size != want:
            raise ValueError(f"{task}: low has {row.size} dims, expected {want} ({self.task_config(task)['low_layout']})")
        prop = row[:PROPRIO_DIM]
        k = PROPRIO_DIM
        wrench = np.zeros(WRENCH_DIM, np.float32)
        if self.force:
            wrench, k = row[k:k + WRENCH_DIM].copy(), k + WRENCH_DIM
        scene = row[k:k + sd]
        imgs = {key: decode_upright(msg["jpeg"][cam][-1]) for key, cam in CAMS.items()}
        return {"observation/image": imgs["image"], "observation/wrist_image": imgs["wrist_image"],
                "observation/state": build_state(prop, scene, self.scene_pad), "observation/wrench": wrench,
                "prompt": m["prompt"]}

    def act(self, task: str, msg: Dict[str, Any]) -> np.ndarray:
        obs = self.make_obs(task, msg)
        with self.lock:
            out = self.policy.infer(obs)
            self.n_calls += 1
        a = np.asarray(out["actions"], np.float32)
        if a.ndim != 2 or a.shape[1] != ACTION_DIM:
            raise ValueError(f"policy returned actions {a.shape}, expected (H, {ACTION_DIM})")
        return np.ascontiguousarray(a[:, self.keep_index(task)])

    def serve_conn(self, conn) -> None:
        while True:
            try:
                msg = conn.recv()
            except (EOFError, ConnectionResetError, OSError):
                break
            try:
                task = self.base_task(msg.get("task"))
                if msg.get("cmd") == "config":
                    conn.send(json.dumps(self.task_config(task)))
                elif msg.get("cmd") == "act":
                    conn.send(pack(self.act(task, msg)))
                else:
                    raise ValueError(f"unknown cmd {msg.get('cmd')!r}")
            except Exception:                    # log and drop this connection: the client's recv raises EOFError, recorded as the episode error
                print(f"[pi05_server] error on {msg.get('cmd')} {msg.get('task')}:\n{traceback.format_exc()}", flush=True)
                break
        conn.close()

    def serve_forever(self, port: int, ready: Optional[threading.Event] = None) -> None:
        lst = Listener(("127.0.0.1", port), authkey=AUTH, backlog=64)     # many evaluation workers connect at once
        if ready is not None:
            ready.set()
        while True:
            threading.Thread(target=self.serve_conn, args=(lst.accept(),), daemon=True).start()


# ---------------------------------------------------------------------------------------------- real server
def find_meta(ckpt: str, config_name: str, explicit: Optional[str]) -> pathlib.Path:
    """meta_fh.json: --meta > assets/<repo_id>/ inside the checkpoint > $FH_PI05_WORK/assets/<config>/<repo_id>/."""
    repo = os.environ.get("FH_REPO_ID", REPO_ID)
    cands = [pathlib.Path(explicit)] if explicit else []
    cands += [pathlib.Path(ckpt) / "assets" / repo / "meta_fh.json"]
    work = pathlib.Path(os.environ.get("FH_PI05_WORK", str(ROOT / os.environ.get("FH_CKPT", "checkpoints") / "pi05")))   # as in pi05_fh.py
    cands += [pathlib.Path(os.environ.get("FH_PI05_ASSETS", str(work / "assets"))) / config_name / repo / "meta_fh.json"]
    for c in cands:
        if c.exists():
            return c
    raise FileNotFoundError(f"meta_fh.json not found; tried {[str(c) for c in cands]} — pass --meta")


def load_policy(config_name: str, ckpt: str, action_horizon: int = 10):
    import pi05_fh  # registers the data transforms and TrainConfigs (needs the openpi environment)
    from openpi.policies import policy_config

    if config_name == "forcevla_fh":        # ForceVLA: same observations (wrench in the state) and actions; model in ../forcevla/
        sys.path.insert(0, str(HERE.parent / "forcevla"))
        import forcevla_fh

        cfg = forcevla_fh.make_config(exp_name="serve", action_horizon=action_horizon)
    else:
        cfg = pi05_fh.make_config(config_name, exp_name="serve", action_horizon=action_horizon)   # must match training
    pi05_fh.register(cfg)
    return policy_config.create_trained_policy(cfg, ckpt), cfg


def warmup(server: PolicyServer) -> None:
    """Run inference once before serving (the first JAX call compiles, about a minute), so the first client does not wait."""
    t = next(iter(server.meta["tasks"]))
    m = server.meta["tasks"][t]
    import cv2

    jp = cv2.imencode(".jpg", np.zeros((128, 128, 3), np.uint8))[1].tobytes()
    low = np.zeros((TO, PROPRIO_DIM + (WRENCH_DIM if server.force else 0) + int(m["scene_dim"])), np.float32)
    t0 = time.time()
    a = server.act(t, {"jpeg": {c: [jp] for c in CAMS.values()}, "low": pack(low)})
    t1 = time.time()
    server.act(t, {"jpeg": {c: [jp] for c in CAMS.values()}, "low": pack(low)})
    print(f"[pi05_server] warmup on {t}: actions {a.shape}, first call {t1 - t0:.1f}s, second {time.time() - t1:.3f}s", flush=True)


# ---------------------------------------------------------------------------------------------- self-test (no model loaded)
class FakePolicy:
    """Records the received observation and checks keys / dtypes / shapes. Returns all zeros (13 dims), or, with a config,
    runs openpi's real transform chain: inputs = InjectDefaultPrompt -> FhInputs -> quantile normalization -> resize 224 /
    tokenize / pad 32; the fake model outputs zero normalized actions, which Unnormalize -> FhOutputs turn into physical units
    (= the midpoint of each dimension's quantile range)."""

    def __init__(self, horizon: int = 10, cfg=None):
        self.horizon, self.last_obs, self.last_model_in = horizon, None, None
        self.in_tf = self.out_tf = None
        if cfg is not None:
            import openpi.transforms as T
            from openpi.training import checkpoints as _ckpt

            dc = cfg.data.create(cfg.assets_dirs, cfg.model)
            ns = _ckpt.load_norm_stats(pathlib.Path(cfg.assets_dirs), dc.asset_id)
            self.in_tf = T.compose([T.InjectDefaultPrompt(None), *dc.data_transforms.inputs,
                                    T.Normalize(ns, use_quantiles=dc.use_quantile_norm), *dc.model_transforms.inputs])
            self.out_tf = T.compose([*dc.model_transforms.outputs, T.Unnormalize(ns, use_quantiles=dc.use_quantile_norm),
                                     *dc.data_transforms.outputs])
            self.action_dim = cfg.model.action_dim

    def infer(self, obs):
        for k in ("observation/image", "observation/wrist_image"):
            assert obs[k].dtype == np.uint8 and obs[k].shape == (128, 128, 3), (k, obs[k].dtype, obs[k].shape)
        assert obs["observation/state"].dtype == np.float32 and obs["observation/state"].ndim == 1
        assert obs["observation/wrench"].shape == (WRENCH_DIM,) and isinstance(obs["prompt"], str)
        self.last_obs = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
        if self.in_tf is None:
            return {"actions": np.zeros((self.horizon, ACTION_DIM), np.float32)}
        x = self.in_tf({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()})
        self.last_model_in = x
        out = self.out_tf({"state": x["state"], "actions": np.zeros((self.horizon, self.action_dim), np.float32)})
        return {"actions": np.asarray(out["actions"], np.float32)}


def selftest(args) -> None:
    from forceharness.data.demos import decode_frame, list_demos, load_demo, proprio, scene_vector, wrench_obs
    from convert_to_lerobot import episode_arrays

    meta_path = pathlib.Path(args.meta) if args.meta else None
    if meta_path is None:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME

        meta_path = HF_LEROBOT_HOME / os.environ.get("FH_REPO_ID", REPO_ID) / "meta_fh.json"
    meta = json.loads(meta_path.read_text())
    variants = [args.config] if args.transforms else ["pi05_fh_f0", "pi05_fh_f2"]
    port = args.port or 47190
    n_ok = 0
    for vi, name in enumerate(variants):
        force = name.endswith("f2")
        cfg = None
        if args.transforms:
            import pi05_fh

            cfg = pi05_fh.make_config(name, exp_name="selftest")
            pi05_fh.register(cfg)
        fake = FakePolicy(cfg=cfg)
        srv = PolicyServer(fake, meta, force, name, ckpt="/fake/ckpt/12345")
        ready = threading.Event()
        threading.Thread(target=srv.serve_forever, args=(port + vi, ready), daemon=True).start()
        ready.wait(10)
        tds = ep_base = None
        if args.transforms:                      # the training data loader's transform chain, to compare the same frame
            import openpi.training.data_loader as _dl

            dc = cfg.data.create(cfg.assets_dirs, cfg.model)
            tds = _dl.transform_dataset(_dl.create_torch_dataset(dc, cfg.model.action_horizon, cfg.model), dc)
            lr = tds._dataset._dataset if hasattr(tds._dataset, "_dataset") else tds._dataset   # unwrap PromptFromLeRobotTask
            ep_base, acc = {}, 0
            for t, m in meta["tasks"].items():
                ep_base[t], acc = acc, acc + int(m["n_episodes"])
        for task, m in meta["tasks"].items():
            conn = Client(("127.0.0.1", port + vi), authkey=AUTH)
            conn.send({"cmd": "config", "task": task})                         # as eval_policy.run_one does
            c = json.loads(conn.recv())
            for k in ("cams", "scene", "scene_keys", "force", "osc_dim", "To", "Ta", "act_norm", "max_T", "ckpt", "ckpt_step"):
                assert k in c, f"config missing {k}"
            osc_dim, To = int(c["osc_dim"]), int(c["To"])
            kp_lo = np.asarray(c["act_norm"]["lo"], np.float32)[osc_dim:osc_dim + 6]
            kp_hi = np.asarray(c["act_norm"]["hi"], np.float32)[osc_dim:osc_dim + 6]
            assert kp_lo.shape == (KP_DIM,) and (kp_lo >= 1).all() and (kp_hi <= 1000).all(), (task, kp_lo, kp_hi)   # really the 6 kp dims
            assert c["force"] == force and c["scene"] is True and To == 1 and int(c["Ta"]) == 5
            # build a message as the evaluation client does: the JPEGs in the npz are encode_frame output (BGR, upside down, q90)
            demos = list_demos(str(pathlib.Path(args.selftest_demos) / task)) if args.selftest_demos else []
            if demos:
                d = load_demo(demos[0])
                T = len(d["action"])
                i = min(T - 1, max(0, T // 2))
                buf = {cam: d["jpeg"][cam] for cam in CAMS.values()}
                jpeg = {cam: [bytes(buf[cam][0][buf[cam][1][i]:buf[cam][1][i + 1]])] for cam in c["cams"]}
                g = d["gripper_qpos"][i] if len(d["gripper_qpos"]) > i else None     # the wiping tool has no gripper: (0, 2) in the npz
                low = [proprio(d["eef_pos"][i], d["eef_quat"][i], g)]
                if c["force"]:
                    low.append(wrench_obs(d, int(d["meta"]["seed"]))[i])
                low.append(scene_vector(d["meta"].get("scene0"), c["scene_keys"])[0])
            else:
                d, i = None, 0
                import cv2

                jp = cv2.imencode(".jpg", np.full((128, 128, 3), 77, np.uint8))[1].tobytes()
                jpeg = {cam: [jp] for cam in c["cams"]}
                low = [np.zeros(PROPRIO_DIM, np.float32)] + ([np.zeros(WRENCH_DIM, np.float32)] if c["force"] else []) \
                    + [np.zeros(5 * len(c["scene_keys"]), np.float32)]
            low = np.concatenate(low).astype(np.float32)[None].repeat(To, 0)
            conn.send({"cmd": "act", "task": task, "jpeg": jpeg, "low": pack(low)})
            acts = unpack(conn.recv())
            conn.close()
            assert acts.shape == (10, osc_dim + KP_DIM) and acts.dtype == np.float32, (task, acts.shape, acts.dtype)
            msg = f"  [{name}] {task}: config ok (osc_dim {osc_dim}, scene_keys {c['scene_keys']}, max_T {c['max_T']}, " \
                  f"kp clamp [{kp_lo.min():g}, {kp_hi.max():g}]); act → {acts.shape} {acts.dtype}"
            ob = fake.last_obs
            if d is not None:                     # the server-side observation == the frame the conversion wrote to the dataset
                e = episode_arrays(d, m["scene_keys"], meta["scene_pad"])
                assert np.array_equal(ob["observation/state"], e["state"][i]), (task, ob["observation/state"], e["state"][i])
                if force:
                    assert np.allclose(ob["observation/wrench"], e["wrench"][i]), task
                for key, cam in CAMS.items():
                    assert np.array_equal(ob[f"observation/{key}"], decode_frame(d, cam, i)), (task, key)
                msg += f"; frame {i}: state / wrench / images == dataset"
            if tds is not None and d is not None:  # compare with the training data loader: tokenized prompt, normalized state, images
                assert int(m["seeds"][0]) == int(d["meta"]["seed"]), (task, m["seeds"][0], d["meta"]["seed"])   # same episode
                lr_ep = ep_base[task]
                idx = int(lr.episode_data_index["from"][lr_ep]) + i
                tr = tds[idx]
                mi = fake.last_model_in
                same_tok = np.array_equal(np.asarray(tr["tokenized_prompt"]), np.asarray(mi["tokenized_prompt"]))
                st_diff = float(np.abs(np.asarray(tr["state"]) - np.asarray(mi["state"])).max())
                img_diff = max(int(np.abs(np.asarray(tr["image"][k]).astype(int) - np.asarray(mi["image"][k]).astype(int)).max())
                               for k in ("base_0_rgb", "left_wrist_0_rgb"))
                assert same_tok, (task, "tokenized prompt differs between training loader and server")
                assert st_diff < 1e-5, (task, st_diff)
                assert img_diff <= 2, (task, img_diff)
                mid = (np.asarray(meta_q(cfg, "q01")) + np.asarray(meta_q(cfg, "q99"))) / 2
                assert np.allclose(acts, mid[srv.keep_index(task)][None].repeat(10, 0), atol=1e-4), (task, acts[0], mid)
                msg += (f"; vs training loader (episode {lr_ep}, idx {idx}): tokens equal "
                        f"({int(np.asarray(mi['tokenized_prompt_mask']).sum())}), state Δ {st_diff:.1e}, image Δ ≤ {img_diff}; "
                        f"zero-output → unnormalized midpoints ok")
            print(msg, flush=True)
            n_ok += 1
        # concurrent clients (evaluation workers each hold one connection)
        errs = []

        def one(t):
            try:
                cc = Client(("127.0.0.1", port + vi), authkey=AUTH)
                cc.send({"cmd": "config", "task": t})
                cf = json.loads(cc.recv())
                lw = np.zeros((1, PROPRIO_DIM + (WRENCH_DIM if cf["force"] else 0) + 5 * len(cf["scene_keys"])), np.float32)
                import cv2

                jp = cv2.imencode(".jpg", np.zeros((128, 128, 3), np.uint8))[1].tobytes()
                for _ in range(3):
                    cc.send({"cmd": "act", "task": t, "jpeg": {cm: [jp] for cm in cf["cams"]}, "low": pack(lw)})
                    assert unpack(cc.recv()).shape[0] == 10
                cc.close()
            except Exception as ex:  # noqa: BLE001
                errs.append(repr(ex))

        ths = [threading.Thread(target=one, args=(t,)) for t in list(meta["tasks"])[:4]]
        [t.start() for t in ths]
        [t.join(60) for t in ths]
        assert not errs, errs
        print(f"  [{name}] 4 concurrent clients ok; total policy calls {srv.n_calls}", flush=True)
    print(f"[selftest] ok: {n_ok} task configs × act round-trips", flush=True)


def meta_q(cfg, which: str):
    from openpi.training import checkpoints as _ckpt

    dc = cfg.data.create(cfg.assets_dirs, cfg.model)
    ns = _ckpt.load_norm_stats(pathlib.Path(cfg.assets_dirs), dc.asset_id)
    return getattr(ns["actions"], which)[:ACTION_DIM]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", choices=["pi05_fh_f0", "pi05_fh_f2", "forcevla_fh"], help="config the checkpoint was trained with (f2 and forcevla_fh have the wrench in the state)")
    ap.add_argument("--ckpt", help="checkpoint step directory (contains params/ and assets/)")
    ap.add_argument("--port", type=int)
    ap.add_argument("--meta", default=None, help="meta_fh.json; default: assets/<repo_id>/meta_fh.json inside the checkpoint")
    ap.add_argument("--act-norm", choices=["global", "task"], default="global",
                    help="act_norm returned to the client (only used to clamp kp): min / max over the whole dataset (default) or of this task")
    ap.add_argument("--action-horizon", type=int, default=10, help="must equal --action-horizon used in training")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--selftest", action="store_true", help="run the protocol with a fake policy, without loading a model")
    ap.add_argument("--selftest-demos", default=None, help="demonstration root (<task>/s*.npz) for the self-test; if given, messages are built from real demo frames and the layout is checked")
    ap.add_argument("--transforms", action="store_true", help="self-test: the fake policy runs openpi's real transform chain (needs --config and the statistics in the assets)")
    args = ap.parse_args()
    if args.selftest:
        if args.transforms and not args.config:
            ap.error("--transforms needs --config")
        selftest(args)
        return
    if not (args.config and args.ckpt and args.port):
        ap.error("--config, --ckpt and --port are required (or use --selftest)")
    meta_path = find_meta(args.ckpt, args.config, args.meta)
    meta = json.loads(meta_path.read_text())
    policy, cfg = load_policy(args.config, args.ckpt, args.action_horizon)
    srv = PolicyServer(policy, meta, force=args.config in ("pi05_fh_f2", "forcevla_fh"), config_name=args.config, ckpt=args.ckpt,
                       act_norm_scope=args.act_norm, horizon=cfg.model.action_horizon)
    print(f"[pi05_server] {args.config} ckpt {args.ckpt} (step {ckpt_step_of(args.ckpt)}), meta {meta_path}, "
          f"tasks {list(meta['tasks'])}", flush=True)
    if not args.no_warmup:
        warmup(srv)
    print(f"[pi05_server] ready on 127.0.0.1:{args.port}", flush=True)
    srv.serve_forever(args.port)


if __name__ == "__main__":
    main()
