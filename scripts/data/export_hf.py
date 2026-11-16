"""Build the Hugging Face release of the HiddenPhys expert demonstrations.

Input: the folder written by ``scripts/hiddenphys/record_demos*.py``: ``<demos>/<task>/s<seed>.npz`` plus
``<demos>/<task>/index.jsonl`` (one line per attempted episode, failed ones included).

Output (``--out``), one Hugging Face dataset repository:

    README.md                       dataset card (from docs/dataset_card.md, with the counts filled in)
    raw/<task>/episode_<seed>.npz   the recorded episodes, unchanged except for a cleaned metadata record
    raw/<task>/index.jsonl          one line per released episode (seed, steps, hidden variable, verdict)
    raw/attempts.json               attempted / released episodes per task
    meta/, data/                    (with --lerobot) the LeRobot v2.1 dataset used to train the pi0.5 / ForceVLA baselines

    python scripts/data/export_hf.py --demos /path/to/demos --out hf_export            # raw part only (numpy is enough)
    python scripts/data/export_hf.py --demos /path/to/demos --out hf_export --lerobot  # + LeRobot part (needs `lerobot`)
    python scripts/data/export_hf.py --out hf_export --card-only --arxiv-id 2611.xxxxx --repo-id <org>/<name>  # refresh README.md
    python scripts/data/export_hf.py --out hf_export --push --repo-id <org>/<name>    # upload a finished export

Nothing is uploaded unless ``--push`` and ``--repo-id`` are both given, and ``--push`` refuses while the card still has the
``ARXIV_ID`` placeholder (the arXiv link in the card is what links the dataset to the paper page on the Hub).  The metadata of every episode is rewritten to a
cleaned record, and the export stops if any released string still looks like a local path or contains CJK text.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from forceharness.data.demos import load_demo  # noqa: E402
from forceharness.hiddenphys.tasks import HIDDENPHYS_TASKS  # noqa: E402

CARD = ROOT / "docs" / "dataset_card.md"
RECORDED_WITH = "expert force controller (hand-written state machine over the ForceHarness force primitives)"
# Fields dropped from every episode's metadata (worker / GPU ids, wall-clock timestamps, file bookkeeping); everything else
# is kept, because replay rebuilds the episode from it (episode_cfg, start_after_pregrasp, prims_args, scene0, ...).
META_DROP = {"egl", "recorded_at", "file", "bytes", "wall_s"}
# Values of episode_cfg that only named local output locations during recording.
EPISODE_CFG_NEUTRAL = {"out_dir": "runs", "run_name": "demos", "note": ""}
# Per-episode summary written to raw/<task>/index.jsonl (hidden variables differ per task; missing keys are skipped).
INDEX_KEEP = ["task", "seed", "n_steps", "verdict", "verdict_correct", "truth", "label", "theta_u", "door", "damaged",
              "sim_time_s", "peak_wrench_N"]
SUSPICIOUS = re.compile(r"(/users/|/home/|/workspace/|/mnt/|/scratch|/tmp/|[A-Za-z]:\\\\|[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef])")


def _walk_strings(o: Any):
    if isinstance(o, str):
        yield o
    elif isinstance(o, dict):
        for k, v in o.items():
            yield str(k)
            yield from _walk_strings(v)
    elif isinstance(o, (list, tuple)):
        for v in o:
            yield from _walk_strings(v)


def clean_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    out = {k: v for k, v in meta.items() if k not in META_DROP}
    if isinstance(out.get("episode_cfg"), dict):
        out["episode_cfg"] = {k: EPISODE_CFG_NEUTRAL.get(k, v) for k, v in out["episode_cfg"].items()}
    strategy = (out.get("check_result") or {}).get("strategy") if isinstance(out.get("check_result"), dict) else None
    out["recorded_with"] = RECORDED_WITH + (f", strategy {strategy} of the task's expert script" if strategy else "")
    bad = [s for s in _walk_strings(out) if SUSPICIOUS.search(s)]
    if bad:
        raise ValueError(f"metadata of {meta.get('task')} seed {meta.get('seed')} still contains: {bad[:3]}")
    return out


def export_episode(src: Path, dst: Path) -> Dict[str, Any]:
    z = np.load(src, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != "meta_json"}
    meta = json.loads(bytes(z["meta_json"]).decode("utf-8"))
    T = len(arrays["action"])
    for k in ("t", "eef_pos", "eef_quat", "gripper_qpos", "joint_pos", "force_raw", "torque_raw", "R_ft"):
        if k == "gripper_qpos" and len(arrays[k]) == 0:      # the wiping tool has no fingers: robosuite gives an empty array
            continue
        if len(arrays[k]) != T + 1:
            raise ValueError(f"{src}: {k} has {len(arrays[k])} frames, expected actions + 1 = {T + 1}")
    for k in ("kp", "weld", "via_step", "prim"):
        if len(arrays[k]) != T:
            raise ValueError(f"{src}: {k} has {len(arrays[k])} entries, expected {T}")
    m = clean_meta(meta)
    arrays["meta_json"] = np.frombuffer(json.dumps(m).encode("utf-8"), dtype=np.uint8)
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dst, **arrays)                      # uncompressed: the images are JPEG already
    load_demo(str(dst))                          # must load with the released loader
    return m


def export_raw(demos: Path, out: Path, tasks: List[str], n_per_task: int | None) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for task in tasks:
        tdir = demos / task
        idx = [json.loads(line) for line in (tdir / "index.jsonl").read_text().splitlines() if line.strip()] \
            if (tdir / "index.jsonl").exists() else []
        files = sorted(tdir.glob("s*.npz"), key=lambda p: int(p.stem[1:]))
        if n_per_task:
            files = files[:n_per_task]
        lines = []
        for f in files:
            m = export_episode(f, out / "raw" / task / f"episode_{int(f.stem[1:]):05d}.npz")
            lines.append({k: m[k] for k in INDEX_KEEP if k in m})
        (out / "raw" / task / "index.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
        counts[task] = dict(attempted=len(idx) or len(files), released=len(files),
                            frames=int(sum(x["n_steps"] for x in lines)),
                            bytes=int(sum(p.stat().st_size for p in (out / "raw" / task).glob("*.npz"))))
        print(f"[raw] {task}: {counts[task]}", flush=True)
    (out / "raw" / "attempts.json").write_text(json.dumps(counts, indent=1))
    return counts


def write_card(out: Path, counts: Dict[str, Dict[str, int]], arxiv_id: str | None = None, repo_id: str | None = None) -> None:
    rows = "\n".join(f"| `{t}` | {c['released']} | {c['attempted']} | {c['frames']:,} | {c['bytes'] / 1e9:.2f} GB |"
                     for t, c in counts.items())
    tot = {k: sum(c[k] for c in counts.values()) for k in ("released", "attempted", "frames", "bytes")}
    rows += f"\n| **total** | **{tot['released']}** | {tot['attempted']} | {tot['frames']:,} | {tot['bytes'] / 1e9:.2f} GB |"
    text = CARD.read_text(encoding="utf-8").replace("<!-- COUNTS_TABLE -->", rows)
    if arxiv_id:
        if not re.fullmatch(r"\d{4}\.\d{4,5}", arxiv_id):
            sys.exit(f"--arxiv-id should look like 2611.01234, got {arxiv_id!r}")
        text = text.replace("ARXIV_ID", arxiv_id)
    if repo_id:
        text = text.replace("<this repository id>", repo_id)
    (out / "README.md").write_text(text, encoding="utf-8")


def build_lerobot(out: Path) -> None:
    """LeRobot v2.1 dataset at the repository root, built by the same converter the pi0.5 baseline was trained with."""
    conv = ROOT / "scripts" / "baselines" / "pi05" / "convert_to_lerobot.py"
    tmp = out / "_lerobot_build"
    raw_as_demos = out / "_raw_view"
    raw_as_demos.mkdir(exist_ok=True)
    for tdir in sorted((out / "raw").iterdir()):                     # the converter reads <task>/s<seed>.npz
        if not tdir.is_dir():
            continue
        (raw_as_demos / tdir.name).mkdir(exist_ok=True)
        for f in tdir.glob("episode_*.npz"):
            link = raw_as_demos / tdir.name / f"s{int(f.stem.split('_')[1])}.npz"
            if not link.exists():
                link.symlink_to(f.resolve())
    env = dict(**__import__("os").environ, HF_LEROBOT_HOME=str(tmp))
    subprocess.run([sys.executable, str(conv), "--demos", str(raw_as_demos), "--overwrite"], check=True, env=env)
    built = [p for p in tmp.rglob("info.json") if p.parent.name == "meta"]
    if len(built) != 1:
        raise RuntimeError(f"expected one LeRobot dataset under {tmp}, found {len(built)}")
    root = built[0].parent.parent
    for sub in ("meta", "data", "videos", "images"):
        if (root / sub).exists():
            shutil.move(str(root / sub), str(out / sub))
    for extra in root.glob("*.json"):                 # meta_fh.json: dimensions and action ranges the pi0.5 server reads
        shutil.move(str(extra), str(out / extra.name))
    shutil.rmtree(tmp)
    shutil.rmtree(raw_as_demos)
    print(f"[lerobot] written to {out}/meta and {out}/data", flush=True)


def push(out: Path, repo_id: str, private: bool) -> None:
    from huggingface_hub import HfApi

    card = (out / "README.md").read_text(encoding="utf-8")
    if "ARXIV_ID" in card or "<this repository id>" in card:
        sys.exit("README.md still has placeholders; rerun with --card-only --arxiv-id <id> --repo-id <org>/<name> first")
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_large_folder(repo_id=repo_id, repo_type="dataset", folder_path=str(out))
    print(f"[push] uploaded {out} to https://huggingface.co/datasets/{repo_id}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--demos", type=Path, help="recorded demonstrations: <demos>/<task>/s<seed>.npz + index.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tasks", nargs="+", default=HIDDENPHYS_TASKS)
    ap.add_argument("--n-per-task", type=int, default=None, help="release at most this many episodes per task (lowest seeds)")
    ap.add_argument("--lerobot", action="store_true", help="also build the LeRobot v2.1 part (requires the lerobot package)")
    ap.add_argument("--card-only", action="store_true", help="rewrite README.md of an existing export (counts from raw/attempts.json)")
    ap.add_argument("--arxiv-id", help="fills the paper link and citation in the card, e.g. 2611.01234")
    ap.add_argument("--push", action="store_true", help="upload --out to the Hugging Face Hub (needs --repo-id and a login)")
    ap.add_argument("--repo-id")
    ap.add_argument("--public", action="store_true", help="create the Hub repository as public (default: private)")
    args = ap.parse_args()
    if args.demos:
        if (args.out / "raw").exists():
            sys.exit(f"{args.out}/raw exists; choose a new --out")
        counts = export_raw(args.demos, args.out, args.tasks, args.n_per_task)
        write_card(args.out, counts, args.arxiv_id, args.repo_id)
        if args.lerobot:
            build_lerobot(args.out)
    if args.card_only:
        counts = json.loads((args.out / "raw" / "attempts.json").read_text(encoding="utf-8"))
        write_card(args.out, counts, args.arxiv_id, args.repo_id)
        print(f"[card] {args.out}/README.md rewritten", flush=True)
    if args.push:
        if not args.repo_id:
            sys.exit("--push needs --repo-id")
        push(args.out, args.repo_id, private=not args.public)
    return 0


if __name__ == "__main__":
    sys.exit(main())
