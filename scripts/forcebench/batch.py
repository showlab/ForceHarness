#!/usr/bin/env python3
"""Run a list of ForceBench jobs: start scripts/forcebench/run.sh for each job, at most a given number at a time per GPU, one
run directory per job. Jobs whose run directory already has eval/ are skipped, so an interrupted batch can be resumed.

  python3 scripts/forcebench/batch.py configs/forcebench/jobs_example.yaml --gpus 0:2
  python3 scripts/forcebench/batch.py --row sol_force --gpus 0:1,1:1               # one paper row: 8 tasks x seeds 300-309
  python3 scripts/forcebench/batch.py --row astra_force --tasks nut_thread --dry-run  # print what would run, run nothing

A job list (JSON or YAML) is a list of jobs; `run` and `task` are required:
  {"run": "fb_dr_sol_force", "task": "radmanso_drawer_sort_table", "seed": 300, "n": 10, "row": "sol_force"}
  {"run": "fb_dr_scripted", "task": "radmanso_drawer_sort_table", "seed": 300, "n": 4, "args": ["--agent", "scripted"]}
A job is one evaluator process with n parallel environments on seeds seed .. seed+n-1 (defaults: the seeds of the settings).
The policy-server arguments of a job are assembled from the settings file (default configs/forcebench/paper_settings.yaml):
the common server arguments, then the model and condition of the job's `row` (if it names one), then the per-task paper
overrides, then the job's own `args`; for repeated options the last one wins. The evaluator arguments (EVAL_ARGS) and the
values of ENV_SPACING and MAX_STEPS come from the same file unless the job sets `eval_args`. With --no-settings only the job's
own `args` and `eval_args` are used. Each job runs under `timeout` (default 5400 s). LLM jobs make paid API calls.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_SETTINGS = REPO / "configs" / "forcebench" / "paper_settings.yaml"
TASK_ABBREV = {"weight_sort": "ws", "rattle_sort": "rs", "faucet_turn": "ft", "faucet_force_turn": "fft", "stove_turn_on": "st",
               "stove_press_turn": "spt", "radmanso_drawer_sort_table": "dr", "nut_thread": "nt"}


def fb_base() -> Path:
    """Same default as scripts/forcebench/env.sh."""
    return Path(os.environ.get("FB_BASE") or Path.home() / "forcebench")


def runs_dir() -> Path:
    return Path(os.environ.get("FB_RUNS") or fb_base() / "runs")


def load_file(path: Path) -> Any:
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:                       # pragma: no cover - PyYAML is a dependency of forceharness
            raise SystemExit(f"reading {path} needs PyYAML (pip install pyyaml): {e}")
        return yaml.safe_load(text)
    return json.loads(text)


def load_settings(path: Optional[Path]) -> Dict[str, Any]:
    return dict(load_file(path) or {}) if path else {}


def _strs(xs) -> List[str]:
    return [str(x) for x in (xs or [])]


def task_args(settings: Dict[str, Any], task: str) -> List[str]:
    """Per-task policy-server arguments of the settings (applied in every row)."""
    t = (settings.get("tasks") or {}).get(task) or {}
    out = _strs(t.get("args"))
    if t.get("prim_overrides") is not None:
        out += ["--prim-overrides", json.dumps(t["prim_overrides"], separators=(",", ":"))]
    return out


def server_args(settings: Dict[str, Any], job: Dict[str, Any]) -> List[str]:
    """Policy-server arguments of one job: common, row (model + condition), task, then the job's own (last one wins)."""
    args = _strs((settings.get("server") or {}).get("args"))
    row_name = job.get("row")
    if row_name:
        rows = settings.get("rows") or {}
        if row_name not in rows:
            raise SystemExit(f"job {job.get('run')}: unknown row {row_name!r} (settings have: {sorted(rows)})")
        row = rows[row_name]
        try:
            args += _strs(settings["models"][row["model"]])
            args += _strs(settings["conditions"][row["condition"]])
        except KeyError as e:
            raise SystemExit(f"row {row_name!r}: unknown model or condition {e}")
    if settings:
        args += task_args(settings, job["task"])
    return args + _strs(job.get("args"))


def eval_env(settings: Dict[str, Any], job: Dict[str, Any], default_eval_args: Optional[str]) -> Dict[str, str]:
    """Environment variables for run.sh that configure the evaluator."""
    ev = settings.get("evaluator") or {}
    if "eval_args" in job:
        eval_args = str(job["eval_args"])
    elif default_eval_args is not None:
        eval_args = default_eval_args
    else:
        eval_args = " ".join(_strs(ev.get("args")))
    env = {"EVAL_ARGS": eval_args}
    for key, var in (("env_spacing", "ENV_SPACING"), ("max_steps", "MAX_STEPS")):
        if ev.get(key) is not None:
            env[var] = str(ev[key])
    return env


def seed_range(settings: Dict[str, Any], row_name: Optional[str] = None) -> Dict[str, int]:
    seeds = dict(settings.get("seeds") or {"start": 300, "count": 10})
    if row_name and ((settings.get("rows") or {}).get(row_name) or {}).get("seeds"):
        seeds.update(settings["rows"][row_name]["seeds"])
    return {"start": int(seeds["start"]), "count": int(seeds["count"])}


def row_jobs(settings: Dict[str, Any], row: str, tasks: List[str], prefix: str = "fb_") -> List[Dict[str, Any]]:
    """The jobs of one row: one run per task over the row's seeds."""
    s = seed_range(settings, row)
    return [dict(run=f"{prefix}{TASK_ABBREV.get(t, t)}_{row}", task=t, seed=s["start"], n=s["count"], row=row) for t in tasks]


def resolve(settings: Dict[str, Any], job: Dict[str, Any], default_eval_args: Optional[str] = None) -> Dict[str, Any]:
    """Everything run.sh needs for one job: the policy-server arguments and the environment variables."""
    for key in ("run", "task"):
        if not job.get(key):
            raise SystemExit(f"job without {key!r}: {job}")
    s = seed_range(settings, job.get("row"))
    env = {"RUN": str(job["run"]), "TASK": str(job["task"]), "SEED": str(job.get("seed", s["start"])),
           "N": str(job.get("n", s["count"]))}
    env.update(eval_env(settings, job, default_eval_args))
    return {"job": job, "args": server_args(settings, job), "env": env}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs", nargs="?", help="job list (.json, .yaml or .yml)")
    ap.add_argument("--row", action="append", default=[], help="add the jobs of this row of the settings (repeatable)")
    ap.add_argument("--tasks", default=None, help="comma-separated tasks for --row (default: every task in the settings)")
    ap.add_argument("--run-prefix", default="fb_", help="prefix of the run names that --row creates")
    ap.add_argument("--settings", type=Path, default=DEFAULT_SETTINGS)
    ap.add_argument("--no-settings", action="store_true", help="use only the jobs' own args / eval_args")
    ap.add_argument("--gpus", default="0:1", help="GPU:max concurrent jobs, comma-separated, e.g. 0:2,1:3")
    ap.add_argument("--port-base", type=int, default=8300)
    ap.add_argument("--log-dir", type=Path, default=None, help="default: $FB_LOG/batch_<name>")
    ap.add_argument("--timeout", type=int, default=5400, help="seconds per job")
    ap.add_argument("--eval-args", default=None, help="evaluator arguments for jobs without eval_args (default: from the settings)")
    ap.add_argument("--dry-run", action="store_true", help="print the resolved jobs and run nothing")
    a = ap.parse_args()

    settings = {} if a.no_settings else load_settings(a.settings)
    jobs: List[Dict[str, Any]] = list(load_file(Path(a.jobs)) or []) if a.jobs else []
    if a.row:
        if not settings:
            raise SystemExit("--row needs the settings file")
        tasks = a.tasks.split(",") if a.tasks else list((settings.get("tasks") or {}))
        for row in a.row:
            jobs += row_jobs(settings, row, tasks, a.run_prefix)
    if not jobs:
        raise SystemExit("nothing to run: give a job list and/or --row")
    resolved = [resolve(settings, j, a.eval_args) for j in jobs]
    if len({r["env"]["RUN"] for r in resolved}) != len(resolved):
        raise SystemExit("run names must be unique")

    runs = runs_dir()
    pending = [r for r in resolved if not (runs / r["env"]["RUN"] / "eval").exists()]
    skipped = len(resolved) - len(pending)
    if a.dry_run:
        for r in pending:
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in r["env"].items())
            print(f"{env} bash {HERE / 'run.sh'} {shlex.join(r['args'])}")
        print(f"# {len(pending)} jobs to run, {skipped} skipped (eval/ exists under {runs})")
        return

    slots = {int(g): int(c) for g, c in (x.split(":") for x in a.gpus.split(","))}
    name = Path(a.jobs).stem if a.jobs else "_".join(a.row)
    log_dir = a.log_dir or Path(os.environ.get("FB_LOG") or fb_base() / "logs") / f"batch_{name}"
    log_dir.mkdir(parents=True, exist_ok=True)
    running = []                                   # (resolved job, gpu, Popen, log file)
    port = a.port_base
    print(f"{time.strftime('%T')} {len(pending)} jobs to run, {skipped} skipped (eval/ exists)", flush=True)
    while pending or running:
        for item in list(running):
            r, gpu, proc, fh = item
            if proc.poll() is not None:
                fh.close()
                running.remove(item)
                summ = runs / r["env"]["RUN"] / "summary.md"
                line = next((l for l in summ.read_text(encoding="utf-8").splitlines() if "ForceBench's criteria" in l), "") \
                    if summ.exists() else "(no summary)"
                print(f"{time.strftime('%T')} done  {r['env']['RUN']} gpu{gpu} rc={proc.returncode} {line[:90]}", flush=True)
        busy = {g: sum(1 for _, gg, _, _ in running if gg == g) for g in slots}
        for g in slots:
            while pending and busy[g] < slots[g]:
                r = pending.pop(0)
                env = dict(os.environ, **r["env"], GPU=str(g), PORT=str(port))
                port += 1
                fh = open(log_dir / f"{r['env']['RUN']}.log", "w")
                proc = subprocess.Popen(["timeout", str(a.timeout), "bash", str(HERE / "run.sh"), *r["args"]], env=env, cwd=str(HERE),
                                        stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                running.append((r, g, proc, fh))
                busy[g] += 1
                print(f"{time.strftime('%T')} start {r['env']['RUN']} gpu{g} port{port - 1}", flush=True)
                time.sleep(3)
        time.sleep(10)
    print(f"{time.strftime('%T')} ALLDONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
