"""Environment checks for ForceHarness, written for humans and for coding agents.

    python -m forceharness.doctor                      # core checks + every target, human-readable
    python -m forceharness.doctor --target sim         # sim | forcebench | real | all
    python -m forceharness.doctor --target real --probe-robot      # also try a read-only ssh to the robot PC
    python -m forceharness.doctor --backend mypkg.robot:MyBackend  # check a custom backend class against the contract
    python -m forceharness.doctor --json               # machine-readable report

Every check returns PASS, WARN, FAIL or SKIP with a one-line hint.  Nothing here moves a robot, spends API credit,
or prints a secret: API keys are reported only as set / not set.  Exit code 0 means no FAIL.
"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

ROOT = Path(__file__).resolve().parents[1]
PKG = Path(__file__).resolve().parent

# GPUs without RT cores cannot render Isaac Sim (ForceBench); substrings matched against `nvidia-smi` names.
NO_RT_CORES = ("H100", "H200", "H800", "A100", "A800", "A30", "V100", "P100", "T4", "B100", "B200", "GH200")
ISAAC_MIN_DRIVER = (550, 90, 7)


@dataclass
class Check:
    target: str
    name: str
    status: str          # PASS | WARN | FAIL | SKIP
    detail: str
    hint: str = ""


def _version(mod: str) -> Optional[str]:
    try:
        m = importlib.import_module(mod)
    except Exception:
        return None
    return str(getattr(m, "__version__", "installed"))


def _run(cmd: List[str], timeout: float = 10.0) -> Optional[str]:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return None


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass


# ----------------------------------------------------------------------------------------------------------- core
def check_core() -> List[Check]:
    out: List[Check] = []
    v = sys.version_info
    out.append(Check("core", "python", "PASS" if v >= (3, 10) else "FAIL", platform.python_version(),
                     "" if v >= (3, 10) else "ForceHarness needs Python >= 3.10"))
    out.append(Check("core", "platform", "PASS", f"{platform.system()} {platform.release()} {platform.machine()}"))
    for mod, need, why in (("numpy", True, ""), ("scipy", True, ""), ("yaml", True, ""), ("PIL", True, ""),
                           ("matplotlib", True, ""), ("openai", False, "needed for OpenAI models"),
                           ("anthropic", False, "needed for Claude models"), ("dotenv", False, "reads .env")):
        ver = _version(mod)
        out.append(Check("core", f"import {mod}", "PASS" if ver else ("FAIL" if need else "WARN"), ver or "missing",
                         "" if ver else f"pip install -e .   (from the repository root){'; ' + why if why else ''}"))
    prompts = sorted(p.relative_to(PKG).as_posix() for p in (PKG / "prompts").rglob("*.md"))
    out.append(Check("core", "prompt files", "PASS" if prompts else "FAIL", f"{len(prompts)} files",
                     "" if prompts else "forceharness/prompts is missing; reinstall the package"))
    _load_dotenv()
    for var, what in (("FH_API_KEY", "OpenAI (GPT-6 Astra / Sol)"), ("OPENAI_API_KEY", "OpenAI fallback"),
                      ("ANTHROPIC_API_KEY", "Anthropic (Claude)")):
        out.append(Check("core", f"env {var}", "PASS" if os.environ.get(var) else "SKIP",
                         "set" if os.environ.get(var) else "not set",
                         "" if os.environ.get(var) else f"needed only for {what} episodes; put it in .env"))
    return out


# ------------------------------------------------------------------------------------------------------------ sim
def check_sim() -> List[Check]:
    out: List[Check] = []
    mj = _version("mujoco")
    if mj is None:
        out.append(Check("sim", "mujoco", "FAIL", "missing", "pip install -e '.[sim]'   (installs mujoco==3.2.7)"))
    else:
        out.append(Check("sim", "mujoco", "PASS" if mj == "3.2.7" else "FAIL", mj,
                         "" if mj == "3.2.7" else "robosuite 1.5.1 needs mujoco==3.2.7 exactly: pip install mujoco==3.2.7"))
    rs = _version("robosuite")
    out.append(Check("sim", "robosuite", "PASS" if rs and rs.startswith("1.5") else ("FAIL" if rs is None else "WARN"),
                     rs or "missing", "" if rs and rs.startswith("1.5") else "pip install robosuite==1.5.1"))
    cv = _version("cv2")
    cv_new = bool(cv and re.match(r"(\d+)\.(\d+)", cv) and tuple(map(int, re.match(r"(\d+)\.(\d+)", cv).groups())) >= (4, 13))
    out.append(Check("sim", "opencv", "PASS" if cv and not cv_new else "WARN", cv or "missing",
                     "needed for camera images and demo recording: pip install 'opencv-python-headless<4.13'" if not cv else
                     ("versions >= 4.13 are untested here; pip install 'opencv-python-headless<4.13'" if cv_new else "")))
    gl = os.environ.get("MUJOCO_GL", "")
    sysname = platform.system()
    if sysname == "Darwin":
        ok = gl in ("", "cgl", "glfw")
        out.append(Check("sim", "MUJOCO_GL", "PASS" if ok else "WARN", gl or "unset (ForceHarness uses cgl on macOS)",
                         "" if ok else "on macOS use MUJOCO_GL=cgl (off-screen) or glfw"))
    elif sysname == "Linux":
        smi = _run(["nvidia-smi", "-L"])
        if gl in ("", "egl"):
            out.append(Check("sim", "MUJOCO_GL", "PASS" if smi else "WARN", f"{gl or 'egl (default)'}; "
                             f"{'NVIDIA GPU found' if smi else 'no nvidia-smi'}",
                             "" if smi else "without a GPU use MUJOCO_GL=osmesa (slow) or install EGL drivers"))
            if smi:
                out.append(Check("sim", "EGL device", "PASS", "set MUJOCO_EGL_DEVICE_ID to a free GPU",
                                 "EGL device numbering can differ from nvidia-smi numbering; check with a short render"))
        else:
            out.append(Check("sim", "MUJOCO_GL", "PASS", gl))
    else:
        out.append(Check("sim", "MUJOCO_GL", "WARN", f"{sysname}: untested", "use Linux or macOS"))
    if mj == "3.2.7" and rs:
        out.append(Check("sim", "next step", "PASS", "python scripts/check_sim.py --tasks weight_sort",
                         "builds the scene, checks the force sign, runs the expert once (no API key needed)"))
    return out


# ----------------------------------------------------------------------------------------------------- forcebench
def _driver_version() -> Optional[tuple]:
    s = _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"])
    if not s:
        return None
    m = re.match(r"(\d+)\.(\d+)(?:\.(\d+))?", s.strip().splitlines()[0])
    return tuple(int(x or 0) for x in m.groups()) if m else None


def check_forcebench(requested: bool = True) -> List[Check]:
    out: List[Check] = []
    sysname = platform.system()
    if sysname != "Linux":
        return [Check("forcebench", "os", "FAIL" if requested else "SKIP", sysname,
                      "ForceBench (Isaac Sim 6.0.1 / Isaac Lab 3.0) needs Linux x86_64, Ubuntu 22.04 or newer")]
    libc = platform.libc_ver()
    lv = tuple(int(x) for x in libc[1].split(".")[:2]) if libc[1] else (0, 0)
    out.append(Check("forcebench", "glibc", "PASS" if lv >= (2, 35) else "FAIL", ".".join(map(str, lv)),
                     "" if lv >= (2, 35) else "Isaac Sim 6 needs glibc >= 2.35 (Ubuntu 22.04); use a container otherwise"))
    names = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
    if not names:
        out.append(Check("forcebench", "gpu", "FAIL", "no nvidia-smi", "an NVIDIA RTX-class GPU is required"))
    else:
        first = names.strip().splitlines()[0]
        bad = any(k in first for k in NO_RT_CORES)
        out.append(Check("forcebench", "gpu", "FAIL" if bad else "PASS", first,
                         "this GPU has no RT cores; Isaac Sim rendering will fail" if bad else ""))
    dv = _driver_version()
    if dv:
        ok = dv >= ISAAC_MIN_DRIVER
        out.append(Check("forcebench", "driver", "PASS" if ok else "WARN", ".".join(map(str, dv)),
                         "" if ok else "below 550.90.07: Isaac refuses to render unless you opt in to the driver-check "
                                       "workaround described in docs/forcebench.md"))
    root = os.environ.get("FORCEBENCH_ROOT")
    out.append(Check("forcebench", "FORCEBENCH_ROOT", "PASS" if root and Path(root).is_dir() else "WARN", root or "not set",
                     "" if root and Path(root).is_dir() else "set it after running scripts/forcebench/install.sh"))
    for mod in ("websockets", "msgpack"):
        ver = _version(mod)
        out.append(Check("forcebench", f"import {mod}", "PASS" if ver else "FAIL", ver or "missing",
                         "" if ver else "pip install -e '.[forcebench]'"))
    return out


# ----------------------------------------------------------------------------------------------------------- real
def _real_config_path() -> Optional[Path]:
    for p in (os.environ.get("FH_ROBOT_CONFIG"), ROOT / "configs/real/robot.json"):
        if p and Path(p).is_file():
            return Path(p)
    return None


def _find_ssh_host(cfg) -> Optional[str]:
    """The robot-PC ssh host: ``robot_pc.ssh_host`` in configs/real/robot.json (other schemas: any *host key)."""
    host = ((cfg or {}).get("robot_pc") or {}).get("ssh_host") if isinstance(cfg, dict) else None
    if isinstance(host, str) and host and "<" not in host:
        return host
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            if isinstance(v, str) and not k.startswith("_") and re.search(r"(ssh_)?host$", k) and v and "<" not in v:
                return v
            if isinstance(v, dict):
                h = _find_ssh_host(v)
                if h:
                    return h
    return None


def check_real(probe_robot: bool = False) -> List[Check]:
    out: List[Check] = []
    cfg_path = _real_config_path()
    if cfg_path is None:
        out.append(Check("real", "robot config", "WARN", "configs/real/robot.json not found",
                         "copy configs/real/robot.example.json to configs/real/robot.json and fill it in "
                         "(or set FH_ROBOT_CONFIG); see docs/real_robot.md"))
        cfg = None
    else:
        try:
            cfg = json.loads(cfg_path.read_text())
            left = re.findall(r"<[^<>]+>", cfg_path.read_text())
            out.append(Check("real", "robot config", "WARN" if left else "PASS", str(cfg_path),
                             f"unfilled placeholders: {', '.join(sorted(set(left))[:5])}" if left else ""))
        except Exception as e:
            cfg = None
            out.append(Check("real", "robot config", "FAIL", f"{cfg_path}: {e}", "fix the JSON"))
    out.append(Check("real", "ssh client", "PASS" if shutil.which("ssh") else "FAIL", shutil.which("ssh") or "missing",
                     "" if shutil.which("ssh") else "the workstation reaches the robot PC over ssh"))
    rs = _version("pyrealsense2")
    out.append(Check("real", "pyrealsense2", "PASS" if rs else "WARN", rs or "missing",
                     "" if rs else "needed for the wrist / exterior RealSense cameras: pip install pyrealsense2"))
    if rs:
        try:
            import pyrealsense2 as rs2
            devs = [d.get_info(rs2.camera_info.name) for d in rs2.context().query_devices()]
            out.append(Check("real", "realsense devices", "PASS" if devs else "WARN", f"{len(devs)}: {', '.join(devs)}",
                             "" if devs else "no camera connected"))
        except Exception as e:
            out.append(Check("real", "realsense devices", "WARN", f"query failed: {e}"))
    host = _find_ssh_host(cfg) if cfg else None
    if probe_robot:
        if not host:
            out.append(Check("real", "robot PC ssh", "SKIP", "no host in the robot config"))
        else:
            r = _run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, "echo ok; uname -sr; python3 --version"],
                     timeout=15)
            ok = bool(r and r.startswith("ok"))
            out.append(Check("real", "robot PC ssh", "PASS" if ok else "FAIL", (r or "no answer").strip().replace("\n", " | "),
                             "" if ok else f"set up key-based ssh to {host} (BatchMode); the harness never asks for passwords"))
    else:
        out.append(Check("real", "robot PC ssh", "SKIP", "not probed", "rerun with --probe-robot (read-only: runs echo/uname)"))
    out.append(Check("real", "force channel", "SKIP", "needs the robot",
                     "follow skills/forceharness-setup/references/force-check.md (read-only noise test first)"))
    return out


# -------------------------------------------------------------------------------------------------- custom backend
def check_backend(spec: str) -> List[Check]:
    from forceharness.backends.base import OPTIONAL_MEMBERS, describe_backend

    mod, _, cls = spec.partition(":")
    try:
        obj = getattr(importlib.import_module(mod), cls)
    except Exception as e:
        return [Check("backend", spec, "FAIL", f"cannot import: {e}", "use the form package.module:ClassName")]
    d = describe_backend(obj)
    out = [Check("backend", "required members", "FAIL" if d["missing_required"] else "PASS",
                 "missing: " + ", ".join(d["missing_required"]) if d["missing_required"] else "all present",
                 "see forceharness/backends/base.py" if d["missing_required"] else "")]
    for key in ("rezero_bias", "set_kp", "kp"):
        out.append(Check("backend", f"optional {key}", "PASS" if key in d["optional_present"] else "WARN",
                         "present" if key in d["optional_present"] else "absent", OPTIONAL_MEMBERS[key]))
    return out


# ----------------------------------------------------------------------------------------------------------- main
TARGETS: dict = {"sim": check_sim, "forcebench": check_forcebench, "real": check_real}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", choices=["sim", "forcebench", "real", "all"], default="all")
    ap.add_argument("--probe-robot", action="store_true", help="read-only ssh to the robot PC configured in configs/real/robot.json")
    ap.add_argument("--backend", help="check a custom backend class, e.g. mypkg.robot:MyBackend")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    checks = check_core()
    for name, fn in TARGETS.items():
        if args.target in (name, "all"):
            if name == "real":
                checks += fn(args.probe_robot)
            elif name == "forcebench":
                checks += fn(requested=args.target == "forcebench")    # on a Mac, "all" should not fail because of Isaac
            else:
                checks += fn()
    if args.backend:
        checks += check_backend(args.backend)
    if args.json:
        print(json.dumps([asdict(c) for c in checks], indent=1))
    else:
        width = max(len(c.name) for c in checks)
        cur = None
        for c in checks:
            if c.target != cur:
                cur = c.target
                print(f"\n[{cur}]")
            print(f"  {c.status:4s}  {c.name:<{width}}  {c.detail}" + (f"\n        -> {c.hint}" if c.hint and c.status != "PASS" else ""))
        n = {s: sum(c.status == s for c in checks) for s in ("PASS", "WARN", "FAIL", "SKIP")}
        print(f"\n{n['PASS']} pass, {n['WARN']} warn, {n['FAIL']} fail, {n['SKIP']} skip")
    return 1 if any(c.status == "FAIL" for c in checks) else 0


if __name__ == "__main__":
    sys.exit(main())
