#!/usr/bin/env bash
# Shared settings for the ForceBench scripts (install.sh, run.sh, batch.py). Source it, do not execute it:
#   source scripts/forcebench/env.sh
# Every variable below can be overridden by exporting it before sourcing. By default everything (the ForceBench checkout,
# the three Python environments, assets, runs and logs) lives under one work directory FB_BASE (default ~/forcebench).
#
# Requirements (docs/forcebench.md): Linux with glibc >= 2.35 (Ubuntu 22.04 or 24.04), an NVIDIA RTX GPU with RT cores and
# >= 16 GB of memory, driver >= 550.90.07 (or the opt-in FB_SKIP_DRIVER_CHECK=1 below), and about 40 GB of free disk.
#
# Using an existing simulator environment: set FB_SIM_ENV to the prefix of a venv or conda environment that has (or should
# get) Isaac Sim 6.0.1 + Isaac Lab 3.0.0b2.post1, or set FORCEBENCH_SIM_PYTHON to its interpreter. For a source install of
# Isaac Lab, point FORCEBENCH_SIM_PYTHON at a two-line wrapper script that runs: exec /path/to/IsaacLab/isaaclab.sh -p "$@"
set -a
FH_CODE="${FH_CODE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # ForceHarness repository root
FB_BASE="${FB_BASE:-$HOME/forcebench}"
FORCEBENCH_ROOT="${FORCEBENCH_ROOT:-$FB_BASE/ForceBench}"                 # ForceBench checkout, pinned to FB_COMMIT
FB_REPO_URL="${FB_REPO_URL:-https://github.com/forcebench-platform/ForceBench.git}"
FB_COMMIT=31acf7353d2f723985e6576a5708f67a81707d61                         # the commit the paper's results were obtained with
FB_PYTHON_VERSION=3.12.13
FB_SIM_ENV="${FB_SIM_ENV:-$FB_BASE/envs/sim}"                             # Isaac Sim + Isaac Lab + forcebench (Python 3.12)
FORCEBENCH_SIM_PYTHON="${FORCEBENCH_SIM_PYTHON:-$FB_SIM_ENV/bin/python}"  # ForceBench's own name for the simulator interpreter
FB_TOOLS_ENV="${FB_TOOLS_ENV:-$FB_BASE/envs/tools}"                       # lightweight asset tools (usd-core 25.11)
FB_TOOLS_PY="${FB_TOOLS_PY:-$FB_TOOLS_ENV/bin/python}"
FB_HARNESS_ENV="${FB_HARNESS_ENV:-$FB_BASE/envs/harness}"                 # ForceHarness policy server (no simulator inside)
FB_HARNESS_PY="${FB_HARNESS_PY:-$FB_HARNESS_ENV/bin/python}"
FB_RUNS="${FB_RUNS:-$FB_BASE/runs}"                                       # one directory per run (run.sh / batch.py)
FB_LOG="${FB_LOG:-$FB_BASE/logs}"
FB_ASSET_SRC="${FB_ASSET_SRC:-$FB_BASE/asset-source}"                     # downloaded third-party assets (ForceBench's downloader)
FB_DL_CACHE="${FB_DL_CACHE:-$FB_BASE/download-cache}"
FORCEBENCH_ASSET_DIR="${FORCEBENCH_ASSET_DIR:-$FB_BASE/assets}"           # assets imported and verified by ForceBench
FORCEBENCH_DATA_ROOT="${FORCEBENCH_DATA_ROOT:-$FB_BASE/data}"             # ForceBench requires these three to be distinct
FORCEBENCH_SCRATCH_ROOT="${FORCEBENCH_SCRATCH_ROOT:-$FB_BASE/scratch}"
FORCEBENCH_ARCHIVE_ROOT="${FORCEBENCH_ARCHIVE_ROOT:-$FB_BASE/archive}"
FB_KIT_ARGS="${FB_KIT_ARGS:-}"                                            # extra Kit settings for the evaluator, space-separated
set +a

mkdir -p "$FB_LOG" "$FB_RUNS" "$FORCEBENCH_DATA_ROOT" "$FORCEBENCH_SCRATCH_ROOT" "$FORCEBENCH_ARCHIVE_ROOT"

# Optional: keep the caches of Kit / Omniverse, pip, uv, Warp and CUDA (several GB) under FB_BASE instead of $HOME, e.g. when
# $HOME is a small or network-mounted file system. Off by default because it changes HOME for the commands that follow.
if [ "${FB_ISOLATE_HOME:-0}" = "1" ]; then
  export REAL_HOME="${REAL_HOME:-$HOME}"
  export HOME="$FB_BASE/home"
  mkdir -p "$HOME" "$FB_BASE/tmp"
  export XDG_CACHE_HOME="$HOME/.cache" XDG_DATA_HOME="$HOME/.local/share" XDG_CONFIG_HOME="$HOME/.config"
  export PIP_CACHE_DIR="$HOME/.cache/pip" UV_CACHE_DIR="$HOME/.cache/uv" UV_PYTHON_INSTALL_DIR="$HOME/.local/share/uv/python"
  export CUDA_CACHE_PATH="$HOME/.nv/ComputeCache" WARP_CACHE_PATH="$HOME/.cache/warp" HF_HOME="$HOME/.cache/huggingface"
  export TMPDIR="$FB_BASE/tmp"
fi
export UV="${UV:-$(command -v uv || true)}"                               # optional; used to install Python and create venvs

# Opt-in workaround for drivers older than the minimum of Isaac Sim 6.0.1's RTX renderer (550.90.07). Without it the renderer
# refuses to start ("rtx driver verification failed") and the evaluator fails later with a CUDA error. This only turns the
# version check off: it still needs an RTX GPU with RT cores, it runs outside NVIDIA's tested configuration, and results
# obtained with it should say so.
if [ "${FB_SKIP_DRIVER_CHECK:-0}" = "1" ]; then
  echo "[forcebench/env.sh] WARNING: FB_SKIP_DRIVER_CHECK=1 disables Isaac Sim's RTX driver version check." \
       "This is outside NVIDIA's tested configuration (needs an RTX GPU with RT cores); report it with any results." >&2
  case " $FB_KIT_ARGS " in
    *" --/rtx/verifyDriverVersion/enabled=false "*) ;;
    *) FB_KIT_ARGS="${FB_KIT_ARGS:+$FB_KIT_ARGS }--/rtx/verifyDriverVersion/enabled=false" ;;
  esac
  export FB_KIT_ARGS
fi

# Privacy: PRIVACY_CONSENT is never set here, so no usage data is sent.
# NVIDIA license: these scripts never accept the Isaac Sim EULA for you. Read NVIDIA's terms first (see docs/forcebench.md);
# only if you accept them, export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=Y in your own shell.
fb_require_eula() {
  case "${OMNI_KIT_ACCEPT_EULA:-}" in
    [Yy]|[Yy][Ee][Ss]|1) return 0 ;;
  esac
  {
    echo "Isaac Sim needs its NVIDIA license (EULA) to be accepted, and these scripts do not accept it for you."
    echo "Read the terms linked from the Isaac Lab pip installation guide"
    echo "(https://isaac-sim.github.io/IsaacLab/v3.0.0-beta/source/setup/installation/isaaclab_pip_installation.html)"
    echo "and, only if you accept them, run in this shell:"
    echo "  export ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=Y"
  } >&2
  return 3
}
# Commands pass FB_KIT_ARGS to the evaluator as ${FB_KIT_ARGS:+"--kit_args=$FB_KIT_ARGS"}: the "=" form is required, otherwise
# argparse takes the value (which starts with "--") for an option.
