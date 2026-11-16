#!/usr/bin/env bash
# Install ForceBench (pinned) and the three Python environments used to run ForceHarness on it. Run the steps in order; each
# step can be re-run and reuses finished work.
#   bash scripts/forcebench/install.sh repo python     # clone ForceBench at the pinned commit; Python 3.12.13 (via uv if present)
#   bash scripts/forcebench/install.sh tools assets    # asset tools env (usd-core 25.11); download + verify the assets of the 8 tasks
#   bash scripts/forcebench/install.sh sim             # simulator env: torch 2.11.0+cu128, isaaclab[isaacsim]==3.0.0b2.post1, forcebench
#   bash scripts/forcebench/install.sh harness         # policy-server env: this repository with its [forcebench] extra
#   bash scripts/forcebench/install.sh prepare         # import assets into $FORCEBENCH_ASSET_DIR, doctor, Nut checks (no GPU)
#   GPU=0 bash scripts/forcebench/install.sh warmup    # first Isaac Sim start (needs the EULA variables, see below)
#   GPU=0 bash scripts/forcebench/install.sh validate  # ForceBench's hold check (weight_sort, 10 steps; TASKS="a b" for others)
#   bash scripts/forcebench/install.sh all             # = repo python tools assets sim harness prepare (no GPU needed)
# Locations come from env.sh (FB_BASE, FORCEBENCH_ROOT, FB_SIM_ENV or FORCEBENCH_SIM_PYTHON, FB_HARNESS_ENV, ...); an existing
# venv / conda environment given there is reused. Package versions follow the "observed reference environment" of ForceBench's
# docs/getting_started_main.md. Packages come from the official indexes (PyPI, PyTorch, NVIDIA); override FB_PYPI_INDEX,
# FB_TORCH_INDEX or FB_NVIDIA_INDEX only if you must. ISAACLAB_WHEEL=/path/to/isaaclab-3.0.0b2.post1-py3-none-any.whl installs
# a wheel you downloaded yourself (its SHA256 is checked).
# NVIDIA EULA: this script never accepts it. After reading NVIDIA's terms, and only if you accept them, export
# ACCEPT_EULA=Y OMNI_KIT_ACCEPT_EULA=YES ISAACSIM_ACCEPT_EULA=Y yourself before the warmup and validate steps.
set -euo pipefail
. "$(dirname "$0")/env.sh"
mkdir -p "$FB_BASE"
cd "$FB_BASE"

FB_PYPI_INDEX="${FB_PYPI_INDEX:-https://pypi.org/simple}"
FB_TORCH_INDEX="${FB_TORCH_INDEX:-https://download.pytorch.org/whl/cu128}"
FB_NVIDIA_INDEX="${FB_NVIDIA_INDEX:-https://pypi.nvidia.com}"
ISAACLAB_VERSION=3.0.0b2.post1                                          # pulls Isaac Sim 6.0.1.0 through its [isaacsim] extra
ISAACLAB_WHEEL_SHA256=dd32886588479ffd70f7348019aeac1582eb9ad16c40244f41d9f458f96f73c3   # published by ForceBench
SIM_PINS=(torch==2.11.0 numpy==2.3.1 usd-core==25.11)
SIM_EXTRAS=(opencv-python-headless==4.13.0.90 h5py==3.16.0 msgpack==1.2.1 websockets==12.0)
GPU=${GPU:-0}                                                           # CUDA device index (CUDA_DEVICE_ORDER=PCI_BUS_ID matches nvidia-smi)

ensure_env() {  # $1 = interpreter, $2 = prefix where a venv is created if that interpreter does not exist yet
  local py=$1 prefix=$2
  if [ -x "$py" ]; then "$py" -V; return 0; fi
  if [ "$py" != "$prefix/bin/python" ]; then echo "interpreter $py does not exist" >&2; exit 1; fi
  if [ -n "$UV" ]; then
    "$UV" venv --python "$FB_PYTHON_VERSION" --seed "$prefix"
  else
    "${FB_PYTHON:-python3.12}" -m venv "$prefix"
  fi
  "$py" -m pip install --quiet --upgrade pip
  "$py" -V
}

verify_sha256() {  # $1 = file, $2 = expected SHA256
  local got
  got=$(sha256sum "$1" | cut -d' ' -f1)
  [ "$got" = "$2" ] || { echo "SHA256 mismatch for $1: got $got, expected $2" >&2; exit 5; }
}

step_repo() {
  if [ ! -d "$FORCEBENCH_ROOT/.git" ]; then git clone "$FB_REPO_URL" "$FORCEBENCH_ROOT"; fi
  if [ "$(git -C "$FORCEBENCH_ROOT" rev-parse HEAD)" != "$FB_COMMIT" ]; then
    git -C "$FORCEBENCH_ROOT" fetch --quiet origin
    git -C "$FORCEBENCH_ROOT" checkout --quiet "$FB_COMMIT"
  fi
  git -C "$FORCEBENCH_ROOT" log --oneline -1
}

step_python() {
  if [ -n "$UV" ]; then
    "$UV" python install "$FB_PYTHON_VERSION"
    "$UV" python find "$FB_PYTHON_VERSION"
  else
    command -v "${FB_PYTHON:-python3.12}" >/dev/null || {
      echo "need uv (https://docs.astral.sh/uv/) or a Python 3.12 interpreter (set FB_PYTHON=/path/to/python3.12)" >&2; exit 1; }
    "${FB_PYTHON:-python3.12}" -V
  fi
}

step_tools() {
  ensure_env "$FB_TOOLS_PY" "$FB_TOOLS_ENV"
  "$FB_TOOLS_PY" -m pip install -e "$FORCEBENCH_ROOT[assets]" --index-url "$FB_PYPI_INDEX"
  (cd "$FORCEBENCH_ROOT" && "$FB_TOOLS_PY" -m forcebench list-tasks --json) > "$FB_LOG/forcebench_tasks.json"
  echo "ForceBench task list -> $FB_LOG/forcebench_tasks.json"
}

step_assets() {
  mkdir -p "$FB_ASSET_SRC" "$FB_DL_CACHE"
  (cd "$FORCEBENCH_ROOT" && "$FB_TOOLS_PY" tools/download_assets.py --source-dir "$FB_ASSET_SRC" --cache-dir "$FB_DL_CACHE" --workers 4)
  du -sh "$FB_ASSET_SRC" "$FB_DL_CACHE"
  echo "Downloaded assets keep their upstream licenses (some, e.g. the kitchen scenes, are CC BY-NC 4.0);" \
       "see NOTICE.md and docs/assets.md in $FORCEBENCH_ROOT."
}

step_sim() {
  ensure_env "$FORCEBENCH_SIM_PYTHON" "$FB_SIM_ENV"
  local py="$FORCEBENCH_SIM_PYTHON" spec
  "$py" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "the simulator environment needs Python 3.12, found " + sys.version.split()[0])'
  "$py" -m pip install torch==2.11.0 --index-url "$FB_TORCH_INDEX"
  spec="isaaclab[isaacsim]==$ISAACLAB_VERSION"
  if [ -n "${ISAACLAB_WHEEL:-}" ]; then
    verify_sha256 "$ISAACLAB_WHEEL" "$ISAACLAB_WHEEL_SHA256"
    spec="isaaclab[isaacsim] @ file://$(cd "$(dirname "$ISAACLAB_WHEEL")" && pwd)/$(basename "$ISAACLAB_WHEEL")"
  fi
  if ! "$py" -m pip install "$spec" "${SIM_PINS[@]}" --index-url "$FB_PYPI_INDEX" --extra-index-url "$FB_NVIDIA_INDEX"; then
    [ -z "${ISAACLAB_WHEEL:-}" ] || exit 1
    # This prerelease is hosted by NVIDIA and may not resolve through an index list. Fetch the wheel itself from NVIDIA's
    # index, check the SHA256 that ForceBench publishes, and install that file.
    echo "[sim] isaaclab==$ISAACLAB_VERSION did not resolve; downloading the wheel from $FB_NVIDIA_INDEX and checking its SHA256" >&2
    local wd="$FB_BASE/wheels"
    mkdir -p "$wd"
    "$py" -m pip download --no-deps "isaaclab==$ISAACLAB_VERSION" --index-url "$FB_NVIDIA_INDEX" -d "$wd"
    verify_sha256 "$wd/isaaclab-$ISAACLAB_VERSION-py3-none-any.whl" "$ISAACLAB_WHEEL_SHA256"
    "$py" -m pip install "isaaclab[isaacsim] @ file://$wd/isaaclab-$ISAACLAB_VERSION-py3-none-any.whl" "${SIM_PINS[@]}" \
      --index-url "$FB_PYPI_INDEX" --extra-index-url "$FB_NVIDIA_INDEX"
  fi
  "$py" -m pip install "${SIM_EXTRAS[@]}" --index-url "$FB_PYPI_INDEX"
  "$py" -m pip install -e "$FORCEBENCH_ROOT" --index-url "$FB_PYPI_INDEX" --extra-index-url "$FB_NVIDIA_INDEX"
  "$py" -m pip freeze > "$FB_LOG/sim_pip_freeze.txt"
  "$py" - <<'EOF'
import importlib.metadata as md
for p in ["isaacsim", "isaaclab", "torch", "numpy", "usd-core", "warp-lang", "opencv-python-headless", "websockets"]:
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING", e)
import torch; print("torch.cuda", torch.version.cuda)
EOF
}

step_harness() {
  ensure_env "$FB_HARNESS_PY" "$FB_HARNESS_ENV"
  "$FB_HARNESS_PY" -m pip install -e "$FH_CODE[forcebench]" --index-url "$FB_PYPI_INDEX"
  "$FB_HARNESS_PY" -c 'import forceharness, msgpack, websockets; print("policy-server environment ok")'
}

step_prepare() {
  cd "$FORCEBENCH_ROOT"
  "$FORCEBENCH_SIM_PYTHON" tools/prepare_assets.py --source-dir "$FB_ASSET_SRC" --asset-dir "$FORCEBENCH_ASSET_DIR" --check-only || true
  "$FORCEBENCH_SIM_PYTHON" tools/prepare_assets.py --source-dir "$FB_ASSET_SRC" --asset-dir "$FORCEBENCH_ASSET_DIR"
  "$FORCEBENCH_SIM_PYTHON" -m forcebench doctor --asset-dir "$FORCEBENCH_ASSET_DIR"
  "$FORCEBENCH_SIM_PYTHON" tools/prepare_nut_assets.py --asset-dir "$FORCEBENCH_ASSET_DIR" --check-only
  "$FORCEBENCH_SIM_PYTHON" tools/verify_nut_asset_composition.py --asset-dir "$FORCEBENCH_ASSET_DIR"
}

step_warmup() {
  fb_require_eula || exit 3
  cd "$FORCEBENCH_ROOT"
  local out
  out="$FORCEBENCH_SCRATCH_ROOT/validation/warmup-$(date +%Y%m%d-%H%M%S)"
  mkdir -p "$(dirname "$out")"
  # The first start downloads NVIDIA extensions and can take more than ten minutes.
  CUDA_VISIBLE_DEVICES=$GPU "$FORCEBENCH_SIM_PYTHON" scripts/evaluation/inference_forcebench_vectorized.py \
    --task weight_sort --seed_start 300 --num_envs 1 --max_steps 1 --smoke_hold \
    --output_dir "$out" --headless --enable_cameras --device cuda:0 ${FB_KIT_ARGS:+"--kit_args=$FB_KIT_ARGS"} | tee "$out.log"
  # Isaac's exit code is 0 even when it failed (simulation_app.close() swallows the exception): the result line decides.
  grep -q '^FORCEBENCH_VECTOR_RESULT=' "$out.log" || { echo "warmup FAILED (no FORCEBENCH_VECTOR_RESULT line), see $out.log" >&2; exit 4; }
  echo "warmup ok -> $out"
}

step_validate() {
  fb_require_eula || exit 3
  cd "$FORCEBENCH_ROOT"
  local out
  out="$FORCEBENCH_SCRATCH_ROOT/validation/check-$(date +%Y%m%d-%H%M%S)"
  # simpy_kit.sh forwards FB_KIT_ARGS, which validate_simulation.py itself does not pass to the evaluator.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU "$FORCEBENCH_SIM_PYTHON" tools/validate_simulation.py --python "$FH_CODE/scripts/forcebench/simpy_kit.sh" \
    --assets "$FORCEBENCH_ASSET_DIR" --output "$out" --tasks ${TASKS:-weight_sort} \
    --steps 10 --timeout 1200
  echo "validate ok -> $out"
}

step_all() {
  for s in repo python tools assets sim harness prepare; do run_step "$s"; done
}

run_step() {
  echo "===== [$(date '+%F %T')] step $1 ====="
  "step_$1"
  echo "===== [$(date '+%F %T')] step $1 done ====="
}

[ $# -gt 0 ] || { awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1; }
for s in "$@"; do
  declare -F "step_$s" >/dev/null || { echo "unknown step: $s" >&2; exit 1; }
  run_step "$s"
done
