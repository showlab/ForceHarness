#!/bin/bash
# Distribution-shift test for Diffusion Policy: evaluate a model trained on the original distribution (no retraining) on the
# shifted variant <task>_ood. The shifted variants exist for weight_sort, com_sort, wipe_hard and peg_in_hole_hard.
# No API calls. See docs/baselines.md.
# Usage: bash scripts/baselines/dp/dp_shift_eval.sh <task> <variant> <gpu> <port>
#   variant  a trained variant of dp_pipeline.sh, e.g. dp or dp_force (reads $FH_CKPT/<variant>_<task>/ema_last.pt)
#   gpu      CUDA device index (nvidia-smi order) for the policy server
#   port     local TCP port of the policy server
# Environment (all optional; relative paths are taken from the repository root):
#   FH_CKPT, FH_RUNS, FH_TORCH_PY, FH_PY, EGL_DEVICE  as in dp_pipeline.sh; results go to $FH_RUNS/learned_shift/<variant>_<task>_ood/
#   EVAL_SEEDS    test seeds (default 9100-9149)
set -u
TASK=$1; VAR=$2; GPU=$3; PORT=$4
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 1
FH_CKPT=${FH_CKPT:-checkpoints}; FH_RUNS=${FH_RUNS:-runs}
TPY=${FH_TORCH_PY:-python3}; FH_PY=${FH_PY:-python3}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=${MUJOCO_GL:-egl}
CKPT=$FH_CKPT/${VAR}_$TASK/ema_last.pt
RUN=$FH_RUNS/learned_shift/${VAR}_${TASK}_ood
mkdir -p "$RUN"
EGL=${EGL_DEVICE:-$GPU}
[ -f "$CKPT" ] || { echo "no checkpoint $CKPT"; exit 1; }
echo "[$(date +%T)] serve $CKPT on GPU $GPU port $PORT -> eval ${TASK}_ood"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  "$TPY" scripts/baselines/dp/dp_server.py --ckpt "$CKPT" --port "$PORT" > "$RUN/server.log" 2>&1 &
SPID=$!
# Wait until the server is ready (its log prints "on 127.0.0.1"); on a busy machine this can take minutes.
for i in $(seq 1 120); do grep -q "on 127.0.0.1" "$RUN/server.log" 2>/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 5; done
grep -q "on 127.0.0.1" "$RUN/server.log" 2>/dev/null || { echo "[$(date +%T)] FAILED: server not ready, see $RUN/server.log"; kill $SPID 2>/dev/null; exit 1; }
OMP_NUM_THREADS=1 "$FH_PY" scripts/baselines/dp/eval_policy.py --task "${TASK}_ood" --name "$VAR" --ports "$PORT" --seeds "${EVAL_SEEDS:-9100-9149}" \
  --run "$RUN" --egl "$EGL" --workers 6 > "$RUN/eval.log" 2>&1
kill $SPID
echo "[$(date +%T)] done $VAR ${TASK}_ood: $(tail -n 1 "$RUN/eval.log")"
