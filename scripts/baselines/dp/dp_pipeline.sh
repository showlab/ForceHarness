#!/bin/bash
# Diffusion Policy pipeline for one task and one variant: train -> start the inference server -> evaluate on the test seeds
# -> stop the server. No API calls. See docs/baselines.md.
# Usage: bash scripts/baselines/dp/dp_pipeline.sh <task> <variant> <gpu> <port> [steps]
#   task     weight_sort | com_sort | wipe_hard | stack_gentle | peg_in_hole_hard | door_latch | lock_or_stiff
#   variant  dp | dp_force | dp50 | dp50_force | dpnoscene | dpnoscene_force
#              *force*    add the 6-D wrist wrench to the low-dim observation (train_dp.py --force)
#              dp50*      use only the first 50 demos by seed (data-size ablation; otherwise up to 200)
#              *noscene*  no noisy scene vector: images + proprioception (+ force), the ForceBench-style input (--no-scene)
#   gpu      CUDA device index (nvidia-smi order) for training and for the policy server
#   port     local TCP port of the policy server
#   steps    training steps (default 40000)
# Environment (all optional; relative paths are taken from the repository root):
#   FH_DEMOS      demonstration root with <task>/s*.npz              (default demos/hiddenphys)
#   FH_CKPT       checkpoint root, writes <variant>_<task>/              (default checkpoints)
#   FH_RUNS       results root, writes learned/<variant>_<task>/        (default runs)
#   FH_TORCH_PY   python of the torch environment (training, server)  (default python3)
#   FH_PY         python of the simulator environment (eval client)   (default python3)
#   EGL_DEVICE    MUJOCO_EGL_DEVICE_ID used by the eval client         (default: <gpu>; EGL and CUDA numbering can differ
#                 on multi-GPU machines, so check which EGL device is the GPU you want)
#   EVAL_SEEDS    test seeds (default 9100-9119, the 20 seeds shared by all rows; 9100-9149 for 50 episodes)
#   RECORD_LOG    if set, first wait until this demo-recorder log reports "[record] <task>: ... saved"
#                 (useful when recording and training run concurrently)
set -u
TASK=$1; VAR=$2; GPU=$3; PORT=$4; STEPS=${5:-40000}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 1
FH_DEMOS=${FH_DEMOS:-demos/hiddenphys}; FH_CKPT=${FH_CKPT:-checkpoints}; FH_RUNS=${FH_RUNS:-runs}
TPY=${FH_TORCH_PY:-python3}; FH_PY=${FH_PY:-python3}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=${MUJOCO_GL:-egl}
DEMOS=$FH_DEMOS/$TASK
OUT=$FH_CKPT/${VAR}_$TASK
RUN=$FH_RUNS/learned/${VAR}_$TASK
mkdir -p "$OUT" "$RUN"
EGL=${EGL_DEVICE:-$GPU}
FORCE_FLAG=""; [[ "$VAR" == *force* ]] && FORCE_FLAG="--force"
NDEMO=200; [[ "$VAR" == dp50* ]] && NDEMO=50
SCENE_FLAG=""; [[ "$VAR" == *noscene* ]] && SCENE_FLAG="--no-scene"

if [ -n "${RECORD_LOG:-}" ]; then
  echo "[$(date +%T)] wait for demos of $TASK in $RECORD_LOG"
  until grep -q "^\[record\] $TASK: .* saved" "$RECORD_LOG" 2>/dev/null; do sleep 30; done
fi

# Train unless a finished checkpoint for this many steps already exists.
if [ ! -f "$OUT/ema_last.pt" ] || ! grep -q "\"step\": $STEPS" "$OUT/train_log.jsonl" 2>/dev/null; then
  echo "[$(date +%T)] train $VAR $TASK on GPU $GPU"
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU OMP_NUM_THREADS=2 \
    "$TPY" scripts/baselines/dp/train_dp.py --demos "$DEMOS" --out "$OUT" $FORCE_FLAG --n-demos $NDEMO $SCENE_FLAG --steps "$STEPS" \
    --save-every 10000 --workers 8 > "$OUT/train.log" 2>&1
fi

[ -f "$OUT/ema_last.pt" ] || { echo "[$(date +%T)] FAILED: training produced no checkpoint, see $OUT/train.log"; exit 1; }
echo "[$(date +%T)] serve + eval $VAR $TASK (GPU $GPU, EGL $EGL, port $PORT)"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU \
  "$TPY" scripts/baselines/dp/dp_server.py --ckpt "$OUT/ema_last.pt" --port "$PORT" > "$OUT/server.log" 2>&1 &
SPID=$!
# Wait until the server is ready (its log prints "on 127.0.0.1"); on a busy machine this can take minutes, and clients that
# connect too early fail.
for i in $(seq 1 120); do grep -q "on 127.0.0.1" "$OUT/server.log" 2>/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 5; done
grep -q "on 127.0.0.1" "$OUT/server.log" 2>/dev/null || { echo "[$(date +%T)] FAILED: server not ready, see $OUT/server.log"; kill $SPID 2>/dev/null; exit 1; }
OMP_NUM_THREADS=1 "$FH_PY" scripts/baselines/dp/eval_policy.py --task "$TASK" --name "$VAR" --ports "$PORT" --seeds "${EVAL_SEEDS:-9100-9119}" \
  --run "$RUN" --egl "$EGL" --workers 6 > "$OUT/eval.log" 2>&1
kill $SPID
echo "[$(date +%T)] done $VAR $TASK: $(tail -n 1 "$OUT/eval.log")"
