#!/bin/bash
# Evaluate a multi-task openpi model (pi0.5 f0 / f2 or ForceVLA) on HiddenPhys. Starts one inference server
# (pi05_server.py, openpi environment) and runs the evaluation client (scripts/baselines/dp/eval_policy.py, simulator
# environment) for the 7 tasks and the 4 shifted variants in parallel, all connected to that server. Same test seeds and the
# same physical success check as every other row. Shifted variants are not retrained: the server maps <task>_ood to <task>
# (PolicyServer.base_task), so they get the original task's config and prompt. No API calls. See docs/baselines.md.
# Usage: bash scripts/baselines/pi05/eval_multitask.sh <config> <checkpoint step dir> <gpu> <port> <name>
#   config   pi05_fh_f0 | pi05_fh_f2 | forcevla_fh
#   name     row name in the results, e.g. pi05_f0, pi05_f2, forcevla
# Environment (all optional; relative paths are taken from the repository root):
#   FH_RUNS        results root: learned/<name>_<task>/, learned_shift/<name>_<task>_ood/   (default runs)
#   FH_OPENPI_PY   python of the openpi environment, for the server     (default python3)
#   FH_PY          python of the simulator environment, for the clients (default python3)
#   EGL_DEVICE     MUJOCO_EGL_DEVICE_ID used by the clients (default: <gpu>; EGL and CUDA numbering can differ)
#   EVAL_SEEDS     test seeds (default 9100-9119)
#   SET            id (the 7 tasks) | shift (the 4 shifted variants) | all (default)
#   FH_SERVE_MEM_FRACTION  JAX memory fraction of the server (default 0.35, as used on a 24 GB GPU; it replaces any
#                  XLA_PYTHON_CLIENT_MEM_FRACTION exported for training)
#   OPENPI_ROOT and OPENPI_DATA_HOME are passed on to the server.
set -u
CFG=$1; CKPT=$2; GPU=$3; PORT=$4; NAME=$5
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$ROOT" || exit 1
FH_RUNS=${FH_RUNS:-runs}; OPY=${FH_OPENPI_PY:-python3}; FH_PY=${FH_PY:-python3}
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}" MUJOCO_GL=${MUJOCO_GL:-egl}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${FH_SERVE_MEM_FRACTION:-0.35} TF_CPP_MIN_LOG_LEVEL=2
EGL=${EGL_DEVICE:-$GPU}
SEEDS=${EVAL_SEEDS:-9100-9119}
RUN=$FH_RUNS/learned; RSHIFT=$FH_RUNS/learned_shift
mkdir -p "$RUN" "$RSHIFT"
SET=${SET:-all}
LOG=$RUN/${NAME}_server.log; [ "$SET" = all ] || LOG=$RUN/${NAME}_${SET}_server.log
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPU "$OPY" scripts/baselines/pi05/pi05_server.py --config "$CFG" --ckpt "$CKPT" \
  --port "$PORT" > "$LOG" 2>&1 &
SPID=$!
# The server compiles and runs two warm-up inferences before it prints "ready on 127.0.0.1"; clients that connect earlier fail.
for i in $(seq 1 240); do grep -q "ready on 127.0.0.1" "$LOG" 2>/dev/null && break; kill -0 $SPID 2>/dev/null || break; sleep 5; done
grep -q "ready on 127.0.0.1" "$LOG" 2>/dev/null || { echo "[$(date +%T)] FAILED: server not ready, see $LOG"; kill $SPID 2>/dev/null; exit 1; }
echo "[$(date +%T)] server ready: $CFG $CKPT on GPU $GPU port $PORT"
PIDS=()
[ "$SET" = shift ] || for t in weight_sort com_sort wipe_hard stack_gentle peg_in_hole_hard door_latch lock_or_stiff; do
  OMP_NUM_THREADS=1 "$FH_PY" scripts/baselines/dp/eval_policy.py --task $t --name "$NAME" --ports "$PORT" --seeds "$SEEDS" \
    --run "$RUN/${NAME}_$t" --egl "$EGL" --workers 4 > "$RUN/${NAME}_$t.eval.log" 2>&1 &
  PIDS+=($!)
done
[ "$SET" = id ] || for t in weight_sort com_sort wipe_hard peg_in_hole_hard; do
  OMP_NUM_THREADS=1 "$FH_PY" scripts/baselines/dp/eval_policy.py --task ${t}_ood --name "$NAME" --ports "$PORT" \
    --seeds "$SEEDS" --run "$RSHIFT/${NAME}_${t}_ood" --egl "$EGL" --workers 3 > "$RSHIFT/${NAME}_${t}_ood.eval.log" 2>&1 &
  PIDS+=($!)
done
wait "${PIDS[@]}"
kill $SPID
echo "[$(date +%T)] done $NAME ($SET):"; for f in "$RUN"/${NAME}_*.eval.log "$RSHIFT"/${NAME}_*.eval.log; do [ -f "$f" ] && tail -n 1 "$f"; done
