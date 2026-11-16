#!/usr/bin/env bash
# One ForceBench run: start the ForceHarness policy server, run our evaluator copy (one action per control tick), stop the
# server and merge the records of both sides into $FB_RUNS/$RUN/{merged.jsonl,summary.md}.
#   RUN=dr_scripted TASK=radmanso_drawer_sort_table SEED=300 N=4 GPU=0 bash scripts/forcebench/run.sh --agent scripted
#   RUN=dr_sol TASK=radmanso_drawer_sort_table SEED=300 N=10 bash scripts/forcebench/run.sh \
#       --agent llm --model gpt-6.1-sol --provider openai-responses --reasoning-effort low
# Arguments after the script name go to scripts/forcebench/policy_server.py unchanged. The episodes use seeds SEED..SEED+N-1
# as N parallel environments of one Isaac process. API keys come from the environment or from the repository's .env file,
# which the policy server reads (variables already set in the environment take precedence).
#
# The defaults are the paper's setting: RGB-only perception (the policy server's default --perception rgb) and an evaluator
# that sends the camera calibration but no object geometry. Privileged mode (noisy true object positions instead of images)
# needs both --perception privileged and EVAL_ARGS=--privileged_geometry. configs/forcebench/paper_settings.yaml lists the
# remaining per-task settings of the paper; scripts/forcebench/batch.py applies them.
#
# Environment: RUN and TASK (required); SEED=300, N=4, GPU=0 (CUDA device index), PORT=8102, MAX_STEPS=1500 (100 s at 15 Hz),
# ENV_SPACING=30 (metres between parallel environments; at ForceBench's default of 6 m the kitchen scenes darken each other
# and thin objects blur in later environments), EVAL_ARGS (extra evaluator arguments), STARTUP_TIMEOUT=600 and
# STALL_TIMEOUT=480 (seconds; watchdogs, see below; STALL_TIMEOUT=0 disables the second one).
# Isaac's exit code is always 0 (simulation_app.close() swallows exceptions), so the outcome is read from the
# FORCEBENCH_VECTOR_RESULT line of eval.log and from eval/seed_*/summary.json, never from an exit code.
set -euo pipefail
. "$(dirname "$0")/env.sh"
: "${RUN:?set RUN=<run name>}" "${TASK:?set TASK=<ForceBench task, e.g. weight_sort>}"
GPU=${GPU:-0}; SEED=${SEED:-300}; N=${N:-4}; PORT=${PORT:-8102}; MAX_STEPS=${MAX_STEPS:-1500}
ENV_SPACING=${ENV_SPACING:-30}
EVAL_ARGS=${EVAL_ARGS:-}
fb_require_eula || exit 3
[ -x "$FORCEBENCH_SIM_PYTHON" ] || { echo "simulator Python not found: $FORCEBENCH_SIM_PYTHON (install.sh sim, or set FORCEBENCH_SIM_PYTHON)" >&2; exit 2; }
[ -x "$FB_HARNESS_PY" ] || { echo "policy-server Python not found: $FB_HARNESS_PY (install.sh harness, or set FB_HARNESS_PY)" >&2; exit 2; }

# The policy server's perception must match what the evaluator sends.
perception=rgb; prev=""
for a in "$@"; do
  case "$a" in --perception=*) perception=${a#*=} ;; esac
  if [ "$prev" = "--perception" ]; then perception=$a; fi
  prev=$a
done
privileged=0
for a in $EVAL_ARGS; do
  case "$a" in --privileged_geometry) privileged=1 ;; --no-privileged_geometry) privileged=0 ;; esac
done
if [ "$perception" = privileged ] && [ "$privileged" = 0 ]; then
  echo "--perception privileged needs EVAL_ARGS=--privileged_geometry (by default the evaluator sends no object geometry)" >&2; exit 2
fi
if [ "$perception" != privileged ] && [ "$privileged" = 1 ]; then
  echo "warning: EVAL_ARGS has --privileged_geometry but perception is $perception: true geometry is sent but not used" >&2
fi

R=$FB_RUNS/$RUN
mkdir -p "$R"
[ -e "$R/eval" ] && { echo "$R/eval already exists; choose another RUN name" >&2; exit 2; }

"$FB_HARNESS_PY" "$FH_CODE/scripts/forcebench/policy_server.py" --task "$TASK" --port "$PORT" \
  --run-name "$RUN" --out "$FB_RUNS" --max-steps "$MAX_STEPS" "$@" > "$R/server.log" 2>&1 < /dev/null &
SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
for _ in $(seq 60); do grep -q FH_POLICY_SERVER_READY "$R/server.log" 2>/dev/null && break; sleep 1; done
grep -q FH_POLICY_SERVER_READY "$R/server.log" || { echo "the policy server did not start:" >&2; tail -n 20 "$R/server.log" >&2; exit 3; }

cd "$FORCEBENCH_ROOT"
# Startup watchdog: Kit sometimes hangs during startup on a thread lock (futex, CPU ~ 0, not yet loaded) for tens of minutes.
# If the evaluator prints no progress line within STARTUP_TIMEOUT seconds it is killed with -9 and started once more under a
# new log name (no episode has started yet, so the policy server keeps running).
run_eval() {
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES=$GPU "$FORCEBENCH_SIM_PYTHON" "$FH_CODE/scripts/forcebench/eval_forcebench.py" --task "$TASK" --seed_start "$SEED" --num_envs "$N" \
    --env_spacing "$ENV_SPACING" --policy_host 127.0.0.1 --policy_port "$PORT" --policy_adapter forcebench_ws --execute_chunk 1 \
    --max_steps "$MAX_STEPS" --save_rollouts --save_images --output_dir "$R/eval" --headless --enable_cameras --device cuda:0 \
    ${FB_KIT_ARGS:+"--kit_args=$FB_KIT_ARGS"} ${EVAL_ARGS:-} > "$R/eval.log" 2>&1 < /dev/null &
  local epid=$! t=0 last_size=0 still=0
  while kill -0 "$epid" 2>/dev/null; do
    sleep 10; t=$((t + 10))
    if [ "$t" -ge "${STARTUP_TIMEOUT:-600}" ] && ! grep -aq 'FORCEBENCH_VECTOR_PROGRESS\|FORCEBENCH_VECTOR_RESULT' "$R/eval.log"; then
      echo "no evaluator progress after ${t}s: startup hang, kill -9" >&2; kill -9 "$epid" 2>/dev/null || true; break
    fi
    # Stall watchdog: the evaluator can keep spinning (CPU 50-170%) inside Isaac / PhysX without making progress, and SIGTERM
    # does not stop it. A stall is declared when neither the evaluator log nor the policy-server log has grown for
    # STALL_TIMEOUT seconds; while a model thinks, the policy server logs every API request, so model episodes are not killed.
    if [ "${STALL_TIMEOUT:-480}" -gt 0 ] && grep -aq 'FORCEBENCH_VECTOR_PROGRESS' "$R/eval.log"; then
      local size; size=$(( $(stat -c %s "$R/eval.log" 2>/dev/null || echo 0) + $(stat -c %s "$R/server.log" 2>/dev/null || echo 0) ))
      if [ "$size" -eq "$last_size" ]; then still=$((still + 10)); else still=0; last_size=$size; fi
      if [ "$still" -ge "${STALL_TIMEOUT:-480}" ]; then
        echo "no new evaluator output for ${still}s: stalled, kill -9" >&2
        echo "FH_EVAL_STALLED last=$(grep -a 'FORCEBENCH_VECTOR_PROGRESS' "$R/eval.log" | tail -n 1)" >> "$R/eval.log"
        kill -9 "$epid" 2>/dev/null || true; break
      fi
    fi
  done
  wait "$epid" 2>/dev/null || true
}
run_eval
if ! grep -aq '^FORCEBENCH_VECTOR_RESULT=' "$R/eval.log" && ! grep -aq 'FORCEBENCH_VECTOR_PROGRESS' "$R/eval.log"; then
  mv "$R/eval.log" "$R/eval.log.startup_hang1"; rm -rf "$R/eval"
  echo "starting the evaluator once more" >&2; run_eval
fi
grep -a '^FORCEBENCH_VECTOR_RESULT=' "$R/eval.log" | cut -c1-400 || {
  echo "the evaluator printed no result line:" >&2; grep -aE "Traceback|Error" "$R/eval.log" | tail -n 20 >&2 || true; }
sleep 5          # let the policy server save the episodes that the evaluator ended early
kill $SPID 2>/dev/null || true
wait $SPID 2>/dev/null || true
"$FB_HARNESS_PY" "$FH_CODE/scripts/forcebench/collect.py" --run "$R"
if grep -aq '^FH_EVAL_STALLED' "$R/eval.log"; then      # no per-episode summaries: online successes are the episodes the evaluator closed
  { echo; echo "**The evaluator stalled and was killed by the watchdog**: $(grep -a '^FH_EVAL_STALLED' "$R/eval.log" | tail -n 1 | cut -c1-120)."
    echo "The success column above is not usable. Episodes whose harness record has stop_reason ended_by_evaluator succeeded online" \
         "(FH_ENV_SUCCESS lines in eval.log), provided every other episode was done before the stall; rerun unfinished episodes."; } >> "$R/summary.md"
fi
