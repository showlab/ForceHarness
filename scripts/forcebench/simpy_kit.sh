#!/usr/bin/env bash
# Interpreter wrapper for ForceBench's tools/validate_simulation.py, which starts the evaluator as a subprocess through its
# --python option and does not forward Kit arguments. This wrapper appends --kit_args=$FB_KIT_ARGS (e.g. the opt-in driver
# check workaround of env.sh) and passes everything else through unchanged.
exec "${FORCEBENCH_SIM_PYTHON:?source scripts/forcebench/env.sh first}" "$@" ${FB_KIT_ARGS:+"--kit_args=$FB_KIT_ARGS"}
