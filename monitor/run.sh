#!/usr/bin/env bash
# Start the ForceHarness monitor:  bash monitor/run.sh               (demo episodes and review only)
#                                  bash monitor/run.sh --allow-live  (also real-robot episodes)
# Extra arguments go to `python -m fhmon` (see --help). Then open http://localhost:8765/
# The monitor runs under $FH_PYTHON, else python3, and starts the runner scripts with the same interpreter
# unless --python says otherwise. If <repository>/.env exists, its variables (API keys) are loaded into the
# monitor process only; the monitor never writes a key anywhere.
set -eo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
PY="${FH_PYTHON:-python3}"
ENV_ARGS=()
if [ -f "$ROOT/.env" ]; then
  ENV_ARGS=(--env-file "$ROOT/.env")
fi
cd "$HERE"
exec "$PY" -m fhmon "${ENV_ARGS[@]}" "$@"
