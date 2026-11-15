#!/usr/bin/env bash
# Existing-environment entry. Public start.sh h also prepares an empty pod.
set -Eeuo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
unset MUJOCO_GL LD_LIBRARY_PATH PYTHONHOME
exec python3 scripts/run_session_h.py --mode "${FORESIGHT_MODE:-full}" \
  --max-minutes "${FORESIGHT_MAX_MINUTES:-75}" "$@"
