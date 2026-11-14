#!/usr/bin/env bash
# Start the bounded evaluation runner without replacing any existing tmux session.
set -Eeuo pipefail
FORESIGHT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$FORESIGHT_ROOT"
if ! command -v tmux >/dev/null 2>&1; then
  echo 'tmux is unavailable. Run in this terminal: python3 scripts/run_experiments_v2.py --mode pilot'
  exit 1
fi
if [[ $# -gt 0 ]]; then
  echo 'Use FORESIGHT_MODE=smoke or FORESIGHT_MAX_MINUTES=45 as environment overrides.' >&2
  exit 2
fi
FORESIGHT_MODE="${FORESIGHT_MODE:-pilot}"
FORESIGHT_MAX_MINUTES="${FORESIGHT_MAX_MINUTES:-45}"
[[ "$FORESIGHT_MODE" == smoke || "$FORESIGHT_MODE" == pilot ]] || { echo 'Invalid FORESIGHT_MODE'; exit 2; }
[[ "$FORESIGHT_MAX_MINUTES" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'Invalid FORESIGHT_MAX_MINUTES'; exit 2; }
if tmux has-session -t '=foresight-v2' 2>/dev/null; then
  echo 'Session foresight-v2 already exists. Inspect it with: tmux attach -t foresight-v2'
  exit 1
fi
mkdir -p logs
FORESIGHT_LOG="$FORESIGHT_ROOT/logs/experiments_v2_$(date -u +%Y%m%dT%H%M%SZ).log"
printf -v FORESIGHT_COMMAND 'cd %q && env %q %q python3 scripts/run_experiments_v2.py --mode %q --max-minutes %q 2>&1 | tee %q; exec bash' \
  "$FORESIGHT_ROOT" "FORESIGHT_VENV=${FORESIGHT_VENV:-/workspace/mujoco_playground/rss2025/.venv}" \
  "FORESIGHT_PYTHON=${FORESIGHT_PYTHON:-}" "$FORESIGHT_MODE" "$FORESIGHT_MAX_MINUTES" "$FORESIGHT_LOG"
tmux new-session -d -s foresight-v2 "$FORESIGHT_COMMAND"
echo "Started tmux session: foresight-v2"
echo "Watch: tail -f $FORESIGHT_LOG"
echo 'Attach: tmux attach -t foresight-v2'
echo 'The wall-time limit stops experiment processes, not pod billing.'
