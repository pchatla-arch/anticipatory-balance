#!/usr/bin/env bash
# ONE command for a pod, fresh or restarted: bootstrap everything, then launch a session
# inside tmux (survives SSH drops) followed by finish_and_stop.sh (save, push, verify, stop).
#
# From the pod's shell (fresh pod: this clones first; restarted pod: it just updates):
#   bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/session-d/scripts/start.sh) d
# or, if the repo is already on disk:
#   bash /workspace/anticipatory-balance/scripts/start.sh d
#
# Argument: the session letter (a, b, c, d). Env: BRANCH (default: the branch this file is from),
# NO_STOP=1 to leave the pod running at the end.
set -euo pipefail
SESSION=${1:?usage: start.sh <a|b|c|d|e|f|g|h>}
case "$SESSION" in a|b|c|d|e|f|g|h) ;; *) echo 'Invalid session letter' >&2; exit 2;; esac
# H uses the bounded isolated workflow, preserving historical evidence.
if [[ "$SESSION" == h ]]; then
  export FORESIGHT_CODE_REF="${FORESIGHT_CODE_REF:-${BRANCH:-session-d}}"
  [[ "$FORESIGHT_CODE_REF" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$FORESIGHT_CODE_REF" != *..* ]] || { echo 'Invalid code ref' >&2; exit 2; }
  H_LAUNCH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/foresight-h.XXXXXX")
  trap 'rm -rf -- "$H_LAUNCH_DIR"' EXIT
  curl -fsSL --connect-timeout 15 --max-time 90 \
    "https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/$FORESIGHT_CODE_REF/scripts/start_session_h.sh" \
    -o "$H_LAUNCH_DIR/start_session_h.sh"
  bash -n "$H_LAUNCH_DIR/start_session_h.sh"
  shift
  bash "$H_LAUNCH_DIR/start_session_h.sh" "${1:-h}"
  exit $?
fi
BRANCH=${BRANCH:-session-d}
REPO=/workspace/anticipatory-balance
if [ ! -d "$REPO/.git" ]; then
  git clone -q -b "$BRANCH" https://github.com/pchatla-arch/anticipatory-balance.git "$REPO"
fi
BRANCH="$BRANCH" bash "$REPO/scripts/bootstrap_pod.sh"
cd "$REPO"
LOG="logs/session_${SESSION}_$(date +%Y%m%d-%H%M).txt"
mkdir -p logs
if tmux has-session -t '=session' 2>/dev/null; then
  echo 'An existing session is present; inspect it with: tmux attach -t session' >&2
  echo 'No session was killed and no new experiment was started.' >&2
  exit 1
fi
CMD="source /workspace/mujoco_playground/rss2025/.venv/bin/activate && cd $REPO && bash scripts/run_session_${SESSION}.sh 2>&1 | tee $LOG; NO_STOP=${NO_STOP:-0} bash scripts/finish_and_stop.sh 2>&1 | tee -a $LOG; echo; echo '--- session ended; press Enter to close ---'; read"
tmux new-session -d -s session "bash -lc \"$CMD\""
echo
echo "Session $SESSION started in tmux (log: $REPO/$LOG)."
echo "  watch:   tmux attach -t session      (detach with Ctrl-b d; it keeps running)"
echo "  stages:  grep '===' $REPO/$LOG"
echo "At the end it saves, pushes (asks for your GitHub token once), verifies, and stops the pod."
