#!/usr/bin/env bash
# Run after a session (or chain it: `...; bash scripts/finish_and_stop.sh`):
#   1. commit any results not yet committed and push
#   2. save final checkpoints into the repo and push
#   3. verify that GitHub has everything (nothing unpushed, working tree clean)
#   4. stop THIS pod (runpodctl + the pod's own id); never terminates
# If any step fails, the pod is left running and the reason is printed.
# Wait for a running session first:
#   while pgrep -f run_session_b.sh >/dev/null; do sleep 60; done; bash scripts/finish_and_stop.sh
set -uo pipefail
cd "$(dirname "$0")/.."
# Never stop the pod right after a session that failed at once: require a results commit
# from the last 12 hours or an explicit FORCE_STOP=1.
if [ "${FORCE_STOP:-0}" != "1" ] && ! git log -1 --since="12 hours ago" --format=%s -- results | grep -q .; then
  echo "No results committed in the last 12 hours: saving/pushing only, pod left running (FORCE_STOP=1 to override)."
  NO_STOP=1
fi
echo "=== [$(date +%H:%M:%S)] finish_and_stop: commit results"
git add results 2>/dev/null || true
git commit -qm "Results autosave $(date -u +%Y-%m-%dT%H:%MZ)" 2>/dev/null || echo "(no new results to commit)"
echo "=== save checkpoints"
bash scripts/save_checkpoints.sh || { echo "checkpoint save failed; pod left running"; exit 1; }
echo "=== push"
git push 2>&1 | tail -1
echo "=== verify"
git fetch -q origin
branch=$(git rev-parse --abbrev-ref HEAD)
ahead=$(git rev-list --count "origin/$branch..HEAD" 2>/dev/null || echo "?")
dirty=$(git status --porcelain | grep -v '^?? logs/' | wc -l)
if [ "$ahead" != "0" ] || [ "$dirty" != "0" ]; then
  echo "NOT SAFE: $ahead unpushed commit(s), $dirty uncommitted change(s); pod left running"
  git status -sb | head -5
  exit 1
fi
echo "GitHub is up to date with $branch ($(git rev-parse --short HEAD))."
if [ "${NO_STOP:-0}" = "1" ]; then
  echo "pod left running"
elif [ -n "${RUNPOD_POD_ID:-}" ] && command -v runpodctl >/dev/null; then
  echo "=== stopping pod $RUNPOD_POD_ID in 60 s (Ctrl-C to keep it running)"
  sleep 60
  runpodctl stop pod "$RUNPOD_POD_ID" && echo "stop requested"
else
  echo "runpodctl or RUNPOD_POD_ID not available; stop the pod from the RunPod page"
fi
