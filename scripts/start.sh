#!/usr/bin/env bash
# Install this wrapper as scripts/start.sh beside the hardened start_empty_pod.sh.
# The wrapper downloads the real entry point to a regular file before invoking it.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash start.sh f
       bash start.sh smoke
       bash start.sh retry-publish RUN_ID

f, f0, release  Run F0: the bounded 64-scenario release diagnostic.
smoke           Run the same diagnostic with 32 scenarios.

F0 checks the controller release contract with frozen policies and synthetic
detections. It does not implement or run the new lookahead or rendered-depth
experiments. Results are published automatically when Git credentials permit.
By default it requests this pod's stop after the current run's upload is verified.
Set NO_STOP=1 to keep the pod running. Missing RunPod controls or an upload
failure leave the pod running and print the required manual action.

The normal destination is /workspace/foresight-runner. The launcher reuses the
existing /workspace/mujoco_playground/rss2025/.venv when it passes preflight.

Overrides:
  FORESIGHT_CODE_REF=foresight/experiments-v2  (a published SHA pins the code)
  FORESIGHT_SETUP_ROOT=/workspace
  FORESIGHT_REPO=/workspace/foresight-runner
  FORESIGHT_VENV=/workspace/mujoco_playground/rss2025/.venv
  FORESIGHT_MAX_MINUTES=20
  FORESIGHT_AUTO_STOP=1                      (NO_STOP=1 overrides it)

retry-publish uploads an already finished RUN_ID without starting experiments.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ ${1:-} == --help || ${1:-} == -h || ${1:-} == help ]]; then
  usage
  exit 0
fi

SESSION=${1:-f}
case "$SESSION" in
  f|f0|release) FORESIGHT_MODE=release ;;
  smoke) FORESIGHT_MODE=smoke ;;
  retry-publish) ;;
  *) usage >&2; exit 2 ;;
esac

FORESIGHT_SETUP_ROOT=${FORESIGHT_SETUP_ROOT:-/workspace}
FORESIGHT_REPO=${FORESIGHT_REPO:-$FORESIGHT_SETUP_ROOT/foresight-runner}
FORESIGHT_VENV=${FORESIGHT_VENV:-$FORESIGHT_SETUP_ROOT/mujoco_playground/rss2025/.venv}
FORESIGHT_CODE_REF=${FORESIGHT_CODE_REF:-foresight/experiments-v2}
FORESIGHT_MAX_MINUTES=${FORESIGHT_MAX_MINUTES:-20}
FORESIGHT_AUTO_STOP=${FORESIGHT_AUTO_STOP:-1}
for FORESIGHT_PATH in "$FORESIGHT_SETUP_ROOT" "$FORESIGHT_REPO" "$FORESIGHT_VENV"; do
  [[ "$FORESIGHT_PATH" == /* ]] || die 'Setup, checkout, and environment paths must be absolute.'
done

if [[ "$SESSION" == retry-publish ]]; then
  [[ $# == 2 && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || {
    printf 'Usage: bash start.sh retry-publish RUN_ID\n' >&2
    exit 2
  }
  FORESIGHT_RUN_DIR="$FORESIGHT_REPO/results/$2"
  [[ -f "$FORESIGHT_RUN_DIR/run.json" ]] || die 'The named run has no run.json record.'
  [[ -f "$FORESIGHT_REPO/scripts/publish_experiment_results.py" ]] || die 'The result publisher is missing from the runner checkout.'
  command -v python3 >/dev/null || die 'python3 is required to publish existing results.'
  FORESIGHT_RUN_STATUS=$(python3 - "$FORESIGHT_RUN_DIR/run.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as source:
    record = json.load(source)
code = record.get("exit_code")
if record.get("status") == "running" or type(code) is not int or not 0 <= code <= 255:
    raise SystemExit("The named run has not recorded a final exit status.")
print(code)
PY
  )
  python3 "$FORESIGHT_REPO/scripts/publish_experiment_results.py" \
    --run-dir "$FORESIGHT_RUN_DIR" --log "$FORESIGHT_RUN_DIR/console.log" \
    --exit-code "$FORESIGHT_RUN_STATUS"
  exit $?
fi

(( $# <= 1 )) || { usage >&2; exit 2; }
[[ "$FORESIGHT_CODE_REF" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$FORESIGHT_CODE_REF" != *..* ]] || \
  die 'FORESIGHT_CODE_REF must be a Git branch, tag, or full commit SHA.'
command -v curl >/dev/null || die 'curl is required to download the launcher.'

export FORESIGHT_SETUP_ROOT FORESIGHT_REPO FORESIGHT_VENV
export FORESIGHT_CODE_REF FORESIGHT_MODE FORESIGHT_MAX_MINUTES
export FORESIGHT_AUTO_STOP
if [[ "$FORESIGHT_MODE" == release ]]; then
  printf 'Starting F0: 64 paired scenarios, three release contracts, frozen policies.\n'
else
  printf 'Starting F0 smoke: 32 paired scenarios, three release contracts, frozen policies.\n'
fi
printf 'This is the release diagnostic; new lookahead and rendered-depth experiments are not included.\n'
printf 'Runner checkout: %s\nExperiment budget: %s minutes after setup.\n' "$FORESIGHT_REPO" "$FORESIGHT_MAX_MINUTES"
printf 'After verified upload: automatic pod stop requested=%s, NO_STOP=%s.\n' "$FORESIGHT_AUTO_STOP" "${NO_STOP:-0}"

FORESIGHT_LAUNCH_DIR=$(mktemp -d "${TMPDIR:-/tmp}/foresight-launch.XXXXXX")
trap 'rm -rf -- "$FORESIGHT_LAUNCH_DIR"' EXIT
FORESIGHT_ENTRY="$FORESIGHT_LAUNCH_DIR/start_empty_pod.sh"
FORESIGHT_ENTRY_URL="https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/$FORESIGHT_CODE_REF/scripts/start_empty_pod.sh"
if ! curl --fail --show-error --silent --location --connect-timeout 15 --max-time 90 \
    --retry 2 --retry-delay 1 --output "$FORESIGHT_ENTRY" "$FORESIGHT_ENTRY_URL"; then
  die 'Could not download the launcher. No experiment was started.'
fi
[[ -s "$FORESIGHT_ENTRY" ]] || die 'The downloaded launcher is empty.'
IFS= read -r FORESIGHT_FIRST_LINE < "$FORESIGHT_ENTRY"
[[ "$FORESIGHT_FIRST_LINE" == '#!/usr/bin/env bash' || "$FORESIGHT_FIRST_LINE" == '#!/bin/bash' ]] || \
  die 'The download is not a Bash launcher.'
bash -n "$FORESIGHT_ENTRY" || die 'The downloaded launcher did not pass Bash syntax validation.'

# Invoke the regular downloaded file; its worker uses the published checkout.
# Keep the EXIT trap active so this temporary download is removed on completion.
bash "$FORESIGHT_ENTRY"
