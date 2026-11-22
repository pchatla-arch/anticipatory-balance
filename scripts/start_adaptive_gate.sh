#!/usr/bin/env bash
# Public entry point for the isolated frozen-walker actuator feasibility gate.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash start_adaptive_gate.sh [gate|smoke|calibrate-identity]
       bash start_adaptive_gate.sh retry-publish RUN_ID

Run the gain-compensation feasibility gate using the published frozen G1 walker.
No PPO training or online estimator is run. The gate asks whether a bounded,
privileged correction can recover a deliberately degraded actuator response.

Default: gate, with a 120-minute attempt budget measured from this launcher,
including setup; smoke defaults to 30 minutes and calibrate-identity to 45 minutes. Ten minutes are reserved for
result publication and stop checks. This is not a guaranteed billing cap:
failed upload or unavailable RunPod controls leave the pod running.

Before GPU work the launcher verifies Git push access (saved credentials, or a
GitHub access token entered at a hidden prompt), runs in tmux
adaptive-gate, automatically uploads compact results, then requests this pod's
Stop only after its exact upload is verified and its GPU is idle.

Overrides:
  ADAPTIVE_MAX_MINUTES=120                 (integer 15..120; smoke default 30)
  FORESIGHT_CODE_REF=portfolio/adaptive-execution-gate
  FORESIGHT_SETUP_ROOT=/workspace
  FORESIGHT_REPO=/workspace/adaptive-execution-gate
  FORESIGHT_VENV=/workspace/mujoco_playground/rss2025/.venv
  FORESIGHT_AUTO_STOP=1                    (NO_STOP=1 disables stopping)
  FORESIGHT_GITHUB_TOKEN=...                (optional; else prompted if needed)
  FORESIGHT_SKIP_GITHUB_LOGIN=1             (skip all GitHub checks)

An existing working venv is checked without upgrades. Fresh setup uses the
repository's pinned bootstrap. retry-publish does not rerun experiments.
EOF
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
if [[ ${1:-} == --help || ${1:-} == -h || ${1:-} == help ]]; then usage; exit 0; fi
ADAPTIVE_PROFILE=${1:-gate}
case "$ADAPTIVE_PROFILE" in gate|smoke|calibrate-identity|retry-publish) ;; *) usage >&2; exit 2 ;; esac
FORESIGHT_SETUP_ROOT=${FORESIGHT_SETUP_ROOT:-/workspace}
FORESIGHT_REPO=${FORESIGHT_REPO:-$FORESIGHT_SETUP_ROOT/adaptive-execution-gate}
FORESIGHT_VENV=${FORESIGHT_VENV:-$FORESIGHT_SETUP_ROOT/mujoco_playground/rss2025/.venv}
FORESIGHT_CODE_REF=${FORESIGHT_CODE_REF:-portfolio/adaptive-execution-gate}
FORESIGHT_AUTO_STOP=${FORESIGHT_AUTO_STOP:-1}
for ADAPTIVE_PATH in "$FORESIGHT_SETUP_ROOT" "$FORESIGHT_REPO" "$FORESIGHT_VENV"; do
  [[ "$ADAPTIVE_PATH" == /* ]] || die 'Setup, checkout, and venv paths must be absolute.'
done
[[ "$FORESIGHT_AUTO_STOP" == 0 || "$FORESIGHT_AUTO_STOP" == 1 ]] || die 'FORESIGHT_AUTO_STOP must be 0 or 1.'
[[ "$FORESIGHT_CODE_REF" =~ ^[A-Za-z0-9][A-Za-z0-9._/-]*$ && "$FORESIGHT_CODE_REF" != *..* ]] || die 'Invalid source revision.'

if [[ "$ADAPTIVE_PROFILE" == retry-publish ]]; then
  [[ $# == 2 && "$2" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$ ]] || { usage >&2; exit 2; }
  ADAPTIVE_RUN="$FORESIGHT_REPO/results/$2"
  [[ -f "$ADAPTIVE_RUN/run.json" ]] || die 'The named run has no run.json.'
  [[ -f "$FORESIGHT_REPO/scripts/publish_experiment_results.py" ]] || die 'The publisher is missing.'
  ADAPTIVE_EXIT=$(python3 - "$ADAPTIVE_RUN/run.json" <<'PY'
import json, sys
record = json.load(open(sys.argv[1], encoding='utf-8'))
code = record.get('exit_code')
if record.get('status') == 'running' or type(code) is not int or not 0 <= code <= 255:
    raise SystemExit('The run has no final exit status.')
print(code)
PY
  )
  exec python3 "$FORESIGHT_REPO/scripts/publish_experiment_results.py" \
    --run-dir "$ADAPTIVE_RUN" --log "$ADAPTIVE_RUN/console.log" --exit-code "$ADAPTIVE_EXIT"
fi
(( $# <= 1 )) || { usage >&2; exit 2; }
if [[ "$ADAPTIVE_PROFILE" == smoke ]]; then
  ADAPTIVE_MAX_MINUTES=${ADAPTIVE_MAX_MINUTES:-30}
elif [[ "$ADAPTIVE_PROFILE" == calibrate-identity ]]; then
  ADAPTIVE_MAX_MINUTES=${ADAPTIVE_MAX_MINUTES:-45}
else
  ADAPTIVE_MAX_MINUTES=${ADAPTIVE_MAX_MINUTES:-120}
fi
[[ "$ADAPTIVE_MAX_MINUTES" =~ ^[0-9]+$ ]] || die 'ADAPTIVE_MAX_MINUTES must be an integer from 15 to 120.'
(( 10#$ADAPTIVE_MAX_MINUTES >= 15 && 10#$ADAPTIVE_MAX_MINUTES <= 120 )) || die 'ADAPTIVE_MAX_MINUTES must be an integer from 15 to 120.'
if [[ "$ADAPTIVE_PROFILE" == calibrate-identity ]]; then
  (( 10#$ADAPTIVE_MAX_MINUTES <= 45 )) || die 'Identity calibration budget must not exceed 45 minutes.'
fi
ADAPTIVE_STARTED_EPOCH=$(date +%s)
ADAPTIVE_DEADLINE_EPOCH=$((ADAPTIVE_STARTED_EPOCH + 10#$ADAPTIVE_MAX_MINUTES * 60))
export ADAPTIVE_PROFILE ADAPTIVE_MAX_MINUTES ADAPTIVE_STARTED_EPOCH ADAPTIVE_DEADLINE_EPOCH
export FORESIGHT_SETUP_ROOT FORESIGHT_REPO FORESIGHT_VENV FORESIGHT_CODE_REF FORESIGHT_AUTO_STOP
# ADAPTIVE_DETERMINISTIC=auto|on|off: bit-exact physics and XLA ops (technical note AE-008); off is the escape hatch.
[[ "${ADAPTIVE_DETERMINISTIC:-auto}" == auto || "${ADAPTIVE_DETERMINISTIC:-}" == on || "${ADAPTIVE_DETERMINISTIC:-}" == off ]] || die 'ADAPTIVE_DETERMINISTIC must be auto, on or off.'
export ADAPTIVE_DETERMINISTIC="${ADAPTIVE_DETERMINISTIC:-auto}"
command -v curl >/dev/null || die 'curl is required to download the launcher.'
printf 'Adaptive execution: %s; %s-minute attempt budget including setup. No PPO training.\n' "$ADAPTIVE_PROFILE" "$ADAPTIVE_MAX_MINUTES"
printf 'Only a verified upload permits an automatic stop request; check RunPod if any step fails.\n'
ADAPTIVE_DOWNLOAD=$(mktemp -d "${TMPDIR:-/tmp}/adaptive-gate-launch.XXXXXX")
trap 'rm -rf -- "$ADAPTIVE_DOWNLOAD"' EXIT
ADAPTIVE_ENTRY="$ADAPTIVE_DOWNLOAD/start_adaptive_gate_pod.sh"
curl --fail --show-error --silent --location --connect-timeout 15 --max-time 90 \
  --retry 2 --retry-delay 1 --output "$ADAPTIVE_ENTRY" \
  "https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/$FORESIGHT_CODE_REF/scripts/start_adaptive_gate_pod.sh" || die 'Could not download the launcher. No experiment started.'
IFS= read -r ADAPTIVE_FIRST_LINE < "$ADAPTIVE_ENTRY"
[[ "$ADAPTIVE_FIRST_LINE" == '#!/usr/bin/env bash' ]] || die 'The download is not the expected Bash launcher.'
bash -n "$ADAPTIVE_ENTRY" || die 'The launcher has invalid Bash syntax.'
bash "$ADAPTIVE_ENTRY"
