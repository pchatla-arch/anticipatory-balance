#!/usr/bin/env bash
# Public entry point for the isolated frozen-walker actuator feasibility gate.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash start_adaptive_gate.sh [gate|smoke]
       bash start_adaptive_gate.sh check-access  # test current checkout credentials
       bash start_adaptive_gate.sh repair-access # configure credentials on an existing checkout
       bash start_adaptive_gate.sh retry-publish RUN_ID

Run the gain-compensation feasibility gate using the published frozen G1 walker.
No PPO training or online estimator is run. The gate asks whether a bounded,
privileged correction can recover a deliberately degraded actuator response.

Default: gate, with a 120-minute attempt budget measured from this launcher,
including setup; smoke defaults to 30 minutes. Ten minutes are reserved for
result publication and stop checks. This is not a guaranteed billing cap:
failed upload or unavailable RunPod controls leave the pod running.

Before GPU work the launcher verifies Git push access (saved credentials, or a
GitHub access token entered at a hidden prompt), runs in tmux
adaptive-gate, automatically uploads compact results, then requests this pod's
Stop only after its exact upload is verified and its GPU is idle.

For a revision downloaded from raw.githubusercontent.com, set FORESIGHT_CODE_REF
to the same commit so the checkout and inner launcher match the download.

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
case "$ADAPTIVE_PROFILE" in gate|smoke|check-access|repair-access|retry-publish) ;; *) usage >&2; exit 2 ;; esac
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

if [[ "$ADAPTIVE_PROFILE" == check-access ]]; then
  (( $# == 1 )) || { usage >&2; exit 2; }
  [[ -f "$FORESIGHT_REPO/scripts/publish_experiment_results.py" ]] || die 'Set FORESIGHT_REPO to an existing project checkout with the results publisher.'
  exec python3 - "$FORESIGHT_REPO" <<'PYACCESS'
from pathlib import Path
import sys
import tempfile

repo = Path(sys.argv[1])
sys.dont_write_bytecode = True
sys.path.insert(0, str(repo / 'scripts'))
from publish_experiment_results import Git, PublicationError

with tempfile.TemporaryDirectory() as scratch:
    git = Git(repo, Path(scratch) / 'index')
    try:
        git.validate_origin()
        git.run('-c', 'core.hooksPath=/dev/null', 'push', '--dry-run', '--quiet',
                'origin', 'HEAD:refs/heads/foresight/credential-probe',
                operation='GitHub write-access check', network=True)
    except PublicationError as exc:
        sys.exit(f'{exc} Check pod credentials, network access and the origin configuration.')
print('GitHub accepted the dry-run push from this checkout. No ref was created.')
PYACCESS
fi

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
if [[ "$ADAPTIVE_PROFILE" == smoke || "$ADAPTIVE_PROFILE" == repair-access ]]; then
  ADAPTIVE_MAX_MINUTES=${ADAPTIVE_MAX_MINUTES:-30}
else
  ADAPTIVE_MAX_MINUTES=${ADAPTIVE_MAX_MINUTES:-120}
fi
[[ "$ADAPTIVE_MAX_MINUTES" =~ ^[0-9]+$ ]] || die 'ADAPTIVE_MAX_MINUTES must be an integer from 15 to 120.'
(( 10#$ADAPTIVE_MAX_MINUTES >= 15 && 10#$ADAPTIVE_MAX_MINUTES <= 120 )) || die 'ADAPTIVE_MAX_MINUTES must be an integer from 15 to 120.'
ADAPTIVE_STARTED_EPOCH=$(date +%s)
ADAPTIVE_DEADLINE_EPOCH=$((ADAPTIVE_STARTED_EPOCH + 10#$ADAPTIVE_MAX_MINUTES * 60))
export ADAPTIVE_PROFILE ADAPTIVE_MAX_MINUTES ADAPTIVE_STARTED_EPOCH ADAPTIVE_DEADLINE_EPOCH
export FORESIGHT_SETUP_ROOT FORESIGHT_REPO FORESIGHT_VENV FORESIGHT_CODE_REF FORESIGHT_AUTO_STOP
command -v curl >/dev/null || die 'curl is required to download the launcher.'
if [[ "$ADAPTIVE_PROFILE" == repair-access ]]; then
  echo 'Configuring GitHub access for the existing pod checkout.'
else
  printf 'Adaptive execution: %s; %s-minute attempt budget including setup. No PPO training.\n' "$ADAPTIVE_PROFILE" "$ADAPTIVE_MAX_MINUTES"
  printf 'Only a verified upload permits an automatic stop request; check RunPod if any step fails.\n'
fi
ADAPTIVE_DOWNLOAD=$(mktemp -d "${TMPDIR:-/tmp}/adaptive-gate-launch.XXXXXX")
trap 'rm -rf -- "$ADAPTIVE_DOWNLOAD"' EXIT
ADAPTIVE_ENTRY="$ADAPTIVE_DOWNLOAD/start_adaptive_gate_pod.sh"
curl --fail --show-error --silent --location --connect-timeout 15 --max-time 90 \
  --retry 2 --retry-delay 1 --output "$ADAPTIVE_ENTRY" \
  "https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/$FORESIGHT_CODE_REF/scripts/start_adaptive_gate_pod.sh" || die 'Could not download the launcher. No experiment started.'
IFS= read -r ADAPTIVE_FIRST_LINE < "$ADAPTIVE_ENTRY"
[[ "$ADAPTIVE_FIRST_LINE" == '#!/usr/bin/env bash' ]] || die 'The download is not the expected Bash launcher.'
bash -n "$ADAPTIVE_ENTRY" || die 'The launcher has invalid Bash syntax.'
if [[ "$ADAPTIVE_PROFILE" == repair-access ]]; then
  bash "$ADAPTIVE_ENTRY" --repair-access
else
  bash "$ADAPTIVE_ENTRY"
fi
