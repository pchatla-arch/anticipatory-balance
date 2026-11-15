#!/usr/bin/env bash
# Downloadable entry point: install an empty pod and start the bounded experiment.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash start_session_h_pod.sh

Installs missing Git/curl/CA/Python/tmux/util-linux/GitHub CLI system tools,
clones the Foresight release, installs its pinned Python 3.12 GPU environment,
restores published frozen policies, and runs the Session H experiments in tmux:
foresight-h. No PPO training is run.

Overrides:
  FORESIGHT_SETUP_ROOT=/workspace
  FORESIGHT_CODE_REF=session-d   (use a published SHA for a pinned run)
  FORESIGHT_REPO=/workspace/foresight-session-h  (isolated experiment checkout)
  FORESIGHT_MODE=full                        (or smoke for the small gates)
  FORESIGHT_MAX_MINUTES=75                    (smoke defaults to 20; after setup)
  FORESIGHT_SKIP_GITHUB_LOGIN=1               (trust existing Git credentials;
                                              bypass login/push-permission checks)
  FORESIGHT_AUTO_STOP=1                       (request stop after verified upload)

Python environment setup has its own 25-minute limit and can resume a matching
interrupted installation. Initial system-tool installation has separate bounded
steps. Existing unrelated environments are reused only after preflight;
they are never upgraded by this entry point. Automatic stop requires an explicit
FORESIGHT_AUTO_STOP=1, verified result upload, and working RunPod controls.
NO_STOP=1 overrides automatic stop. Missing GitHub credentials abort before the
experiment; missing stop controls leave the pod running after publication.
In either case, correct the issue or stop the pod in the RunPod dashboard.
The default GitHub flow requires repository push permission before GPU work.
FORESIGHT_SKIP_GITHUB_LOGIN=1 explicitly bypasses that check and trusts your
existing Git credentials; publication may fail if they cannot push this repo.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# > 1 )) || [[ $# == 1 && $1 != --worker ]]; then usage >&2; exit 2; fi
FORESIGHT_SETUP_ROOT="${FORESIGHT_SETUP_ROOT:-/workspace}"
FORESIGHT_CODE_REF="${FORESIGHT_CODE_REF:-session-d}"
FORESIGHT_MODE="${FORESIGHT_MODE:-full}"
[[ "$FORESIGHT_MODE" != h ]] || FORESIGHT_MODE=full
if [[ "$FORESIGHT_MODE" == smoke ]]; then
  FORESIGHT_MAX_MINUTES="${FORESIGHT_MAX_MINUTES:-20}"
else
  FORESIGHT_MAX_MINUTES="${FORESIGHT_MAX_MINUTES:-75}"
fi
FORESIGHT_SKIP_GITHUB_LOGIN="${FORESIGHT_SKIP_GITHUB_LOGIN:-0}"
[[ "$FORESIGHT_MODE" == full || "$FORESIGHT_MODE" == smoke ]] || { echo 'Invalid FORESIGHT_MODE: use full or smoke'; exit 2; }
[[ "$FORESIGHT_MAX_MINUTES" =~ ^[0-9]*[1-9][0-9]*([.][0-9]+)?$ ]] || { echo 'FORESIGHT_MAX_MINUTES must be a finite number of at least 1 minute'; exit 2; }
[[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 || "$FORESIGHT_SKIP_GITHUB_LOGIN" == 1 ]] || { echo 'Invalid FORESIGHT_SKIP_GITHUB_LOGIN'; exit 2; }
[[ "$FORESIGHT_SETUP_ROOT" == /* ]] || { echo 'FORESIGHT_SETUP_ROOT must be absolute'; exit 2; }
FORESIGHT_REPO="${FORESIGHT_REPO:-$FORESIGHT_SETUP_ROOT/foresight-session-h}"
FORESIGHT_VENV="${FORESIGHT_VENV:-$FORESIGHT_SETUP_ROOT/mujoco_playground/rss2025/.venv}"
FORESIGHT_AUTO_STOP="${FORESIGHT_AUTO_STOP:-0}"
[[ "$FORESIGHT_REPO" == /* && "$FORESIGHT_VENV" == /* ]] || { echo 'Repository and venv paths must be absolute'; exit 2; }
[[ "$FORESIGHT_AUTO_STOP" == 0 || "$FORESIGHT_AUTO_STOP" == 1 ]] || { echo 'Invalid FORESIGHT_AUTO_STOP'; exit 2; }

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ ${1:-} == --worker ]]; then
  cd "$FORESIGHT_REPO"
  exec 9> "$FORESIGHT_SETUP_ROOT/.foresight-setup.lock"
  flock -n 9 || die 'Another Foresight setup is already running.'
  printf '\n=== Foresight setup/run %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Source revision: %s\n' "$(git rev-parse HEAD)"
  export FORESIGHT_SETUP_ROOT FORESIGHT_REPO FORESIGHT_VENV FORESIGHT_AUTO_STOP
  export PYTHONPATH="$FORESIGHT_REPO${PYTHONPATH:+:$PYTHONPATH}"
  unset LD_LIBRARY_PATH MUJOCO_GL PYTHONHOME
  mkdir -p logs
  exec > >(tee -a logs/fresh_pod.log) 2>&1
  exec python3 scripts/session_h_pod.py --mode "$FORESIGHT_MODE" --max-minutes "$FORESIGHT_MAX_MINUTES"
fi

command -v nvidia-smi >/dev/null || die 'This pod has no exposed NVIDIA GPU. Select a GPU pod first.'
# An existing working environment is judged by preflight. Check the pinned
# fresh runtime before installing system tools when no such environment exists.
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits
if [[ ! -x "$FORESIGHT_VENV/bin/python" ]]; then
  FORESIGHT_DRIVERS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
  while IFS= read -r FORESIGHT_DRIVER; do
    FORESIGHT_MAJOR="${FORESIGHT_DRIVER%%.*}"
    [[ "$FORESIGHT_MAJOR" =~ ^[0-9]+$ && "$FORESIGHT_MAJOR" -ge 580 ]] || \
      die 'The pinned fresh runtime needs driver 580 or newer. No installation or experiment was started.'
  done <<< "$FORESIGHT_DRIVERS"
fi

# Serialize even the initial package-manager work, before flock is available.
mkdir -p "$FORESIGHT_SETUP_ROOT"
FORESIGHT_ENTRY_LOCK="$FORESIGHT_SETUP_ROOT/.foresight-entry.lock"
FORESIGHT_CLONE_STAGE=''
if ! mkdir "$FORESIGHT_ENTRY_LOCK" 2>/dev/null; then
  FORESIGHT_OLD_PID=''
  [[ ! -f "$FORESIGHT_ENTRY_LOCK/pid" ]] || read -r FORESIGHT_OLD_PID < "$FORESIGHT_ENTRY_LOCK/pid"
  if [[ "$FORESIGHT_OLD_PID" =~ ^[0-9]+$ ]] && ! kill -0 "$FORESIGHT_OLD_PID" 2>/dev/null; then
    rm -- "$FORESIGHT_ENTRY_LOCK/pid"
    rmdir -- "$FORESIGHT_ENTRY_LOCK" || die 'Unrecognized content in the old entry lock; it was preserved.'
    mkdir "$FORESIGHT_ENTRY_LOCK" || die 'Another launcher acquired the entry lock.'
  else
    die 'Another fresh-pod launcher is active.'
  fi
fi
printf '%s\n' "$BASHPID" > "$FORESIGHT_ENTRY_LOCK/pid"
entry_cleanup() {
  if [[ -n "$FORESIGHT_CLONE_STAGE" ]]; then rm -rf -- "$FORESIGHT_CLONE_STAGE"; fi
  if [[ -n "$FORESIGHT_ENTRY_LOCK" ]]; then
    rm -f -- "$FORESIGHT_ENTRY_LOCK/pid"
    rmdir -- "$FORESIGHT_ENTRY_LOCK" 2>/dev/null || true
  fi
}
trap entry_cleanup EXIT

apt_step() {
  local apt_options=(-o DPkg::Lock::Timeout=120 -o Acquire::Retries=2 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30)
  if command -v timeout >/dev/null; then
    timeout --kill-after=30s 8m env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" "$@"
  else
    env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" "$@"
  fi
}
FORESIGHT_MISSING=0
for FORESIGHT_TOOL in git curl python3 tmux flock findmnt timeout; do
  command -v "$FORESIGHT_TOOL" >/dev/null || FORESIGHT_MISSING=1
done
[[ -s /etc/ssl/certs/ca-certificates.crt ]] || FORESIGHT_MISSING=1
FORESIGHT_NEED_GH=0
if [[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 ]] && ! command -v gh >/dev/null; then FORESIGHT_NEED_GH=1; fi
if (( FORESIGHT_MISSING )); then
  [[ $(id -u) == 0 ]] || die 'Missing system tools need root. Use the normal root shell on an Ubuntu/Debian RunPod image.'
  command -v apt-get >/dev/null || die 'This installer expects an Ubuntu/Debian pod image with apt-get.'
  echo 'Installing required system tools: git, curl, CA certificates, python3, tmux, and util-linux.'
  apt_step update -qq
  FORESIGHT_PACKAGES=(git curl ca-certificates python3 tmux util-linux coreutils)
  apt_step install -y --no-install-recommends "${FORESIGHT_PACKAGES[@]}"
fi
if (( FORESIGHT_NEED_GH )); then
  if [[ $(id -u) == 0 ]] && command -v apt-get >/dev/null; then
    if (( ! FORESIGHT_MISSING )); then apt_step update -qq || true; fi
    apt_step install -y --no-install-recommends gh || true
  fi
  if ! command -v gh >/dev/null; then
    die 'GitHub CLI could not be installed. With working Git credentials, retry using FORESIGHT_SKIP_GITHUB_LOGIN=1. No experiment has started. The pod is still running; stop it in the RunPod dashboard if leaving.'
  fi
fi

mkdir -p "$FORESIGHT_SETUP_ROOT"
exec 8> "$FORESIGHT_SETUP_ROOT/.foresight-launch.lock"
flock -n 8 || die 'Another fresh-pod launcher is active.'
if tmux has-session -t '=foresight-h' 2>/dev/null; then
  echo 'Session foresight-h already exists. View it with: tmux attach -t foresight-h'
  echo 'If the old run has ended, exit its shell before restarting this command.'
  exit 0
fi
if tmux has-session -t '=foresight-v2' 2>/dev/null; then
  die 'The existing foresight-v2 session must finish first. Inspect with: tmux attach -t foresight-v2'
fi
FORESIGHT_GPU_PIDS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ "$FORESIGHT_GPU_PIDS" =~ [0-9] ]]; then
  die 'A GPU process is already active. Let it finish before changing the checkout or starting another run.'
fi

git check-ref-format --allow-onelevel "$FORESIGHT_CODE_REF" >/dev/null || \
  [[ "$FORESIGHT_CODE_REF" =~ ^[0-9a-f]{40}$ ]] || die 'Invalid FORESIGHT_CODE_REF'
if [[ ! -e "$FORESIGHT_REPO" ]]; then
  echo 'Downloading the project and published checkpoint bundles.'
  FORESIGHT_CLONE_STAGE="$(mktemp -d "$FORESIGHT_SETUP_ROOT/.foresight-clone.XXXXXX")"
  GIT_TERMINAL_PROMPT=0 git clone --filter=blob:none --no-checkout \
    https://github.com/pchatla-arch/anticipatory-balance.git "$FORESIGHT_CLONE_STAGE/repo"
  FORESIGHT_CHECKOUT="$FORESIGHT_CLONE_STAGE/repo"
else
  [[ -e "$FORESIGHT_REPO/.git" ]] || die "Existing path is not a Git checkout: $FORESIGHT_REPO"
  [[ -z "$(git -C "$FORESIGHT_REPO" status --porcelain --untracked-files=no)" ]] || \
    die 'The project has uncommitted tracked changes. They were preserved; save them before switching to this release.'
  FORESIGHT_CHECKOUT="$FORESIGHT_REPO"
fi
GIT_TERMINAL_PROMPT=0 git -C "$FORESIGHT_CHECKOUT" fetch --no-tags \
  https://github.com/pchatla-arch/anticipatory-balance.git "$FORESIGHT_CODE_REF"
git -C "$FORESIGHT_CHECKOUT" -c advice.detachedHead=false checkout --detach FETCH_HEAD
if [[ -n "$FORESIGHT_CLONE_STAGE" ]]; then
  mv -- "$FORESIGHT_CHECKOUT" "$FORESIGHT_REPO"
  rmdir -- "$FORESIGHT_CLONE_STAGE"
  FORESIGHT_CLONE_STAGE=''
fi
[[ -f "$FORESIGHT_REPO/scripts/bootstrap_fresh_pod.sh" ]] || die 'Selected revision lacks the fresh-pod installer.'
[[ -f "$FORESIGHT_REPO/scripts/session_h_pod.py" ]] || die 'Selected revision lacks the reviewed pod session worker.'
mkdir -p "$FORESIGHT_REPO/logs"

if [[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 ]]; then
  if gh auth status --hostname github.com >/dev/null 2>&1; then
    gh auth setup-git --hostname github.com
  elif [[ -t 0 ]]; then
    echo
    echo 'Complete this GitHub browser login before GPU work starts so results can be uploaded.'
    echo 'No access token needs to be pasted into chat.'
    if gh auth login --hostname github.com --git-protocol https --web; then
      gh auth setup-git --hostname github.com
    else
      die 'GitHub login was not completed. No experiment has started; rerun after login. The pod is still running; stop it in the RunPod dashboard if leaving.'
    fi
  else
    die 'No terminal for GitHub login. Run gh auth login --hostname github.com --git-protocol https --web, or use existing Git credentials with FORESIGHT_SKIP_GITHUB_LOGIN=1. The pod is still running; stop it in the RunPod dashboard if leaving.'
  fi
  if ! FORESIGHT_CAN_PUSH=$(gh api repos/pchatla-arch/anticipatory-balance \
      --jq '.permissions.push' 2>/dev/null); then
    die 'Could not verify GitHub push permission for pchatla-arch/anticipatory-balance. No experiment has started. Correct repository access and retry, or stop the still-running pod in the RunPod dashboard.'
  fi
  [[ "$FORESIGHT_CAN_PUSH" == true ]] || \
    die 'This GitHub account/token does not have verified push permission for pchatla-arch/anticipatory-balance. No experiment has started. Correct repository access and retry, or stop the still-running pod in the RunPod dashboard.'
fi
printf -v FORESIGHT_WORKER 'cd %q && env %q %q %q %q %q %q %q %q bash scripts/start_session_h_pod.sh --worker 8>&-' \
  "$FORESIGHT_REPO" "FORESIGHT_SETUP_ROOT=$FORESIGHT_SETUP_ROOT" "FORESIGHT_CODE_REF=$FORESIGHT_CODE_REF" \
  "FORESIGHT_REPO=$FORESIGHT_REPO" "FORESIGHT_VENV=$FORESIGHT_VENV" \
  "FORESIGHT_MODE=$FORESIGHT_MODE" "FORESIGHT_MAX_MINUTES=$FORESIGHT_MAX_MINUTES" \
  "FORESIGHT_AUTO_STOP=$FORESIGHT_AUTO_STOP" "NO_STOP=${NO_STOP:-0}"
tmux new-session -d -s foresight-h "$FORESIGHT_WORKER"
exec 8>&-
rm -- "$FORESIGHT_ENTRY_LOCK/pid"
rmdir -- "$FORESIGHT_ENTRY_LOCK"
FORESIGHT_ENTRY_LOCK=''
echo
echo 'Setup and Session H experiments are running in tmux: foresight-h'
echo "Isolated checkout: $FORESIGHT_REPO"
echo "Live log: $FORESIGHT_REPO/logs/fresh_pod.log"
echo 'Detach without stopping: Ctrl-B, then D.'
echo "Limits from here: 25 minutes for Python setup, then $FORESIGHT_MAX_MINUTES minutes for mode $FORESIGHT_MODE."
echo "Automatic stop requested: $FORESIGHT_AUTO_STOP (NO_STOP=${NO_STOP:-0}); requires verified upload and RunPod controls."
echo 'If upload or stop controls fail, the pod stays running; stop it manually in the RunPod dashboard.'
echo 'Next: tmux attach -t foresight-h'
