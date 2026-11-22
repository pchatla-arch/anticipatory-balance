#!/usr/bin/env bash
# Downloadable entry point: install an empty pod and start the bounded experiment.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash start_adaptive_gate_pod.sh

Installs missing Git/curl/CA/Python/tmux/util-linux/GitHub CLI system tools,
clones the Foresight release, installs its pinned Python 3.12 GPU environment,
restores published frozen policies, and runs the adaptive actuator gate experiments in tmux:
adaptive-gate. No PPO training is run.

The public start_adaptive_gate.sh wrapper establishes the original 120-minute
attempt deadline (smoke 30 minutes), including setup and a 10-minute cleanup
reserve. Call this entry through that wrapper. No PPO training is run.

Overrides:
  FORESIGHT_SETUP_ROOT=/workspace
  FORESIGHT_CODE_REF=portfolio/adaptive-execution-gate
  FORESIGHT_REPO=/workspace/adaptive-execution-gate
  FORESIGHT_VENV=/workspace/mujoco_playground/rss2025/.venv
  FORESIGHT_SKIP_GITHUB_LOGIN=1  (trust existing Git credentials)
  FORESIGHT_AUTO_STOP=1         (NO_STOP=1 overrides)

The known working venv is checked, never upgraded. Fresh setup uses the pinned
bootstrap. Only a verified result upload can authorize this pod's stop, and a
busy GPU or missing RunPod controls withholds it. This is an attempt budget, not
a guaranteed billing ceiling. Check the RunPod dashboard after completion.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# > 1 )) || [[ $# == 1 && $1 != --worker ]]; then usage >&2; exit 2; fi
FORESIGHT_SETUP_ROOT="${FORESIGHT_SETUP_ROOT:-/workspace}"
FORESIGHT_CODE_REF="${FORESIGHT_CODE_REF:-portfolio/adaptive-execution-gate}"
ADAPTIVE_PROFILE="${ADAPTIVE_PROFILE:-gate}"
[[ "$ADAPTIVE_PROFILE" == gate || "$ADAPTIVE_PROFILE" == smoke ]] || { echo 'Invalid ADAPTIVE_PROFILE'; exit 2; }
[[ ${ADAPTIVE_STARTED_EPOCH:-} =~ ^[0-9]+$ && ${ADAPTIVE_DEADLINE_EPOCH:-} =~ ^[0-9]+$ ]] || {
  echo 'Use start_adaptive_gate.sh to establish the original attempt deadline.'; exit 2;
}
ADAPTIVE_DURATION=$((ADAPTIVE_DEADLINE_EPOCH - ADAPTIVE_STARTED_EPOCH))
(( ADAPTIVE_DURATION >= 900 && ADAPTIVE_DURATION <= 7200 )) || { echo 'Invalid attempt duration'; exit 2; }
FORESIGHT_SKIP_GITHUB_LOGIN="${FORESIGHT_SKIP_GITHUB_LOGIN:-0}"
[[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 || "$FORESIGHT_SKIP_GITHUB_LOGIN" == 1 ]] || { echo 'Invalid FORESIGHT_SKIP_GITHUB_LOGIN'; exit 2; }
[[ "$FORESIGHT_SETUP_ROOT" == /* ]] || { echo 'FORESIGHT_SETUP_ROOT must be absolute'; exit 2; }
FORESIGHT_REPO="${FORESIGHT_REPO:-$FORESIGHT_SETUP_ROOT/adaptive-execution-gate}"
FORESIGHT_VENV="${FORESIGHT_VENV:-$FORESIGHT_SETUP_ROOT/mujoco_playground/rss2025/.venv}"
FORESIGHT_AUTO_STOP="${FORESIGHT_AUTO_STOP:-1}"
[[ "$FORESIGHT_REPO" == /* && "$FORESIGHT_VENV" == /* ]] || { echo 'Repository and venv paths must be absolute'; exit 2; }
[[ "$FORESIGHT_AUTO_STOP" == 0 || "$FORESIGHT_AUTO_STOP" == 1 ]] || { echo 'Invalid FORESIGHT_AUTO_STOP'; exit 2; }

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v timeout >/dev/null || die 'GNU timeout is required to bound setup work.'
# Clamp setup commands to the original deadline, preserving 10 minutes for cleanup.
bounded() {
  local cap=$1 remaining
  shift
  local timeout_options=()
  if [[ ${1:-} == --interactive ]]; then timeout_options=(--foreground); shift; fi
  remaining=$((ADAPTIVE_DEADLINE_EPOCH - $(date +%s) - 600))
  (( remaining > 0 )) || die 'The attempt budget is exhausted. No further setup or experiment will start; check RunPod.'
  (( cap <= remaining )) || cap=$remaining
  timeout "${timeout_options[@]}" --signal=TERM --kill-after=15s "${cap}s" "$@"
}

github_login() {
  bounded 30 gh api --hostname github.com user --jq '.login' 2>/dev/null
}

require_github_access() {
  local github_user can_push
  if ! github_user=$(github_login) || [[ "$github_user" != pchatla-arch ]]; then
    die 'GitHub must authenticate as pchatla-arch in this shell. Run gh auth login --hostname github.com --git-protocol https on the pod; check GH_TOKEN/GITHUB_TOKEN overrides if set. No experiment has started. The pod is still running; stop it in the RunPod dashboard if leaving.'
  fi
  if ! can_push=$(bounded 30 gh api --hostname github.com repos/pchatla-arch/anticipatory-balance \
      --jq '.permissions.push' 2>/dev/null); then
    die 'Could not verify GitHub push permission for pchatla-arch/anticipatory-balance. No experiment has started. Correct repository access and retry, or stop the still-running pod in the RunPod dashboard.'
  fi
  [[ "$can_push" == true ]] || \
    die 'The pchatla-arch account/token does not have verified push permission for pchatla-arch/anticipatory-balance. No experiment has started. Correct repository access and retry, or stop the still-running pod in the RunPod dashboard.'
}

if [[ ${1:-} == --worker ]]; then
  cd "$FORESIGHT_REPO"
  exec 9> "$FORESIGHT_SETUP_ROOT/.foresight-setup.lock"
  flock -n 9 || die 'Another Foresight setup is already running.'
  printf '\n=== Foresight setup/run %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Source revision: %s\n' "$(git rev-parse HEAD)"
  export FORESIGHT_SETUP_ROOT FORESIGHT_REPO FORESIGHT_VENV FORESIGHT_AUTO_STOP
  export ADAPTIVE_PROFILE ADAPTIVE_STARTED_EPOCH ADAPTIVE_DEADLINE_EPOCH
  export PYTHONPATH="$FORESIGHT_REPO${PYTHONPATH:+:$PYTHONPATH}"
  unset LD_LIBRARY_PATH MUJOCO_GL PYTHONHOME
  mkdir -p logs
  exec > >(tee -a logs/fresh_pod.log) 2>&1
  # An existing tmux server can have different credential environment variables
  # from the launching shell. Recheck before setup or any paid experiment work.
  if [[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 ]]; then require_github_access; fi
  exec python3 scripts/adaptive_gate_pod.py --profile "$ADAPTIVE_PROFILE"
fi

command -v nvidia-smi >/dev/null || die 'This pod has no exposed NVIDIA GPU. Select a GPU pod first.'
# An existing working environment is judged by preflight. Check the pinned
# fresh runtime before installing system tools when no such environment exists.
bounded 15 nvidia-smi --query-gpu=name,driver_version --format=csv,noheader,nounits
if [[ ! -x "$FORESIGHT_VENV/bin/python" ]]; then
  FORESIGHT_DRIVERS="$(bounded 15 nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
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
  bounded 480 env DEBIAN_FRONTEND=noninteractive apt-get "${apt_options[@]}" "$@"
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
if tmux has-session -t '=adaptive-gate' 2>/dev/null; then
  echo 'Session adaptive-gate already exists. View it with: tmux attach -t adaptive-gate'
  echo 'If the old run has ended, exit its shell before restarting this command.'
  exit 0
fi
if tmux has-session -t '=foresight-h' 2>/dev/null; then
  die 'The existing Session H must finish first. Inspect with: tmux attach -t foresight-h'
fi
if tmux has-session -t '=foresight-v2' 2>/dev/null; then
  die 'The existing foresight-v2 session must finish first. Inspect with: tmux attach -t foresight-v2'
fi
FORESIGHT_GPU_PIDS="$(bounded 15 nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
if [[ "$FORESIGHT_GPU_PIDS" =~ [0-9] ]]; then
  die 'A GPU process is already active. Let it finish before changing the checkout or starting another run.'
fi

git check-ref-format --allow-onelevel "$FORESIGHT_CODE_REF" >/dev/null || \
  [[ "$FORESIGHT_CODE_REF" =~ ^[0-9a-f]{40}$ ]] || die 'Invalid FORESIGHT_CODE_REF'
if [[ ! -e "$FORESIGHT_REPO" ]]; then
  echo 'Downloading the project and published checkpoint bundles.'
  FORESIGHT_CLONE_STAGE="$(mktemp -d "$FORESIGHT_SETUP_ROOT/.foresight-clone.XXXXXX")"
  bounded 600 env GIT_TERMINAL_PROMPT=0 git clone --filter=blob:none --no-checkout \
    https://github.com/pchatla-arch/anticipatory-balance.git "$FORESIGHT_CLONE_STAGE/repo"
  FORESIGHT_CHECKOUT="$FORESIGHT_CLONE_STAGE/repo"
else
  [[ -e "$FORESIGHT_REPO/.git" ]] || die "Existing path is not a Git checkout: $FORESIGHT_REPO"
  [[ -z "$(git -C "$FORESIGHT_REPO" status --porcelain --untracked-files=no)" ]] || \
    die 'The project has uncommitted tracked changes. They were preserved; save them before switching to this release.'
  FORESIGHT_CHECKOUT="$FORESIGHT_REPO"
fi
bounded 300 env GIT_TERMINAL_PROMPT=0 git -C "$FORESIGHT_CHECKOUT" fetch --no-tags \
  https://github.com/pchatla-arch/anticipatory-balance.git "$FORESIGHT_CODE_REF"
bounded 600 git -C "$FORESIGHT_CHECKOUT" -c advice.detachedHead=false checkout --detach FETCH_HEAD
if [[ -n "$FORESIGHT_CLONE_STAGE" ]]; then
  mv -- "$FORESIGHT_CHECKOUT" "$FORESIGHT_REPO"
  rmdir -- "$FORESIGHT_CLONE_STAGE"
  FORESIGHT_CLONE_STAGE=''
fi
[[ -f "$FORESIGHT_REPO/scripts/bootstrap_fresh_pod.sh" ]] || die 'Selected revision lacks the fresh-pod installer.'
[[ -f "$FORESIGHT_REPO/scripts/adaptive_gate_pod.py" ]] || die 'Selected revision lacks the reviewed pod session worker.'
mkdir -p "$FORESIGHT_REPO/logs"

if [[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 ]]; then
  FORESIGHT_GITHUB_USER=$(github_login) || FORESIGHT_GITHUB_USER=''
  if [[ "$FORESIGHT_GITHUB_USER" != pchatla-arch ]]; then
    if [[ -n ${GH_TOKEN:-} || -n ${GITHUB_TOKEN:-} ]]; then
      die 'GH_TOKEN/GITHUB_TOKEN does not authenticate as pchatla-arch. Correct or unset that override, then log in as pchatla-arch on the pod. No experiment has started. The pod is still running; stop it in the RunPod dashboard if leaving.'
    fi
    # Reuse the requested account when it is already stored by GitHub CLI.
    if bounded 30 gh auth switch --hostname github.com --user pchatla-arch >/dev/null 2>&1; then
      FORESIGHT_GITHUB_USER=$(github_login) || FORESIGHT_GITHUB_USER=''
    fi
    if [[ "$FORESIGHT_GITHUB_USER" != pchatla-arch ]]; then
      if [[ -t 0 ]]; then
        echo
        echo 'Log in to GitHub as pchatla-arch before GPU work starts.'
        echo 'Choose Paste an authentication token to enter your key directly on this pod, or choose browser login.'
        echo 'Do not paste the token into chat or put it in the repository URL.'
        bounded 600 --interactive gh auth login --hostname github.com --git-protocol https || \
          die 'GitHub login was not completed. No experiment has started; rerun after login. The pod is still running; stop it in the RunPod dashboard if leaving.'
      else
        die 'No terminal for GitHub login as pchatla-arch. Run gh auth login --hostname github.com --git-protocol https on the pod, or use existing Git credentials with FORESIGHT_SKIP_GITHUB_LOGIN=1. No experiment has started. The pod is still running; stop it in the RunPod dashboard if leaving.'
      fi
    fi
  fi
  require_github_access
  bounded 30 gh auth setup-git --hostname github.com
  git -C "$FORESIGHT_REPO" config --local credential.https://github.com.username pchatla-arch
  echo 'GitHub verified: pchatla-arch has repository push permission.'
fi
ADAPTIVE_WORKER_ENV=(
  "FORESIGHT_SETUP_ROOT=$FORESIGHT_SETUP_ROOT" "FORESIGHT_CODE_REF=$FORESIGHT_CODE_REF"
  "FORESIGHT_REPO=$FORESIGHT_REPO" "FORESIGHT_VENV=$FORESIGHT_VENV"
  "ADAPTIVE_PROFILE=$ADAPTIVE_PROFILE" "ADAPTIVE_STARTED_EPOCH=$ADAPTIVE_STARTED_EPOCH"
  "ADAPTIVE_DEADLINE_EPOCH=$ADAPTIVE_DEADLINE_EPOCH"
  "FORESIGHT_AUTO_STOP=$FORESIGHT_AUTO_STOP" "NO_STOP=${NO_STOP:-0}"
  "FORESIGHT_SKIP_GITHUB_LOGIN=$FORESIGHT_SKIP_GITHUB_LOGIN"
  "ADAPTIVE_DETERMINISTIC=${ADAPTIVE_DETERMINISTIC:-auto}"
)
printf -v FORESIGHT_WORKER 'cd %q && env ' "$FORESIGHT_REPO"
for ADAPTIVE_ASSIGNMENT in "${ADAPTIVE_WORKER_ENV[@]}"; do
  printf -v ADAPTIVE_QUOTED '%q ' "$ADAPTIVE_ASSIGNMENT"
  FORESIGHT_WORKER+="$ADAPTIVE_QUOTED"
done
FORESIGHT_WORKER+='bash scripts/start_adaptive_gate_pod.sh --worker 8>&-'

tmux new-session -d -s adaptive-gate "$FORESIGHT_WORKER"
exec 8>&-
rm -- "$FORESIGHT_ENTRY_LOCK/pid"
rmdir -- "$FORESIGHT_ENTRY_LOCK"
FORESIGHT_ENTRY_LOCK=''
echo
echo 'Setup and the adaptive actuator gate are running in tmux: adaptive-gate'
echo "Isolated checkout: $FORESIGHT_REPO"
echo "Live log: $FORESIGHT_REPO/logs/fresh_pod.log"
echo 'Detach without stopping: Ctrl-B, then D.'
echo "The original attempt deadline includes setup; the worker reserves 10 minutes for publication/stop."
echo "Automatic stop requested: $FORESIGHT_AUTO_STOP (NO_STOP=${NO_STOP:-0}); requires verified upload and RunPod controls."
echo 'If upload or stop controls fail, the pod stays running; stop it manually in the RunPod dashboard.'
echo 'Next: tmux attach -t adaptive-gate'
