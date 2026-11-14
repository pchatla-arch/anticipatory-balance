#!/usr/bin/env bash
# Downloadable entry point: install an empty pod and start the bounded experiment.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash start_empty_pod.sh

Installs missing Git/curl/CA/Python/tmux/util-linux/GitHub CLI system tools,
clones the Foresight release, installs its pinned Python 3.12 GPU environment,
restores published policies, and runs the smoke + pilot in tmux: foresight-run.

Overrides:
  FORESIGHT_SETUP_ROOT=/workspace
  FORESIGHT_CODE_REF=foresight/experiments-v2   (use a published SHA for a pinned run)
  FORESIGHT_MODE=pilot                       (or smoke)
  FORESIGHT_MAX_MINUTES=45                   (experiment budget, after setup)
  FORESIGHT_SKIP_GITHUB_LOGIN=1               (save locally; upload may fail)

The setup has its own 25-minute limit and can resume a matching interrupted
installation. Existing unrelated environments are reused only after preflight;
they are never upgraded by this entry point. The pod remains running afterward.
EOF
}

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
if (( $# > 1 )) || [[ $# == 1 && $1 != --worker ]]; then usage >&2; exit 2; fi
FORESIGHT_SETUP_ROOT="${FORESIGHT_SETUP_ROOT:-/workspace}"
FORESIGHT_CODE_REF="${FORESIGHT_CODE_REF:-foresight/experiments-v2}"
FORESIGHT_MODE="${FORESIGHT_MODE:-pilot}"
FORESIGHT_MAX_MINUTES="${FORESIGHT_MAX_MINUTES:-45}"
FORESIGHT_SKIP_GITHUB_LOGIN="${FORESIGHT_SKIP_GITHUB_LOGIN:-0}"
[[ "$FORESIGHT_MODE" == pilot || "$FORESIGHT_MODE" == smoke ]] || { echo 'Invalid FORESIGHT_MODE'; exit 2; }
[[ "$FORESIGHT_MAX_MINUTES" =~ ^[0-9]+([.][0-9]+)?$ ]] || { echo 'Invalid FORESIGHT_MAX_MINUTES'; exit 2; }
[[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 || "$FORESIGHT_SKIP_GITHUB_LOGIN" == 1 ]] || { echo 'Invalid FORESIGHT_SKIP_GITHUB_LOGIN'; exit 2; }
[[ "$FORESIGHT_SETUP_ROOT" == /* ]] || { echo 'FORESIGHT_SETUP_ROOT must be absolute'; exit 2; }
FORESIGHT_REPO="$FORESIGHT_SETUP_ROOT/anticipatory-balance"
FORESIGHT_VENV="$FORESIGHT_SETUP_ROOT/mujoco_playground/rss2025/.venv"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

if [[ ${1:-} == --worker ]]; then
  cd "$FORESIGHT_REPO"
  exec 9> "$FORESIGHT_SETUP_ROOT/.foresight-setup.lock"
  flock -n 9 || die 'Another Foresight setup is already running.'
  printf '\n=== Foresight setup/run %s ===\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf 'Source revision: %s\n' "$(git rev-parse HEAD)"
  export FORESIGHT_SETUP_ROOT FORESIGHT_VENV
  export PYTHONPATH="$FORESIGHT_REPO${PYTHONPATH:+:$PYTHONPATH}"
  unset LD_LIBRARY_PATH MUJOCO_GL PYTHONHOME
  if [[ -x "$FORESIGHT_VENV/bin/python" && ! -e "$FORESIGHT_SETUP_ROOT/.foresight-bootstrap/state.json" ]]; then
    echo 'Checking the existing environment before reusing it; no package installation will run.'
    timeout --kill-after=30s 5m "$FORESIGHT_VENV/bin/python" scripts/preflight_experiments_v2.py \
      --out logs/reused_environment_preflight.json
  else
    echo 'Installing or verifying the pinned environment. Setup limit: 25 minutes.'
    timeout --kill-after=30s 25m bash scripts/bootstrap_fresh_pod.sh
  fi
  echo 'Environment ready. Starting the bounded experiment.'
  export FORESIGHT_PYTHON="$FORESIGHT_VENV/bin/python"
  python3 scripts/run_experiments_v2.py --mode "$FORESIGHT_MODE" --max-minutes "$FORESIGHT_MAX_MINUTES"
  exit 0
fi

command -v nvidia-smi >/dev/null || die 'This pod has no exposed NVIDIA GPU. Select a GPU pod first.'
FORESIGHT_DRIVERS="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits)"
while IFS= read -r FORESIGHT_DRIVER; do
  FORESIGHT_MAJOR="${FORESIGHT_DRIVER%%.*}"
  [[ "$FORESIGHT_MAJOR" =~ ^[0-9]+$ && "$FORESIGHT_MAJOR" -ge 580 ]] || \
    die 'The pinned Warp runtime requires host driver 580 or newer. Select a compatible GPU pod; no installation was attempted.'
done <<< "$FORESIGHT_DRIVERS"

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
    echo 'GitHub CLI is unavailable. Continuing with local results and existing Git credentials.'
    FORESIGHT_SKIP_GITHUB_LOGIN=1
  fi
fi

mkdir -p "$FORESIGHT_SETUP_ROOT"
exec 8> "$FORESIGHT_SETUP_ROOT/.foresight-launch.lock"
flock -n 8 || die 'Another fresh-pod launcher is active.'
if tmux has-session -t '=foresight-run' 2>/dev/null; then
  echo 'Session foresight-run already exists. View it with: tmux attach -t foresight-run'
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
[[ -f "$FORESIGHT_REPO/scripts/start_empty_pod.sh" ]] || die 'Selected revision lacks the fresh-pod entry point.'
mkdir -p "$FORESIGHT_REPO/logs"

printf -v FORESIGHT_WORKER 'cd %q && env %q %q %q %q bash scripts/start_empty_pod.sh --worker 8>&- 2>&1 | tee -a logs/fresh_pod.log; exec bash' \
  "$FORESIGHT_REPO" "FORESIGHT_SETUP_ROOT=$FORESIGHT_SETUP_ROOT" "FORESIGHT_CODE_REF=$FORESIGHT_CODE_REF" \
  "FORESIGHT_MODE=$FORESIGHT_MODE" "FORESIGHT_MAX_MINUTES=$FORESIGHT_MAX_MINUTES"
tmux new-session -d -s foresight-run "$FORESIGHT_WORKER"
exec 8>&-
rm -- "$FORESIGHT_ENTRY_LOCK/pid"
rmdir -- "$FORESIGHT_ENTRY_LOCK"
FORESIGHT_ENTRY_LOCK=''
echo
echo 'Setup and experiments are running in tmux: foresight-run'
echo "Live log: $FORESIGHT_REPO/logs/fresh_pod.log"
echo 'View: tmux attach -t foresight-run'
echo 'Detach without stopping: Ctrl-B, then D.'
echo 'Limits: 25 minutes for setup, then 45 minutes by default for experiments.'
echo 'The script does not stop the pod or its billing.'

if [[ "$FORESIGHT_SKIP_GITHUB_LOGIN" == 0 ]]; then
  if gh auth status --hostname github.com >/dev/null 2>&1; then
    gh auth setup-git --hostname github.com
  elif [[ -t 0 ]]; then
    echo
    echo 'Setup continues in the background. Complete this GitHub browser login to enable result uploads.'
    echo 'No access token needs to be pasted into chat. If login is canceled, a local review ZIP is retained.'
    if gh auth login --hostname github.com --git-protocol https --web; then
      gh auth setup-git --hostname github.com
    else
      echo 'GitHub login was not completed. Setup continues; results will be saved locally.'
    fi
  else
    echo 'No interactive terminal for GitHub login. Later run: gh auth login --hostname github.com --git-protocol https --web'
    echo 'Then run: gh auth setup-git --hostname github.com'
  fi
fi
echo 'Next: tmux attach -t foresight-run'
