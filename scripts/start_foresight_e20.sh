#!/usr/bin/env bash
# Public entry point. Run on the pod whose GPU you intend to use.
set -euo pipefail
case "${1:-}" in
  --plan)
    echo 'E20: 512 calibration scenarios at 100 ms, then 4 x 512 fresh scenarios only after a calibration match. Frozen seed-1 skills. No PPO training.'
    echo '--run prepares an isolated checkout, checks GitHub write access, reuses or installs the pinned GPU environment, runs the protocol, and publishes. Add --stop to request shutdown after verified publication.'
    echo '--retry-publish retries the last upload without running physics. Paid work starts only with --run.'
    exit 0 ;;
  --run|--retry-publish) action=$1 ;;
  *) echo 'Usage: start_foresight_e20.sh --plan | {--run|--retry-publish} [--stop]' >&2; exit 2 ;;
esac
[[ $# -le 2 && ( -z "${2:-}" || "$2" == --stop ) ]] || { echo 'Unexpected option.' >&2; exit 2; }
die() { echo "$*" >&2; exit 1; }
for utility in git curl flock timeout python3 nohup; do command -v "$utility" >/dev/null || die "Missing on pod: $utility"; done
root=${FORESIGHT_SETUP_ROOT:-/workspace}
[[ "$root" == /* && -d "$root" && -w "$root" ]] || die "Provide a writable, absolute FORESIGHT_SETUP_ROOT ($root)."
export FORESIGHT_SETUP_ROOT="$root"
exec 9>"$root/.foresight-setup.lock"
flock -n 9 || die 'Another project job owns this workspace; leave it running.'
repo="$root/foresight-e20"
url=https://github.com/pchatla-arch/anticipatory-balance.git
scratch=$(mktemp -d "$root/.foresight-e20-start.XXXXXX")
trap 'rm -rf -- "$scratch"' EXIT
if [[ ! -e "$repo" ]]; then
  [[ "$action" == --run ]] || die "No existing E20 run at $repo; nothing to upload."
  ref=${FORESIGHT_CODE_REF:-review/foresight-paper-20261121}
  git check-ref-format --allow-onelevel "$ref" >/dev/null || die 'Invalid code ref.'
  git -C "$scratch" init -q checkout
  git -C "$scratch/checkout" remote add origin "$url"
  GIT_TERMINAL_PROMPT=0 timeout 300 git -C "$scratch/checkout" -c credential.interactive=never fetch --quiet --depth=1 origin "$ref"
  git -C "$scratch/checkout" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
  [[ -f "$scratch/checkout/scripts/run_e20_pod.py" ]] || die "Selected source lacks the E20 pod worker. Check the code ref."
  mv -- "$scratch/checkout" "$repo"
fi
[[ "$(git -C "$repo" config --get remote.origin.url)" == "$url" ]] || die 'Unexpected E20 checkout origin.'
[[ -z "$(git -C "$repo" status --porcelain --untracked-files=no)" ]] || die 'E20 checkout has tracked edits.'
if [[ -n "${FORESIGHT_CODE_REF:-}" && "$FORESIGHT_CODE_REF" =~ ^[0-9a-f]{40}$ ]]; then
  [[ "$(git -C "$repo" rev-parse HEAD)" == "$FORESIGHT_CODE_REF" ]] || die 'Existing E20 checkout differs from requested commit.'
fi
[[ -f "$repo/scripts/run_e20_pod.py" ]] || die "Selected checkout lacks the E20 pod worker. Check the code ref."
if [[ "$action" == --run ]]; then
  [[ ! -e "$repo/logs/e20_attempt.json" ]] || die 'An attempt is already recorded. See logs/e20_pod.log; use --retry-publish after an upload error.'
  command -v nvidia-smi >/dev/null || die 'No NVIDIA tools detected on this pod.'
  gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || die 'Cannot check GPU availability.'
  [[ -z "${gpu_pids//[[:space:]]/}" ]] || die 'GPU already has a compute process; leave that job running.'
fi
# This published revision of the credential helper supports both old and new pod layouts.
helper_ref=b2357abca938cc11942f9d385bc687fe1ceeb1dd
helper="$scratch/check-access"
curl --max-time 40 --retry 2 -fsSL "https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/$helper_ref/scripts/start_adaptive_gate_pod.sh" -o "$helper"
[[ "$(git hash-object "$helper")" == 1efa629c25737833d98a80380f6e455bac502e75 ]] || die 'Credential helper failed verification.'
now=$(date +%s)
FORESIGHT_REPO="$repo" ADAPTIVE_PROFILE=repair-access ADAPTIVE_STARTED_EPOCH="$now" ADAPTIVE_DEADLINE_EPOCH="$((now+1800))" bash "$helper" --repair-access
cd "$repo"
export FORESIGHT_AUTO_STOP=0
if [[ "${2:-}" == --stop ]]; then
  command -v runpodctl >/dev/null && [[ -n "${RUNPOD_POD_ID:-}" ]] || die 'Requested stop requires runpodctl and RUNPOD_POD_ID on the pod.'
  export FORESIGHT_AUTO_STOP=1
fi
mkdir -p logs
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH MUJOCO_GL PYTHONHOME
# Keep fd 9 locked in the worker across terminal disconnects.
nohup python3 scripts/run_e20_pod.py "$action" >> logs/e20_pod.log 2>&1 </dev/null &
echo "E20 $action started (PID $!). Monitor: tail -f $repo/logs/e20_pod.log"
echo 'If authentication or publication fails the pod may still be billing; check the RunPod dashboard.'
