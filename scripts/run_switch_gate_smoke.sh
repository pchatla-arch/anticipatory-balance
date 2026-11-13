#!/usr/bin/env bash
# Run the fixed-checkpoint stabilization diagnostic in the existing pod venv.
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_switch_gate_smoke.sh

Runs the simulator-free evaluation tests, verifies the existing GPU/checkpoints,
then runs 32 paired scenarios with six comparison arms and 32 reference rollouts.
Publishes compact sanitized results to the dedicated foresight/pod-results branch.
No packages are installed and no training, rendering, or code/main changes are made.

Optional environment overrides (use existing paths):
  FORESIGHT_VENV     Default: /workspace/mujoco_playground/rss2025/.venv
  FORESIGHT_NOMINAL  Default: <repo>/logs/nominal_s1/G1Impact-20261113-063725/
                             checkpoints/000202342400
  FORESIGHT_BRACE    Default: <repo>/logs/brace_s1/G1ImpactBrace-20261113-065807/
                             checkpoints/000202342400
  FORESIGHT_PUBLISH_RESULTS  Default: 1; set to 0 for an offline/debug run.

Creates a unique results directory and an external log under the repository.
Prints the review ZIP path on exit; raw NPZ traces remain in the results directory.
An ordinary preflight/run error still packages and publishes available files.
Publication failures print a retry command that does not rerun the GPU diagnostic.
GPU execution of this proposed diagnostic remains to be validated on the pod.
EOF
}

if (( $# )); then
  if [[ $# == 1 && ( "$1" == --help || "$1" == -h ) ]]; then
    usage
    exit 0
  fi
  printf 'Unsupported arguments. Use --help or the documented environment overrides.\n' >&2
  exit 2
fi

FORESIGHT_PUBLISH_RESULTS="${FORESIGHT_PUBLISH_RESULTS:-1}"
if [[ "$FORESIGHT_PUBLISH_RESULTS" != 0 && "$FORESIGHT_PUBLISH_RESULTS" != 1 ]]; then
  printf 'FORESIGHT_PUBLISH_RESULTS must be 0 or 1.\n' >&2
  exit 2
fi

FORESIGHT_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$FORESIGHT_REPO_ROOT"
FORESIGHT_VENV="${FORESIGHT_VENV:-/workspace/mujoco_playground/rss2025/.venv}"
FORESIGHT_NOMINAL="${FORESIGHT_NOMINAL:-$FORESIGHT_REPO_ROOT/logs/nominal_s1/G1Impact-20261113-063725/checkpoints/000202342400}"
FORESIGHT_BRACE="${FORESIGHT_BRACE:-$FORESIGHT_REPO_ROOT/logs/brace_s1/G1ImpactBrace-20261113-065807/checkpoints/000202342400}"

# Reserve a new log directory atomically. The evaluation output must remain absent.
mkdir -p "$FORESIGHT_REPO_ROOT/logs" "$FORESIGHT_REPO_ROOT/results"
FORESIGHT_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
FORESIGHT_LOG_DIR="$(mktemp -d "$FORESIGHT_REPO_ROOT/logs/switch_gate_smoke_v1_${FORESIGHT_STAMP}_XXXXXX")"
FORESIGHT_TAG="${FORESIGHT_LOG_DIR##*/}"
FORESIGHT_LOG="$FORESIGHT_LOG_DIR/run.log"
FORESIGHT_REVIEW_ZIP="$FORESIGHT_LOG_DIR/review.zip"
FORESIGHT_OUT="$FORESIGHT_REPO_ROOT/results/$FORESIGHT_TAG"
# This fallback lets a missing-venv failure still produce a ZIP without installing anything.
FORESIGHT_ARCHIVE_PYTHON="$(type -P python3 || type -P python || true)"

exec 3>&1 4>&2
exec > >(tee -a "$FORESIGHT_LOG") 2>&1
FORESIGHT_TEE_PID=$!

finish() {
  local diagnostic_status=$?
  local run_status=$diagnostic_status
  local log_status archive_status publication_status
  local publisher="$FORESIGHT_REPO_ROOT/scripts/publish_switch_gate_results.py"
  trap - EXIT
  set +e
  # Close the logging pipe and wait until all preceding output is on disk.
  exec 1>&3 2>&4
  wait "$FORESIGHT_TEE_PID"
  log_status=$?
  if (( run_status == 0 && log_status != 0 )); then
    run_status=$log_status
  fi
  printf '\nRun exit status: %s\nResults directory: %s\nConsole log: %s\n' \
    "$run_status" "$FORESIGHT_OUT" "$FORESIGHT_LOG" | tee -a "$FORESIGHT_LOG"

  if [[ -z "$FORESIGHT_ARCHIVE_PYTHON" ]]; then
    printf 'Could not create review ZIP: no existing Python interpreter is available.\n' >&2
    (( run_status != 0 )) || run_status=1
    exit "$run_status"
  fi
  "$FORESIGHT_ARCHIVE_PYTHON" - "$FORESIGHT_REVIEW_ZIP" "$FORESIGHT_OUT" "$FORESIGHT_LOG" <<'PY'
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

archive, output, log = map(Path, sys.argv[1:])
with ZipFile(archive, mode="x", compression=ZIP_DEFLATED) as z:
    for name in ("summary.json", "manifest.json", "scenarios.csv", "episodes.csv"):
        source = output / name
        if source.is_file():
            z.write(source, arcname=name)
    if log.is_file():
        z.write(log, arcname="run.log")
PY
  archive_status=$?
  if (( archive_status == 0 )); then
    printf 'Review ZIP: %s\n' "$FORESIGHT_REVIEW_ZIP"
  else
    printf 'Could not create review ZIP at %s; preserve the log/results above.\n' \
      "$FORESIGHT_REVIEW_ZIP" >&2
    (( run_status != 0 )) || run_status=$archive_status
  fi

  if [[ "$FORESIGHT_PUBLISH_RESULTS" == 0 ]]; then
    printf 'Results publication disabled by FORESIGHT_PUBLISH_RESULTS=0.\n'
    exit "$run_status"
  fi
  printf '\nPublishing compact results to foresight/pod-results.\n'
  if ! type -P git >/dev/null; then
    printf 'Results publication failed: Git is not available on PATH.\n' >&2
    publication_status=127
  elif [[ ! -f "$publisher" ]]; then
    printf 'Results publisher is missing: %s\n' "$publisher" >&2
    publication_status=1
  else
    "$FORESIGHT_ARCHIVE_PYTHON" "$publisher" \
      --run-dir "$FORESIGHT_OUT" --log "$FORESIGHT_LOG" --exit-code "$diagnostic_status"
    publication_status=$?
  fi
  if (( publication_status != 0 )); then
    printf 'Results publication failed; the diagnostic will not be rerun.\n' >&2
    if (( archive_status == 0 )); then
      printf 'Local review ZIP retained: %s\n' "$FORESIGHT_REVIEW_ZIP" >&2
    fi
    printf 'Retry publication only, using the existing artifacts:\n  cd %q && ' \
      "$FORESIGHT_REPO_ROOT" >&2
    printf '%q ' "$FORESIGHT_ARCHIVE_PYTHON" "$publisher" \
      --run-dir "$FORESIGHT_OUT" --log "$FORESIGHT_LOG" --exit-code "$diagnostic_status" >&2
    printf '\n' >&2
    (( run_status != 0 )) || run_status=$publication_status
  else
    printf 'Results are published at the GitHub URL above. Say done when ready for review.\n'
  fi
  exit "$run_status"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

printf 'Foresight stabilization smoke diagnostic\nRepository: %s\nVenv: %s\n' \
  "$FORESIGHT_REPO_ROOT" "$FORESIGHT_VENV"
printf 'Results directory: %s\nConsole log: %s\n' "$FORESIGHT_OUT" "$FORESIGHT_LOG"
printf 'Publish compact results on exit: %s\n' "$FORESIGHT_PUBLISH_RESULTS"
if [[ -e "$FORESIGHT_OUT" ]]; then
  printf 'Output already exists; refusing to reuse it: %s\n' "$FORESIGHT_OUT" >&2
  exit 1
fi
if [[ ! -r "$FORESIGHT_VENV/bin/activate" || ! -x "$FORESIGHT_VENV/bin/python" ]]; then
  printf 'Existing venv is missing or incomplete: %s\nSet FORESIGHT_VENV to the working rss2025 venv; no environment was created.\n' \
    "$FORESIGHT_VENV" >&2
  exit 1
fi

# Match the successful Session A environment. Reuse it without resolving dependencies.
source "$FORESIGHT_VENV/bin/activate"
FORESIGHT_PYTHON="$FORESIGHT_VENV/bin/python"
# Preserve the initial stdlib Python so a venv/dependency failure can still be published.
FORESIGHT_ARCHIVE_PYTHON="${FORESIGHT_ARCHIVE_PYTHON:-$FORESIGHT_PYTHON}"
export PYTHONPATH="$FORESIGHT_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export JAX_DEFAULT_MATMUL_PRECISION=highest
unset LD_LIBRARY_PATH MUJOCO_GL

printf '\nRunning simulator-free evaluation tests and CLI check.\n'
"$FORESIGHT_PYTHON" -m unittest discover -s tests -p 'test_eval_*.py' -v
"$FORESIGHT_PYTHON" scripts/eval_switch_gate.py --help

printf '\nChecking GPU, source imports, and restoring both saved checkpoints.\n'
"$FORESIGHT_PYTHON" - "$FORESIGHT_REPO_ROOT" "$FORESIGHT_VENV" \
  "$FORESIGHT_NOMINAL" "$FORESIGHT_BRACE" <<'PY'
from importlib import metadata
from pathlib import Path
import sys

root, venv, nominal, brace = map(Path, sys.argv[1:])
print("Python:", sys.executable, flush=True)
if Path(sys.prefix).resolve() != venv.resolve():
    raise RuntimeError("Python did not use the selected existing venv")
for name in ("jax", "jaxlib", "brax", "mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang"):
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = "not installed under this distribution name"
    print(f"{name}: {version}", flush=True)

import jax
import mujoco_playground
from foresight.eval import rollout as R

print("Playground:", mujoco_playground.__file__, flush=True)
print("Foresight rollout:", R.__file__, flush=True)
if Path(R.__file__).resolve() != (root / "foresight/eval/rollout.py").resolve():
    raise RuntimeError("Python imported a different Foresight checkout")
devices = jax.devices()
print("JAX devices:", devices, flush=True)
if not any(device.platform == "gpu" for device in devices):
    raise RuntimeError("JAX has no GPU device; stopping before the smoke rollout")
for label, path in (("nominal", nominal), ("brace", brace)):
    path = path.resolve()
    if not path.is_dir() or not (path / "ppo_network_config.json").is_file():
        raise FileNotFoundError(f"{label}: incomplete or missing checkpoint: {path}")
    if R.latest_checkpoint(path) != path:
        raise RuntimeError(f"{label}: checkpoint resolved unexpectedly")
    R.load_policy(path)
    print(f"{label}: restored {path}", flush=True)
PY

printf '\nStarting 32 shared scenarios: 192 comparison rollouts plus 32 references.\n'
"$FORESIGHT_PYTHON" -u scripts/eval_switch_gate.py \
  --nominal "$FORESIGHT_NOMINAL" \
  --brace "$FORESIGHT_BRACE" \
  --impl warp --episodes 32 --batches 1 --lead 0.5 --hold 0.5 --steps 300 \
  --recovery_window 2.5 --command_vx 0.5 --seed 100 \
  --out "$FORESIGHT_OUT"

printf '\nDiagnostic finished. Read summary.status and the three no-projectile arms.\n'
printf 'Both switch_failure and inconclusive are completed experiments, not a safety pass.\n'
