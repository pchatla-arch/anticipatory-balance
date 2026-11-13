#!/usr/bin/env bash
# Session C (~1.5 GPU-h): belief and decision layer.
#   response model fit (CPU, seconds) -> E7 depth pilot (optional, timeboxed)
#   -> E10 decision frontier (reactive, always, TTC threshold sweep, belief rule sweep, oracle;
#      3 latencies; shared trajectories; 1024 episodes x 2 batches)
# Uses the committed checkpoints (checkpoints/nominal_s1, checkpoints/brace2_s1): no training.
#   bash scripts/run_session_c.sh 2>&1 | tee logs/session_c_$(date +%Y%m%d-%H%M).txt; bash scripts/finish_and_stop.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_DEFAULT_MATMUL_PRECISION=highest
unset LD_LIBRARY_PATH MUJOCO_GL || true
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
# Refuse to run outside the Playground venv (a plain shell has no jax; the session would
# fail at once and a chained finish_and_stop.sh would then stop the pod).
if ! $PY -c "import jax, absl" 2>/dev/null; then
  if [ -x /workspace/mujoco_playground/rss2025/.venv/bin/python ]; then
    echo "activating /workspace/mujoco_playground/rss2025/.venv"
    # shellcheck disable=SC1091
    source /workspace/mujoco_playground/rss2025/.venv/bin/activate
  else
    echo "ERROR: jax not importable and no venv found; run scripts/bootstrap_pod.sh first" >&2; exit 1
  fi
fi
IMPL=${IMPL:-warp}
EPISODES=${EPISODES:-1024}
BATCHES=${BATCHES:-2}
mkdir -p logs results
done_stage() { [ -f "results/$1/summary.json" ]; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }

stamp "response model from E6b"
$PY scripts/fit_response_model.py --episodes results/E6b_oracle_v2/episodes.csv --out results/response_model.json

if ! done_stage E7_depth_pilot; then
  stamp "E7 depth-render pilot (optional)"
  timeout 20m $PY scripts/depth_pilot.py --impl $IMPL --num_envs 64 --steps 150 --out results/E7_depth_pilot \
    || echo "E7 pilot did not complete (exit $?): parametric detections stay in use"
fi

if ! done_stage E10_frontier; then
  stamp "E10 decision frontier"
  $PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 \
      --impl $IMPL --episodes $EPISODES --batches $BATCHES --latencies 0,5,15 --out results/E10_frontier
fi

stamp "commit results"
git add results || true
git commit -m "Session C results: response model, E7 depth pilot, E10 decision frontier" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session C done"
