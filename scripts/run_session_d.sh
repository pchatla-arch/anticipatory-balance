#!/usr/bin/env bash
# Session D (~2.5 GPU-h): is sensing rate the binding constraint?
#   E8  observation model from rendered depth (detector noise, detection rate, FOV; timeboxed)
#   E11 decision frontier at 20 Hz and 50 Hz detections (E10 supplies 10 Hz), 3 latencies,
#       same paired scenarios; belief logs dumped for E9
#   E9  belief calibration (reliability, time-to-hit error, when the belief resolves)
#   E11 rate x latency summary figures and the crossover rate
# Uses the committed checkpoints: no training.
#   bash scripts/run_session_d.sh 2>&1 | tee logs/session_d_$(date +%Y%m%d-%H%M).txt; bash scripts/finish_and_stop.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_DEFAULT_MATMUL_PRECISION=highest
unset LD_LIBRARY_PATH MUJOCO_GL || true
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
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

for f in checkpoints/nominal_s1 checkpoints/brace2_s1 results/response_model.json results/E10_frontier/summary.json; do
  [ -e "$f" ] || { echo "ERROR: missing $f (Session C must have run)" >&2; exit 1; }
done

# --- E8: observation model from rendered depth (optional, timeboxed)
if ! done_stage E8_depth_detector; then
  stamp "E8 depth detector on rendered frames"
  timeout 25m $PY scripts/depth_detector.py --impl $IMPL --num_envs 128 --steps 200 --render_every 2 --res 96 \
      --out results/E8_depth_detector || echo "E8 did not complete (exit $?): parametric detector numbers stay in use"
fi

# --- E11: the frontier at 20 Hz and 50 Hz (period 2 and 1); reactive and oracle come from E10
for P in 2 1; do
  if ! done_stage E11_rate_p$P; then
    stamp "E11 decision frontier at period $P ($((50 / P)) Hz)"
    $PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 \
        --impl $IMPL --episodes $EPISODES --batches $BATCHES --latencies 0,5,15 --period $P \
        --methods always,ttc_threshold,belief_commit --out results/E11_rate_p$P
  fi
done

# --- E10 at 10 Hz has no belief dumps (it predates them): rerun just the calibration condition there.
if ! done_stage E10_frontier_calib; then
  stamp "E10 calibration condition at 10 Hz (belief rule, c_prep 0.1, for E9)"
  $PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 \
      --impl $IMPL --episodes $EPISODES --batches 1 --latencies 0,5,15 --period 5 \
      --methods reactive,belief_commit --c_preps 0.1 --out results/E10_frontier_calib
fi

# --- E9 and the E11 summary (CPU, seconds)
stamp "E9 belief calibration"
$PY scripts/calibration.py --dirs results/E10_frontier_calib,results/E11_rate_p2,results/E11_rate_p1 --out results/E9_calibration
stamp "E11 rate x latency summary"
$PY scripts/plot_rate_sweep.py --dirs results/E10_frontier,results/E11_rate_p2,results/E11_rate_p1 --out results/E11_rate_sweep

stamp "commit results"
git add results || true
git commit -m "Session D results: E8 depth detector, E11 rate sweep, E9 calibration" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session D done"
