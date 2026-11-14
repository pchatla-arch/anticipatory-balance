#!/usr/bin/env bash
# Session F (~1.5 GPU-h): the lookahead rule and the loop closed over rendered depth.
#   response model refit with the lead-conditioned BRACE term (CPU, seconds)
#   E16 frontier for belief_lead (myopic, lead-conditioned, no time-to-form gate) and
#       belief_voi (one-step lookahead with an explicit observation branch) at 10 and 50 Hz,
#       3 latencies, same paired scenarios as E15                                  ~40 min
#   E17 closed loop from rendered depth vs the parametric detector, 4 rules,
#       10 Hz, 0 and 100 ms added latency                                          ~25 min
#   bash scripts/run_session_f.sh 2>&1 | tee logs/session_f_$(date +%Y%m%d-%H%M).txt; bash scripts/finish_and_stop.sh
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
for f in checkpoints/nominal_s1 checkpoints/brace2_s1 results/E6b_oracle_v2/episodes.csv results/E15_rule_fixed_p5/summary.json; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

stamp "response model with the lead-conditioned BRACE term"
$PY scripts/fit_response_model.py --episodes results/E6b_oracle_v2/episodes.csv --out results/response_model.json

FRONT="$PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 --impl $IMPL --episodes $EPISODES --batches $BATCHES"
for P in 5 1; do
  if ! done_stage E16_lookahead_p$P; then
    stamp "E16 lookahead rules at period $P ($((50 / P)) Hz)"
    $FRONT --latencies 0,5,15 --period $P --methods reactive,belief_lead,belief_voi --calib_params "" \
        --out results/E16_lookahead_p$P
  fi
done

if ! done_stage E17_depth_closed_loop; then
  stamp "E17 closed loop from rendered depth (10 Hz, no added latency)"
  timeout 40m $PY scripts/eval_depth_closed_loop.py --impl $IMPL --episodes 256 --batches 2 --period 5 \
      --out results/E17_depth_closed_loop || echo "E17 did not complete (exit $?)"
fi
if ! done_stage E17_depth_closed_loop_lat100; then
  stamp "E17 closed loop from rendered depth (10 Hz, 100 ms added latency)"
  timeout 40m $PY scripts/eval_depth_closed_loop.py --impl $IMPL --episodes 256 --batches 2 --period 5 --extra_latency 5 \
      --out results/E17_depth_closed_loop_lat100 || echo "E17 (latency) did not complete (exit $?)"
fi

stamp "commit results"
git add results || true
git commit -m "Session F results: lead-conditioned response model, E16 lookahead rules, E17 closed loop from depth" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session F done"
