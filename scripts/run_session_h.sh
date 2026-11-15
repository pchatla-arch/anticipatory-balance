#!/usr/bin/env bash
# Session H (~1 GPU-h): two follow-ups from Session G.
#   E17c closed loop from depth with the parked projectile moved out of sensor range
#        (E17b's 21% false alarms were the parked object 6 m ahead), 0 and 100 ms added latency
#   E18b hold ablation: BRACE released 0.3 / 0.6 / 1.0 s after the belief's closest approach,
#        threshold and lead-conditioned rule, 10 Hz, latency 0 and 100 ms
#   bash scripts/run_session_h.sh 2>&1 | tee logs/session_h_$(date +%Y%m%d-%H%M).txt; bash scripts/finish_and_stop.sh
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
for f in checkpoints/nominal_s1 checkpoints/brace2_s1 results/response_model.json; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done

for L in 0 5; do
  tag=E17c_depth_lat$((L * 20))
  if ! done_stage $tag; then
    stamp "E17c closed loop from rendered depth, parked object out of range ($((L * 20)) ms added latency)"
    timeout 45m $PY scripts/eval_depth_closed_loop.py --impl $IMPL --episodes 256 --batches 2 --period 5 --extra_latency $L \
        --out results/$tag || echo "$tag did not complete (exit $?)"
  fi
done

FRONT="$PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 --impl $IMPL --episodes $EPISODES --batches $BATCHES"
for RA in 0.6 1.0; do
  tag=E18b_hold_$RA
  if ! done_stage $tag; then
    stamp "E18b hold ablation: release $RA s after closest approach"
    $FRONT --latencies 0,5 --period 5 --release_after $RA --methods ttc_threshold,belief_lead \
        --thetas 0.3,0.4 --c_preps 0.05,0.1 --calib_params "" --out results/$tag
  fi
done

stamp "commit results"
git add results || true
git commit -m "Session H results: E17c depth loop with the parked object out of range, E18b hold ablation" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session H done"
