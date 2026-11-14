#!/usr/bin/env bash
# Session E (~2.5 GPU-h): the oracle gap, ablations and the end-to-end baseline.
#   E12 perfect-sensing frontier (50 Hz, zero latency/noise/misses, full FOV)   ~15 min
#   E13 ablations at 50 Hz, zero latency: commitment window, tau_brace, mass prior ~45 min
#   E14 object-aware PPO walker (detections as inputs, heavy impacts in training) ~25 min train
#       + its evaluation on the frontier scenarios at 10 Hz / 100 ms and 50 Hz / 0 ms
#   E12 summary ladder (CPU)
# Uses the committed checkpoints; trains one new policy (aware_s1).
#   bash scripts/run_session_e.sh 2>&1 | tee logs/session_e_$(date +%Y%m%d-%H%M).txt; bash scripts/finish_and_stop.sh
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
STEPS=${STEPS:-200000000}
NENVS=${NENVS:-4096}
mkdir -p logs results
done_stage() { [ -f "results/$1/summary.json" ]; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
newest_run() { ls -td "$1"/G1Impact*-[0-9]*-[0-9]* 2>/dev/null | head -1; }
has_final_ckpt() {
  local run; run=$(newest_run "$1" || true)
  [ -n "$run" ] && [ "$(ls -d "$run"/checkpoints/[0-9]* 2>/dev/null | wc -l)" -ge 1 ] && \
    grep -q "Done training\|Training and checkpoints complete" "logs/$(basename "$1")_train.txt" 2>/dev/null
}
for f in checkpoints/nominal_s1 checkpoints/brace2_s1 results/response_model.json results/E10_frontier/summary.json; do
  [ -e "$f" ] || { echo "ERROR: missing $f (Session C must have run)" >&2; exit 1; }
done
FRONT="$PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 --impl $IMPL --episodes $EPISODES --batches $BATCHES"

# --- E12: perfect sensing (the rule + skill alone, no perception limits)
if ! done_stage E12_perfect; then
  stamp "E12 perfect-sensing frontier"
  $FRONT --latencies 0 --period 1 --sigma0 0 --sigma1 0 --miss_prob 0 --fov_deg 360 --max_range 20 \
      --methods always,ttc_threshold,belief_commit --calib_params "" --out results/E12_perfect
fi

# --- E13: ablations at 50 Hz, zero latency (the regime where the rule is not sensing-limited)
abl() {  # abl <name> <flags...>
  local name=$1; shift
  if ! done_stage "E13_$name"; then
    stamp "E13 ablation $name"
    $FRONT --latencies 0 --period 1 --methods ttc_threshold,belief_commit --calib_params "" "$@" --out "results/E13_$name"
  fi
}
abl commit_5   --commit_steps 5
abl commit_25  --commit_steps 25
abl tau_0.10   --tau_brace 0.10
abl tau_0.35   --tau_brace 0.35
abl prior_wide --mass_prior 1,32
abl prior_point --mass_prior 8,8.01

# --- E14: object-aware PPO walker
if has_final_ckpt logs/aware_s1; then echo "reuse finished training run in logs/aware_s1"
else
  stamp "E14 train object-aware walker ($STEPS steps)"
  $PY scripts/train_skill.py --skill aware --impl $IMPL --domain_randomization \
      --num_timesteps $STEPS --num_envs $NENVS --seed 1 --use_tb --num_videos 0 \
      --logdir logs/aware_s1 2>&1 | tee logs/aware_s1_train.txt
fi
done_stage aware_s1 || $PY scripts/save_run.py --run_dir "$(newest_run logs/aware_s1)" --name aware_s1 --log logs/aware_s1_train.txt
AW="$(newest_run logs/aware_s1)/checkpoints"
bash scripts/save_checkpoints.sh aware_s1 || echo "(checkpoint copy skipped)"
if ! done_stage E14_aware_ppo; then
  stamp "E14 evaluate the object-aware walker (10 Hz / 100 ms)"
  $PY scripts/eval_decision_frontier.py --nominal "$AW" --impl $IMPL --episodes $EPISODES --batches $BATCHES \
      --env_skill aware --period 5 --latencies 5 --methods reactive --calib_params "" --out results/E14_aware_ppo
fi
if ! done_stage E14_aware_ppo_50hz; then
  stamp "E14 evaluate the object-aware walker (50 Hz / 0 ms)"
  $PY scripts/eval_decision_frontier.py --nominal "$AW" --impl $IMPL --episodes $EPISODES --batches $BATCHES \
      --env_skill aware --period 1 --latencies 0 --methods reactive --calib_params "" --out results/E14_aware_ppo_50hz
fi

stamp "E12 oracle-gap ladder"
$PY scripts/oracle_gap.py --out results/E12_oracle_gap

stamp "commit results"
git add results checkpoints || true
git commit -m "Session E results: E12 perfect sensing and ladder, E13 ablations, E14 object-aware PPO" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session E done"
