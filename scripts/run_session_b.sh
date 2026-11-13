#!/usr/bin/env bash
# Session B (~1.5 GPU-h): the corrected preparation gate.
#   E6a transition diagnostics with BRACE v1 (switch with/without projectile, zero-command control)
#   E2d walking fall surface, heavy regime, led aim   -> E3d LIP overlay
#   E5b BRACE v2 trained from walking states (handover from the frozen walker)
#   E6b oracle lead-time curve with BRACE v2, shared replayed trajectories
# Reuses logs/nominal_s1 and logs/brace_s1 from Session A when present (trains them otherwise).
#   bash scripts/run_session_b.sh 2>&1 | tee logs/session_b_$(date +%Y%m%d-%H%M).txt
set -euo pipefail
cd "$(dirname "$0")/.."
export JAX_DEFAULT_MATMUL_PRECISION=highest
unset LD_LIBRARY_PATH MUJOCO_GL || true
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python}
STEPS=${STEPS:-200000000}
NENVS=${NENVS:-4096}
IMPL=${IMPL:-warp}
mkdir -p logs results

newest_run() { ls -td "$1"/G1Impact*-[0-9]*-[0-9]* 2>/dev/null | head -1; }
done_stage() { [ -f "results/$1/summary.json" ]; }
stamp() { echo; echo "=== [$(date +%H:%M:%S)] $*"; }
has_final_ckpt() {
  local run; run=$(newest_run "$1" || true)
  [ -n "$run" ] && [ "$(ls -d "$run"/checkpoints/[0-9]* 2>/dev/null | wc -l)" -ge 1 ] && \
    grep -q "Done training\|Training and checkpoints complete" "logs/$(basename "$1")_train.txt" 2>/dev/null
}
ckpt_of() {
  local run; run=$(newest_run "logs/$1" || true)
  if [ -z "$run" ] || [ ! -d "$run/checkpoints" ]; then
    echo "ERROR: no training run with checkpoints under logs/$1" >&2; exit 1
  fi
  echo "$run/checkpoints"
}
train() {  # train <skill> <seed> <tag> [extra flags...]
  local skill=$1 seed=$2 tag=$3; shift 3
  if has_final_ckpt "logs/$tag"; then echo "reuse finished training run in logs/$tag"
  else
    stamp "train $tag ($STEPS steps)"
    $PY scripts/train_skill.py --skill "$skill" --impl $IMPL --domain_randomization \
        --num_timesteps $STEPS --num_envs $NENVS --seed "$seed" --use_tb --num_videos 0 \
        --logdir "logs/$tag" "$@" 2>&1 | tee "logs/${tag}_train.txt"
  fi
  done_stage "$tag" || $PY scripts/save_run.py --run_dir "$(newest_run logs/$tag)" --name "$tag" \
      --log "logs/${tag}_train.txt"
}

train nominal 1 nominal_s1
NOM=$(ckpt_of nominal_s1)
train brace 1 brace_s1
BR1=$(ckpt_of brace_s1)

# --- E6a: what does switching do, with and without a projectile? (BRACE v1, heavy regime)
if ! done_stage E6a_transition_v1; then
  stamp "E6a transition diagnostics (BRACE v1)"
  $PY scripts/eval_oracle_brace.py --nominal "$NOM" --brace "$BR1" --impl $IMPL \
      --episodes_per_lead 256 --batches 1 --out results/E6a_transition_v1
fi

# --- E2d: walking fall surface in the heavy regime with the led aim
if ! done_stage E2d_walk_heavy; then
  stamp "E2d walking fall surface (heavy, led aim)"
  $PY scripts/eval_fall_rate.py --checkpoint "$NOM" --impl $IMPL \
      --masses 4,8,16 --speeds 3,4,6,8 --episodes_per_cell 128 --batches 2 \
      --command_vx 0.5 --out results/E2d_walk_heavy
fi
if ! done_stage E3d_lip_walk; then
  stamp "E3d LIP overlay (walking, heavy)"
  $PY scripts/lip_overlay.py --episodes results/E2d_walk_heavy/episodes.csv --out results/E3d_lip_walk
fi

# --- E5b: BRACE v2, handed over from the frozen walker, impacts during training
train brace 1 brace2_s1 --brace_handover_ckpt "$NOM"
BR2=$(ckpt_of brace2_s1)

# --- E6b: oracle lead-time curve with BRACE v2, shared trajectories, controls
if ! done_stage E6b_oracle_v2; then
  stamp "E6b oracle lead-time curve (BRACE v2)"
  $PY scripts/eval_oracle_brace.py --nominal "$NOM" --brace "$BR2" --impl $IMPL \
      --episodes_per_lead 256 --batches 2 --out results/E6b_oracle_v2
fi

stamp "commit results"
git add results || true
git commit -m "Session B results: E6a transition diagnostics, E2d/E3d heavy walking surface, BRACE v2, E6b oracle" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session B done"
