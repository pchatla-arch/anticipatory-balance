#!/usr/bin/env bash
# Session A (one shot, ~2.5 GPU-h on an RTX 6000 Ada):
#   E4 contact validation -> E1 nominal x3 seeds -> E5 brace -> E2 fall-rate surface
#   -> E3 LIP overlay -> E6 oracle brace lead-time curve -> save + commit + push.
# Run from the repo root inside the rss2025 venv, in tmux:
#   bash scripts/run_session_a.sh 2>&1 | tee logs/session_a_$(date +%Y%m%d-%H%M).txt
# Re-running skips stages whose results/ directory already has a summary.json.
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

has_final_ckpt() {  # a run dir whose checkpoints/ holds the final step dir
  local run; run=$(newest_run "$1") || return 1
  [ -n "$run" ] && [ -d "$run/checkpoints" ] && \
    [ "$(ls -d "$run"/checkpoints/[0-9]* 2>/dev/null | wc -l)" -ge 1 ] && \
    grep -q "Done training\|Training and checkpoints complete" "logs/$(basename "$1")_train.txt" 2>/dev/null
}

train() {  # train <skill> <seed>  -> logs/<skill>_s<seed>/<run>/checkpoints
  local skill=$1 seed=$2 tag="${1}_s${2}"
  if done_stage "$tag"; then echo "skip $tag (results/$tag/summary.json exists)"; return; fi
  if has_final_ckpt "logs/$tag"; then
    echo "reuse finished training run in logs/$tag (saving results only)"
  else
    stamp "train $tag ($STEPS steps)"
    $PY scripts/train_skill.py --skill "$skill" --impl $IMPL --domain_randomization \
        --num_timesteps $STEPS --num_envs $NENVS --seed "$seed" --use_tb --num_videos 0 \
        --logdir "logs/$tag" 2>&1 | tee "logs/${tag}_train.txt"
  fi
  $PY scripts/save_run.py --run_dir "$(newest_run logs/$tag)" --name "$tag" \
      --log "logs/${tag}_train.txt"
}

# --- E4: contact validation (no policy needed, ~2 min incl. compile)
if ! done_stage E4_contact; then
  stamp "E4 contact validation"
  $PY scripts/validate_contact.py --impl $IMPL --reps 2 --out results/E4_contact
fi

# --- E1: nominal skill, 3 seeds, all trained with per-episode projectile resampling.
# (results/nominal_s1_naive is the Nov 12 policy that saw a projectile only in each lane's
#  first episode; kept as the projectile-naive baseline.)
train nominal 1
train nominal 2
train nominal 3
ckpt_of() {  # ckpt_of <tag> -> <run>/checkpoints, or exit with a clear message
  local run; run=$(newest_run "logs/$1" || true)
  if [ -z "$run" ] || [ ! -d "$run/checkpoints" ]; then
    echo "ERROR: no training run with checkpoints under logs/$1" >&2; exit 1
  fi
  echo "$run/checkpoints"
}
NOM=$(ckpt_of nominal_s1)

# --- E5: BRACE skill
train brace 1
BR=$(ckpt_of brace_s1)

# --- E2: reactive fall-rate surface (nominal seed 1)
if ! done_stage E2_fall_rate; then
  stamp "E2 fall-rate surface"
  $PY scripts/eval_fall_rate.py --checkpoint "$NOM" --impl $IMPL \
      --masses 1,2,4,8 --speeds 1,2,3,4 --episodes_per_cell 128 --batches 2 \
      --out results/E2_fall_rate
fi

# --- E3: LIP / capture-point overlay (CPU, seconds)
if ! done_stage E3_lip_overlay; then
  stamp "E3 LIP overlay"
  $PY scripts/lip_overlay.py --episodes results/E2_fall_rate/episodes.csv --out results/E3_lip_overlay
fi

# --- E6: oracle brace lead-time curve
if ! done_stage E6_oracle; then
  stamp "E6 oracle brace"
  $PY scripts/eval_oracle_brace.py --nominal "$NOM" --brace "$BR" --impl $IMPL \
      --episodes_per_lead 256 --batches 2 --out results/E6_oracle
fi

# --- save to GitHub
stamp "commit results"
git add results logs/*.txt || true
git commit -m "Session A results: E1 nominal x3, E5 brace, E2 fall-rate, E3 LIP, E4 contact, E6 oracle" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session A done"
