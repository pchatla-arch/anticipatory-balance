#!/usr/bin/env bash
# Session G (~2.5 GPU-h): verification of the corrected timing/hold contract, the lookahead
# ladder on one hold contract, the depth loop with consistent validity, held-out scenarios,
# and a second skill seed pair.
#   E6c  readiness measured relative to contact and the active skill; release reasons     ~10 min
#   E18  rule family on one response model and hold contract, 10 Hz, 3 latencies:
#        threshold, belief_lead (act now), belief_wait (blind wait), belief_voi (observe)  ~30 min
#   E18h held-out scenarios (fresh seed) for the parameters selected on E18                ~10 min
#   E17b closed loop from depth: capture-time validity, median detector, false alarms     ~30 min
#   E19  seed pair 2: walker s2 + BRACE v2 s2 trained, frontier at 10 Hz                  ~60 min
#   bash scripts/run_session_g.sh 2>&1 | tee logs/session_g_$(date +%Y%m%d-%H%M).txt; bash scripts/finish_and_stop.sh
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
reuse_published_checkpoint() {
  # A fresh pod has the published bundles but none of the old training logs.
  # Validate and restore the newest numeric checkpoint on CPU before trusting it.
  # A partial published bundle is an error, not permission to spend an hour
  # silently replacing the trained policies used by the existing results.
  local tag=$1 bundle="checkpoints/$1"
  [ -e "$bundle" ] || return 2
  if JAX_PLATFORMS=cpu "$PY" - "$bundle" <<'PY'
import json
from pathlib import Path
import sys

bundle = Path(sys.argv[1]).resolve()
if not bundle.is_dir():
    raise SystemExit(f"Published checkpoint is not a directory: {bundle}")
steps = [p for p in bundle.iterdir() if p.is_dir() and p.name.isdigit()]
if not steps:
    raise SystemExit(f"Published bundle has no numeric checkpoint: {bundle}")
checkpoint = max(steps, key=lambda p: int(p.name))
required = ('manifest.ocdbt', 'ppo_network_config.json', '_METADATA',
            '_CHECKPOINT_METADATA', 'commit_success.txt')
for name in required:
    path = checkpoint / name
    if not path.is_file() or path.stat().st_size == 0:
        raise SystemExit(f"Incomplete published checkpoint: missing or empty {path}")
for name in ('ppo_network_config.json', '_METADATA', '_CHECKPOINT_METADATA'):
    with (checkpoint / name).open() as stream:
        value = json.load(stream)
    if not isinstance(value, dict) or not value:
        raise SystemExit(f"Invalid checkpoint metadata: {checkpoint / name}")
    if name == '_CHECKPOINT_METADATA' and not value.get('commit_timestamp_nsecs'):
        raise SystemExit(f"Checkpoint has no completed commit: {checkpoint}")
manifests = list(checkpoint.rglob('manifest.ocdbt'))
for manifest in manifests:
    data_dir = manifest.parent / 'd'
    payloads = [p for p in data_dir.iterdir() if p.is_file()] if data_dir.is_dir() else []
    if not payloads or any(p.stat().st_size == 0 for p in payloads):
        raise SystemExit(f"Incomplete OCDBT payload directory: {data_dir}")
    for payload in payloads:
        with payload.open('rb') as stream:
            if stream.read(80).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise SystemExit(f"OCDBT payload is an unfetched LFS pointer: {payload}")
# A structural inventory cannot prove that every OCDBT reference resolves.
# The same loader used by evaluation checks the actual data before training is skipped.
from foresight.eval import rollout as R
R.load_policy(checkpoint)
print(f"Verified published checkpoint on CPU: {checkpoint}", flush=True)
PY
  then
    echo "reuse published checkpoint checkpoints/$tag; no retraining or log recovery needed"
    return 0
  fi
  echo "ERROR: published checkpoints/$tag failed validation/restore. Repair the bundle before rerunning; no replacement training was started." >&2
  return 1
}
train() {  # train <skill> <seed> <tag> [extra flags...]
  local skill=$1 seed=$2 tag=$3; shift 3
  local reuse_status=0
  reuse_published_checkpoint "$tag" || reuse_status=$?
  if [ "$reuse_status" -eq 0 ]; then return 0; fi
  if [ "$reuse_status" -ne 2 ]; then return "$reuse_status"; fi
  if has_final_ckpt "logs/$tag"; then echo "reuse finished training run in logs/$tag"
  else
    stamp "train $tag ($STEPS steps)"
    $PY scripts/train_skill.py --skill "$skill" --impl $IMPL --domain_randomization \
        --num_timesteps $STEPS --num_envs $NENVS --seed "$seed" --use_tb --num_videos 0 \
        --logdir "logs/$tag" "$@" 2>&1 | tee "logs/${tag}_train.txt"
  fi
  done_stage "$tag" || $PY scripts/save_run.py --run_dir "$(newest_run logs/$tag)" --name "$tag" --log "logs/${tag}_train.txt"
  bash scripts/save_checkpoints.sh "$tag" || echo "(checkpoint copy skipped)"
}
for f in checkpoints/nominal_s1 checkpoints/brace2_s1 results/response_model.json; do
  [ -e "$f" ] || { echo "ERROR: missing $f" >&2; exit 1; }
done
grep -q c_lead results/response_model.json || $PY scripts/fit_response_model.py --episodes results/E6b_oracle_v2/episodes.csv --out results/response_model.json

FRONT="$PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 --impl $IMPL"

# --- E6c: readiness relative to contact and the active skill, release reasons
if ! done_stage E6c_readiness; then
  stamp "E6c readiness and hold verification (BRACE v2, shared trajectories)"
  $PY scripts/eval_oracle_brace.py --nominal checkpoints/nominal_s1 --brace checkpoints/brace2_s1 --impl $IMPL \
      --modes brace --leads=-1,0.1,0.2,0.3,0.5 --episodes_per_lead 512 --batches 1 --nono_launch_controls --out results/E6c_readiness
fi

# --- E18: the rule family on one hold contract (corrected release), 10 Hz
if ! done_stage E18_rule_family_p5; then
  stamp "E18 rule family at 10 Hz (threshold, lead, blind wait, observe)"
  $FRONT --episodes $EPISODES --batches $BATCHES --latencies 0,5,15 --period 5 \
      --methods reactive,oracle,ttc_threshold,belief_lead,belief_wait,belief_voi --calib_params "" \
      --out results/E18_rule_family_p5
fi
# --- E18h: held-out scenarios for the operating points selected on E18 (fresh seed, fixed parameters)
if ! done_stage E18h_heldout_p5; then
  stamp "E18h held-out evaluation (seed 400)"
  $FRONT --episodes $EPISODES --batches $BATCHES --latencies 0,5,15 --period 5 --seed 400 \
      --methods reactive,ttc_threshold,belief_lead,belief_wait,belief_voi --thetas 0.3 --c_preps 0.05,0.1 --calib_params "" \
      --out results/E18h_heldout_p5
fi

# --- E17b: closed loop from depth, consistent validity and the calibrated (median) detector
for L in 0 5; do
  tag=E17b_depth_lat$((L * 20))
  if ! done_stage $tag; then
    stamp "E17b closed loop from rendered depth (10 Hz, $((L * 20)) ms added latency)"
    timeout 45m $PY scripts/eval_depth_closed_loop.py --impl $IMPL --episodes 256 --batches 2 --period 5 --extra_latency $L \
        --out results/$tag || echo "$tag did not complete (exit $?)"
  fi
done

# --- E19: a second seed pair (walker s2, BRACE v2 from it), frontier at 10 Hz
if ! done_stage E19_seed2_p5; then
  train nominal 2 nominal_s2
  NOM2=checkpoints/nominal_s2
  train brace 2 brace2_s2 --brace_handover_ckpt "$NOM2"
  stamp "E19 frontier at 10 Hz with seed pair 2"
  $PY scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s2 --brace checkpoints/brace2_s2 --impl $IMPL \
      --episodes $EPISODES --batches $BATCHES --latencies 0,5 --period 5 \
      --methods reactive,oracle,ttc_threshold,belief_lead,belief_voi --thetas 0.3,0.4 --c_preps 0.05,0.1,0.2 --calib_params "" \
      --out results/E19_seed2_p5
fi

stamp "commit results"
git add results checkpoints || true
git commit -m "Session G results: E6c readiness, E18 rule family + held-out, E17b depth loop, E19 seed pair 2" || true
git push || echo "push failed; run 'git push' manually"
stamp "Session G done"
