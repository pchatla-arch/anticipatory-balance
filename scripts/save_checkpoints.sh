#!/usr/bin/env bash
# Copy the final checkpoint of every finished training run under logs/ into the repo
# (checkpoints/<tag>/<step>/, a few MB each) and push, so a lost pod never costs a retrain.
# Safe to run while a session is in progress (it only reads logs/ and writes checkpoints/).
#   bash scripts/save_checkpoints.sh            # all tags under logs/
#   bash scripts/save_checkpoints.sh nominal_s1 # one tag
# Load later with: --checkpoint checkpoints/<tag>   (rollout.latest_checkpoint picks the step)
set -euo pipefail
cd "$(dirname "$0")/.."
tags=("$@")
if [ ${#tags[@]} -eq 0 ]; then
  tags=($(ls logs 2>/dev/null | grep -E '^[a-z0-9]+_s[0-9]+$' || true))
fi
saved=()
for tag in "${tags[@]}"; do
  run=$(ls -td "logs/$tag"/G1Impact*-[0-9]*-[0-9]* 2>/dev/null | head -1 || true)
  [ -n "$run" ] || { echo "skip $tag: no run dir"; continue; }
  step=$(ls -d "$run"/checkpoints/[0-9]* 2>/dev/null | sort | tail -1 || true)
  [ -n "$step" ] || { echo "skip $tag: no checkpoint yet"; continue; }
  grep -q "Done training\|Training and checkpoints complete" "logs/${tag}_train.txt" 2>/dev/null \
    || { echo "skip $tag: training not finished"; continue; }
  dest="checkpoints/$tag/$(basename "$step")"
  if [ -d "$dest" ]; then echo "already saved: $dest"; continue; fi
  mkdir -p "checkpoints/$tag"
  cp -r "$step" "$dest"
  cp "$run/checkpoints/config.json" "checkpoints/$tag/env_config.json" 2>/dev/null || true
  echo "saved $dest ($(du -sh "$dest" | cut -f1))"
  saved+=("$tag")
done
if [ ${#saved[@]} -gt 0 ]; then
  git add checkpoints
  git commit -q -m "Checkpoints: ${saved[*]}" && git push || echo "push failed; run 'git push' manually"
fi
