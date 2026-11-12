# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Save a training run's results into results/<name>/ for commit.

Extracts eval scalars from the run's TensorBoard event files to CSV, plots the learning
curve to PNG, copies the env/PPO config JSONs and the console log, and writes a
summary.json (final reward, steps, train time, checkpoint path on the pod).

  python scripts/save_run.py --run_dir logs/bench_s1/G1Impact-20261112-042331 \
      --name bench_s1 --log logs_bench_s1.txt
"""
import glob
import json
import os
import re
import shutil

from absl import app, flags

_RUN = flags.DEFINE_string("run_dir", None, "Playground run dir (contains checkpoints/)", required=True)
_NAME = flags.DEFINE_string("name", None, "results/<name>", required=True)
_LOG = flags.DEFINE_string("log", None, "console log captured with tee (optional)")


def read_tb_scalars(run_dir):
  """Return {tag: [(step, value), ...]} from TensorBoard event files, if any."""
  try:
    from tensorboard.backend.event_processing import event_accumulator as ea
  except ImportError:
    return {}
  out = {}
  for f in glob.glob(os.path.join(run_dir, "**", "events.out.tfevents.*"), recursive=True):
    acc = ea.EventAccumulator(f, size_guidance={"scalars": 0}); acc.Reload()
    for tag in acc.Tags().get("scalars", []):
      out.setdefault(tag, []).extend((e.step, e.value) for e in acc.Scalars(tag))
  return {k: sorted(v) for k, v in out.items()}


def parse_console(log_path):
  """Fallback: parse 'step: reward=x' lines and timings from the console log."""
  curve, train_time, jit_time = [], None, None
  if not log_path or not os.path.exists(log_path):
    return curve, jit_time, train_time
  for line in open(log_path, errors="ignore"):
    m = re.match(r"\s*(\d+): reward=(-?[\d.]+)", line)
    if m:
      curve.append((int(m.group(1)), float(m.group(2))))
    m = re.search(r"Time to train: ([\d.]+)", line)
    if m: train_time = float(m.group(1))
    m = re.search(r"Time to JIT compile: ([\d.]+)", line)
    if m: jit_time = float(m.group(1))
  return curve, jit_time, train_time


def main(_):
  out = os.path.join("results", _NAME.value)
  os.makedirs(out, exist_ok=True)
  scalars = read_tb_scalars(_RUN.value)
  curve, jit_time, train_time = parse_console(_LOG.value)
  if "eval/episode_reward" in scalars:
    curve = scalars["eval/episode_reward"]

  with open(os.path.join(out, "eval_reward.csv"), "w") as f:
    f.write("env_steps,episode_reward\n")
    for s, v in curve:
      f.write(f"{s},{v}\n")
  for tag, vals in scalars.items():
    safe = tag.replace("/", "_")
    with open(os.path.join(out, f"tb_{safe}.csv"), "w") as f:
      f.write("step,value\n"); f.writelines(f"{s},{v}\n" for s, v in vals)

  try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if curve:
      xs, ys = zip(*curve)
      plt.figure(figsize=(5, 3.2)); plt.plot([x / 1e6 for x in xs], ys, marker="o", ms=3)
      plt.xlabel("environment steps (M)"); plt.ylabel("eval episode reward"); plt.title(_NAME.value)
      plt.grid(alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(out, "eval_reward.png"), dpi=150)
  except ImportError:
    pass

  for name in ("config.json", "ppo_params.json"):
    for p in glob.glob(os.path.join(_RUN.value, "**", name), recursive=True):
      shutil.copy(p, os.path.join(out, name)); break
  if _LOG.value and os.path.exists(_LOG.value):
    shutil.copy(_LOG.value, os.path.join(out, "console.log"))

  # Numeric step directories only (the folder also holds config.json), ordered by step.
  ckpts = sorted((p for p in glob.glob(os.path.join(_RUN.value, "checkpoints", "*"))
                  if os.path.isdir(p) and os.path.basename(p).isdigit()),
                 key=lambda p: int(os.path.basename(p)))
  summary = {
      "run_dir_on_pod": os.path.abspath(_RUN.value),
      "final_checkpoint_on_pod": ckpts[-1] if ckpts else None,
      "num_eval_points": len(curve),
      "final_env_steps": curve[-1][0] if curve else None,
      "final_eval_reward": curve[-1][1] if curve else None,
      "best_eval_reward": max(v for _, v in curve) if curve else None,
      "jit_time_s": jit_time, "train_time_s": train_time,
  }
  json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=2)
  print(json.dumps(summary, indent=2)); print("saved to", out)


if __name__ == "__main__":
  app.run(main)
