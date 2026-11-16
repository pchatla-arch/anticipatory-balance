# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Rebuild TensorBoard event files from the scalar CSVs that save_run.py committed
(results/<run>/tb_*.csv). The original event files stayed on the pods.

  python scripts/csv_to_tensorboard.py --out tensorboard
  tensorboard --logdir tensorboard
"""
import argparse
import csv
import glob
import os

from tensorboardX import SummaryWriter


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--results", default="results")
  ap.add_argument("--out", default="tensorboard")
  args = ap.parse_args()
  runs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(args.results, "*", "tb_*.csv"))})
  for run in runs:
    name = os.path.basename(run)
    w = SummaryWriter(os.path.join(args.out, name))
    n = 0
    for path in sorted(glob.glob(os.path.join(run, "tb_*.csv"))):
      base = os.path.basename(path)[3:-4]                 # eval_episode_reward_tracking_lin_vel
      section, _, rest = base.partition("_")              # eval | training
      tag = f"{section}/{rest}"
      with open(path) as f:
        rows = list(csv.reader(f))
      start = 1 if rows and not rows[0][0].replace(".", "", 1).isdigit() else 0
      for row in rows[start:]:
        if len(row) >= 2:
          try:
            w.add_scalar(tag, float(row[1]), int(float(row[0])))
            n += 1
          except ValueError:
            pass
    w.close()
    print(f"{name}: {n} points")


if __name__ == "__main__":
  main()
