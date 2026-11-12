# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E2: reactive fall-rate surface of the frozen nominal controller over (mass, speed).

One batch covers the whole grid (cells x episodes_per_cell envs) plus a near-miss control
block (same count as one cell) that measures the walking policy's base fall rate with the
projectile passing by. Several batches (seeds) are concatenated.

  python scripts/eval_fall_rate.py --checkpoint logs/nominal_s1/<run>/checkpoints \
      --impl warp --out results/E2_fall_rate

Outputs in --out: episodes.csv (one row per episode), grid.csv (per-cell rates),
fall_rate_heatmap.png, summary.json.
"""
import json
import os
import time

import jax
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R

_CKPT = flags.DEFINE_string("checkpoint", None, "nominal skill checkpoint dir", required=True)
_IMPL = flags.DEFINE_string("impl", "warp", "mjx impl: warp | jax")
_MASSES = flags.DEFINE_list("masses", ["1", "2", "4", "8"], "projectile masses, kg")
_SPEEDS = flags.DEFINE_list("speeds", ["1", "2", "3", "4"], "projectile speeds, m/s")
_EPC = flags.DEFINE_integer("episodes_per_cell", 128, "episodes per (mass, speed) cell per batch")
_BATCHES = flags.DEFINE_integer("batches", 2, "independent batches (seeds)")
_STEPS = flags.DEFINE_integer("steps", 300, "control steps per episode")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward velocity command, m/s")
_OUT = flags.DEFINE_string("out", "results/E2_fall_rate", "output dir")
_SEED = flags.DEFINE_integer("seed", 0, "base seed")


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  masses = [float(m) for m in _MASSES.value]
  speeds = [float(s) for s in _SPEEDS.value]
  cells = [(m, s) for m in masses for s in speeds]
  epc = _EPC.value
  n = len(cells) * epc + epc            # + near-miss control block
  mass_arr = np.array([m for m, _ in cells for _ in range(epc)] + [masses[len(masses) // 2]] * epc)
  speed_arr = np.array([s for _, s in cells for _ in range(epc)] + [speeds[len(speeds) // 2]] * epc)
  miss_arr = np.array([False] * (len(cells) * epc) + [True] * epc)

  policy = R.load_policy(_CKPT.value)
  env = R.make_env(skill="nominal", impl=_IMPL.value)
  wenv = R.batched_env(env, mass_arr)
  scenario = lambda s: R.set_scenario(env, s, speed=speed_arr, is_miss=miss_arr,
                                      command=[_VX.value, 0.0, 0.0])
  print(f"grid {len(masses)}x{len(speeds)} cells x {epc} eps + {epc} miss-control = {n} envs/batch")

  rows = []
  for b in range(_BATCHES.value):
    t0 = time.time()
    info, traj = R.rollout(env, wenv, policy, jax.random.PRNGKey(_SEED.value + b),
                           _STEPS.value, scenario_fn=scenario)
    tab = R.episode_table(info, traj, env.dt)
    tab["batch"] = np.full(n, b)
    rows.append(tab)
    print(f"batch {b}: {time.time()-t0:.0f}s  launched {tab['launched'].mean():.3f}  "
          f"contact(hits) {tab['contact'][~miss_arr].mean():.3f}  "
          f"fell(hits) {tab['fell'][~miss_arr].mean():.3f}  "
          f"fell(miss-control) {tab['fell'][miss_arr].mean():.3f}  "
          f"fell_before_launch {tab['fell_before_launch'].mean():.3f}")
  tab = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
  R.write_csv(os.path.join(_OUT.value, "episodes.csv"), tab)

  # Per-cell aggregates. Primary population = all scheduled hits; secondary = contacts only.
  grid = []
  for m, s in cells:
    sel = (tab["mass"] == m) & (tab["speed"] == s) & (~tab["is_miss"])
    c = sel & tab["contact"]
    grid.append({
        "mass": m, "speed": s, "momentum": m * s, "n": int(sel.sum()),
        "contact_rate": float(tab["contact"][sel].mean()),
        "fall_rate": float(tab["fell"][sel].mean()),
        "fall_rate_given_contact": float(tab["fell"][c].mean()) if c.any() else float("nan"),
        "progress_m": float(tab["progress_m"][sel].mean()),
    })
  miss = tab["is_miss"]
  control = {"n": int(miss.sum()), "fall_rate": float(tab["fell"][miss].mean()),
             "contact_rate": float(tab["contact"][miss].mean()),
             "progress_m": float(tab["progress_m"][miss].mean())}
  with open(os.path.join(_OUT.value, "grid.csv"), "w") as f:
    keys = list(grid[0].keys())
    f.write(",".join(keys) + "\n")
    for g in grid:
      f.write(",".join(f"{g[k]:.4g}" if isinstance(g[k], float) else str(g[k]) for k in keys) + "\n")
  summary = {"checkpoint": str(R.latest_checkpoint(_CKPT.value)), "masses": masses,
             "speeds": speeds, "episodes_per_cell": epc * _BATCHES.value,
             "command_vx": _VX.value, "near_miss_control": control,
             "overall_fall_rate_hits": float(tab["fell"][~miss].mean()),
             "fell_before_launch_rate": float(tab["fell_before_launch"].mean())}
  json.dump(summary, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  print(json.dumps(summary, indent=2))

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fr = np.array([[next(g["fall_rate"] for g in grid if g["mass"] == m and g["speed"] == s)
                    for s in speeds] for m in masses])
    fig, ax = plt.subplots(figsize=(5.2, 4))
    im = ax.imshow(fr, origin="lower", vmin=0, vmax=1, cmap="magma", aspect="auto")
    ax.set_xticks(range(len(speeds)), [f"{s:g}" for s in speeds])
    ax.set_yticks(range(len(masses)), [f"{m:g}" for m in masses])
    ax.set_xlabel("projectile speed (m/s)"); ax.set_ylabel("projectile mass (kg)")
    ax.set_title(f"Reactive controller fall rate (control: {control['fall_rate']:.2f})")
    for i in range(len(masses)):
      for j in range(len(speeds)):
        ax.text(j, i, f"{fr[i, j]:.2f}", ha="center", va="center",
                color="w" if fr[i, j] < 0.6 else "k", fontsize=9)
    fig.colorbar(im, ax=ax, label="P(fall)")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT.value, "fall_rate_heatmap.png"), dpi=160)
  except Exception as e:  # plotting is optional on the pod
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
