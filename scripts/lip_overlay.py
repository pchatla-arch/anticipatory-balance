# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E3: LIP / capture-point prediction vs the empirical fall-rate surface from E2.

For every contact episode in E2's episodes.csv, take the pre-impact state (CoM height and
velocity, foot positions) and the projectile's horizontal direction, compute
    omega = sqrt(g / z_com),  x_cp = x + xdot / omega,  J_max = m_robot * omega * margin
and compare the incoming momentum J_in = m_proj * |v_xy| (times (1 + e) if --restitution
is given) with J_max. Reports the confusion matrix against the measured outcome, the
per-cell predicted fall fraction, and draws the analytic boundary over the heatmap.

  python scripts/lip_overlay.py --episodes results/E2_fall_rate/episodes.csv \
      --out results/E3_lip_overlay
"""
import csv
import json
import os

import numpy as np
from absl import app, flags

from foresight.lip import capture_point as lip

_EP = flags.DEFINE_string("episodes", None, "E2 episodes.csv", required=True)
_OUT = flags.DEFINE_string("out", "results/E3_lip_overlay", "output dir")
_MROBOT = flags.DEFINE_float("robot_mass", 33.34, "G1 total mass, kg (Playground model: 33.34)")
_E = flags.DEFINE_float("restitution", 0.0, "coefficient of restitution for J_in = m v (1+e)")


def load(path):
  with open(path) as f:
    rows = list(csv.DictReader(f))
  cols = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
  return cols


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  d = load(_EP.value)
  sel = (d["contact"] > 0) & (d["is_miss"] == 0)
  idx = np.where(sel)[0]
  pred, jmax, jin = np.zeros(len(d["mass"]), bool), np.full(len(d["mass"]), np.nan), np.full(len(d["mass"]), np.nan)
  for i in idx:
    z = max(float(d["com_z_pre"][i]), 0.3)
    poly = lip.SupportPolygon.from_feet([d["lfoot_x_pre"][i], d["lfoot_y_pre"][i]],
                                        [d["rfoot_x_pre"][i], d["rfoot_y_pre"][i]])
    x = np.array([d["com_x_pre"][i], d["com_y_pre"][i]])
    xd = np.array([d["com_vx_pre"][i], d["com_vy_pre"][i]])
    pv = np.array([d["proj_vx_pre"][i], d["proj_vy_pre"][i]])
    if np.linalg.norm(pv) < 1e-6:
      continue
    push_dir = pv / np.linalg.norm(pv)
    cp = lip.capture_point(x, xd, z)
    jmax[i] = lip.j_max(_MROBOT.value, z, poly, cp, push_dir)
    jin[i] = d["mass"][i] * np.linalg.norm(pv) * (1.0 + _E.value)
    pred[i] = jin[i] > jmax[i]
  ok = sel & ~np.isnan(jmax)
  fell = d["fell"][ok] > 0
  p = pred[ok]
  cm = {"n": int(ok.sum()),
        "pred_fall_and_fell": int((p & fell).sum()), "pred_fall_not_fell": int((p & ~fell).sum()),
        "pred_stand_but_fell": int((~p & fell).sum()), "pred_stand_and_stood": int((~p & ~fell).sum())}
  cm["accuracy"] = (cm["pred_fall_and_fell"] + cm["pred_stand_and_stood"]) / max(cm["n"], 1)
  cm["jmax_median_Ns"] = float(np.nanmedian(jmax[ok]))
  cm["jmax_iqr_Ns"] = [float(np.nanpercentile(jmax[ok], 25)), float(np.nanpercentile(jmax[ok], 75))]
  cm["z_com_pre_median_m"] = float(np.median(d["com_z_pre"][ok]))
  print(json.dumps(cm, indent=2))

  masses = sorted(set(d["mass"][sel])); speeds = sorted(set(d["speed"][sel]))
  grid = []
  for m in masses:
    for s in speeds:
      c = ok & (d["mass"] == m) & (d["speed"] == s)
      grid.append({"mass": m, "speed": s, "n": int(c.sum()),
                   "empirical_fall": float((d["fell"][c] > 0).mean()) if c.any() else np.nan,
                   "lip_pred_fall": float(pred[c].mean()) if c.any() else np.nan,
                   "jmax_median": float(np.nanmedian(jmax[c])) if c.any() else np.nan})
  with open(os.path.join(_OUT.value, "lip_grid.csv"), "w") as f:
    keys = list(grid[0].keys()); f.write(",".join(keys) + "\n")
    for g in grid:
      f.write(",".join(f"{g[k]:.4g}" for k in keys) + "\n")
  json.dump(cm, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)

  # Per-episode predictions for later belief/decision work.
  with open(os.path.join(_OUT.value, "episodes_lip.csv"), "w") as f:
    f.write("mass,speed,contact,fell,j_in,j_max,lip_pred_fall\n")
    for i in np.where(ok)[0]:
      f.write(f"{d['mass'][i]:g},{d['speed'][i]:g},{int(d['contact'][i])},{int(d['fell'][i])},"
              f"{jin[i]:.4g},{jmax[i]:.4g},{int(pred[i])}\n")

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    emp = np.array([[next(g["empirical_fall"] for g in grid if g["mass"] == m and g["speed"] == s)
                     for s in speeds] for m in masses])
    fig, ax = plt.subplots(figsize=(5.4, 4))
    im = ax.imshow(emp, origin="lower", vmin=0, vmax=1, cmap="magma", aspect="auto",
                   extent=[-0.5, len(speeds) - 0.5, -0.5, len(masses) - 0.5])
    ax.set_xticks(range(len(speeds)), [f"{s:g}" for s in speeds])
    ax.set_yticks(range(len(masses)), [f"{m:g}" for m in masses])
    # Analytic boundary m*v*(1+e) = J_max (median): plotted in index coordinates.
    J = cm["jmax_median_Ns"]
    sx = np.linspace(min(speeds), max(speeds), 100)
    my = J / (sx * (1.0 + _E.value))
    def to_idx(vals, grid_vals):
      return np.interp(vals, grid_vals, np.arange(len(grid_vals)))
    ax.plot(to_idx(sx, speeds), to_idx(my, masses), "c--", lw=2,
            label=f"LIP: m·v = J_max ≈ {J:.0f} N·s")
    ax.set_ylim(-0.5, len(masses) - 0.5)
    ax.set_xlabel("projectile speed (m/s)"); ax.set_ylabel("projectile mass (kg)")
    ax.set_title(f"Empirical P(fall) with LIP boundary (acc {cm['accuracy']:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    fig.colorbar(im, ax=ax, label="P(fall)")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT.value, "lip_overlay.png"), dpi=160)
  except Exception as e:
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
