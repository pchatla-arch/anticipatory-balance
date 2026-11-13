# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Report figures from the E2/E3 outputs.

  python scripts/plot_fall_surfaces.py --standing results/E2c_heavy_standing \
      --walking results/E2b_heavy --lip results/E3c_lip_standing --out results/figures

fall_surfaces.png : recorded failure rate over (speed, mass) on log2 axes, standing and
                    walking side by side, values and contact rates in the cells, iso-momentum
                    lines, and the zero-step LIP reference m*v = J_max on the standing panel.
lip_scatter.png   : per-contact incoming momentum vs the LIP budget J_max, coloured by outcome;
                    the diagonal is the prediction boundary.
"""
import csv
import json
import os

import numpy as np
from absl import app, flags
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

_ST = flags.DEFINE_string("standing", "results/E2c_heavy_standing", "E2 dir, standing")
_WK = flags.DEFINE_string("walking", "results/E2b_heavy", "E2 dir, walking")
_LIP = flags.DEFINE_string("lip", "results/E3c_lip_standing", "E3 dir (standing)")
_OUT = flags.DEFINE_string("out", "results/figures", "output dir")
_ORACLE = flags.DEFINE_string("oracle", "results/E6_oracle", "E6 dir (lead_curve.csv)")

DARK, MID, ACCENT, LIGHT = "#14213d", "#4a5568", "#d9772b", "#f7f6f1"


def grid(path):
  rows = list(csv.DictReader(open(os.path.join(path, "grid.csv"))))
  return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def panel(ax, g, title, norm, lip_j=None, show_cbar=False):
  masses, speeds = np.unique(g["mass"]), np.unique(g["speed"])
  # Cell edges at geometric midpoints (log2 axes).
  def edges(v):
    lv = np.log2(v); mid = (lv[1:] + lv[:-1]) / 2
    return 2 ** np.concatenate([[lv[0] - (mid[0] - lv[0])], mid, [lv[-1] + (lv[-1] - mid[-1])]])
  me, se = edges(masses), edges(speeds)
  Z = np.full((len(masses), len(speeds)), np.nan)
  C = np.full_like(Z, np.nan)
  for i, m in enumerate(masses):
    for j, s in enumerate(speeds):
      k = (g["mass"] == m) & (g["speed"] == s)
      Z[i, j] = g["fall_rate"][k][0]; C[i, j] = g["contact_rate"][k][0]
  pc = ax.pcolormesh(se, me, Z, cmap="magma", norm=norm, edgecolors=LIGHT, linewidth=1.5)
  for i, m in enumerate(masses):
    for j, s in enumerate(speeds):
      col = "w" if Z[i, j] < 0.55 else DARK
      ax.text(s, m * 1.12, f"{Z[i, j]*100:.0f}%", ha="center", va="center", color=col,
              fontsize=10, fontweight="bold")
      ax.text(s, m / 1.15, f"contact {C[i, j]*100:.0f}%", ha="center", va="center", color=col,
              fontsize=6.5, alpha=0.9)
  ax.set_xscale("log", base=2); ax.set_yscale("log", base=2)
  ax.set_xticks(speeds, [f"{v:g}" for v in speeds]); ax.set_yticks(masses, [f"{v:g}" for v in masses])
  ax.minorticks_off()
  ax.set_xlim(se[0], se[-1]); ax.set_ylim(me[0], me[-1])
  ax.set_xlabel("projectile speed (m/s)"); ax.set_title(title, fontsize=11, loc="left", color=DARK)
  # Iso-momentum lines m*v = const.
  v = np.linspace(se[0], se[-1], 200)
  for J in (16, 32, 64, 128):
    m = J / v
    ok = (m >= me[0]) & (m <= me[-1])
    if ok.any():
      ax.plot(v[ok], m[ok], color="#9fb3c8", lw=0.9, ls=":", zorder=3)
      vi = v[ok][-1] if m[ok][-1] <= me[0] * 1.3 else v[ok][0]
      ax.text(v[ok][0] * 1.02, m[ok][0] / 1.08, f"{J} N·s", color="#9fb3c8", fontsize=7,
              ha="left", va="top", rotation=-28)
  if lip_j is not None:
    m = lip_j / v
    ok = (m >= me[0]) & (m <= me[-1])
    ax.plot(v[ok], m[ok], color="#5ad0d6", lw=2.2, ls="--", zorder=4,
            label=f"zero-step LIP: m·v = J_max ≈ {lip_j:.0f} N·s")
    ax.legend(loc="upper right", fontsize=8, frameon=True, facecolor=LIGHT, edgecolor="none")
  return pc


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  gs, gw = grid(_ST.value), grid(_WK.value)
  lip = json.load(open(os.path.join(_LIP.value, "summary.json")))
  norm = Normalize(0, 1)
  fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True, constrained_layout=True)
  fig.patch.set_facecolor("white")
  panel(axes[0], gs, "Standing (zero command): recorded failure rate", norm, lip_j=lip["jmax_median_Ns"])
  pc = panel(axes[1], gw, "Walking at 0.5 m/s: recorded failure rate", norm)
  axes[0].set_ylabel("projectile mass (kg)")
  cb = fig.colorbar(pc, ax=axes, shrink=0.9, pad=0.02); cb.set_label("P(failure)")
  fig.suptitle("Frozen PPO walker under projectile impacts: 128 scheduled hits per cell; "
               "dotted lines are equal incoming momentum", fontsize=10, color=MID)
  fig.savefig(os.path.join(_OUT.value, "fall_surfaces.png"), dpi=200)

  # Per-episode LIP scatter.
  rows = list(csv.DictReader(open(os.path.join(_LIP.value, "episodes_lip.csv"))))
  jin = np.array([float(r["j_in"]) for r in rows]); jmax = np.array([float(r["j_max"]) for r in rows])
  fell = np.array([int(r["fell"]) for r in rows]) == 1
  fig, ax = plt.subplots(figsize=(5.6, 4.6), constrained_layout=True)
  rng = np.random.RandomState(0)
  jit = lambda x: x * np.exp(rng.normal(0, 0.02, len(x)))   # small log-jitter: cells overlap
  ax.scatter(jit(jmax[~fell]), jit(jin[~fell]), s=9, color="#2b6f8e", alpha=0.45, lw=0, label=f"stood (n={int((~fell).sum())})")
  ax.scatter(jit(jmax[fell]), jit(jin[fell]), s=9, color=ACCENT, alpha=0.45, lw=0, label=f"fell (n={int(fell.sum())})")
  lim = [4, 300]
  ax.plot(lim, lim, color=DARK, lw=1.2, ls="--", label="J_in = J_max (prediction boundary)")
  ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(4, 40); ax.set_ylim(lim)
  ax.set_xticks([4, 6, 8, 10, 15, 20, 30, 40], ["4", "6", "8", "10", "15", "20", "30", "40"])
  ax.set_yticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
  ax.minorticks_off()
  ax.set_xlabel("zero-step LIP budget J_max = m ω (x_edge − x_cp)  (N·s)")
  ax.set_ylabel("incoming momentum m·v  (N·s)")
  acc = lip["accuracy"]; n = lip["n"]
  ax.set_title(f"Standing contacts: LIP predicts failure above the line\n"
               f"agreement {acc*100:.1f}% of {n}; almost all errors are survivors above the line",
               fontsize=9.5, loc="left", color=DARK)
  ax.legend(fontsize=8, loc="lower right", frameon=True, facecolor=LIGHT, edgecolor="none")
  ax.grid(True, which="major", color="#e3e6ea", lw=0.6)
  fig.savefig(os.path.join(_OUT.value, "lip_scatter.png"), dpi=200)

  # Binned: P(failure) vs incoming momentum / LIP budget, with Wilson 95% intervals.
  ratio = jin / jmax
  edges = np.array([0.4, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0, 5.6, 8.0, 11.3, 16.0, 32.0])
  xs, ps, lo, hi, ns = [], [], [], [], []
  for a, b in zip(edges[:-1], edges[1:]):
    k = (ratio >= a) & (ratio < b)
    n_ = int(k.sum())
    if n_ < 10:
      continue
    p_ = fell[k].mean(); z = 1.96
    den = 1 + z * z / n_; c = (p_ + z * z / (2 * n_)) / den
    h = z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / den
    xs.append(np.sqrt(a * b)); ps.append(p_); lo.append(c - h); hi.append(c + h); ns.append(n_)
  xs, ps, lo, hi = map(np.array, (xs, ps, lo, hi))
  fig, ax = plt.subplots(figsize=(6.2, 4.2), constrained_layout=True)
  ax.fill_between(xs, lo, hi, color=ACCENT, alpha=0.18, lw=0, label="95% interval")
  ax.plot(xs, ps, "o-", color=ACCENT, lw=2, ms=5, label="recorded failure rate")
  ax.axvline(1.0, color=DARK, ls="--", lw=1.2)
  ax.text(1.05, 0.93, "zero-step LIP boundary\n(J_in = J_max)", fontsize=8, color=DARK, va="top")
  for x, n_, p_ in zip(xs, ns, ps):
    ax.text(x, min(p_ + 0.06, 0.98), f"n={n_}", ha="center", fontsize=6.5, color=MID)
  ax.set_xscale("log"); ax.set_xlim(0.5, 30); ax.set_ylim(0, 1.02)
  ax.set_xticks([0.5, 1, 2, 4, 8, 16], ["0.5", "1", "2", "4", "8", "16"]); ax.minorticks_off()
  ax.set_xlabel("incoming momentum / LIP budget   (m·v) / J_max")
  ax.set_ylabel("P(failure)")
  ax.set_title("Standing contacts: failure rises past the zero-step capture bound; 50% near 2x the budget", fontsize=10, loc="left", color=DARK)
  ax.grid(True, color="#e3e6ea", lw=0.6); ax.legend(fontsize=8, loc="center left", frameon=False)
  fig.savefig(os.path.join(_OUT.value, "lip_ratio_curve.png"), dpi=200)

  # Oracle lead-time curve (E6, BRACE v1).
  lc = os.path.join(_ORACLE.value, "lead_curve.csv")
  if os.path.exists(lc):
    rows = list(csv.DictReader(open(lc)))
    rows = [r for r in rows if "mode" not in r or (r["mode"] == "brace" and r["launch"] == "1")]
    L = np.array([float(r["lead_s"]) for r in rows]); fr = np.array([float(r["fall_rate"]) for r in rows])
    cr = np.array([float(r["contact_rate"]) for r in rows]); pm = np.array([float(r["progress_m"]) for r in rows])
    mid = (L >= 0) & (L < 5)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    a1.plot(L[mid], fr[mid], "o-", color=ACCENT, lw=2, label="switch to BRACE at lead")
    a1.axhline(fr[L == -1][0], color=DARK, ls="--", lw=1.2, label=f"CONTINUE ({fr[L == -1][0]*100:.1f}%)")
    if (L >= 5).any():
      a1.axhline(fr[L >= 5][0], color=MID, ls=":", lw=1.2, label=f"BRACE from start ({fr[L >= 5][0]*100:.0f}%)")
    a1.set_ylim(0, 1); a1.set_xlabel("switch lead before nominal arrival (s)"); a1.set_ylabel("recorded failure rate")
    a1.set_title("Switching to BRACE v1 raises failure with lead time", fontsize=10, loc="left", color=DARK)
    a1.legend(fontsize=8, frameon=False, loc="lower right"); a1.grid(True, color="#e3e6ea", lw=0.6)
    a2.plot(L[mid], cr[mid], "s-", color="#2b6f8e", lw=2, label="detected contact rate")
    a2.plot(L[mid], pm[mid] / pm[L == -1][0], "^-", color="#6a7179", lw=2, label="progress relative to CONTINUE")
    a2.set_ylim(0, 1.05); a2.set_xlabel("switch lead before nominal arrival (s)")
    a2.set_title("Contact coverage stays near 30%; progress falls", fontsize=10, loc="left", color=DARK)
    a2.legend(fontsize=8, frameon=False, loc="center right"); a2.grid(True, color="#e3e6ea", lw=0.6)
    fig.savefig(os.path.join(_OUT.value, "oracle_v1_lead_curve.png"), dpi=200)
  print("saved", _OUT.value)


if __name__ == "__main__":
  app.run(main)
