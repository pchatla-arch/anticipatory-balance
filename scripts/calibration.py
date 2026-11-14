# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E9: belief calibration from the per-step logs dumped by eval_decision_frontier.py
(--calib_params), for every detection period x latency found under the given result dirs.

  Reliability: at each decision step after the first detection and before contact (or before
  the nominal arrival on near-misses), the belief's hit probability p_hit against whether the
  object eventually made contact. A calibrated belief has observed frequency ~ p_hit per bin.
  Timing: E_b[t_hit] - true time to contact at those steps, as a function of the true time to
  contact, per latency; plus the time before contact at which p_hit first exceeds 0.5.

  python scripts/calibration.py --dirs results/E10_frontier,results/E11_rate_p2,results/E11_rate_p1 \
      --out results/E9_calibration
"""
import glob
import json
import os

import numpy as np
from absl import app, flags

_DIRS = flags.DEFINE_list("dirs", ["results/E10_frontier"], "frontier result dirs with calib/ dumps")
_OUT = flags.DEFINE_string("out", "results/E9_calibration", "output dir")
_BINS = flags.DEFINE_integer("bins", 10, "probability bins")


def wilson(k, n, z=1.96):
  if n == 0:
    return np.nan, np.nan
  p = k / n
  d = 1 + z * z / n
  c = (p + z * z / (2 * n)) / d
  h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return c - h, c + h


def analyse(path):
  d = np.load(path)
  p = d["p_hit"].astype(np.float32)          # (T, N)
  th = d["t_hit"].astype(np.float32)
  nobs = d["n_obs"]
  T, N = p.shape
  dt = float(d["dt"]); period = int(d["period"]); lat = int(d["latency"])
  cs = d["contact_step"]; miss = d["is_miss"]; contact = d["contact"]
  # Outcome and the last usable step per episode.
  nominal_step = np.round(d["t_hit_nominal"] / dt).astype(int)
  end = np.where(cs >= 0, cs, nominal_step)
  t = np.arange(T)[:, None]
  usable = (nobs > 0) & (t < end[None, :]) & (t >= d["launch_step"][None, :])
  outcome = np.broadcast_to(contact[None, :], (T, N))
  # Reliability over usable decision steps (every `period` steps, as the rule sees them).
  usable &= (t % period == 0)
  pp = p[usable]; oo = outcome[usable]
  edges = np.linspace(0, 1, _BINS.value + 1)
  rel = []
  for i in range(_BINS.value):
    m = (pp >= edges[i]) & (pp < edges[i + 1] + (1e-9 if i == _BINS.value - 1 else 0))
    n = int(m.sum()); k = int(oo[m].sum())
    lo, hi = wilson(k, n)
    rel.append({"bin_lo": float(edges[i]), "bin_hi": float(edges[i + 1]), "n": n,
                "p_mean": float(pp[m].mean()) if n else np.nan, "freq": k / n if n else np.nan,
                "lo": lo, "hi": hi})
  ece = float(sum(r["n"] * abs(r["freq"] - r["p_mean"]) for r in rel if r["n"]) / max(len(pp), 1))
  brier = float(np.mean((pp - oo) ** 2)) if len(pp) else np.nan
  # Timing on hits with contact: belief t_hit vs true time to contact, binned by truth.
  hit_ep = (~miss) & (cs >= 0)
  tt = (cs[None, :] - t) * dt                       # true time to contact, s
  m2 = usable & hit_ep[None, :] & np.isfinite(th) & (p > 0.2)
  err = (th - tt)[m2]; truth = tt[m2]
  tb = np.arange(0, 1.61, 0.2)
  timing = []
  for i in range(len(tb) - 1):
    m = (truth >= tb[i]) & (truth < tb[i + 1])
    timing.append({"ttc_lo": float(tb[i]), "ttc_hi": float(tb[i + 1]), "n": int(m.sum()),
                   "err_mean": float(err[m].mean()) if m.any() else np.nan,
                   "err_std": float(err[m].std()) if m.any() else np.nan})
  # When does the belief resolve? First step with p_hit > 0.5 on contact episodes, as time before contact.
  resolve = []
  for j in np.where(hit_ep)[0]:
    k = np.where((p[:, j] > 0.5) & (np.arange(T) < cs[j]))[0]
    resolve.append((cs[j] - k[0]) * dt if len(k) else np.nan)
  resolve = np.array(resolve)
  # Same for near-misses: fraction whose p_hit ever exceeds 0.5 (would fool a 0.5 threshold).
  miss_ep = miss & (cs < 0)
  fooled = float(np.mean(p[:, miss_ep].max(axis=0) > 0.5)) if miss_ep.any() else np.nan
  return {"file": os.path.basename(path), "period": period, "latency_steps": lat, "rate_hz": 1 / (period * dt),
          "latency_ms": lat * dt * 1000, "n_episodes": int(N), "n_steps_used": int(len(pp)),
          "ece": ece, "brier": brier, "reliability": rel, "timing": timing,
          "resolve_s_before_contact_median": float(np.nanmedian(resolve)),
          "resolve_s_before_contact_p25": float(np.nanpercentile(resolve, 25)),
          "resolve_s_before_contact_p75": float(np.nanpercentile(resolve, 75)),
          "never_resolved_frac": float(np.mean(np.isnan(resolve))),
          "near_miss_p_over_half_frac": fooled}


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  files = sorted(f for d in _DIRS.value for f in glob.glob(os.path.join(d, "calib", "*.npz")))
  if not files:
    raise SystemExit("no calib dumps found under " + ",".join(_DIRS.value))
  res = [analyse(f) for f in files]
  # Merge batches of the same (period, latency): keep the first per key for plotting; all in json.
  json.dump({"results": res}, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  with open(os.path.join(_OUT.value, "calibration.csv"), "w") as f:
    f.write("period,rate_hz,latency_ms,n_episodes,n_steps,ece,brier,resolve_median_s,resolve_p25_s,resolve_p75_s,never_resolved,near_miss_over_half\n")
    for r in res:
      f.write(f"{r['period']},{r['rate_hz']:.3g},{r['latency_ms']:.0f},{r['n_episodes']},{r['n_steps_used']},"
              f"{r['ece']:.4f},{r['brier']:.4f},{r['resolve_s_before_contact_median']:.3f},"
              f"{r['resolve_s_before_contact_p25']:.3f},{r['resolve_s_before_contact_p75']:.3f},"
              f"{r['never_resolved_frac']:.3f},{r['near_miss_p_over_half_frac']:.3f}\n")
    for r in res:
      print(f"{r['rate_hz']:5.1f} Hz  lat {r['latency_ms']:4.0f} ms  ECE {r['ece']:.3f}  Brier {r['brier']:.3f}  "
            f"resolves {r['resolve_s_before_contact_median']:.2f} s before contact "
            f"(IQR {r['resolve_s_before_contact_p25']:.2f}-{r['resolve_s_before_contact_p75']:.2f}), "
            f"never {r['never_resolved_frac']:.2f}; near-miss p>0.5: {r['near_miss_p_over_half_frac']:.2f}")
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keyed = {}
    for r in res:
      keyed.setdefault((r["period"], r["latency_steps"]), r)
    periods = sorted({k[0] for k in keyed}, reverse=True); lats = sorted({k[1] for k in keyed})
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(lats), 2)))
    fig, axes = plt.subplots(2, len(periods), figsize=(4.6 * len(periods), 8), constrained_layout=True, squeeze=False)
    for j, per in enumerate(periods):
      ax = axes[0, j]; ax.plot([0, 1], [0, 1], "--", color="#9aa0a6", lw=0.8)
      for i, lat in enumerate(lats):
        r = keyed.get((per, lat))
        if not r:
          continue
        pts = [b for b in r["reliability"] if b["n"] >= 20]
        ax.errorbar([b["p_mean"] for b in pts], [b["freq"] for b in pts],
                    yerr=[[b["freq"] - b["lo"] for b in pts], [b["hi"] - b["freq"] for b in pts]],
                    fmt="o-", ms=4, color=cols[i], label=f"latency {r['latency_ms']:.0f} ms (ECE {r['ece']:.2f})")
      ax.set_title(f"detections at {r['rate_hz']:.0f} Hz", loc="left", fontsize=10)
      ax.set_xlabel("belief p_hit"); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.grid(True, color="#e3e6ea", lw=0.6)
      ax.legend(fontsize=8, frameon=False)
      ax2 = axes[1, j]; ax2.axhline(0, color="#9aa0a6", lw=0.8)
      for i, lat in enumerate(lats):
        r = keyed.get((per, lat))
        if not r:
          continue
        tm = [b for b in r["timing"] if b["n"] >= 20]
        x = [(b["ttc_lo"] + b["ttc_hi"]) / 2 for b in tm]
        ax2.errorbar(x, [b["err_mean"] for b in tm], yerr=[b["err_std"] for b in tm], fmt="o-", ms=4, color=cols[i],
                     label=f"latency {r['latency_ms']:.0f} ms")
      ax2.set_xlabel("true time to contact (s)"); ax2.set_ylabel("E_b[t_hit] - truth (s)"); ax2.grid(True, color="#e3e6ea", lw=0.6)
    axes[0, 0].set_ylabel("observed contact frequency")
    fig.suptitle("E9 belief calibration: reliability (top) and time-to-hit error (bottom)", fontsize=11)
    fig.savefig(os.path.join(_OUT.value, "calibration.png"), dpi=170)
  except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
