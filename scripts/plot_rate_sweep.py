# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E11: detection rate x latency sweep, assembled from frontier runs at different detector periods
(E10 at period 5 = 10 Hz, plus E11 runs at periods 2 and 1 = 20 and 50 Hz).

Outputs, in --out:
  frontier_grid.png   rows = detection rate, columns = latency, each the decision frontier
  rate_curves.png     best failure rate per rule against detection rate, one line per latency;
                      and the false-preparation rate each rule needs for that best failure rate
  sweep.csv / summary.json   the numbers behind both, plus the crossover (the lowest rate at which
                      the belief rule reaches a lower failure rate than the threshold rule)

  python scripts/plot_rate_sweep.py --dirs results/E10_frontier,results/E11_rate_p2,results/E11_rate_p1 \
      --out results/E11_rate_sweep
"""
import json
import os

import numpy as np
from absl import app, flags

_DIRS = flags.DEFINE_list("dirs", ["results/E10_frontier", "results/E11_rate_p2", "results/E11_rate_p1"], "frontier dirs")
_OUT = flags.DEFINE_string("out", "results/E11_rate_sweep", "output dir")
_DT = flags.DEFINE_float("dt", 0.02, "control step, s")
_FP_MAX = flags.DEFINE_float("fp_max", 0.5, "false-preparation budget for the 'best under budget' column")


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  runs = []
  for d in _DIRS.value:
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
      print("missing", p); continue
    s = json.load(open(p)); s["dir"] = d; runs.append(s)
  if not runs:
    raise SystemExit("no frontier summaries found")
  dt = _DT.value
  runs.sort(key=lambda s: -s["period"])
  periods = [s["period"] for s in runs]
  lats = sorted({r["latency_steps"] for s in runs for r in s["rows"] if r["method"] in ("ttc_threshold", "belief_commit")})
  react = next((r for s in runs for r in s["rows"] if r["method"] == "reactive"), None)
  oracle = next((r for s in runs for r in s["rows"] if r["method"] == "oracle"), None)

  table = []
  for s in runs:
    for L in lats:
      row = {"period": s["period"], "rate_hz": 1 / (s["period"] * dt), "latency_steps": L, "latency_ms": L * dt * 1000}
      for m in ("ttc_threshold", "belief_commit"):
        pts = [r for r in s["rows"] if r["method"] == m and r["latency_steps"] == L]
        if not pts:
          continue
        best = min(pts, key=lambda r: r["fall_rate_hits"])
        under = [r for r in pts if r["false_prep_rate"] <= _FP_MAX.value]
        bu = min(under, key=lambda r: r["fall_rate_hits"]) if under else None
        row[f"{m}_best_fall"] = best["fall_rate_hits"]; row[f"{m}_best_fp"] = best["false_prep_rate"]
        row[f"{m}_best_param"] = best["param"]
        row[f"{m}_fall_under_budget"] = bu["fall_rate_hits"] if bu else np.nan
        row[f"{m}_fp_under_budget"] = bu["false_prep_rate"] if bu else np.nan
        # Area-type summary: mean failure rate over the rule's own frontier points, weighted by nothing.
        row[f"{m}_min_fp_at_fall_le_0.6"] = min([r["false_prep_rate"] for r in pts if r["fall_rate_hits"] <= 0.6], default=np.nan)
      if "ttc_threshold_fall_under_budget" in row and "belief_commit_fall_under_budget" in row:
        row["belief_minus_ttc_under_budget"] = row["belief_commit_fall_under_budget"] - row["ttc_threshold_fall_under_budget"]
      table.append(row)
  # Crossover per latency: lowest rate (largest period first -> walk ascending rate) where belief < ttc under budget.
  cross = {}
  for L in lats:
    rows = sorted([r for r in table if r["latency_steps"] == L], key=lambda r: r["rate_hz"])
    hit = next((r for r in rows if r.get("belief_minus_ttc_under_budget", np.nan) < 0), None)
    cross[str(int(L * dt * 1000))] = hit["rate_hz"] if hit else None
  cols = sorted({k for r in table for k in r}, key=lambda k: (k not in ("period", "rate_hz", "latency_steps", "latency_ms"), k))
  with open(os.path.join(_OUT.value, "sweep.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for r in table:
      f.write(",".join(f"{r.get(c, np.nan):.4g}" if isinstance(r.get(c, np.nan), float) else str(r.get(c, "")) for c in cols) + "\n")
  json.dump({"dirs": _DIRS.value, "fp_budget": _FP_MAX.value, "reactive": react, "oracle": oracle,
             "crossover_rate_hz_by_latency_ms": cross, "rows": table}, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  for r in table:
    print(f"{r['rate_hz']:5.1f} Hz lat {r['latency_ms']:4.0f} ms | threshold best {r.get('ttc_threshold_best_fall', np.nan):.3f} "
          f"@fp {r.get('ttc_threshold_best_fp', np.nan):.2f} | belief best {r.get('belief_commit_best_fall', np.nan):.3f} "
          f"@fp {r.get('belief_commit_best_fp', np.nan):.2f} | under fp<={_FP_MAX.value}: thr {r.get('ttc_threshold_fall_under_budget', np.nan):.3f} "
          f"belief {r.get('belief_commit_fall_under_budget', np.nan):.3f}")
  print("crossover rate (Hz) by latency (ms):", cross)

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"ttc_threshold": "#2b6f8e", "belief_commit": "#d9772b", "always": "#6a7179", "reactive": "#14213d", "oracle": "#3a9d5d"}
    fig, axes = plt.subplots(len(runs), len(lats), figsize=(4.4 * len(lats), 3.6 * len(runs)), constrained_layout=True,
                             sharex=True, sharey=True, squeeze=False)
    for i, s in enumerate(runs):
      for j, L in enumerate(lats):
        ax = axes[i, j]
        for m in ("ttc_threshold", "belief_commit"):
          pts = sorted([r for r in s["rows"] if r["method"] == m and r["latency_steps"] == L], key=lambda r: r["false_prep_rate"])
          if pts:
            ax.plot([r["false_prep_rate"] for r in pts], [r["fall_rate_hits"] for r in pts], "o-", ms=4, color=col[m],
                    label={"ttc_threshold": "time-to-contact threshold", "belief_commit": "belief commit rule"}[m])
        if react:
          ax.plot([react["false_prep_rate"]], [react["fall_rate_hits"]], "*", ms=12, color=col["reactive"], label="reactive")
        if oracle:
          ax.plot([oracle["false_prep_rate"]], [oracle["fall_rate_hits"]], "*", ms=12, color=col["oracle"], label="oracle")
        al = [r for r in s["rows"] if r["method"] == "always" and r["latency_steps"] == L]
        if al:
          ax.plot([al[0]["false_prep_rate"]], [al[0]["fall_rate_hits"]], "s", ms=7, color=col["always"], label="always")
        ax.set_title(f"{1/(s['period']*dt):.0f} Hz, latency {L*dt*1000:.0f} ms", loc="left", fontsize=9)
        ax.grid(True, color="#e3e6ea", lw=0.6); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
        if i == len(runs) - 1:
          ax.set_xlabel("false preparation rate")
        if j == 0:
          ax.set_ylabel("failure rate on hits")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("E11 decision frontier by detection rate (rows) and latency (columns)", fontsize=11)
    fig.savefig(os.path.join(_OUT.value, "frontier_grid.png"), dpi=170)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), constrained_layout=True)
    styles = ["-", "--", ":"]
    for k, L in enumerate(lats):
      rows = sorted([r for r in table if r["latency_steps"] == L], key=lambda r: r["rate_hz"])
      x = [r["rate_hz"] for r in rows]
      for m in ("ttc_threshold", "belief_commit"):
        axes[0].plot(x, [r.get(f"{m}_fall_under_budget", np.nan) for r in rows], "o" + styles[k % 3], color=col[m], ms=4,
                     label=f"{'threshold' if m == 'ttc_threshold' else 'belief'}, {L*dt*1000:.0f} ms")
        axes[1].plot(x, [r.get(f"{m}_fp_under_budget", np.nan) for r in rows], "o" + styles[k % 3], color=col[m], ms=4)
    if react:
      axes[0].axhline(react["fall_rate_hits"], color=col["reactive"], lw=0.8, label="reactive")
    if oracle:
      axes[0].axhline(oracle["fall_rate_hits"], color=col["oracle"], lw=0.8, label="oracle")
    axes[0].set_xscale("log"); axes[1].set_xscale("log")
    axes[0].set_xlabel("detection rate (Hz)"); axes[0].set_ylabel(f"best failure rate with false-prep <= {_FP_MAX.value:.0%}")
    axes[1].set_xlabel("detection rate (Hz)"); axes[1].set_ylabel("false-preparation rate at that point")
    for ax in axes:
      ax.grid(True, color="#e3e6ea", lw=0.6, which="both")
      ax.set_xticks([1/(p*dt) for p in periods]); ax.set_xticklabels([f"{1/(p*dt):.0f}" for p in periods])
    axes[0].legend(fontsize=7, frameon=False, ncol=2)
    fig.suptitle("E11: what sensing rate the belief rule needs", fontsize=11)
    fig.savefig(os.path.join(_OUT.value, "rate_curves.png"), dpi=170)
  except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
