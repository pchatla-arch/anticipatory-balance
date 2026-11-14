# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E12: where does the oracle gap go? A ladder of frontier runs that remove one limitation
at a time, read off at a fixed false-preparation budget:

  reactive  ->  rule @ 10 Hz (E10)  ->  rule @ 50 Hz (E11)  ->  rule with perfect sensing
  (E12: 50 Hz, zero latency, zero noise, no misses, full FOV)  ->  oracle (privileged timing)
  plus, when present, the object-aware PPO walker (E14) and the E13 ablations.

Each rung's best failure rate under the budget is the number; the differences are the
shares of the gap attributable to detection rate, latency/noise, and the rule+skill itself.

  python scripts/oracle_gap.py --out results/E12_oracle_gap
"""
import glob
import json
import os

import numpy as np
from absl import app, flags

_RUNGS = flags.DEFINE_list("rungs", ["10 Hz:results/E10_frontier", "25 Hz:results/E11_rate_p2", "50 Hz:results/E11_rate_p1",
                                     "perfect sensing:results/E12_perfect"], "label:dir of frontier runs, in ladder order")
_AWARE = flags.DEFINE_list("aware", ["object-aware PPO:results/E14_aware_ppo"], "label:dir of E14 runs (reactive rows)")
_ABL = flags.DEFINE_string("ablations_glob", "results/E13_*", "E13 frontier dirs")
_FP = flags.DEFINE_float("fp_max", 0.5, "false-preparation budget")
_OUT = flags.DEFINE_string("out", "results/E12_oracle_gap", "output dir")


def best_under(rows, method, fp, lat=None):
  pts = [r for r in rows if r["method"] == method and r["false_prep_rate"] <= fp and (lat is None or r["latency_steps"] == lat)]
  return min(pts, key=lambda r: r["fall_rate_hits"]) if pts else None


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  ladder = []
  react = oracle = None
  for item in _RUNGS.value:
    label, d = item.split(":", 1)
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
      print("missing", p); continue
    s = json.load(open(p)); rows = s["rows"]
    react = react or next((r for r in rows if r["method"] == "reactive"), None)
    oracle = oracle or next((r for r in rows if r["method"] == "oracle"), None)
    lat0 = 0
    for m in ("ttc_threshold", "belief_commit"):
      b = best_under(rows, m, _FP.value, lat0)
      if b:
        ladder.append({"rung": label, "method": m, "fall_rate_hits": b["fall_rate_hits"], "false_prep_rate": b["false_prep_rate"],
                       "param": b["param"], "progress_m": b["progress_m"], "latency_steps": lat0, "dir": d})
  for item in _AWARE.value:
    label, d = item.split(":", 1)
    p = os.path.join(d, "summary.json")
    if os.path.exists(p):
      for r in json.load(open(p))["rows"]:
        if r["method"] == "reactive":
          ladder.append({"rung": label, "method": "aware_ppo", "fall_rate_hits": r["fall_rate_hits"], "false_prep_rate": np.nan,
                         "param": 0, "progress_m": r["progress_m"], "latency_steps": r["latency_steps"], "dir": d,
                         "contact_rate_hits": r["contact_rate_hits"]})
  abl = []
  for d in sorted(glob.glob(_ABL.value)):
    p = os.path.join(d, "summary.json")
    if not os.path.exists(p):
      continue
    s = json.load(open(p))
    for m in ("ttc_threshold", "belief_commit"):
      b = best_under(s["rows"], m, _FP.value, 0)
      if b:
        abl.append({"ablation": os.path.basename(d), "method": m, "fall_rate_hits": b["fall_rate_hits"],
                    "false_prep_rate": b["false_prep_rate"], "param": b["param"], "rule": s.get("rule"), "detector": s.get("detector"),
                    "period": s.get("period")})
  out = {"fp_budget": _FP.value, "reactive": react, "oracle": oracle, "ladder": ladder, "ablations": abl}
  if react and oracle:
    gap = react["fall_rate_hits"] - oracle["fall_rate_hits"]
    out["gap_total"] = gap
    for m in ("ttc_threshold", "belief_commit"):
      steps = [react["fall_rate_hits"]] + [x["fall_rate_hits"] for x in ladder if x["method"] == m] + [oracle["fall_rate_hits"]]
      names = ["reactive"] + [x["rung"] for x in ladder if x["method"] == m] + ["oracle"]
      out[f"shares_{m}"] = [{"from": names[i], "to": names[i + 1], "reduction": steps[i] - steps[i + 1],
                             "share_of_gap": (steps[i] - steps[i + 1]) / gap if gap else np.nan} for i in range(len(steps) - 1)]
  json.dump(out, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2, default=float)
  with open(os.path.join(_OUT.value, "ladder.csv"), "w") as f:
    f.write("rung,method,fall_rate_hits,false_prep_rate,param,progress_m\n")
    if react:
      f.write(f"reactive,reactive,{react['fall_rate_hits']:.4f},{react['false_prep_rate']:.4f},0,{react['progress_m']:.3f}\n")
    for x in ladder:
      f.write(f"{x['rung']},{x['method']},{x['fall_rate_hits']:.4f},{x['false_prep_rate']:.4f},{x['param']},{x['progress_m']:.3f}\n")
    if oracle:
      f.write(f"oracle,oracle,{oracle['fall_rate_hits']:.4f},{oracle['false_prep_rate']:.4f},{oracle['param']},{oracle['progress_m']:.3f}\n")
  print(open(os.path.join(_OUT.value, "ladder.csv")).read())
  for m in ("ttc_threshold", "belief_commit"):
    for sh in out.get(f"shares_{m}", []):
      print(f"{m:14} {sh['from']:>16} -> {sh['to']:<16} reduction {sh['reduction']:+.3f}  share {sh['share_of_gap']:+.0%}")
  for a in abl:
    print(f"ablation {a['ablation']:28} {a['method']:14} fall {a['fall_rate_hits']:.3f} @fp {a['false_prep_rate']:.2f}")

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.2), constrained_layout=True)
    col = {"ttc_threshold": "#2b6f8e", "belief_commit": "#d9772b", "aware_ppo": "#8e44ad"}
    rungs = [x for x in _RUNGS.value]
    names = ["reactive"] + [r.split(":")[0] for r in rungs] + ["oracle"]
    xs = np.arange(len(names))
    for m in ("ttc_threshold", "belief_commit"):
      ys = [react["fall_rate_hits"] if react else np.nan]
      for r in rungs:
        lab = r.split(":")[0]
        v = next((x["fall_rate_hits"] for x in ladder if x["method"] == m and x["rung"] == lab), np.nan)
        ys.append(v)
      ys.append(oracle["fall_rate_hits"] if oracle else np.nan)
      ax.plot(xs, ys, "o-", color=col[m], label={"ttc_threshold": "time-to-contact threshold", "belief_commit": "belief commit rule"}[m])
    for x in ladder:
      if x["method"] == "aware_ppo":
        ax.axhline(x["fall_rate_hits"], color=col["aware_ppo"], ls="--", lw=1, label=f"{x['rung']} (no decision layer)")
    ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel(f"failure rate on hits (false-prep <= {_FP.value:.0%})")
    ax.set_ylim(0, 1); ax.grid(True, color="#e3e6ea", lw=0.6); ax.legend(fontsize=8, frameon=False)
    ax.set_title("E12 oracle-gap ladder: each rung removes one limitation", loc="left", fontsize=10)
    fig.savefig(os.path.join(_OUT.value, "ladder.png"), dpi=170)
  except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
