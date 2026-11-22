"""Plot a complete calibration sweep from a published pilot or E20 run."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from summarize_threshold_sweep import summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    rows = [r for r in summarize(args.evidence) if r["stage"] == "calibration" and r["stratum"] == "all"]
    reference, = [r for r in rows if r["method"] == "belief_lead"]
    ttc = sorted((r for r in rows if r["method"] == "ttc_threshold"), key=lambda r: r["param"])
    target = reference["applied_brace_s"]
    frozen_files = [p for p in (args.evidence / "freeze.json", args.evidence / "frozen.json") if p.exists()]
    freeze = json.loads(frozen_files[0].read_text())
    chosen, = [r for r in ttc if r["param"] == freeze["theta_s"]]
    parameters = [r["param"] for r in ttc]
    duration = [r["applied_brace_s"] for r in ttc]
    fig, ax = plt.subplots(figsize=(6.0, 3.55), layout="constrained")
    ax.axhspan(target - .020, target + .020, color="#d2ece4", label="Calibration band (±20 ms)")
    ax.axhline(target, color="#1e7164", lw=1.5, ls="--", label="Belief duration")
    ax.plot(parameters, duration, marker="o", color="#384a69", lw=1.6, label="Deterministic TTC")
    ax.scatter(chosen["param"], chosen["applied_brace_s"], marker="D", s=85, facecolors="none",
               edgecolors="#bd6227", linewidth=2, zorder=5, label="Closest threshold")
    ax.set_xticks(parameters, [f"{v:.2f}" for v in parameters], rotation=30)
    ax.set_xlabel("TTC trigger threshold (s)")
    ax.set_ylabel("Mean applied brace time (s / episode)")
    ax.grid(axis="y", color="#e3e7e6", lw=.75)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.24), fontsize=8.5)
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
