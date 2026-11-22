#!/usr/bin/env python3
"""Figures for the AA228 paper on Online Actuator Adaptation, from result files only.

  python scripts/make_oaa_figures.py --bench results/bench_dev_cpu_v0 \
      --authority results/authority_dev_v1_speed3.0,results/authority_dev_v1_speed1.0 --out paper/figures

Every number plotted is read from summary.json / authority.json / trace npz files; nothing is
typed in. Figures: (1) recovery vs harmful intervention per method; (2) authority study:
requested/oracle and zero-authority steps by fault level, speed bound and joint group;
(3) time series of one return-to-healthy episode and one sensor-scale episode.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

COLORS = {"none": "#6a7179", "oracle": "#3a9d5d", "estimated_inverse": "#d9772b",
          "heuristic_gate": "#2b6f8e", "candidate": "#8e44ad", "robust_ppo": "#b03a2e"}
LABELS = {"none": "no adaptation", "oracle": "oracle (true gain)", "estimated_inverse": "estimated inverse",
          "heuristic_gate": "heuristic gate", "candidate": "identification-aware", "robust_ppo": "robust PPO"}


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e3e6ea", "grid.linewidth": 0.6,
                         "legend.frameon": False, "figure.dpi": 150})
    return plt


def fig_tradeoff(summary, out, telemetry="declared"):
    plt = style()
    cells = summary["cells"]
    fault = [k for k in cells if k.split("|")[0].startswith("gain") and k.split("|")[1] == telemetry]
    healthy = [k for k in cells if not k.split("|")[0].startswith("gain") and k.split("|")[1] == telemetry]
    methods = sorted({k.split("|")[2] for k in cells})
    fig, ax = plt.subplots(figsize=(3.4, 2.8), constrained_layout=True)
    for m in methods:
        loss = np.mean([cells[k]["mean_postfault_loss"] for k in fault if k.endswith("|" + m) and cells[k]["mean_postfault_loss"] is not None])
        harm = np.mean([cells[k]["mean_applied_offset_rad"] for k in healthy if k.endswith("|" + m) and cells[k]["mean_applied_offset_rad"] is not None])
        ax.plot(harm, loss, "o", ms=7, color=COLORS.get(m, "k"), label=LABELS.get(m, m))
    ax.set_xlabel("mean correction applied to a healthy plant (rad)")
    ax.set_ylabel("post-fault tracking loss on gain faults (m/s)$^2$")
    ax.set_title(f"Recovery against harmful intervention ({telemetry} telemetry)", loc="left", fontsize=8)
    ax.legend(fontsize=7)
    fig.savefig(out / "tradeoff.pdf")
    fig.savefig(out / "tradeoff.png")


def fig_authority(auth_dirs, out, actuators):
    plt = style()
    names = actuators["names"]

    def group(a):
        n = names[a]
        return "ankle" if "ankle" in n else ("knee" if "knee" in n else "hip")

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), constrained_layout=True)
    styles = ["-", "--", ":"]
    for si, d in enumerate(auth_dirs):
        res = json.load(open(Path(d) / "authority.json"))
        speed = next(iter(res.values()))["config"]["max_speed"]
        rows = sorted([v for v in res.values() if v["telemetry"] == "declared"], key=lambda v: v["gain"])
        gains = [v["gain"] for v in rows]
        for gi, grp in enumerate(("hip", "knee", "ankle")):
            req = [np.mean([p["requested_over_oracle"] for p in v["per_actuator"] if group(p["actuator"]) == grp]) for v in rows]
            zero = [np.mean([p["zero_authority_fraction_post"] for p in v["per_actuator"] if group(p["actuator"]) == grp]) for v in rows]
            col = ["#2b6f8e", "#3a9d5d", "#d9772b"][gi]
            axes[0].plot(gains, req, "o" + styles[si % 3], color=col, ms=4, label=f"{grp}, speed bound {speed:g} rad/s")
            axes[1].plot(gains, zero, "o" + styles[si % 3], color=col, ms=4)
    axes[0].set_xlabel("fault gain g"); axes[0].set_ylabel("requested / oracle correction")
    axes[1].set_xlabel("fault gain g"); axes[1].set_ylabel("fraction of post-onset steps with zero authority")
    axes[0].set_ylim(0, 1.05); axes[1].set_ylim(0, 1.0)
    axes[0].legend(fontsize=6, ncol=2)
    fig.suptitle("Authority left by the declared information (offline replay, declared telemetry)", fontsize=8)
    fig.savefig(out / "authority.pdf")
    fig.savefig(out / "authority.png")


def fig_timeseries(bench_dir, out, actuators):
    plt = style()
    leg = np.asarray(actuators["leg_mask"])
    panels = [("gain_070_return", "declared", "unannounced return to healthy gain"),
              ("sensor_scale_080", "declared", "healthy motor, effort sensor reads 80%")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.2), constrained_layout=True, sharex="col")
    for col, (family, tele, title) in enumerate(panels):
        for method in ("estimated_inverse", "heuristic_gate", "candidate"):
            path = Path(bench_dir) / f"trace_{family}_{tele}_{method}.npz"
            if not path.exists():
                continue
            d = np.load(path)
            i = 0
            t = np.arange(d["g_hat"].shape[0]) * 0.02
            g_true = np.where(d["gain_true_active"][:, i], 0.7 if family.startswith("gain") else 1.0, 1.0)
            axes[0, col].plot(t, d["g_hat"][:, i, leg].mean(-1), color=COLORS[method], lw=1, label=LABELS[method])
            axes[1, col].plot(t, np.abs(d["effective"][:, i, :]).max(-1), color=COLORS[method], lw=1)
        axes[0, col].plot(t, g_true, "k--", lw=1, label="true gain")
        axes[0, col].set_title(title, loc="left", fontsize=8)
        axes[0, col].set_ylabel("gain estimate (leg mean)")
        axes[1, col].set_ylabel("max |applied offset| (rad)")
        axes[1, col].set_xlabel("time (s)")
        if "ret" in d.files:
            for ax in axes[:, col]:
                ax.axvline(d["onset"][i] * 0.02, color="#999", lw=0.8)
                if d["ret"][i] < 2**29:
                    ax.axvline(d["ret"][i] * 0.02, color="#999", lw=0.8, ls=":")
    axes[0, 0].legend(fontsize=6)
    fig.savefig(out / "timeseries.pdf")
    fig.savefig(out / "timeseries.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--authority", default="")
    parser.add_argument("--actuators", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("paper/figures"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = json.load(open(args.bench / "summary.json"))
    actuators_path = args.actuators or (args.bench / "actuators.json")
    if not actuators_path.exists():
        candidates = glob.glob(str(args.bench.parent / "*/actuators.json"))
        actuators_path = Path(candidates[0])
    actuators = json.load(open(actuators_path))
    fig_tradeoff(summary, args.out)
    if args.authority:
        fig_authority(args.authority.split(","), args.out, actuators)
    fig_timeseries(args.bench, args.out, actuators)
    print("figures written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
