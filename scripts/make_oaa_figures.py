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


def segments_from_command(cmd):
    """Boundaries of the command schedule, read from the commanded (vx, vy, yaw) signal itself.

    Returns a list of (name, start, end) with end exclusive. Names follow the walk-turn-stop-resume
    schedule when four segments are found; otherwise they are numbered.
    """
    change = np.any(np.abs(np.diff(cmd, axis=0)) > 1e-6, axis=1)
    starts = [0] + [int(i) + 1 for i in np.flatnonzero(change)]
    ends = starts[1:] + [cmd.shape[0]]
    names = ["walk", "turn", "stop", "resume"] if len(starts) == 4 else [f"seg{i}" for i in range(len(starts))]
    return list(zip(names, starts, ends))


def segment_speeds(trace_path):
    """Per schedule segment: mean achieved forward speed, mean |commanded forward speed| and the
    mean speed-command error (achieved minus commanded), each averaged over episodes after the
    per-episode means. Ticks before each episode's onset index are excluded from every segment,
    for every family (the healthy family carries a nominal onset), so that all arms are compared
    over the same post-onset window. Commanded speed can differ between scenarios, so the error
    column is the like-for-like quantity; during the commanded stop it equals the achieved speed.
    """
    d = np.load(trace_path)
    cmd, vel, onset = d["command"], d["velocity"], d["onset"]
    rows = []
    for name, a, b in segments_from_command(cmd[:, 0, :]):
        ach, com, err = [], [], []
        for e in range(cmd.shape[1]):
            lo = max(a, int(onset[e]) + 1)
            if lo >= b:
                continue
            v = vel[lo:b, e, 0]
            c = np.abs(cmd[lo:b, e, 0])
            ach.append(float(v.mean())); com.append(float(c.mean())); err.append(float((v - c).mean()))
        f = lambda xs: float(np.mean(xs)) if xs else np.nan
        rows.append((name, f(ach), f(com), f(err)))
    return rows


def fig_command_following(bench_dir, out, family="gain_070", telemetry="declared",
                          healthy_family="healthy"):
    """Speed-command error per schedule segment, by method, with the healthy walker as the
    reference. Reads the per-step traces written with --trace. The uncorrected faulted walker is
    expected to overshoot the command and to keep moving during the commanded stop."""
    plt = style()
    bench_dir = Path(bench_dir)
    methods = ["none", "estimated_inverse", "heuristic_gate", "oracle"]
    data = {}
    ph = bench_dir / f"trace_{healthy_family}_{telemetry}_none.npz"
    if ph.exists():
        data["healthy_none"] = segment_speeds(ph)
    for m in methods:
        p = bench_dir / f"trace_{family}_{telemetry}_{m}.npz"
        if p.exists():
            data[m] = segment_speeds(p)
    if not any(m in data for m in methods):
        return
    segs = [r[0] for r in next(iter(data.values()))]
    x = np.arange(len(segs))
    width = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    for k, (m, rows) in enumerate(data.items()):
        color = "#cfd4da" if m == "healthy_none" else COLORS[m]
        label = "healthy walker, no correction" if m == "healthy_none" else LABELS[m]
        ax.bar(x + k * width, [r[3] for r in rows], width, color=color, label=label)
    ax.axhline(0, color="k", lw=1.0, ls="--")
    ax.set_xticks(x + 0.4 - width / 2)
    ax.set_xticklabels(segs)
    ax.set_ylabel("achieved − commanded speed (m/s)")
    ax.set_title(f"Speed-command error, {family.replace('gain_0', 'gain 0.')}, {telemetry.replace('_', ' ')} telemetry",
                 loc="left", fontsize=8)
    ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
    fig.savefig(out / f"command_following_{family}_{telemetry}.pdf")
    fig.savefig(out / f"command_following_{family}_{telemetry}.png")
    return data


def fig_telemetry_curve(bench_dirs, out, families=("gain_085", "gain_070", "gain_055")):
    """Post-fault loss per method across telemetry presets: the value of timely, accurate
    information. bench_dirs maps preset name -> directory with summary.json."""
    plt = style()
    order = [p for p in ("exact", "declared_fast", "declared") if p in bench_dirs]
    summaries = {p: json.load(open(Path(bench_dirs[p]) / "summary.json")) for p in order}
    methods = ["none", "estimated_inverse", "heuristic_gate", "oracle"]
    fig, axes = plt.subplots(1, len(families), figsize=(2.3 * len(families), 2.4), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, fam in zip(axes, families):
        for m in methods:
            ys = []
            for p in order:
                cell = summaries[p]["cells"].get(f"{fam}|{p}|{m}")
                ys.append(cell["mean_postfault_loss"] if cell else np.nan)
            ax.plot(range(len(order)), ys, "o-", color=COLORS[m], label=LABELS[m], ms=4, lw=1)
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([{"exact": "exact", "declared_fast": "declared,\nno latency", "declared": "declared,\n20–60 ms"}[p] for p in order], fontsize=6)
        ax.set_title(fam.replace("_", " "), loc="left", fontsize=8)
    axes[0].set_ylabel("post-fault tracking loss (m/s)$^2$")
    axes[0].legend(fontsize=6)
    fig.savefig(out / "telemetry_curve.pdf")
    fig.savefig(out / "telemetry_curve.png")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", type=Path, required=True)
    parser.add_argument("--authority", default="")
    parser.add_argument("--actuators", type=Path, default=None)
    parser.add_argument("--telemetry-benches", default="",
                        help="comma-separated preset=dir pairs for the telemetry curve, e.g. exact=results/x/exact,declared=results/x/declared")
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
    cf = fig_command_following(args.bench, args.out)
    if cf:
        json.dump(cf, open(args.out / "command_following_gain_070_declared.json", "w"), indent=1)
    if args.telemetry_benches:
        pairs = dict(kv.split("=", 1) for kv in args.telemetry_benches.split(","))
        fig_telemetry_curve(pairs, args.out)
    print("figures written to", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
