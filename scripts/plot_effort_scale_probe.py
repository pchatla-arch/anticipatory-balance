"""Plot a measured matched-state trial from probe_effort_scale_ambiguity.py."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("record", type=Path)
p.add_argument("figure", type=Path)
args = p.parse_args()
trial, = [x for x in json.loads(args.record.read_text())["cases"] if x["error_rad"] == 0.1]
fig, axes = plt.subplots(1, 2, figsize=(9, 3.35), sharey=True, layout="constrained")
for ax, field, title in zip(axes, ("observed_effort_nm", "physical_effort_nm"),
                            ("Torque reported to controller", "Physical joint torque")):
    for index, (label, world) in enumerate((("Weak motor", "weak_motor"), ("Healthy motor, scaled sensor", "scaled_sensor"))):
        row = trial["worlds"][world]
        x0, x1 = (row[a][field] for a in ("leave_target", "apply_correction"))
        y = 1 - index
        ax.plot([x0, x1], [y, y], lw=2, color="#738591")
        ax.scatter([x0], [y], s=78, facecolor="white", edgecolor="#15525d", linewidth=2, zorder=4)
        ax.scatter([x1], [y], s=78, color="#15525d", zorder=4)
        for x, dy in ((x0, 0.17), (x1, -0.24)):
            ax.text(x, y + dy, f"{x:g}", ha="center", fontsize=10)
    ax.set_title(title, loc="left", weight="bold", fontsize=12)
    ax.set_xlim(4.55, 11.15)
    ax.set_ylim(-0.5, 1.55)
    ax.set_xlabel("Torque (N·m)")
    ax.set_yticks([0, 1], ["Healthy motor,\nscaled sensor", "Weak motor"])
    ax.spines[["right", "top", "left"]].set_visible(False)
    ax.grid(axis="x", color="#e4e9e9")
axes[1].axvline(trial["healthy_reference_nm"], color="#bd612e", linestyle=":", zorder=0)
axes[1].text(7.64, 1.35, "Healthy target 7.5", color="#8b481f", fontsize=9)
axes[0].scatter([], [], s=57, facecolor="white", edgecolor="#15525d", label="Before correction")
axes[0].scatter([], [], s=57, color="#15525d", label="After correction")
fig.legend(loc="outside lower center", ncol=2, frameon=False)
fig.savefig(args.figure, dpi=155, bbox_inches="tight")
