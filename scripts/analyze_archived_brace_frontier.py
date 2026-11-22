"""Recompute deterministic duration frontiers at 100 ms on the E18 development archive."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


SOURCE = "results/E18_rule_family_p5/episodes.npz"
DT = 0.02
TTC, BELIEF = "ttc_threshold", "belief_lead"


def arm(data, method, par):
    prefix = f"{method}__p{par:g}__lat5__"
    return {k: np.asarray(data[prefix + k]) for k in ("fell", "brace_steps", "is_miss")}


def summarize(method, cost, data):
    miss = data["is_miss"].astype(bool)
    steps = data["brace_steps"]
    return dict(method=method, setting=cost, n_hits=int((~miss).sum()), n_misses=int(miss.sum()),
                mean_recorded_brace_s=float(steps.mean() * DT),
                hit_failures=int(data["fell"][~miss].sum()), prepared_misses=int((steps[miss] > 0).sum()))


def compare(archive):
    # A second, explicitly exploratory operating point, with the belief setting changed.
    a, b = arm(archive, TTC, .30), arm(archive, BELIEF, .02)
    misses = a["is_miss"].astype(bool)
    if not np.array_equal(misses, b["is_miss"]):
        raise ValueError("Unaligned intended-miss labels")
    brace = DT * (a["brace_steps"].astype(float) - b["brace_steps"])
    falls = a["fell"].astype(float) - b["fell"].astype(float)
    prep = (a["brace_steps"] > 0).astype(float) - (b["brace_steps"] > 0)
    h, m = np.flatnonzero(~misses), np.flatnonzero(misses)
    rng = np.random.default_rng(52706)
    draws = []
    for _ in range(100):
        hi = rng.choice(h, size=(100, len(h)), replace=True)
        mi = rng.choice(m, size=(100, len(m)), replace=True)
        draws.append(np.column_stack(((brace[hi].sum(1) + brace[mi].sum(1)) / len(misses),
                                      falls[hi].mean(1), prep[mi].mean(1))))
    boot = np.concatenate(draws)
    point = [brace.mean(), falls[h].mean(), prep[m].mean()]
    return {name: {"estimate":float(point[i]), "paired_95pct":np.quantile(boot[:, i],[.025,.975]).tolist()}
            for i,name in enumerate(("duration_s","hit_failure_difference","near_miss_preparation_difference"))}


def render(rows, dest):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.55), sharex=True, layout="constrained")
    for ax, name, n, heading in zip(axes, ("hit_failures", "prepared_misses"), ("n_hits", "n_misses"),
                                    ("Failure on scheduled hits", "Preparation on scheduled near-misses")):
        for method, label, color, marker in [(TTC, "Time-to-contact threshold", "#14576a", "o"),
                                               (BELIEF, "Lead-conditioned belief", "#a94c25", "s")]:
            group=sorted((r for r in rows if r["method"]==method), key=lambda r:r["mean_recorded_brace_s"])
            xs=[r["mean_recorded_brace_s"] for r in group]
            ys=[100*r[name]/r[n] for r in group]
            ax.plot(xs, ys, color=color, lw=1.5, marker=marker, ms=4.6, label=label)
        ax.set_title(heading,loc="left",fontsize=10,weight="bold")
        ax.set_ylabel("Episodes (%)")
        ax.grid(axis="y",color="#d9dfdf",linewidth=.6)
        ax.spines[["right","top"]].set_visible(False)
    axes[0].legend(frameon=False,fontsize=8,loc="upper right")
    axes[-1].set_xlabel("Mean recorded time in BRACE per encounter (s)")
    fig.savefig(dest / "frontier.pdf",metadata={'CreationDate':None,'ModDate':None})
    fig.savefig(dest / "frontier.png",dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    dest=args.out or args.root / "analysis/deterministic_frontier"
    dest.mkdir(parents=True,exist_ok=True)
    path=args.root/SOURCE
    payload=path.read_bytes()
    git_blob=hashlib.sha1(b"blob "+str(len(payload)).encode()+b"\0"+payload).hexdigest()
    if git_blob != "799a33bec5cc41f56f6f96d6a019b438d4887976":
        raise ValueError(f"Unexpected E18 archive: {git_blob}")
    with np.load(path,allow_pickle=False) as data:
        rows=[summarize(method,par,arm(data,method,par)) for method,settings in
              [(TTC,(.20,.30,.40,.60,.80)),(BELIEF,(.02,.05,.10,.20,.40,.80))] for par in settings]
        sensitivity=compare(data)
    with (dest/"operating_points.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    record={"archive_git_blob":git_blob,"scope":"Retrospective development-set analysis, 10 Hz, 100 ms added delay",
            "direction":"TTC 0.30 s minus belief 0.02","uncertainty":"10,000 paired bootstrap draws stratified by scheduled hit/miss, fixed original stratum sizes, conditional on these settings",
            "paired_comparison":sensitivity}
    (dest/"paired_comparison.json").write_text(json.dumps(record,indent=2)+"\n")
    render(rows,dest)


if __name__ == "__main__":
    main()
