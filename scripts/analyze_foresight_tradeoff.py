"""Reproduce the 100 ms matched-preparation comparison from archived episode arrays.

From the repository root: python3 scripts/analyze_foresight_tradeoff.py
Requires NumPy and Matplotlib. The source artifact hashes pin the data used in the paper.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DATA = {
    "development": ("E18_rule_family_p5", "799a33bec5cc41f56f6f96d6a019b438d4887976"),
    "held_out_scenarios": ("E18h_heldout_p5", "249e92e0ae0c084aff631ed092c78a50f77270d8"),
    "second_skill_pair": ("E19_seed2_p5", "1b475315c11838a957263e8c3a06f48f1c5bdef7"),
}
ARMS = {
    "reactive": "reactive__p0__lat0",  # The evaluator shares this sensor-independent arm across delays.
    "ttc": "ttc_threshold__p0.3__lat5",
    "belief": "belief_lead__p0.05__lat5",
}
FIELDS = ("hit_failure", "brace_seconds", "near_miss_preparation", "progress_m")
DT = 0.02
SEED = 22821
REPLICATES = 10000


def read_archive(results, name):
    directory, expected = DATA[name]
    path = results / directory / "episodes.npz"
    raw = path.read_bytes()
    sha = hashlib.sha1(f"blob {len(raw)}\0".encode() + raw).hexdigest()
    if sha != expected:
        raise ValueError(f"Unexpected Git blob for {path}: {sha}")
    arms = {}
    with np.load(path, allow_pickle=False) as archive:
        hit = ~archive[ARMS["reactive"] + "__is_miss"].astype(bool)
        if not 0 < hit.sum() < len(hit):
            raise ValueError("Both scheduled hits and near-misses are required")
        for arm, key in ARMS.items():
            if not np.array_equal(hit, ~archive[key + "__is_miss"].astype(bool)):
                raise ValueError("Scheduled hit labels differ between paired conditions")
            falls = archive[key + "__fell"]
            steps = archive[key + "__brace_steps"]
            progress = archive[key + "__progress_m"]
            if not np.isin(falls, [0, 1]).all() or (steps < 0).any():
                raise ValueError(f"Invalid episode records for {key}")
            values = np.column_stack([falls, DT * steps.astype(float), steps > 0, progress])
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite episode records for {key}")
            arms[arm] = values
    return hit, arms


def reduce_metrics(values, hit):
    return dict(zip(FIELDS, map(float, [
        values[hit, 0].mean(), values[:, 1].mean(),
        values[~hit, 2].mean(), values[:, 3].mean(),
    ])))


def bootstrap(values, hit, seed):
    """Paired, stratified percentile intervals, conditional on the archived mixture of scenarios."""
    rng = np.random.default_rng(seed)
    hits, misses = values[hit], values[~hit]
    nh, nm = len(hits), len(misses)
    draws = np.empty((REPLICATES, len(FIELDS)))
    for offset in range(0, REPLICATES, 64):
        count = min(64, REPLICATES - offset)
        h = hits[rng.integers(nh, size=(count, nh))].mean(axis=1)
        m = misses[rng.integers(nm, size=(count, nm))].mean(axis=1)
        pooled = (nh * h + nm * m) / (nh + nm)
        draws[offset:offset + count] = np.column_stack([h[:, 0], pooled[:, 1], m[:, 2], pooled[:, 3]])
    lo, hi = np.quantile(draws, [0.025, 0.975], axis=0)
    return {field: [float(l), float(u)] for field, l, u in zip(FIELDS, lo, hi)}


def summarize(hit, arms, p, seed):
    # Average both realized potential outcomes per row over an independent episode-start coin.
    # This integrates out coin noise. It is an expectation, not a new realized rollout.
    arms = dict(arms, mixture=p * arms["ttc"] + (1 - p) * arms["reactive"])
    delta = arms["belief"] - arms["mixture"]
    return {
        "episodes": int(hit.size), "scheduled_hits": int(hit.sum()),
        "scheduled_near_misses": int((~hit).sum()),
        "arms": {k: reduce_metrics(v, hit) for k, v in arms.items()},
        "realized_scheduled_hit_failures": {
            k: int(v[hit, 0].sum()) for k, v in arms.items() if k != "mixture"
        },
        "belief_minus_mixture": reduce_metrics(delta, hit),
        "conditional_paired_95pct": bootstrap(delta, hit, seed),
    }


def plot(splits, output):
    primary = splits["held_out_scenarios"]
    arms = primary["arms"]
    plt.rcParams.update({"pdf.fonttype": 42})
    fig = plt.figure(figsize=(3.65, 3.65))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1])
    ax = fig.add_subplot(grid[0, :])
    risk = fig.add_subplot(grid[1, 0])
    near_miss = fig.add_subplot(grid[1, 1], sharey=risk)
    for panel in (ax, risk, near_miss):
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(labelsize=8)
        panel.grid(axis="x", color="#e1e5e9", linewidth=.65)
        panel.set_axisbelow(True)
    def xy(arm):
        return arms[arm]["brace_seconds"], 100 * arms[arm]["hit_failure"]
    ax.plot([xy("reactive")[0], xy("ttc")[0]], [xy("reactive")[1], xy("ttc")[1]],
            lw=1.3, color="#727b84", zorder=1)
    for label, offset, name in [("reactive", (5,-1), "Never brace"), ("ttc", (0,8), "TTC"),
                                ("belief", (-49,-11), "Belief"), ("mixture", (-3,11), "Mixture")]:
        x,y=xy(label)
        mark = "D" if label=="belief" else "o"
        ax.scatter(x,y,s=27,marker=mark, facecolors="#00716b" if label=="belief" else "white",
                   edgecolors="#00716b" if label=="belief" else "#343e48",zorder=3)
        ax.annotate(name,(x,y),xytext=offset,textcoords="offset points",fontsize=8)
    ax.set(xlim=(-.025,.70),ylim=(36,73))
    ax.set_yticks([40,50,60,70])
    ax.set_xlabel("Mean recorded brace time (s / episode)",fontsize=8)
    ax.set_ylabel("Hit failure (%)",fontsize=8)
    # Different horizontal scales keep the smaller hit effects legible.
    for panel, field, title, limits, ticks in [
        (risk, "hit_failure", "Hit failure", (-3.5, 4), [-3, 0, 3]),
        (near_miss, "near_miss_preparation", "Near-miss prep.", (-26.5, 2), [-24, -12, 0]),
    ]:
        for index, key in enumerate(["second_skill_pair", "held_out_scenarios"]):
            result = splits[key]
            effect = 100 * result["belief_minus_mixture"][field]
            lo, hi = np.array(result["conditional_paired_95pct"][field]) * 100
            panel.errorbar(effect, index, xerr=[[effect - lo], [hi - effect]], fmt="D",
                           color="#00716b", markersize=4, elinewidth=1.3, capsize=3)
        panel.axvline(0, color="#6e7881", lw=1, ls="--")
        panel.set(xlim=limits, ylim=(-.55, 1.55), yticks=[0, 1], xticks=ticks)
        panel.set_title(title, fontsize=8.5, pad=2, loc="left")
    risk.set_yticklabels(["Second pair", "Fresh scenarios"], fontsize=7)
    near_miss.tick_params(axis="y", left=False, labelleft=False)
    fig.text(.59, .025, "Belief minus mixture (percentage points)", ha="center", fontsize=8)
    fig.subplots_adjust(left=.235, right=.97, bottom=.155, top=.965, hspace=.82, wspace=.34)
    for ext in ["pdf", "png"]:
        fig.savefig(output / f"heldout_tradeoff.{ext}", dpi=200, bbox_inches="tight",pad_inches=.035,
                    metadata={"Creator":"Foresight archive analysis", **({"CreationDate":None} if ext=="pdf" else {})})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--out", type=Path, default=Path("analysis/decision_tradeoff"))
    args = parser.parse_args()
    archives = {name: read_archive(args.results, name) for name in DATA}
    _, dev = archives["development"]
    times = {k: a[:, 1].mean() for k, a in dev.items()}
    if not times["reactive"] < times["belief"] < times["ttc"]:
        raise ValueError("Calibration target is outside the two-arm mixture support")
    p = (times["belief"] - times["reactive"]) / (times["ttc"] - times["reactive"])
    splits = {name: summarize(hit, arms, p, SEED + i)
              for i, (name, (hit, arms)) in enumerate(archives.items())}
    record = {
        "source_commit": "e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f",
        "input_git_blob_sha1": {directory: sha for directory, sha in DATA.values()},
        "development_fixed_ttc_probability": float(p), "detection_rate_hz": 10,
        "added_delay_ms": 100, "arm_keys": ARMS, "recorded_seconds_per_step": DT,
        "interval": {"method":"paired scenario bootstrap, stratified by scheduled hit/miss; percentile",
                     "replicates":REPLICATES,"rng_base_seed":SEED,"split_seeds":{
                         name: SEED+i for i,name in enumerate(DATA)
                      },"held_fixed":"estimated mixing probability, policy checkpoints and scenario mix"},
        "splits": splits,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/"results.json").write_text(json.dumps(record,indent=2,allow_nan=False)+"\n")
    plot(splits,args.out)


if __name__ == "__main__":
    main()
