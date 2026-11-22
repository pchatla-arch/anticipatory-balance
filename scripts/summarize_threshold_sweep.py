"""Summarize a published CPU pilot or full E20 run using its phase-tagged episode records."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import fmean


def truth(row, field):
    value = row[field]
    if value not in ("True", "False"):
        raise ValueError(f"Unexpected {field}: {value}")
    return value == "True"


def summarize(root):
    plan = json.loads((root / "protocol.json").read_text())
    pilot = plan.get("schema") == "foresight.deterministic-cpu-pilot.v1"
    if not pilot and plan.get("experiment") != "Foresight deterministic TTC matched to belief preparation, revision 1":
        raise ValueError("Unsupported E20 or CPU pilot protocol")
    freeze = json.loads((root / ("freeze.json" if pilot else "frozen.json")).read_text())
    verdict = json.loads((root / "verdict.json").read_text())
    if pilot:
        dt = plan["control_horizon"][1]
        reference = tuple(plan["reference_rule"])
        grid = plan["ttc_threshold_grid_seconds"]
    else:
        config = json.loads((root / "calibration/config.json").read_text())
        dt = config["dt"]
        reference = tuple(plan["confirmation"]["frozen_belief"])
        grid = plan["calibration"]["theta_grid_s"]
    def seeds_and_size(stage):
        if pilot:
            return [plan[stage + "_seed"]], plan[stage + "_n"]
        phase = plan[stage]
        return ([phase["seed"]], phase["n"]) if stage == "calibration" else (
            phase["batch_seeds"], phase["episodes_per_seed"])
    result = []
    stages = ["calibration", "confirmation"] if freeze["matched_on_calibration"] else ["calibration"]
    if bool(verdict.get("confirmation_started", True)) != bool(freeze["matched_on_calibration"]):
        raise ValueError("Verdict and registered calibration stopping rule differ")
    for stage in stages:
        seeds, size = seeds_and_size(stage)
        compact = root / (stage + "_episodes.csv")
        paths = [compact] if pilot and compact.is_file() else sorted((root / stage / "episodes").glob("*.csv"))
        if not paths:
            raise ValueError(f"No completed episode files for {stage}")
        raw = []
        for path in paths:
            with path.open() as stream:
                raw.extend(csv.DictReader(stream))
        arms = {}
        for row in raw:
            arm = (row["method"], float(row["param"]))
            sid = row["scenario_id"]
            if "scenario_index" in row:
                seed, batch, index = int(row["scenario_seed"]), int(row["batch"]), int(row["scenario_index"])
                if (batch < 0 or batch >= len(seeds) or seeds[batch] != seed or index < 0 or index >= size
                        or int(row["latency_steps"]) != 5 or sid != f"seed{seed}_env{index:05d}"):
                    raise ValueError("Mismatched raw evaluator scenario, batch or delay")
            if sid in arms.setdefault(arm, {}):
                raise ValueError("Repeated scenario ID in arm")
            arms[arm][sid] = row
        expected_methods = set(grid if stage == "calibration" else [freeze["theta_s"]])
        expected_arms = {("ttc_threshold", theta) for theta in expected_methods} | {reference}
        if set(arms) != expected_arms:
            raise ValueError("Unexpected or missing arm")
        ref = arms[reference]
        expected_ids = {f"seed{seed}_env{i:05d}" for seed in seeds for i in range(size)}
        for (method, param), rows in arms.items():
            if set(rows) != expected_ids:
                raise ValueError("Unexpected or missing scenario ID")
            for sid, row in rows.items():
                if any(ref[sid][f] != row[f] for f in ("mass", "speed", "is_miss", "t_hit_nominal")):
                    raise ValueError("Unpaired scenario metadata")
            for name, miss in (("all", None), ("scheduled_hit", False), ("scheduled_miss", True)):
                obs = [r for r in rows.values() if miss is None or truth(r, "is_miss") == miss]
                finite = [float(r["progress_m"]) for r in obs if math.isfinite(float(r["progress_m"]))]
                result.append(dict(stage=stage, method=method, param=param, stratum=name, n=len(obs),
                                   contacts=sum(truth(r, "contact_before_termination") for r in obs),
                                   failures=sum(truth(r, "fell") for r in obs),
                                   failures_among_contacts=sum(truth(r, "fell") and truth(r, "contact_before_termination") for r in obs),
                                   any_brace=sum(int(r["brace_steps_total"]) > 0 for r in obs),
                                   applied_brace_s=fmean(dt * int(r["brace_steps_total"]) for r in obs) if obs else "",
                                   finite_progress_n=len(finite), finite_mean_progress_m=fmean(finite) if finite else ""))
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = summarize(args.evidence)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
