#!/usr/bin/env python3
"""Read-only verification of a completed WP 2.2 velocity pilot. Requires NumPy."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import re
import zlib

import numpy as np


FAMILIES = ("gain_070", "gain_070_return", "healthy", "sensor_scale_080")
CHANNELS = ("declared", "declared_fast")
METHODS = ("none", "oracle", "heuristic_gate", "candidate", "candidate_velocity")
CHECKPOINT = "14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de"


def read(path):
    return json.loads(path.read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(measured, reported, label, *, atol=1e-8, rtol=1e-7):
    if measured is None or reported is None:
        require(measured is reported, f"{label}: expected {measured}, found {reported}")
        return
    np.testing.assert_allclose(reported, measured, atol=atol, rtol=rtol, equal_nan=False, err_msg=label)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit_cells(root, grid, cells, config, manifest, joint_names):
    ages_and_coverage = {}
    paired_delivery, paired_initial, paired_commands = {}, {}, {}
    dt = config["candidate_model"]["dt_s"]
    equal(.02, dt, "control interval")
    for family, channel, method in grid:
        key = "|".join((family, channel, method))
        suffix = "_".join((family, channel, method))
        with np.load(root / f"trace_{suffix}.npz", allow_pickle=False) as archive:
            trace = {k: archive[k] for k in archive}
        rows = cells[key]
        joint_groups = {g: np.array([g in name for name in joint_names]) for g in ("hip", "knee")}
        alive = trace["alive"]
        t, n = alive.shape
        require((t, n) == (manifest["steps"], manifest["episodes"]), f"{key} trace shape")
        steps = np.arange(t)[:, None]
        for where, mapping, observed in (
            ((family, channel), paired_delivery, trace["pkt_arrival"]),
            (family, paired_initial, trace["authority_decision_q_truth"][0]),
            (family, paired_commands, trace["command"]),
        ):
            fingerprint = digest(observed.tobytes())
            require(mapping.setdefault(where, fingerprint) == fingerprint, f"{key} unpaired draws, commands, or reset")
        require(np.array_equal(trace["pilot_substep_q"][:, :, -1][alive],
                               trace["authority_source_q_truth"][alive]), f"{key} last substep truth")
        mag = np.max(np.abs(trace["effective"]), axis=-1)
        withdrawals = []
        for i, row in enumerate(rows):
            tag = f"{key} scenario {i}"
            term = int(row["termination_step"])
            onset = int(trace["onset"][i])
            ret = int(trace["ret"][i])
            live_i = alive[:, i]
            require(np.array_equal(live_i, (steps[:, 0] <= term) if term >= 0 else np.ones(t, bool)), tag + " live mask")
            require(row["fell"] == (term >= 0) and row["pre_fault_failed"] == (0 <= term < onset), tag + " falls")
            require(onset == row["onset_step"] and (ret if ret < 2**29 else -1) == row["return_step"], tag + " schedule")
            active = (steps[:, 0] >= onset) & (steps[:, 0] < ret) & (family in ("gain_070", "gain_070_return"))
            require(np.array_equal(active, trace["gain_true_active"][:, i]), tag + " actual gain window")
            planar_error = trace["velocity"][:, i] - trace["command"][:, i, :2]
            loss = np.minimum(4., (planar_error.astype(np.float64)**2).sum(axis=1))
            if term >= 0:
                loss[term:] = 4.
            equal(loss, trace["loss"][:, i], tag + " fixed-horizon per-tick loss", atol=1e-6, rtol=3e-6)
            equal(loss[onset:].mean() if onset < t else 0., row["postfault_loss"], tag + " episode loss",
                  atol=2e-6, rtol=1e-5)
            healthy = live_i & ~active
            for mask, metric in ((healthy, "harmful_offset_rad"), (live_i, "mean_applied_offset_rad")):
                mean = np.mean(mag[mask, i], dtype=np.float64) if mask.any() else 0.
                equal(mean, row[metric], tag + " " + metric, atol=2e-7, rtol=1e-5)
            crossed = np.flatnonzero(live_i & (steps[:, 0] >= ret) & (mag[:, i] < np.float32(.01)))
            withdrawal = int(crossed[0] - ret) if len(crossed) else -1
            require(withdrawal == row["withdrawal_steps"], tag + " withdrawal")
            if withdrawal >= 0:
                withdrawals.append(withdrawal)
        src = trace["authority_rls_last_source_step"]
        # Reconstruct the pre-decision cache from individual ordered packet releases.
        slots = trace["authority_release_mask"].shape[-1]
        cache = np.full(src.shape[1:], -1, dtype=int)
        for tick in range(t):
            require(np.array_equal(src[tick][alive[tick]], cache[alive[tick]]), f"{key} pre-decision source at {tick}")
            for slot in range(slots):
                released = trace["authority_release_mask"][tick, :, slot]
                packet_src = trace["authority_packet_source_step"][tick, :, slot]
                valid = trace["authority_packet_valid"][tick, :, slot]
                require(np.all((packet_src[released] >= 0) & (packet_src[released] <= tick)), f"{key} future release")
                cache = np.where(released[:, None] & valid, packet_src[:, None], cache)
        age = steps[:, :, None] - 1 - src
        recorded = read(root / f"hold_{suffix}.json")
        states = trace["pilot_substep_q"]
        substeps = states.shape[2]
        require(substeps == 10 and states.shape[:2] == (t, n), f"{key} wrong hold resolution")
        sensor = manifest["channels"][channel]
        source_pos = trace["authority_rls_last_position"]
        source_vel = trace["authority_rls_last_velocity"]
        # An accepted sample's position and velocity must match the actual source packet.
        ti, ei, ji = np.nonzero(alive[:, :, None] & (src >= 0))
        source_indices = (src[ti, ei, ji], ei, ji)
        require(np.array_equal(source_pos[ti, ei, ji], trace["pkt_position"][source_indices]), f"{key} source position")
        require(np.array_equal(source_vel[ti, ei, ji], trace["pkt_velocity"][source_indices]), f"{key} source velocity")
        measured = {}
        for group, joints in joint_groups.items():
            require(joints.any(), f"{key} missing joint group {group}")
            use = alive[:, :, None] & (src >= 0) & joints
            a = age[use]
            require(np.all(a >= 0), f"{key} negative source age")
            bins, count = np.unique(a, return_counts=True)
            h = {str(k): int(v) for k, v in zip(bins, count)}
            audit = recorded[group]
            require(h == audit["source_age_ticks_histogram"], f"{key} {group} age histogram")
            require(int(np.sum(alive) * np.sum(joints)) == audit["live_joint_ticks"] and len(a) ==
                    audit["source_available_joint_ticks"], f"{key} {group} exposure")
            avg = float(np.mean(a) * 1000. * dt) if len(a) else None
            equal(avg, audit["mean_source_age_ms"], f"{key} {group} age")
            measured[group] = {"live": audit["live_joint_ticks"], "available": len(a), "mean_age_ms": avg}
            if method in ("candidate", "candidate_velocity"):
                covered = use.copy()
                for sub in range(substeps + 1):
                    position = states[:, :, sub - 1] if sub else trace["authority_decision_q_truth"]
                    elapsed = age * dt + (dt * sub / substeps)
                    if method == "candidate":
                        radius = sensor["bound_position_rad"] + manifest["speed_config"]["hold_speed_rad_s"] * elapsed
                    else:
                        radius = sensor["bound_position_rad"] + (np.abs(source_vel) + sensor["bound_velocity_rad_s"]
                                 + .5 * manifest["acceleration_rad_s2"] * elapsed) * elapsed
                    covered &= np.abs(position - source_pos) <= radius
                count_covered = int(covered.sum())
                require(count_covered == audit["whole_hold_covered_joint_ticks"], f"{key} {group} covered count")
                fraction = count_covered / len(a) if len(a) else None
                equal(fraction, audit["whole_hold_coverage"], f"{key} {group} coverage")
                measured[group].update(covered=count_covered, coverage=fraction)
        scheduled = [r for r in rows if r["return_step"] >= 0]
        reached = [r for r in scheduled if r["termination_step"] < 0 or r["termination_step"] >= r["return_step"]]
        measured["withdrawal_counts"] = dict(scheduled=len(scheduled), reached=len(reached), withdrew=len(withdrawals))
        measured["withdrawal_median_steps"] = float(np.median(withdrawals)) if withdrawals else None
        ages_and_coverage[key] = measured
    return ages_and_coverage


def audit_summary(document, reference, grid, cells, evidence, margins):
    require(document["reference"] == reference and set(document["cells"]) == set(cells), f"{reference}: grid")
    comparisons = document["paired_vs_reference"]
    expected = set()
    for family, channel, method in grid:
        key = "|".join((family, channel, method))
        c = document["cells"][key]
        rows = cells[key]
        require(c["n"] == len(rows) and c["invalid_episodes"] == 0, key + " episode count")
        require(c["withdrawal_counts"] == evidence[key]["withdrawal_counts"], key + " withdrawal counts")
        equal(evidence[key]["withdrawal_median_steps"], c["withdrawal_steps_median"], key + " withdrawal median")
        for group in ("hip", "knee"):
            equal(evidence[key][group]["mean_age_ms"], c["mean_source_age_ms"][group], key + " " + group)
        for col, metric in (("postfault_loss", "mean_postfault_loss"), ("progress_m", "mean_progress_m"),
                            ("fell", "fall_rate"), ("harmful_offset_rad", "mean_harmful_offset_rad"),
                            ("mean_applied_offset_rad", "mean_applied_offset_rad"), ("gain_abs_error", "mean_gain_abs_error"),
                            ("lost_packets", "lost_packets_mean")):
            equal(np.mean([r[col] for r in rows]), c[metric], key + " " + metric)
        require(c["pre_fault_failures"] == sum(r["pre_fault_failed"] for r in rows), key + " pre-fault failures")
        require(c["never_withdrew"] == sum(r["return_step"] >= 0 and r["withdrawal_steps"] < 0 for r in rows),
                key + " never-withdrew count")
        require(c["control_bound_violations"] == 0 and all(r["max_correction_rad"] <= .20001 and
                r["max_slew_rad_s"] <= 2.00001 for r in rows), key + " control limits")
        refkey = "|".join((family, channel, reference))
        if refkey not in cells or method == reference or len(rows) <= 1:
            continue
        expected.add(key)
        n = len(rows)
        ref_rows = cells[refkey]
        ix = np.random.default_rng(228).integers(n, size=(2000, n))
        pair = comparisons[key]
        for metric, column in (("postfault_loss", "postfault_loss"), ("harmful_offset_rad", "harmful_offset_rad"),
                               ("progress_m", "progress_m"), ("fall_rate", "fell")):
            delta = np.array([float(r[column]) - float(s[column]) for r, s in zip(rows, ref_rows)])
            equal(delta.mean(), pair[metric]["mean_difference"], f"{key} relative to {reference}: {metric}")
            equal(np.quantile(delta[ix].mean(axis=1), [.025, .975]), pair[metric]["ci95"], key + " paired interval")
            require(pair[metric]["n"] == n, key + " paired n")
        if family in margins:
            within = pair["postfault_loss"]["ci95"][1] <= margins[family]
            require(within == pair["noninferior_loss"], key + " noninferiority")
    require(expected == set(comparisons), f"{reference}: missing or extra comparisons")


def run(root, wiring, source, plan_dir):
    manifest = read(root / "pilot_manifest.json")
    config = read(root / "config.json")
    registration = read(root / "registration.json")
    families = ("healthy",) if wiring else FAMILIES
    channels = ("declared",) if wiring else CHANNELS
    methods = ("candidate_velocity",) if wiring else METHODS
    n, steps = (1, 8) if wiring else (8, 300)
    grid = list(itertools.product(families, channels, methods))
    expected_seed = 911000 if wiring else 811000
    require(manifest["mode"] == ("wiring" if wiring else "development"), "unexpected experiment mode")
    for k, v in dict(episodes=n, steps=steps, batch=n, seed_base=expected_seed, split="dev", impl="jax",
                     platform="cpu", schedule="walk_turn_stop_resume", attempt_budget=1,
                     checkpoint_tree_sha256=CHECKPOINT, expected_episode_rows=len(grid)*n).items():
        require(manifest[k] == v, f"unexpected manifest {k}")
    if source:
        require(manifest["source_commit"] == source, "unexpected experiment source revision")
    require(config["source_commit"] == manifest["source_commit"] and config["devices"] and
            all("cpu" in s.lower() for s in config["devices"]), "runtime/source")
    if plan_dir:
        planned = read(plan_dir / "pilot_manifest.json")
        require(manifest == planned, "executed manifest differs from saved plan")
    for key, vals, regkey in (("families", families, "families"), ("channels", channels, "telemetry"),
                              ("methods", methods, "methods")):
        require(set(manifest[key]) == set(vals) == set(registration[regkey]), f"{key} registration disagreement")
    for k in ("episodes", "steps", "schedule", "split", "baseline"):
        require(registration[k] == manifest[k], f"registration {k}")
    require(config["baseline"] == manifest["baseline"], "comparator binding")
    if not wiring:
        b = manifest["baseline"]
        raw = (root / "frozen_heuristic.json").read_bytes()
        require(b["status"] == "frozen_wp21" and digest(raw) == b["sha256"] and
                re.fullmatch(r"[a-f0-9]{40}", b["source_commit"]) and
                json.loads(raw)["tuned_on"] == b["tuned_on"], "independent comparator identity")
    require(manifest["speed_config"]["hold_speed_rad_s"] == 10 and manifest["acceleration_rad_s2"] == 400,
            "unexpected candidate envelope")
    for key, ch in manifest["channels"].items():
        require((ch["latency_min"], ch["latency_max"]) == ((1, 3) if key == "declared" else (0, 0)) and
                ch["bound_velocity_rad_s"] == .5 and ch["velocity_error_rad_s"] == .5, "channel presets")
    cells = {"|".join(tup): [] for tup in grid}
    fixed_names = {"scenario_id", "split", "phase", "family", "telemetry", "method"}
    bool_names = {"nonfinite", "fell", "pre_fault_failed"}
    with (root / "episodes.csv").open(newline="") as f:
        for item in csv.DictReader(f):
            row = {}
            for field, value in item.items():
                if field in fixed_names:
                    row[field] = value
                elif field in bool_names:
                    require(value in ("True", "False"), "invalid boolean")
                    row[field] = value == "True"
                else:
                    row[field] = float(value)
                    require(math.isfinite(row[field]), f"invalid {field} for {item['scenario_id']}")
            require(row["split"] == "dev" and row["phase"] == "eval" and not row["nonfinite"], "invalid pilot episode")
            cells["|".join(row[k] for k in ("family", "telemetry", "method"))].append(row)
    for family, channel, method in grid:
        rows = cells["|".join((family, channel, method))]
        rows.sort(key=lambda r: r["scenario_id"])
        seed = expected_seed + 37 * (zlib.crc32(family.encode()) % 997)
        require(manifest["family_seeds"][family] == config["family_seeds"][family] == seed, "family seed mapping")
        require([r["scenario_id"] for r in rows] == [f"{family}_seed{seed}_env{i:05d}" for i in range(n)],
                "unexpected, unpaired or duplicate scenario IDs")
        rng = np.random.default_rng(seed)
        onsets = rng.integers(75, 125, size=n, dtype=np.int32)
        returns = (onsets + rng.integers(100, 150, size=n, dtype=np.int32)
                   if family == "gain_070_return" else np.full(n, -1))
        speeds = rng.choice(np.array([.4, .6], dtype=np.float32), size=n)
        for i, row in enumerate(rows):
            require(row["onset_step"] == onsets[i] and row["return_step"] == returns[i] and
                    row["speed_cmd_mps"] == speeds[i], "scenarios differ from registered RNG draws")
            require(row["gain"] == (.70 if family.startswith("gain_070") else 1.0), "wrong injected gain")
    names = read(root / "actuators.json")["names"]
    evidence = audit_cells(root, grid, cells, config, manifest, names)
    margins = registration["noninferiority_margins_loss"]
    for name, ref in (("summary.json", "heuristic_gate"), ("paired_vs_speed.json", "candidate")):
        audit_summary(read(root / name), ref, grid, cells, evidence, margins)
    pooled = {}
    for channel, method, group in itertools.product(channels, methods, ("hip", "knee")):
        records = [evidence["|".join((f, channel, method))][group] for f in families]
        available = sum(rec["available"] for rec in records)
        entry = dict(available=available, live=sum(rec["live"] for rec in records))
        entry["mean_age_ms"] = (sum(rec["mean_age_ms"] * rec["available"] for rec in records
                                    if rec["available"]) / available if available else None)
        if method in ("candidate", "candidate_velocity"):
            entry["covered"] = sum(rec["covered"] for rec in records)
            entry["coverage"] = entry["covered"] / available if available else None
        pooled["|".join((channel, method, group))] = entry
    return dict(status="verified", mode=manifest["mode"], source=manifest["source_commit"],
                cells=len(grid), episode_rows=len(grid)*n, paired_scenarios=len(families)*n,
                nominal_control_transitions=len(grid)*n*steps, hold_and_withdrawal=evidence,
                whole_horizon_joint_ticks_pooled_across_families=pooled)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--expected-source", help="Require an exact source commit.")
    parser.add_argument("--plan", type=Path, help="Compare the executed manifest to the saved frozen plan.")
    parser.add_argument("--wiring", action="store_true", help="Require a 1-episode, 8-tick wiring archive instead.")
    args = parser.parse_args()
    print(json.dumps(run(args.results, args.wiring, args.expected_source, args.plan), indent=2, allow_nan=False))
