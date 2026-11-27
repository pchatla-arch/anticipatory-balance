#!/usr/bin/env python3
"""Recompute post-hoc G3 trajectory diagnostics using only Python's standard library.

Offline:
  python reanalyze_g3_samples.py --input-dir source_frames --output audit.json

The input directory must contain healthy.frames.jsonl, none.frames.jsonl, and
gate.frames.jsonl. Without --input-dir, fetch the three immutable GitHub sources.
No simulator, GPU, pod, package installation, or video decoding is used.

These stride-5 positional diagnostics are not high-rate measured forward-command
tracking loss. In particular, planar speed versus a forward command is a
comparison of different quantities. No statistical or causal benefit is inferred.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from urllib.request import Request, urlopen

REPO = "pchatla-arch/anticipatory-balance"
G2_SOURCE = "6d75add0f2b6ceb601f90a5376818edcce10cfee"
G3_SOURCE = "7ea45d1522e3d70ee950b98eb4aee1fe35013fd5"
G2_RESULTS = "b90253190c36024e3daef9bcf4c25c8d84dc781f"
G3_RESULTS = "55dfa2d2a15790f1caa13b9460faf268081225ab"
G2_DIR = "results/g1_g2c_nominal_20261127T045324Z_6d75add"
G3_DIR = "results/g3_paired_fault_20261127_seed1_7ea45d1"
SOURCE_SPECS = {
    "healthy": (G2_SOURCE, G2_RESULTS, G2_DIR + "/frames.jsonl",
                "c3827e7cf3507e6cc5232d434c01d8fbe65e4d97"),
    "none": (G3_SOURCE, G3_RESULTS, G3_DIR + "/none/frames.jsonl",
             "18a9dd942ce35bfb7a742f2a238b6f881d0f8354"),
    "gate": (G3_SOURCE, G3_RESULTS, G3_DIR + "/heuristic_gate/frames.jsonl",
             "5fc621ef16f519b47afd90322800dfd94a956632"),
}
DT = 0.02
START, END, STRIDE = 200, 900, 5

def blob_url(commit, path):
    return f"https://github.com/{REPO}/blob/{commit}/{path}"

def get_inputs(input_dir):
    all_frames, provenance = {}, {}
    for arm, (source, result, path, expected_blob) in SOURCE_SPECS.items():
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{result}/{path}"
        if input_dir is not None:
            raw = (input_dir / (arm + ".frames.jsonl")).read_bytes()
        else:
            with urlopen(Request(raw_url, headers={"User-Agent": "G3-CPU-audit"}), timeout=45) as response:
                raw = response.read()
        actual_blob = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        if actual_blob != expected_blob:
            raise ValueError(f"{arm}: content differs from the pinned Git blob")
        frames = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        steps = [p["step"] for p in frames]
        if steps != sorted(set(steps)):
            raise ValueError(f"{arm}: steps must be unique and increasing")
        all_frames[arm] = frames
        provenance[arm] = {
            "source_commit": source,
            "result_snapshot_commit": result,
            "path": path,
            "github_url": blob_url(result, path),
            "raw_url": raw_url,
            "git_blob_sha1_verified": actual_blob,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw),
            "total_saved_frames": len(frames),
        }
    return all_frames, provenance

def analyze(frames):
    selected = [p for p in frames if START <= p["step"] <= END]
    if [p["step"] for p in selected] != list(range(START, END + 1, STRIDE)):
        raise ValueError("Missing or irregular saved samples in diagnostic window")
    segments = []
    for p, q in zip(selected, selected[1:]):
        elapsed = (q["step"] - p["step"]) * DT
        dx, dy = q["x"] - p["x"], q["y"] - p["y"]
        distance = math.hypot(dx, dy)
        forward_displacement = dx * math.cos(p["yaw"]) + dy * math.sin(p["yaw"])
        command = p["command_forward_mps"]
        segments.append({
            "dt": elapsed, "distance": distance,
            "planar_speed": distance / elapsed,
            "forward_projected_speed": forward_displacement / elapsed,
            "command": command,
        })
    duration = math.fsum(s["dt"] for s in segments)
    def time_mean(field):
        return math.fsum(s[field] * s["dt"] for s in segments) / duration
    def error_summary(field):
        errors = [(s[field] - s["command"], s["dt"]) for s in segments]
        return {
            "mean_signed_mps": math.fsum(e * dt for e, dt in errors) / duration,
            "mae_mps": math.fsum(abs(e) * dt for e, dt in errors) / duration,
            "rmse_mps": math.sqrt(math.fsum(e * e * dt for e, dt in errors) / duration),
        }
    mean_command = time_mean("command")
    mean_planar = time_mean("planar_speed")
    final = frames[-1]
    return {
        "selected_samples": len(selected),
        "segments": len(segments),
        "window_elapsed_seconds": duration,
        "sampled_path_length_m": math.fsum(s["distance"] for s in segments),
        "mean_planar_speed_mps": mean_planar,
        "mean_start_heading_projected_forward_speed_mps": time_mean("forward_projected_speed"),
        "mean_start_sample_forward_command_mps": mean_command,
        "planar_mean_over_forward_command_pct_DIFFERENT_QUANTITIES":
            100.0 * (mean_planar / mean_command - 1.0),
        "planar_minus_forward_command_diagnostic_DIFFERENT_QUANTITIES":
            error_summary("planar_speed"),
        "sampled_start_heading_forward_projection_error_PROXY":
            error_summary("forward_projected_speed"),
        "max_saved_y_m": max(p["y"] for p in selected),
        "last_saved_frame": {
            "step": final["step"],
            "xy_m": [final["x"], final["y"]],
            "goal_distance_m_from_saved_frame_NOT_run_status":
                math.hypot(final["x"] - 5.0, final["y"]),
        },
    }

def report(frames, provenance):
    pairing = {}
    for step in (195, 200):
        point = {arm: next(p for p in data if p["step"] == step)
                 for arm, data in frames.items()}
        def separation(a, b):
            return math.hypot(point[a]["x"] - point[b]["x"],
                              point[a]["y"] - point[b]["y"])
        pairing[str(step)] = {
            "none_vs_gate_position_distance_m": separation("none", "gate"),
            "none_vs_healthy_position_distance_m": separation("none", "healthy"),
        }
    return {
        "schema_version": 1,
        "analysis_type": "retrospective sampled-trajectory diagnostic",
        "title": "Reliable Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Adaptive Execution",
        "provenance": provenance,
        "method": {
            "independent_seeds": 1,
            "control_period_seconds": DT,
            "saved_stride_control_ticks": STRIDE,
            "selected_saved_ticks_inclusive": [START, END],
            "analyzed_intervals": "[200,205), ... , [895,900); no post-recovery interval",
            "duration_seconds": (END - START) * DT,
            "path_length": "sum hypot(x_next-x, y_next-y)",
            "planar_speed": "segment XY displacement norm / segment elapsed time",
            "forward_projection": "(dx*cos(yaw_start)+dy*sin(yaw_start))/elapsed_time",
            "command": "forward command at each segment's starting saved sample",
            "aggregation": "elapsed-time-weighted means, MAE and RMSE across 140 segments",
            "detour": "maximum saved world-frame y coordinate among the 141 selected samples",
            "pairing": "Euclidean XY separation at the same pre-step saved tick",
            "runner_source": blob_url(G3_SOURCE, "scripts/eval_g3_depth_actuator.py"),
        },
        "results": {arm: analyze(data) for arm, data in frames.items()},
        "pre_fault_pairing": pairing,
        "interpretation_limits": [
            "One seed: no statistical significance or reliability claim.",
            "Healthy reference uses a different source commit from the two G3 arms.",
            "Same seed is approximately paired, not exact replay; pre-fault drift is observed.",
            "Planar speed and commanded forward velocity are different quantities.",
            "A ratio of their means is not forward-command tracking loss or mean tracking error.",
            "Heading projection is a post-hoc finite-displacement proxy, not high-rate measured body velocity.",
            "Commands and heading can change within the five-tick interval; the start sample approximation cannot resolve this.",
            "Saved trajectories omit per-tick g_hat, requested/effective offset and measured forward velocity.",
            "Faster goal arrival and final goal-threshold distance do not alone establish better control.",
            "Final saved frame pose precedes its simulator step; run_status.final_xy is after the final step.",
            "Video decode metadata was read from published audits; this CPU script does not decode videos.",
        ],
        "status_and_source_notes_NOT_recomputed_from_frames": {
            "paired_audit_url": blob_url(G3_RESULTS, G3_DIR + "/paired_independent_audit.json"),
            "g2c_alias": {
                "path": "results/g1_g2c_static_memory_20261127/frames.jsonl",
                "same_git_blob_as_healthy": "c3827e7cf3507e6cc5232d434c01d8fbe65e4d97",
                "interpretation": "Duplicate saved trajectory, not an additional independent trial.",
            },
            "adapter": {
                "status_field": "causal_telemetry_evidence.gain_estimate_leg_mean",
                "published_value": 0.9374097585678101,
                "meaning": "Final RLS g_hat averaged over leg joints, evaluated after recovery; not a time/run mean.",
                "definition_url": blob_url(G3_SOURCE, "foresight/adaptive/single_causal_controller.py"),
            },
            "contact": {
                "status_fields": ["collision_frames", "physical_contact_ticks", "contact_definition"],
                "coverage": ["torso_impact_geom", "pelvis_impact_geom", "left_thigh", "right_thigh"],
                "interpretation": "Zero recorded cylinder contacts under these explicit pairs; not full-body collision certification.",
                "definition_url": blob_url(G3_SOURCE, "foresight/adaptive/obstacle_scene_xml.py"),
            },
            "fall": {
                "runner_check": "After env.step: bool(state.done) or bool(state.info['fell'])",
                "runner_stop_reason": "terminated_or_fell",
                "status_fields": ["last_reason", "goal_reached"],
                "observed_g3_last_reason_both_arms": "goal_reached",
                "interpretation": "The combined termination/fall check did not stop either successful rollout. No separate fall count, per-tick fall flag, detector validation or termination-cause disaggregation is supplied by these saved frames.",
            },
            "existing_claude_review": "https://github.com/" + REPO + "/pull/51#issuecomment-7210669939",
            "review_reconciliation": [
                "Path lengths and pre-fault pairing drift reproduce.",
                "Maximum-y values differ from the rounded values in the review; use this defined recomputation.",
                "Do not promote planar/forward mean ratios into claims of being within 3% forward-command tracking.",
                "Final gain estimate cannot be treated as the fault-window or episode mean.",
                "The review's smoke-versus-healthy 15 cm comparison was not checked here.",
            ],
        },
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, help="Offline three-file source directory")
    parser.add_argument("--output", type=Path, default=Path("sampled_g3_diagnostics.json"))
    args = parser.parse_args()
    frames, provenance = get_inputs(args.input_dir)
    output = report(frames, provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}; verified 3 pinned Git blobs; no simulator or GPU used.")

if __name__ == "__main__":
    main()

