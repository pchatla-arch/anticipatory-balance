#!/usr/bin/env python3
"""Independent post-hoc audit of G1 depth-navigation diagnostic evidence.

This module reads *only* reported outputs and actual recorded frame-level data.
It never supplies simulator ground truth to online planning or control.
Usage:
  python scripts/audit_g1_depth_navigation.py --run PATH --out PATH/audit.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import subprocess


def audit(run_dir: Path):
    status = json.loads((run_dir / "run_status.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    frames_file = run_dir / "frames.jsonl"
    if not frames_file.exists():
        raise ValueError("No frame-level simulator evidence; no completion claim")
    frames = [json.loads(s) for s in frames_file.read_text().splitlines() if s.strip()]
    if not frames:
        raise ValueError("Empty simulator frame evidence")
    steps = [int(row["step"]) for row in frames]
    if steps != sorted(set(steps)):
        raise ValueError("Duplicate, unordered, or nonmonotonic frame source steps")
    if steps[0] != 0:
        raise ValueError("First sampled frame must be simulator tick zero")
    if any(b-a != int(config["capture_every"]) for a, b in zip(steps, steps[1:])):
        raise ValueError("Missing or irregular source frame cadence")
    for row in frames:
        xyz = [float(row["x"]), float(row["y"]), float(row["yaw"])]
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError("Nonfinite robot trajectory state")
    observed = sum(bool(r.get("detected")) for r in frames)
    hits = sum(int(r.get("collisions", 0)) > 0 for r in frames)
    reasons = Counter(str(r.get("navigator_reason", "unknown")) for r in frames)
    if status.get("frames_checked") != len(frames):
        raise ValueError("Summary frame count mismatches raw frames")
    if status.get("detected_frames") != observed:
        raise ValueError("Summary detector count mismatches raw frames")
    if status.get("collision_frames") != hits:
        raise ValueError("Summary collision count mismatches raw frames")
    goal = [float(config["goal_x"]), 0.0]
    last = frames[-1]
    final_xy = list(map(float, status["final_xy"]))
    actual_goal_dist = math.hypot(final_xy[0]-goal[0], final_xy[1]-goal[1])
    if status.get("goal_reached"):
        if actual_goal_dist > 0.4 or hits or status.get("physical_contact_ticks", 0):
            raise ValueError("Goal-success label contradicts position or contact data")
        if status.get("result") != "GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS":
            raise ValueError("Goal-success state has inconsistent declared result")
    if status.get("result") == "GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS" and not status.get("goal_reached"):
        raise ValueError("Successful result without goal-reached evidence")
    full_video = run_dir / "real_simulator_nominal.mp4"
    video = {"present": full_video.is_file()}
    if full_video.is_file():
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,nb_frames,width,height,duration",
             "-of", "json", str(full_video)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
        if proc.returncode:
            raise ValueError("Cannot decode actual simulator video with ffprobe")
        streams = json.loads(proc.stdout).get("streams", [])
        if len(streams) != 1 or streams[0].get("codec_name") != "h264":
            raise ValueError("Unexpected simulator video codec")
        frame_count = int(streams[0]["nb_frames"])
        if frame_count != len(frames) or frame_count != status.get("frames"):
            raise ValueError("Encoded H264 frame count mismatch")
        video.update({"codec": "h264", "frame_count": frame_count,
                      "width": int(streams[0]["width"]),
                      "height": int(streams[0]["height"]),
                      "duration_seconds": float(streams[0]["duration"]),
                      "size_bytes": full_video.stat().st_size})
    return {
        "source_commit": status["source_commit"],
        "status": status.get("result"),
        "goal_reached": bool(status.get("goal_reached")),
        "observed_goal_distance_m": actual_goal_dist,
        "frame_count": len(frames), "first_source_step": steps[0],
        "last_source_step": steps[-1], "detector_positive_frames": observed,
        "detector_fraction": observed / len(frames),
        "contact_positive_frames": hits,
        "physical_contact_ticks": int(status.get("physical_contact_ticks", 0)),
        "navigator_reasons": dict(sorted(reasons.items())),
        "video": video,
        "calibration": json.loads((run_dir / "depth_calibration.json").read_text())
        if (run_dir / "depth_calibration.json").exists() else None,
        "scope": "independent post-hoc integrity check; no hardware/safety guarantee",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.run)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
