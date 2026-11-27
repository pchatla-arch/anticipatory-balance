#!/usr/bin/env python3
"""Independent G3 fault-navigation paired-arm audit, no controller code imported.

Actual results and files, not successful-launch messages, determine completion.
The frozen shared seed, hidden gain schedule and camera/scene configuration
must match between correction-off and the causal heuristic arm. Simulated
fault labels are evaluation-only, never passed into the heuristic policy.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess


COMMON = ("steps", "seed", "depth_res", "capture_every", "goal_x",
          "obstacle_x", "fault_gain", "fault_onset", "fault_return", "render_video")
EXPECTED_METHODS = ("none", "heuristic_gate")


def arm(path: Path, expected_method: str):
    status = json.loads((path / "run_status.json").read_text())
    config = json.loads((path / "config.json").read_text())
    if config.get("adapter") != expected_method or status.get("adapter_method") != expected_method:
        raise ValueError("Method does not match registered arm")
    if status.get("kind") != "G3_DEPTH_ACTUATOR_DEVELOPMENT_PAIRED":
        raise ValueError("Unexpected development experiment kind")
    if status.get("result") == "FAILED_OR_INCOMPLETE":
        raise ValueError("Incomplete scientific arm")
    frames = [json.loads(x) for x in (path / "frames.jsonl").read_text().splitlines() if x.strip()]
    if not frames:
        raise ValueError("No raw real-G1 depth-frame evidence")
    steps = [int(x["step"]) for x in frames]
    if steps != sorted(set(steps)) or steps[0] != 0:
        raise ValueError("Nonunique/nonmonotonic frame steps")
    if any(b-a != config["capture_every"] for a,b in zip(steps,steps[1:])):
        raise ValueError("Missing camera-source frames")
    if len(frames) != status.get("frames_checked"):
        raise ValueError("Raw frame count differs from summary")
    contacts=sum(bool(x.get("collisions",0)) for x in frames)
    detects=sum(bool(x.get("detected")) for x in frames)
    if contacts != status.get("collision_frames") or detects != status.get("detected_frames"):
        raise ValueError("Frame-detection/contact counts inconsistent")
    last_xy = status.get("final_xy")
    dist=math.hypot(last_xy[0]-config["goal_x"],last_xy[1])
    goal=bool(status.get("goal_reached"))
    if goal and (dist>0.4 or contacts or status.get("physical_contact_ticks",0)):
        raise ValueError("False goal success under incomplete/unsafe evidence")
    if goal and status.get("result") != "G3_GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS":
        raise ValueError("Incorrect success result code")
    if not goal and status.get("result") == "G3_GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS":
        raise ValueError("Reported goal success without actual arrival")
    if goal and not status.get("confirmed_obstacle"):
        raise ValueError("Goal success without confirmed depth-derived obstacle")
    # A camera that never confirms an obstacle is a valid NEGATIVE task
    # outcome if the recorded run stopped safely; retain those failures.
    video = path / "real_simulator_g3.mp4"
    if config.get("render_video"):
        if not video.is_file():
            raise ValueError("Missing simulator H264 video")
        p=subprocess.run(["ffprobe","-v","error","-select_streams","v:0",
                          "-show_entries","stream=codec_name,nb_frames,width,height,duration",
                          "-of","json",str(video)],stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE,text=True,timeout=20)
        if p.returncode:
            raise ValueError("Video decode inspection failed")
        media=json.loads(p.stdout)["streams"][0]
        if media["codec_name"]!="h264" or int(media["nb_frames"])!=len(frames):
            raise ValueError("Video frames do not match actual simulator timestamps")
        video_proof={"codec":media["codec_name"],"frames":int(media["nb_frames"]),
                     "seconds":float(media["duration"]),"bytes":video.stat().st_size}
    else:
        video_proof=None
    if expected_method=="heuristic_gate" and not (
            status.get("causal_telemetry_evidence",{}).get("released_packets",0)>0):
        raise ValueError("No causal telemetry released to estimated controller")
    return {"method":expected_method,"source_commit":status["source_commit"],
            "config":config,"goal_reached":goal,"goal_distance_m":dist,
            "final_xy":last_xy,"control_steps_completed":status.get("ended_at_step",-1)+1,
            "collision_frames":contacts,"physical_contact_ticks":status.get("physical_contact_ticks",0),
            "detected_frames":detects,"frames":len(frames),
            "confirmed_obstacle":bool(status.get("confirmed_obstacle")),
            "adapter_nonzero_control_ticks":status.get("adapter_nonzero_control_ticks",0),
            "telemetry":status.get("causal_telemetry_evidence"),
            "last_reason":status.get("last_reason"),"video":video_proof}


def audit(none_dir: Path, heuristic_dir: Path):
    a=arm(none_dir,"none")
    b=arm(heuristic_dir,"heuristic_gate")
    if a["source_commit"] != b["source_commit"]:
        raise ValueError("Not the same frozen scientific source revision")
    for key in COMMON:
        if a["config"].get(key)!=b["config"].get(key):
            raise ValueError("Paired configuration differs on "+key)
    return {
        "kind":"G3_ONE_SEED_PAIRED_DEVELOPMENT_AUDIT",
        "source_commit":a["source_commit"],
        "fault": {"gain":a["config"]["fault_gain"],
                 "onset":a["config"]["fault_onset"],
                 "return":a["config"]["fault_return"]},
        "seed":a["config"]["seed"],
        "none":a,"heuristic_gate":b,
        "heuristic_minus_none_goal_distance_m":b["goal_distance_m"]-a["goal_distance_m"],
        "claim":"Development paired feasibility. n=1 seed. Not statistically reliable, "
                "not a 72-row registered obstacle evaluation, not trained VLA or hardware."
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--none",type=Path,required=True)
    p.add_argument("--heuristic",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.none,args.heuristic)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,sort_keys=True,default=str)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("none","heuristic_gate")},
                     indent=2,sort_keys=True))


if __name__=="__main__":
    main()
