#!/usr/bin/env python3
"""Diagnostic real-G1 nominal obstacle-route pilot with actual head depth and contacts.

D2/D3 exploratory evaluation ONLY. The control loop never receives true obstacle
world coordinates; those are used to evaluate detection/contact afterwards.
A failure to calibrate metric depth, maintain fresh frames, compile the static
cylinder or render video is recorded as a failed gate, never hidden by a
ground-truth fallback.

Example on a pinned G1/JAX pod:
  PYTHONPATH=. python scripts/eval_g1_obstacle_nominal.py \
      --out /workspace/g1_scene_diagnostics/nominal_01 --steps 800

Produces summary.json, frames.jsonl, head_depth_first.npy,
optional head_depth_first.png and an actual rendered scene H.264 MP4.
This is NOT paired actuator adaptation, a VLA or collision recovery.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time


def ground_score(depth, cam_pos, cam_rot, fovy, ray, flip):
    import numpy as np
    from scripts.depth_detector import pixel_dirs, backproject
    directions = pixel_dirs(depth.shape[0], fovy, flip)
    valid = np.isfinite(depth) & (depth > 0.05) & (depth < 50.0)
    xyz = backproject(np.where(valid, depth, 1.0), cam_pos, cam_rot,
                      directions, ray)
    support = valid & (np.abs(xyz[..., 2]) < 0.08)
    return int(support.sum())


def calibrate(depth, cam_pos, cam_rot, fovy):
    options = []
    for ray in (False, True):
        for flip in (False, True):
            n = ground_score(depth, cam_pos, cam_rot, fovy, ray, flip)
            options.append((n, ray, flip))
    best = max(options, key=lambda p: p[0])
    if best[0] < 12:
        raise RuntimeError(
            "No trustworthy ground-pixel depth calibration; "
            "stop rather than infer obstacle from uncalibrated pixels")
    return {"ground_pixels": best[0], "depth_is_ray": best[1],
            "flip_rows": best[2],
            "candidate_ground_pixel_counts": [
                {"count": n, "ray": ray, "flip": flip}
                for n, ray, flip in options]}


def collision_count(model, data, obstacle_geom_id):
    return sum(obstacle_geom_id in (int(data.contact[i].geom1),
                                    int(data.contact[i].geom2))
               for i in range(data.ncon))


def encode_real_frames(frames, output, fps):
    if not frames:
        raise RuntimeError("No actual simulator frames to encode")
    width = frames[0].shape[1]
    height = frames[0].shape[0]
    if shutil.which("ffmpeg"):
        argv = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v",
                f"{width}x{height}", "-r", str(fps), "-i", "-",
                "-an", "-c:v", "libx264", "-crf", "23",
                "-pix_fmt", "yuv420p", str(output)]
        payload = b"".join(im.tobytes() for im in frames)
        process = subprocess.run(argv, input=payload, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE, timeout=120)
        if process.returncode:
            raise RuntimeError("ffmpeg failed (exit " + str(process.returncode) + ")")
    else:
        import imageio.v2 as imageio
        imageio.mimsave(str(output), frames, fps=fps, codec="libx264")
    if not output.is_file() or output.stat().st_size < 1000:
        raise RuntimeError("Rendered MP4 was not written or is empty")
    return int(output.stat().st_size)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--depth-res", type=int, default=96)
    parser.add_argument("--capture-every", type=int, default=5)
    parser.add_argument("--render-video", action="store_true")
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--obstacle-x", type=float, default=2.8)
    args = parser.parse_args(argv)
    if not (200 <= args.steps <= 1200 and 48 <= args.depth_res <= 160
            and args.capture_every in (2, 5, 10)
            and 4.0 <= args.goal_x <= 7.0
            and 2.3 <= args.obstacle_x <= 3.5):
        parser.error("diagnostic parameters outside registered safe range")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("output must be new/empty: never overwrite a prior attempt")
    args.out.mkdir(parents=True, exist_ok=True)
    import numpy as np
    import jax
    import jax.numpy as jp
    import mujoco
    from foresight.adaptive.actuator_env import AdaptiveJoystick
    from foresight.envs.g1_impact import default_config
    from foresight.adaptive.obstacle_depth import ObstacleNavigator, detect_from_depth
    from foresight.adaptive.obstacle_scene_xml import OBSTACLE_GEOM
    from foresight.eval import rollout as R

    source = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                     text=True).strip()
    status = {"kind": "G1_OBSTACLE_NOMINAL_PILOT",
              "result": "FAILED_OR_INCOMPLETE",
              "registered_control": "frozen_nominal_joystick",
              "sensing": "real_Renderer_head_depth",
              "obstacle_ground_truth_available_to_policy": False,
              "gain_fault": False,
              "source_commit": source, "started_utc": datetime.now(timezone.utc).isoformat(),
              "hardware": [str(d) for d in jax.devices()],
              "frames": 0, "collision_frames": 0,
              "contact_definition": "evaluator-only explicit MJCF contact pairs"}
    (args.out / "config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True, default=str) + "\n")
    (args.out / "run_status.json").write_text(json.dumps(status, indent=2) + "\n")
    t0 = time.monotonic()
    frames = []
    log = []
    try:
        cfg = default_config()
        cfg.episode_length = max(args.steps + 25, 500)
        cfg.impact.reset_perturb = 0.0
        cfg.impact.hit_time_range = [1000.0, 1001.0]  # no launched projectile
        cfg.impact.resample_on_autoreset = False
        cfg.impact.static_obstacle.enabled = True
        cfg.impact.static_obstacle.x = args.obstacle_x
        cfg.impact.static_obstacle.y = 0.0
        env = AdaptiveJoystick(config=cfg, config_overrides={"impl": "jax"})
        model = env.mj_model
        obstacle_id = model.geom(OBSTACLE_GEOM).id
        camera_id = model.camera("head_cam").id
        from foresight.adaptive.obstacle_scene_xml import COLLISION_TARGETS
        contact_sensor_indices = [
            int(model.sensor_adr[model.sensor("g1_obstacle_" + tag + "_found").id])
            for _, tag in COLLISION_TARGETS]
        if not any(obstacle_id in (int(model.pair_geom1[i]),
                                  int(model.pair_geom2[i]))
                   for i in range(model.npair)):
            raise RuntimeError("No contact-enabled physical obstacle in real G1 model")
        policy = R.load_policy("checkpoints/nominal_s1/000202342400")
        step_fn = jax.jit(env.step)
        act_fn = jax.jit(policy)
        state = env.reset(jax.random.PRNGKey(args.seed))
        rng = jax.random.PRNGKey(args.seed + 10000)
        cpu_data = mujoco.MjData(model)
        # A real MuJoCo head camera renders physical geometry from the current
        # GPU/host simulator state; position ground truth is never fed to planning.
        depth_renderer = mujoco.Renderer(model, height=args.depth_res,
                                         width=args.depth_res)
        depth_renderer.enable_depth_rendering()
        video_renderer = None
        if args.render_video:
            video_renderer = mujoco.Renderer(model, height=240, width=320)
            free_cam = mujoco.MjvCamera()
            free_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
            free_cam.lookat[:] = [2.5, 0.0, 0.9]
            free_cam.distance = 6.8
            free_cam.azimuth = 95
            free_cam.elevation = -25
        nav = ObstacleNavigator(goal=(args.goal_x, 0.0))
        calibration = None
        reason = "startup"
        status["physical_contact_ticks"] = 0
        status["confirmed_obstacle"] = False
        for t in range(args.steps):
            data = state.data
            pos = np.asarray(data.xpos[env._pelvis_body])
            rot = np.asarray(data.xmat[env._pelvis_body]).reshape(3, 3)
            yaw = float(math.atan2(rot[1, 0], rot[0, 0]))
            if t % args.capture_every == 0:
                cpu_data.qpos[:] = np.asarray(data.qpos)
                cpu_data.qvel[:] = np.asarray(data.qvel)
                mujoco.mj_forward(model, cpu_data)
                headpos = np.asarray(cpu_data.cam_xpos[camera_id]).copy()
                headrot = np.asarray(cpu_data.cam_xmat[camera_id]).reshape(3, 3).copy()
                depth_renderer.update_scene(cpu_data, camera="head_cam")
                depth = np.asarray(depth_renderer.render(), dtype=np.float32).copy()
                if calibration is None:
                    calibration = calibrate(
                        depth, headpos, headrot, model.cam_fovy[camera_id])
                    np.save(args.out / "head_depth_first.npy", depth)
                    (args.out / "depth_calibration.json").write_text(
                        json.dumps(calibration, indent=2) + "\n")
                detected = detect_from_depth(
                    depth, headpos, headrot, model.cam_fovy[camera_id],
                    tuple(pos[:2]), yaw, t,
                    depth_is_ray=calibration["depth_is_ray"],
                    flip_rows=calibration["flip_rows"], radius_m=0.30)
                nav.ingest(detected, observed_step=t)
                status["confirmed_obstacle"] = nav.obstacle is not None
                collisions = collision_count(model, cpu_data, obstacle_id)
                status["collision_frames"] += int(collisions > 0)
                log.append({
                    "step": t, "x": float(pos[0]), "y": float(pos[1]),
                    "yaw": yaw, "collisions": collisions,
                    "detected": detected is not None,
                    "estimated_xy": list(detected.center_xy) if detected else None,
                    "estimate_confidence": detected.confidence if detected else None,
                    "ground_truth_distance_evaluation_only":
                        float(np.linalg.norm(pos[:2] - np.array([args.obstacle_x, 0.0]))),
                })
                if video_renderer is not None:
                    video_renderer.update_scene(cpu_data, camera=free_cam)
                    frames.append(np.asarray(video_renderer.render()).copy())
                    status["frames"] = len(frames)
            if t >= 30 and nav.obstacle is None:
                reason = "stop_no_confirmed_depth_obstacle"
                break
            vx, wz, reason = nav.command(float(pos[0]), float(pos[1]), yaw, t)
            if t % args.capture_every == 0 and log:
                log[-1]["command_forward_mps"] = vx
                log[-1]["command_yaw_rps"] = wz
                log[-1]["navigator_reason"] = reason
            cmd = jp.array([vx, 0.0, wz], dtype=jp.float32)
            # The frozen G1 pilot is a SINGLE environment (unbatched):
            # rollout._with_command uses [:, CMD_SLICE] for vmapped benches
            # and therefore raises IndexError for the (features,) state here.
            obs = dict(state.obs)
            if obs["state"].ndim != 1 or cmd.shape != (3,):
                raise ValueError("Expected one unbatched G1 observation and 3D command")
            obs["state"] = obs["state"].at[R.CMD_SLICE].set(cmd)
            info = dict(state.info)
            info["command"] = cmd
            info["command_nominal"] = cmd
            state = state.replace(info=info, obs=obs)
            rng, key = jax.random.split(rng)
            action, _ = act_fn(obs, key)
            state = step_fn(state, action)
            live_sensor = np.asarray(state.data.sensordata)[contact_sensor_indices]
            if np.any(live_sensor > 0):
                status["physical_contact_ticks"] += 1
                reason = "stop_actual_cylinder_contact"
                break
            if bool(np.asarray(state.done)) or bool(np.asarray(state.info["fell"])):
                reason = "terminated_or_fell"
                break
            if reason == "goal_reached":
                break
        status["ended_at_step"] = t
        status["last_reason"] = reason
        status["detected_frames"] = sum(int(v["detected"]) for v in log)
        status["frames_checked"] = len(log)
        status["goal_reached"] = reason == "goal_reached"
        status["final_xy"] = [float(v) for v in np.asarray(state.data.xpos[env._pelvis_body])[:2]]
        status["elapsed_wall_s"] = time.monotonic() - t0
        (args.out / "frames.jsonl").write_text(
            "".join(json.dumps(v, sort_keys=True) + "\n" for v in log))
        if video_renderer is not None:
            mp4 = args.out / "real_simulator_nominal.mp4"
            status["rendered_video_bytes"] = encode_real_frames(
                frames, mp4, max(1, round(50 / args.capture_every)))
        status["result"] = "COMPLETED_DIAGNOSTIC_NOT_VALIDATED_AVOIDANCE"
        if (status["goal_reached"] and status["confirmed_obstacle"]
                and status["collision_frames"] == 0
                and status["physical_contact_ticks"] == 0):
            status["result"] = "GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS"
    except BaseException as exc:
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)[:500]
        raise
    finally:
        status["completed_utc"] = datetime.now(timezone.utc).isoformat()
        (args.out / "run_status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n")
        try:
            if "depth_renderer" in locals(): depth_renderer.close()
            if "video_renderer" in locals() and video_renderer is not None:
                video_renderer.close()
        except Exception:
            pass
        print(json.dumps(status, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
