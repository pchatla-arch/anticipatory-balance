# G1 obstacle navigation + actuator recovery: GPU demonstration execution plan

**Status 2026-11-26:** obstacle detour prototype added. **Not yet connected to MuJoCo, not physics-tested, not a recorded video.** No pod launched. This document is a work plan, not an experimental result.

## Research claim and demonstration contract

The contribution is uncertainty-aware online actuator adaptation of a **frozen** Unitree G1 walking policy under hidden proportional-servo gain changes and delayed telemetry. Obstacle navigation is a *controlled demonstration task*, not a claimed learned perception or collision-recovery method. All obstacle positions are known to the planner; simulator pose is explicitly privileged navigation information, and is not provided to the gain estimator as the hidden fault label.

Pipeline:

`world pose + known obstacle -> deterministic waypoint planner -> (v_x, yaw rate) -> frozen G1 PPO walking actor -> nominal position targets -> gain estimator/uncertainty-aware execution adapter -> MJX servo + contacts`

Telemetry returns with delay/dropout. Gain labels remain hidden from methods. Task metrics include collisions and clearance, plus the existing actuator adaptation loss/harm/withdrawal metrics.

## Independent execution gates (sole implementation/review operator: ChatGPT; final authorization: Prasuna)

**G0: workspace and provenance** (no physics). Record git SHA, JAX/MuJoCo/Playground/Menagerie versions, frozen `nominal_s1` checkpoint and SHA-256, GPU type, driver/CUDA, current published run contents and original WP2.2 attempt status/lock. Never replay an already-started WP2.2 scientific attempt. The original claim on PR #46 and `scientific_attempt.json` must be reconciled first.

**G1: CPU route geometry**. Run `python -m unittest tests.test_obstacle_route_demo -v` on the proposed branch; test no-intersection, left/right detour, close-start rejection, waypoint advance, yaw saturation, stop. This is a *planning* gate only.

**G2: MuJoCo obstacle integration / CPU smoke**. Add a physical fixed cylinder or box geom to the flat-terrain scene (with contact enabled), expose root world position/yaw in a demo-only wrapper, feed `target_command` outputs to the existing 50 Hz G1 action interface. Verify the prop actually collides when the robot is deliberately commanded through it; verify a safe detour at nominal gain. Record actual minimum torso/foot clearance and contacts, do not rely on waypoint geometry for physical collision safety. If nominal walker cannot reliably execute the route, simplify obstacle placement or fixed waypoints and document failure. Do not modify the frozen actor. Record video from actual MJX/MuJoCo states, not illustrative frames.

**G3: recover existing WP2.2 scientific attempt**. Frozen source `4b20e3e737b790bdb853faf6fc9c099c59f48316`, heuristic `d756c37d0564837a74132c30f8e366320a6bb1a5`, 40 cells, 320 episode rows, 300 control steps, two telemetry conditions, five methods. Original single attempt must be completed/published or explicitly released after verifying its lock. Use `complete_result.py` if physics already complete. Audit `episodes.csv`, `hold_*.json`, `trace_*.npz` and corrected age/withdrawal denominators; preserve failed evidence rather than replaying.

**G4: GPU demo registration only after G2 and G3 state reconciled.** Proposed separate *development* demo experiment, not a secret second WP2.2 run. Paired seeds, 3 methods (none, previously frozen heuristic, velocity envelope), 2 hidden gain schedules (healthy; gain 0.70 with return), 2 navigation layouts (clear; single static obstacle) × 6 paired seeds = **72 episodes**, up to **1200 control steps (24 seconds)** each at 50 Hz. Frozen PPO. One camera path and one navigation planner shared across arms. Each family uses the exact same scene geometry, initial state, fault schedule and telemetry source-indexed draws. Register seed map and paths **before** running. If a stable gain-0.70 non-return family is feasible without exceeding budget, add it in a *new disclosed experiment*; do not change this 72-episode registration after seeing results.

**G5: evidence and video.** Publish immutable source/parameter/checkpoint/scene manifest; raw CSV per-episode metrics and per-step NPZ traces; 3 representative synced videos: healthy obstacle route, degraded correction-off route, degraded adapted route with restoration. Prefer two-camera synced side-by-side comparison, on-screen gain schedule (evaluation-only), confidence, command, adaptation reason, and 1/0 collision/fall status. Display failure clips, not only a cherry-picked success. Select demo clip seeds before evaluation. Make a 60–90 s narrated or captioned video; mp4 H.264 playback verified locally. Rendered scenes must be actual simulator output. Metrics: completion/fall/collision rates, minimum clearance, post-fault tracking loss, time to resume, healthy intervention, telemetry age, withdrawal after gain return, no-fault false intervention, wall-clock compute time.

**G6: independent consistency review (without Claude)**. Recompute summaries with a separate Python/NumPy verifier from raw trajectories; compare against reported figures, time-alignment, fall and collision predicates, and true-source timestamps. Record verification diff; do not invent confidence intervals. For paired metrics bootstrap over distinct scenario seeds and disclose n=6 and wide uncertainty. No hardware/generalization claims.

**G7: coursework paper**. Use the existing `paper/oaa_aa228.tex` as starting point, remove all unexecuted robust-PPO/held-out claims, include equations, primary baseline, separate auxiliary obstacle demo, limitations and AI-use disclosure. Cite explicit source hashes. Distinguish initial 256-row CPU baseline, the WP2.1 tuned 17-evaluation objective, full WP2.2 experiment if completed, and the new GPU demo if completed. No simulated evidence is a physical G1 result.

## GPU choice and budget proposal (NOT an authorization to launch)

Live RunPod CUDA 13.2 catalog on 2026-11-26: community RTX A6000 48 GB ~$0.33/h (Low stock); RTX 6000 Ada 48 GB ~$0.74/h (Low stock); L40S 48 GB ~$0.79/h (Low stock). Secure RTX 6000 Ada catalog ~$1.09/h. All existing pods EXITED at check time. The user previously suspended pod sessions in issue #31; the new request is to review options. **No billable resources may start without an explicit replacement budget authorization.**

Recommend RTX 6000 Ada (48GB) for version continuity if available, or RTX A6000 48GB for budget smoke. Proposed **$8 hard GPU compute ceiling**, at most one GPU and one live pod, with hard stop and separate per-run max times. This is a proposed new budget, not inherited authorization. Container/volume/storage costs and provisioning availability need independent confirmation. Keep publication+stop time reserved, publish partial results and mark incomplete if ceiling reached; verify provider stop acknowledgment.

## Fallback before Dec 4

If obstacle locomotion fails or time runs short, submit the already measured 256-episode CPU baseline and 17-evaluation tuning evidence, plus verified original WP2.2 results if available. A genuine obstacle navigation negative result/video can still be reported, but **do not claim collision recovery, held-out success or robust GPU results without evidence**.
