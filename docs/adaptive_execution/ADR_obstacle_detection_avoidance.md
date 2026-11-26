# ADR-005: Obstacle detection, avoidance, and recovery for the simulated G1

**Decision status:** Selected for implementation (2026-11-26). **Validated scope:** existing source interfaces and historical depth-rendering implementation inspected; the new detector and planner still require integrated unit/physics tests. **Never call this a completed perception or collision-avoidance demonstration before the GPU run.**

## Scientific objective

The experiment tests whether an uncertainty-aware actuator execution layer improves the ability of a **frozen** G1 walking policy to complete a bounded course when servo gain changes unexpectedly. The navigation pipeline must be identical across methods; the gain estimate must not read privileged fault labels. Obstacle detection is a secondary technical subsystem, with its own accuracy, latency and failure metrics. Recovery means **locomotion recovery after an actuator fault** and withdrawal after gain restoration, **not recovery from a physical collision**. Collisions should be avoided, counted and shown if they occur.

## Final selection

**Detection:** Use a G1 **head-camera rendered metric depth image**, back-projected using simulated camera intrinsics/pose, filtered for an above-ground forward blob, and tracked across multiple frames. The detector outputs `(x, y, approximate radius, confidence, source_step)`. Do not give the planner the obstacle's true MuJoCo `geom_xpos` or segmentation label; those are evaluation-only. Explicitly disclose that simulation camera extrinsics, robot odometry, and approximate obstacle size are provided, so this is *not* monocular depth learning or physical-world mapping. Use the existing `scripts/depth_detector.py` conventions, but calibrate depth units/row order on ground pixels before the registered test.

**Avoidance:** Use **perception-conditioned, inflated-circle geometric waypoint avoidance** with closed-loop heading/velocity control and conservative braking on missing/late detections, using `foresight/adaptive/obstacle_route.py` and `obstacle_depth.py`. Planner output goes to the existing **frozen G1 joystick actor**, with the same command limits across methods. A true contact-enabled obstacle must exist in MuJoCo; collision and closest-approach are evaluated from simulator contact/geometry signals *after the run*, never fed into the detector.

**Scene:** One stationary, visually and geometrically observable cylinder in flat terrain and a clear-path control. No moving obstacle, stairs, RGB semantics, whole-body reactive stepping, or manipulation in the registered deadline study.

## Choices considered and why

| Component | Alternative | Decision and reason |
|---|---|---|
| Obstacle source | **Privileged MuJoCo obstacle position / known map** | **Reject as detection claim.** This was the original PR #51 waypoint prototype and remains a useful oracle/debug mode. It bypasses perception, so results cannot claim obstacle *detection*. |
| Obstacle source | RGB detector / foundation VLM or VLA | **Reject for Dec 4.** Extra training or model dependencies, GPU inference latency, uncertain semantics, and no validated labeled static-scene benchmark. Not required to test actuator uncertainty. |
| Obstacle source | Learned monocular depth | **Reject.** Need training, calibration, and error characterization; simulator already supplies metric depth. This would conflate two research problems. |
| Obstacle source | External simulated 2-D lidar | **Defer.** Good physical baseline, but no lidar feed in the inspected G1 integration, adding MJCF and sensor polling without an existing verified stack. |
| Obstacle source | **Head-camera depth** | **Select.** Existing project has MJX-Warp depth renderer and geometric backprojection. No perception network or dataset; detection failures and latency measurable. GPU renderer must pass a separate gate. |
| Obstacle source | Depth plus ground-truth backup | **Reject for primary results.** Silent fallback leaks privileged data; allow a clearly marked oracle-only debug comparison, never same arm. |
| Avoidance | Fixed pre-scripted turn / known-location waypoints | **Reject as autonomy claim; retain as navigation smoke baseline.** Avoidance becomes invalid if obstacle location changes, even inside the simulator. |
| Avoidance | Pure potential field/reactive repulsion | **Reject for main method.** Potential local minima and oscillation can cause stop/go instability, confounding actuator-recovery comparisons. |
| Avoidance | Learned RL obstacle policy | **Defer.** Retraining or a second policy invalidates the frozen-locomotion framing and costs compute/time; more testing needed. |
| Avoidance | Full MPC / dynamic occupancy-grid planner | **Defer.** Unnecessary complexity for one fixed static obstacle; more tuning, computation, and unregistered degrees of freedom. |
| Avoidance | **Depth-confirmed inflated-circle waypoint detour** | **Select.** Deterministic, interpretable, easily tested in isolation, and compatible with the existing velocity/yaw interface. Must stop when geometry or observation is unsafe. |

## Information flow and anti-leakage contract

```
MuJoCo state --[head depth render]--> depth frame at source tick t
                depth + calibrated camera extrinsics --> obstacle pixels
                ground / self / out-of-range filtering --> connected component
                2-frame confirmation and source-age gate --> obstacle estimate
                obstacle estimate + simulated robot pose + goal --> detour planner
                waypoint heading/speed --> frozen G1 PPO policy
                nominal joint targets + causal effort telemetry --> gain belief / correction
                constrained corrected targets --> G1 position servos, MuJoCo contacts
                [evaluation only] true obstacle geom, gain onset/return, falls/collisions
```

No `adaptive_gain`, `adaptive_onset_step`, `adaptive_return_step`, or true obstacle geom may enter online detection/planning or gain-estimation logic. The use of camera pose and simulator robot pose is explicitly a **privileged localization assumption**, disclosed in the paper. The head-camera depth view must actually contain the collision object.

## Defensive control and timing

- Target sensing **10 Hz (one frame every 5 control ticks)**; frozen walker **50 Hz**, physics **500 Hz**.
- Disclose detection latency and route-replan timing. Source timestamps travel with detections. The upper bound for stale depth is initially **10 control ticks / 200 ms**; these are design parameters, *not measured safe thresholds*.
- Require at least **two spatially consistent observations** to confirm an obstacle, with <=0.5 m centroid gate. Before confirmation, stop if an apparent obstacle is inside the 1.6 m near-field.
- If camera frames are absent/stale, a recently confirmed obstacle disappears near the robot, geometry has no clearance, or no bounded route exists: **command zero forward/yaw and record reason**. A new blank frame is not the same as missing camera data.
- Planner inflates radius by an approximate **0.27 m G1 body proxy + 0.25 m margin**. This is *geometric screening*; physical foot/body clearance and contacts must still be measured. Do not present the proxy as certified collision safety.
- Bounded commands: nominal forward <=0.4 m/s for the detour, heading rate <=0.5 rad/s. Slow approach without confirmed obstacle <=0.25 m/s; no gait retraining.
- Pre-register detection/projection constants before measuring paired outcomes. Do not choose successful seeds or render only successful clips.

## Gated validation required before saying the features are 'in place'

| Gate | Required evidence | Failure response |
|---|---|---|
| D0 | Pure-NumPy synthetic point-cloud / 4-connected grouping tests; ground, robot-back, sparse noise rejected; stale/empty frames distinguishable | Fix detector; no claim of physical operation |
| D1 | G1 head-camera depth rendering from a **physical static cylinder** at fixed locations; calibrated ground-row and depth convention | Mark detector unavailable; do not silently use true object coordinates |
| D2 | Record detection rate, false alarms, range error, timing, missed frames and source-age histogram on development scenes | Report limitations; do not tune on held-out data |
| D3 | Prove geometry-clear route for two starting layouts; deliberate straight-through collision test demonstrates collider enabled | Don't proceed to actuator comparisons until collision instrumentation works |
| D4 | Healthy G1 follows route without fall/collision across registered paired seeds; save raw world pose and contact records | Simplify route openly; retain failures |
| D5 | Identical navigator across none, earlier heuristic and velocity-envelope adaptation under gain faults and restoration | Verify no privileged information leakage and compare paired outcomes |
| D6 | Raw-data independent recomputation of detection metrics, collisions, progress, falls, gain restoration and correction withdrawal | Publish failure and logs; withhold success claims |
| D7 | Actual MuJoCo RGB video (not generative frames) of representative preselected paired seed(s); include failures, annotations and source hashes | Never submit simulated narrative as measured footage |

## Decision history and rejected directions

| When / evidence | Decision | Why, impact |
|---|---|---|
| Nov 15: original OAA proposal | Frozen PPO + online bounded actuator correction; start with gain changes | Isolates adaptation of low-level execution under limited compute; retains hardware/software systems contribution |
| Nov 21–22: G0 and early authority replays | Accept G0 GPU nondeterminism within healthy-repeat spread, but G1 **Modify** | Oracle has attainable recovery; conservative fixed-speed hold envelope blocks useful correction under delayed telemetry |
| Nov 22: PR #46 | Compare correction off, oracle, tuned heuristic, speed envelope and velocity envelope under DECLARED and DECLARED_FAST | Separates fault recovery from uncertainty-induced false intervention and timing effects |
| Nov 22: AE-014, issue #49 | Correct delivered sample age and conditional withdrawal denominators | A reported 37-step withdrawal was only one of eight return episodes, not representative unconditional latency |
| Nov 25: issue #31 | Frozen heuristic at `d756c37`; temporarily CPU-only, no held-out or pod sessions | Avoid consuming budget while baseline and experimental provenance were incomplete |
| Nov 26: PR #50 | Publish and independently re-aggregate existing 256-row CPU dev benchmark | Provides real reproducible intermediate evidence; not a substituted WP2.2 velocity result |
| Nov 26: PR #51 | Add single static obstacle + deterministic waypoint routing | Achieves demonstrable walking course with minimal new learned components |
| Nov 26: new user authorization | One Community RTX 6000 Ada, <=6 GPU hours / $4.44; user requests GPU video | Explicitly supersedes no-pod restriction **only for the separately scoped PR #51 demo**, not WP2.2 reruns |
| Nov 26: this ADR | Select metric head-depth connected component + conservative detour; reject ground-truth-based perception and full MPC/VLA | Makes the obstacle detection a genuine observable subsystem with measured error/latency; preserves research causality and Dec 4 feasibility |
| Future gates | Only upgrade to a new detector/planner after archived failure with rationale and new registered scope | Prevents choosing an algorithm after seeing test data and overstating novelty |

## GPU budget and practical fallback

User authorized **one community RTX 6000 Ada 48 GB** at up to $0.74/hour, maximum six compute hours / $4.44, with automatic stop and publication. Provider-specific stock previously returned NONE despite the broad capacity matrix showing Low. Do **not** assume availability from the matrix. Code path uses no billable GPU until the Pod-specific check permits it, the watchdog is enforceable, and provenance is pinned.

If MJX-Warp head-depth rendering fails due to version incompatibility or setup, a **separately disclosed** fallback may use deterministic synthetic range detections with *pre-registered* noise/dropout/delay and no privileged truth in the runtime navigation controller. That would demonstrate planning and adaptation under an **observation model**, NOT validated rendered-depth perception. The main paper and video must label it accurately. A recorded failure of the primary camera detector is itself useful evidence.

## Open limitations

This is a **single-obstacle, flat-ground simulation**, using known camera pose and simulator localization and a nominal obstacle-radius prior. It does not establish general robot scene understanding, robust multi-obstacle navigation, collision recovery, hardware transfer, or human/animal detection. The primary claim remains online actuator uncertainty management, conditional on measured outcomes.
