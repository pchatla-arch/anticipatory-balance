# G1 online actuator adaptation — experiment tracker and Nov 27 completion gates

**Author:** Prasuna Chatla  
**Project:** Online Actuator Adaptation for Reliable Humanoid Locomotion  
**Checkpoint:** frozen G1 PPO, `checkpoints/nominal_s1/000202342400`  
**Evidence cutoff:** Nov 26, 2026 (Pacific; live results through Nov 27 UTC)  
**Submission strategy:** credible actuator-adaptation empirical study first, genuine navigation demonstration second; no invented results.

This is the **canonical experiment checklist** for GitHub PR #51. Existing historical notes and older proposals are background, not separate executed experiments.

## 1. Experiment scoreboard

| ID | Priority | Objective | Verified evidence | State | Next release gate |
| --- | --- | --- | --- | --- | --- |
| E0 / S0 | P0 | Frozen G1/JAX GPU smoke: 2 families × 3 methods × 2 seeds | 12/12 paired rows, CUDA, full audit, GitHub result commit `9a9ed5ea3b3626d709c05d622f0e7e20ec85f876` | **PASS** | Do not rerun |
| E1 / S1a | P0 | Longer 400-step actuator fault, restoration, healthy control and sensor confound | 32/32 paired rows, 16 cells, source `89b6d0dd31020f00032330c76684c727b1f697a0`, published commit `896000b68aa9c7ea6efcd33ea33b4973ed629bb0` | **PASS: data collection** | Analyze progress and false-intervention tradeoff |
| E2 / S1b | P0 | Independent failure-mode analysis, progress/loss balance, truthful withdrawal denominators | S1a summary and raw CSV available, n=2 per cell | **ANALYSIS OPEN** | Recompute per-seed contrasts and decide if a repeat is needed |
| E3 / G2a | P0 | Real static cylinder added to G1 MJCF, explicit physical contacts and sensors, head-camera retained | Pure-Python XML generator implemented, model compile/contacts not yet verified on actual G1 | **CODE IN PROGRESS** | CPU physics fixture passes; pinned G1 compile+forced-contact smoke |
| E4 / G2b | P0 | Real head-camera depth frame and detector calibration against static cylinder | Depth detector and backprojection primitives unit-tested; not tested on stationary collider | **NOT EXECUTED** | Head-camera frame rendered, floor-depth convention calibrated, detection evidence saved |
| E5 / G2c | P1 | Healthy frozen G1 joystick follows conservative detour around physical cylinder | Known-map waypoint planning only, unit-tested; no contact-enabled G1 route | **NOT EXECUTED** | Scene + navigation integrated and held to conservative stopping rules |
| E6 / G3 | P1 | Paired healthy/fault navigation with identical perception/route (none vs heuristic) | No integrated obstacle-and-actuator runner | **NOT EXECUTED** | Only after E5 healthy route works |
| E7 / video | P0 | Authentic simulator RGB video and depth/status overlay | No video from the new online actuator-adaptation obstacle experiment | **NOT EXECUTED** | Actual simulator frames saved and video playback verified |
| E8 / paper | P0 | Reproducible course report, equations, literature, limitations and AI disclosure | Proposal and extensive design draft exist; S0/S1a outputs can be cited | **INCOMPLETE** | Use verified science; do not claim unrun demo success |
| E9 / expanded | P2 | Original 72-episode registered obstacle test and 128-row actuator dev | Registered only; 72-obstacle integration missing | **DEFERRED** | Do not run solely to inflate sample size near deadline |

## 2. Current empirical findings and unresolved controls

S1a gain 0.70: mean velocity tracking loss **0.085517** (no correction), **0.038328** (heuristic), **0.031378** (inverse estimator), **0.028719** (privileged oracle). Heuristic reduces loss by ~55.2% versus none, but forward progress is **2.4409 m vs 3.6219 m** (~32.6% less). This is **not** yet evidence of superior task completion. The oracle's smaller loss also came with less progress; the metric may favor slower movement and needs progress-matched analysis.

Gain 0.70 then restoration: heuristic loss **0.036317** versus none **0.061005**; corrected offset withdrew in **2 of 2** episodes, median **5.5 control ticks (~110 ms)**. Estimated inverse never withdrew in **2 of 2**. Report denominator, not only the conditional median.

Sensor-scale confound (healthy gain, effort readings ×0.8): no correction loss **0.026801**; heuristic **0.031044** (worse), healthy harmful correction **0.043697 rad**. Inverse estimator harmful correction **0.107939 rad**. This is a **negative result** on sensing ambiguity, not a safe adaptation guarantee.

All S1a episodes have **0 falls / 32**, but only two paired seeds per cell: no generalization or population-confidence claim. No physical G1 hardware experiment.

S1b CPU-only task: recompute per-scenario paired differences, report the tracking-error/progress Pareto tradeoff and the missed-restoration counts, diagnose whether the loss metric is speed-sensitive, and specify a candidate sensor-confidence abstention rule **without** silently tuning on held-out outcomes. A new model training or held-out test should not be assumed from the existing development data.

## 3. November 27 (Pacific) minimum deliverable

**Go/no-go order, not automatic background jobs:**

- **G2a gate:** confirm the contact-enabled static cylinder compiles in the real pinned G1 MJCF and produces actual contact under evaluator-only controlled overlap. Stop and publish an error record if it does not.
- **G2b gate:** render a real head-depth frame, calibrate row/depth convention on ground pixels, verify that the obstacle appears at the expected measured range (ground truth used **only for evaluation**). Never pass true geom coordinates into the planner.
- **G2c gate:** healthy route with the frozen walker, record world poses, contacts, geometric clearance and actual simulator frames. If physical detour fails, preserve the failure clip and present navigation as unfinished.
- **Paper gate:** publish a completed, audit-traceable actuator-adaptation results report regardless of whether the obstacle demo succeeds, using exact source hashes and realistic limitations.
- **Video gate:** only call the video a demonstration if RGB frames were rendered from the real simulator trajectory and the MP4 was decoded/played back; otherwise provide a labeled diagnostic still and explain the blocker.

Do **not** hold the paper hostage to a 72-episode obstacle test. A rigorous real-GPU actuator-adaptation study with negative/confounded results is a defensible minimum contribution.

## 4. Budget and safety

Nov 20–27 RunPod posted pod charges at the last check: about **$2.31**, though posted billing may lag recently ended pods. The original focused GPU demonstration budget is **$4.44 cumulative**; other early-November experiments used a separate historical scope and should not be silently attributed to this demo. Preferred one Secure RTX A6000 at **≤$0.69/h** with CUDA host 13.2, no concurrent project pod, visible RunPod ID, GitHub write access checked before physics. A pod-local time monitor is not a guaranteed provider stop; independently verify `EXITED`, and never enter GitHub/RunPod secrets in a tracked file or chat.

## 5. Direct evidence

- S0: [12-row CUDA smoke](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_course_dev_20261127T024308Z_104a34)
- S1a: [32-row long-horizon GPU screen](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_course_dev_20261127T032240Z_5fc7dd)
- [S1a queue registration](experiment_queue_G1_S1A_20261126.json) (completed but original queue record retained as preregistered history)
- [Depth/collision design decision](ADR_obstacle_detection_avoidance.md)
- [G1 GPU obstacle demo plan](G1_obstacle_recovery_demo_plan.md)
- [PR #51](https://github.com/pchatla-arch/anticipatory-balance/pull/51)

## 6. Evidence rule

Every scoreboard status advances only after test output and raw artifacts have been saved, reviewed, and cited. An unexecuted script, passing synthetic unit tests, available GPU, successful GitHub authentication, or published draft paper **does not** establish physical obstacle avoidance, navigation under fault, safe adaptation, a trained VLA, or successful video generation.
