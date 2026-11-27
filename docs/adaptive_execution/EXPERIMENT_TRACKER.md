# G1 online actuator adaptation — experiment tracker and Nov 27 completion gates

**Author:** Prasuna Chatla  
**Project:** Online Actuator Adaptation for Reliable Humanoid Locomotion  
**Checkpoint:** frozen G1 PPO, `checkpoints/nominal_s1/000202342400`  
**Evidence cutoff:** Nov 26, 2026 Pacific / Nov 27 UTC (including independently audited G3 paired CUDA navigation and two video clips)  
**Submission strategy:** credible actuator-adaptation empirical study first, genuine navigation demonstration second; no invented results.

This is the **canonical experiment checklist** for GitHub PR #51. Existing historical notes and older proposals are background, not separate executed experiments.

## 1. Experiment scoreboard

| ID | Priority | Objective | Verified evidence | State | Next release gate |
| --- | --- | --- | --- | --- | --- |
| E0 / S0 | P0 | Frozen G1/JAX GPU smoke: 2 families × 3 methods × 2 seeds | 12/12 paired rows, CUDA, full audit, GitHub result commit `9a9ed5ea3b3626d709c05d622f0e7e20ec85f876` | **PASS** | Do not rerun |
| E1 / S1a | P0 | Longer 400-step actuator fault, restoration, healthy control and sensor confound | 32/32 paired rows, 16 cells, source `89b6d0dd31020f00032330c76684c727b1f697a0`, published commit `896000b68aa9c7ea6efcd33ea33b4973ed629bb0` | **PASS: data collection** | Analyze progress and false-intervention tradeoff |
| E2 / S1b | P0 | Independent failure-mode analysis, tracking/progress balance, withdrawal denominators and confounded effort measurements | [S1b raw-data diagnosis](S1b_raw_data_diagnosis.md) independently recomputes 32 G1 development rows, n=2/cell | **PASS: descriptive analysis; confound remains unsolved** | Avoid claiming corrected identification without another method/experiment |
| E3 / G2a | P0 | Real static cylinder added to G1 MJCF, explicit physical contacts and sensors, head-camera retained | Actual G1 model compiled, 4 static-cylinder contact pairs, 4/4 evaluator-only forced contacts, 0 initial; [result evidence](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g1_g2a_model_contact_20261127/physical_contact.json) | **PASS: G1 contact model** | Next: real head-depth calibration and nominal walking route |
| E4 / G2b | P0 | Real head-camera depth frame and detector calibration against static cylinder | Genuine 96×96 G1 depth; z-depth row order calibrated on 4,183 ground pixels, 22 detections / 160 frames; [raw evidence](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_g2b_nominal_20261127) | **PASS: depth/render integration; persistence limited** | Diagnose observation loss after t105 |
| E5 / G2c | P1 | Healthy frozen G1 follows conservative real-depth detour around a physical stationary cylinder | 1,186 control steps, goal reached ~0.186 m from target, 0 contacts, 238 real H.264 frames, [independently audited evidence](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_g2c_nominal_20261127T045324Z_6d75add) | **PASS: one healthy nominal development trial** | Baseline for G3; no need to rerun |
| E6 / G3 | P1 | Same-seed real-depth navigation under hidden gain 0.70 and restoration at tick900; none versus frozen heuristic | [G3 paired evidence](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1): both reached goal, no contacts; none 1,159 ticks, heuristic 1,111 ticks; 703 corrective ticks, 1,090 released packets; [paired audit](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1/paired_independent_audit.json) | **PASS: integration + paired n=1 development** | Cannot assert general task improvement; no 72-row confirmatory study |
| E7 / video | P0 | Genuine simulator RGB videos from real G1 depth-and-obstacle navigation, baseline vs adaptation | G2c nominal 238-frame H.264; G3 [none 232-frame](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1/none/real_simulator_g3.mp4) and [heuristic 223-frame](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1/heuristic_gate/real_simulator_g3.mp4), decoder and raw-frame audit passed | **PASS: genuine paired simulator videos** | Optional side-by-side edit/captions, no invented frames or hardware claims |
| E8 / paper | P0 | Reproducible research report, equations, measured tables, limitations and AI-use disclosure | [Evidence-grounded course report draft](Course_Report_20261127_Draft.md) covers S0/S1a plus G2c; the G3 one-seed comparison and provenance can now be included | **DRAFT; FINAL PDF/REVIEW OPEN** | Add G3 section, verify disclosure and paper rendering, do not imply held-out evidence |
| E9 / expanded | P2 | Original 72-episode registered obstacle test and 128-row actuator dev | Registered only; 72-obstacle integration missing | **DEFERRED** | Do not run solely to inflate sample size near deadline |

## 2. Current empirical findings and unresolved controls

S1a gain 0.70: mean velocity tracking loss **0.085517** (no correction), **0.038328** (heuristic), **0.031378** (inverse estimator), **0.028719** (privileged oracle). Heuristic reduces loss by ~55.2% versus none, but forward progress is **2.4409 m vs 3.6219 m** (~32.6% less). This is **not** yet evidence of superior task completion. The oracle's smaller loss also came with less progress; the metric may favor slower movement and needs progress-matched analysis.

Gain 0.70 then restoration: heuristic loss **0.036317** versus none **0.061005**; corrected offset withdrew in **2 of 2** episodes, median **5.5 control ticks (~110 ms)**. Estimated inverse never withdrew in **2 of 2**. Report denominator, not only the conditional median.

Sensor-scale confound (healthy gain, effort readings ×0.8): no correction loss **0.026801**; heuristic **0.031044** (worse), healthy harmful correction **0.043697 rad**. Inverse estimator harmful correction **0.107939 rad**. This is a **negative result** on sensing ambiguity, not a safe adaptation guarantee.

All S1a episodes have **0 falls / 32**, but only two paired seeds per cell: no generalization or population-confidence claim. No physical G1 hardware experiment.

S1b CPU-only task: recompute per-scenario paired differences, report the tracking-error/progress Pareto tradeoff and the missed-restoration counts, diagnose whether the loss metric is speed-sensitive, and specify a candidate sensor-confidence abstention rule **without** silently tuning on held-out outcomes. A new model training or held-out test should not be assumed from the existing development data.

**G2a verification:** Real G1 scene check sourced from `15bfd72603d5dac6cecaf2d3cc192d0d10904291` ran in Secure A6000 pod `e9tnqh3jnvdwww`; the CPU MuJoCo model compiled with 29 actuators, two cameras, four explicit static-obstacle contact pairs, zero contacts in the start pose and four contacts under an evaluation-only overlap. The first attempt using raw XML loading failed to resolve a Menagerie asset path; using the same `AdaptiveJoystick` environment asset loader as S1a fixed it. The successful test exited 0 and the RunPod provider verified `EXITED`. **This does not establish closed-loop obstacle detection or route completion.**

## 3. November 27 (Pacific) minimum deliverable

**Go/no-go order, not automatic background jobs:**

- **G2a gate:** confirm the contact-enabled static cylinder compiles in the real pinned G1 MJCF and produces actual contact under evaluator-only controlled overlap. Stop and publish an error record if it does not.
- **G2b gate:** render a real head-depth frame, calibrate row/depth convention on ground pixels, verify that the obstacle appears at the expected measured range (ground truth used **only for evaluation**). Never pass true geom coordinates into the planner.
- **G2c gate:** healthy route with the frozen walker, record world poses, contacts, geometric clearance and actual simulator frames. If physical detour fails, preserve the failure clip and present navigation as unfinished.
- **Paper gate:** publish a completed, audit-traceable actuator-adaptation results report regardless of whether the obstacle demo succeeds, using exact source hashes and realistic limitations.
- **Video gate:** only call the video a demonstration if RGB frames were rendered from the real simulator trajectory and the MP4 was decoded/played back; otherwise provide a labeled diagnostic still and explain the blocker.

Do **not** hold the paper hostage to a 72-episode obstacle test. A rigorous real-GPU actuator-adaptation study with negative/confounded results is a defensible minimum contribution.

**G2b/G2c measured evidence (Nov 27 UTC):** Source `1fd2d776cd06b270c1b099c6ce8d474d55ec4931`, one healthy frozen G1 policy, physical cylinder and head-depth pipeline, 800 control steps with 160 actual simulator RGB frames. The head camera used the calibrated z-depth convention (4183 ground-support pixels); 22 of 160 camera frames contained detected obstacle pixels (not a recall estimate because object visibility changed with route). First confirmed obstacle at t=5, final detection at t=105. Navigator followed the detour for 102 sampled frames, then commanded a conservative stop at t=515 for lost/stale obstacle; goal not reached. Last position x≈2.11, y≈1.57 m. No cylinder contact events were recorded in the sampled frames or control-tick contact sensors. This is a **partial negative navigation outcome**, not a successful obstacle-avoidance course. Real simulator MP4 [here](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g1_g2b_nominal_20261127/real_simulator_nominal.mp4), decoded successfully and raw trajectory published at result commit `eff258a3e11129fbf31a64388229acd5ac8becb2`. The one A6000 pod was provider-confirmed EXITED after publication.

**G2c2 successful nominal route (Nov 27 UTC):** In a separate development run, source `6d75add0f2b6ceb601f90a5376818edcce10cfee`, the depth-built static map was retained after the cylinder moved out of the head-camera view; the controller checked remaining waypoint segments against an inflated obstruction disk, and still stopped on truly stale image delivery or insufficient clearance. The frozen G1 walker **reached its (5 m, 0 m) goal** at step 1185/1200, final pelvis XY (5.013 m, 0.185 m), with **zero recorded static cylinder contact events**. Independent audit of 238 sampled RGB/route frames found 22 initial detection frames, 161 normal detour, 75 depth-map-memory detour, one goal-reached, and minimum evaluator-only center-to-center cylinder distance 1.434 m (not a full-body swept-volume guarantee). [Verified real MP4](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g1_g2c_static_memory_20261127/real_simulator_nominal.mp4), 131,636 bytes, passed independent ffmpeg decode; raw trace and audit published at `foresight/experiment-results` commit `521d00438f5efdf2e1682367baceffe56e83c923`. The pod `sjeltg0z9rw6op` is independently confirmed `EXITED`. **This is one healthy, nominal-gain, simulator-only trial; no actuator adaptation in that scene, no trained VLA, no 72-episode generalization.**

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
