# Draft research framing — Stanford AA228/CS238, December 4, 2026

**Working title:** *When to Correct: Reliable Language-Guided Humanoid Navigation Under Actuator Uncertainty*

**Status:** draft outline only. Measured CPU evidence is sourced from PR #50; end-to-end language/depth/MuJoCo experiments have not yet been completed. Do not use past-tense claims about any unrun evaluation. This draft **does not** assert a trained VLA.

## Proposed abstract (update after integrated results)

Humanoid robots must interpret task goals while maintaining locomotion under changing actuator dynamics and imperfect telemetry. The problem is challenging because corrective intervention can improve performance during a genuine gain fault but can also introduce errors when a healthy actuator is misidentified. This project investigates a hierarchical execution system combining a frozen Unitree G1 walking policy, delayed actuator telemetry, online gain estimation, and uncertainty-constrained target correction. A bounded language interface selects perceived goals; head-camera depth provides obstacle estimates for a deterministic local navigator. The language and navigation stack is treated as an auxiliary integration benchmark rather than a newly trained vision-language-action model. An existing 256-episode CPU development baseline found that an earlier heuristic gate reduced post-onset locomotion loss relative to correction-off by 39.3% in a sustained gain-0.70 scenario and 36.3% in a gain-restoration scenario under declared telemetry. However, that earlier heuristic withdrew correction in only one of eight restoration episodes, motivating explicit release and observation-age analyses. A separately frozen development heuristic was selected from 17 objective evaluations. **[INSERT ONLY after verification: complete WP2.2 velocity-envelope comparisons and true G1 navigation experiment results, obstacle detection and collision outcomes, paired confidence intervals, measured latencies, and video link.]** Limitations include development-only evidence, known simulator localization, a single static obstacle and no physical-robot validation.

## Hypotheses and acceptance rules

- **H1 — Faulted locomotion:** Relative to frozen-policy correction-off, the uncertainty-aware execution layer reduces fixed-horizon velocity error under gain reduction, with paired scenario results and reported falls. Do not claim H1 if not shown for the *final* controller.
- **H2 — Healthy conservatism:** Bound false correction on healthy or sensor-confounded actuators; report intervention rather than hiding the cost in a weighted aggregate.
- **H3 — Timeliness:** Observation source age and velocity-informed hold bounds govern allowable correction; compare declared and fast transport, recording causality, coverage and withdrawal counts.
- **H4 — Navigation integration (secondary):** With exactly the same detected-goal and obstacle-navigator pipeline across methods, adaptation changes route completion/fall/collision behavior under a hidden gain fault.
- **H5 — Language interface (secondary):** A strictly bounded model/grammar goal cannot emit illegal actuator commands, and unsupported/unknown/stale targets trigger a safe stop. An actual VLM accuracy claim requires execution of an image-and-instruction model; the deterministic parser is not VLM evidence.

## Figure/Table plan with source of truth

1. **Architecture:** head depth / goal registry / task parser / navigator / frozen PPO / belief-set adapter / G1 MJX plant. Include rates, causality and privileged localization assumptions.
2. **Existing evidence:** correction-off vs pre-freeze heuristic vs oracle loss in gain 0.70 and gain restoration (256-episode development CPU CSV and PR #50 independent reduction). Clearly caption as *earlier heuristic*.
3. **Tuning:** 17 evaluated objective components and chosen thresholds; selected objective -17.5%, but locomotion loss rose slightly; no claim of unseen generalization.
4. **Primary future result:** PR #46 matched speed vs velocity envelope and frozen heuristic under declared/fast telemetry, with paired CIs and sample-age distributions *only if completed and audited*. Otherwise label not evaluated.
5. **Integrated future result:** obstacle detection range error/false alarms, route completion, falls, collisions and withdrawal counts by actuator method; report all seeds and selection protocol *only if actual camera/MuJoCo evaluation finishes*.
6. **Timeline/video:** synced healthy, faulted no-correction and faulted adaptive examples from the same selected seed; include correction confidence, gain onset/return (evaluation annotation only), source-age and collision flags; preserve failure clips.
7. **Runtime:** measured policy+adapter latency distribution, depth render and model inference separately; controller 50 Hz/20 ms deadline. Do not extrapolate GPU batch throughput to batch-one runtime.

## Methods and reproducibility requirements

- Frozen nominal G1 walker and exact trained checkpoint hash; JAX 0.12.0.1 and MuJoCo 3.16.0 only after verifying local versions.
- Distinguish model, code, deterministic parser and VLM proposal validator. If no real VLM inference ran, title/abstract must say *language-conditioned* and *not* VLA.
- No true obstacle position or gain labels consumed by the actual navigation/adapter. Use simulator robot pose and calibrated camera extrinsics with this assumption clearly disclosed.
- Register scenario seeds, obstacle size, starting pose, time budget and thresholds before running; maintain exact paired random inputs.
- Publish raw CSV, trace NPZ, camera observations or timing metadata, manifest, code/checkpoint hashes and results auditor. Mark incomplete and failed runs explicitly.
- Report no held-out results if none run. Avoid grand claims such as novel VLA policy, certified safe collision avoidance, robust multi-robot generalization or hardware deployment.

## Post-course research extension

Fine-tune and independently evaluate a compact action-chunk VLA (e.g., SmolVLA-like) on target G1 RGB/depth + instruction -> bounded (forward speed, yaw rate) chunks. Compare against deterministic constrained goal navigation at matched observations, timing and fault distributions. Keep the low-level frozen walker plus uncertain actuator correction fixed. This is a new hypothesis requiring collected data, a model training protocol and genuine held-out results, **not** part of the currently implemented system.

Relevant public sources: https://huggingface.co/docs/lerobot/main/smolvla ; https://openvla.github.io/

## Submission integrity

This file is a defensible writing starting point. Only measured claims from source-linked results may appear in final abstract, tables and conclusion. Include the required AI-use disclosure accurately, identify original contributions and list simulator/controller limitations.
