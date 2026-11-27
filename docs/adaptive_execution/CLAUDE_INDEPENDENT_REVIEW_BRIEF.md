# Independent review brief — agreed final project framing

**Manuscript title (retain verbatim):**

**Reliable Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Adaptive Execution**

**Author:** Prasuna Chatla | Stanford University | AA228/CS238, November 2026

**Reviewer:** Claude (or another independent technical reviewer). Review only; do not independently run or modify experiments, create a new GPU pod, publish scientific data or change the title without explicit approval.

## Purpose of the review

This is not a request to invent another project or to redesign the core architecture. The implementation has reached a real G1 MuJoCo/JAX system demonstration: rendered head-camera depth, a physical static obstacle, perception-conditioned waypoint control, a frozen PPO walking policy, and causal online actuator adaptation using hidden gain changes and delayed effort telemetry. We need the **single coherent paper** that honestly combines the navigation system with the actuator adaptation method.

The main scientific limitation is that the most complete controller study is development-only (two paired seeds per family), and the integrated fault-navigation demonstration has only one same-seed pair. The corrected walker also already completes the navigation task without adaptation. The review must not exaggerate the 48-step advantage of the heuristic under a single seed.

## Read these in order

1. [Primary course report](Course_Report_20261127_Draft.md): **the artifact to review and correct**. The title above is fixed.
2. [Canonical experiment tracker](EXPERIMENT_TRACKER.md): completed/blocked stages and exact evidence.
3. [S1b limitations and raw-data diagnosis](S1b_raw_data_diagnosis.md): velocity tracking versus forward progress, restoration, sensor effort-scale confound.
4. [Physical obstacle and perception decision](ADR_obstacle_detection_avoidance.md): information boundaries, simulated depth convention and collision metrics.
5. [PR #51](https://github.com/pchatla-arch/anticipatory-balance/pull/51): source, revision history and methods.

### Verified result sets (do not replace or reinterpret without raw-data audit)

- [S0 frozen walker/GPU smoke](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_course_dev_20261127T024308Z_104a34): 12 complete development rows.
- [S1a actuator adaptation](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_course_dev_20261127T032240Z_5fc7dd): 32 rows, 400 ticks, 16 cells; the heuristic reduces loss but also reduces forward progress; a torque-scale confound triggers harmful corrections; gain-restoration withdrawal 2/2 for the heuristic.
- [G2c healthy depth-guided physical-obstacle navigation](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_g2c_nominal_20261127T045324Z_6d75add): actual simulator frames/video and independent audit; one development route achieved its goal, without recorded obstacle contact.
- [G3 paired gain-fault navigation](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1): same-seed correction-off versus heuristic, both reached the goal, both zero recorded obstacle contacts; heuristic 1111 steps versus correction-off 1159. One seed only and not a robust statistical effect.

## Specific review tasks

1. **Claim hierarchy:** Is “Reliable Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Adaptive Execution” supportable as the paper title, with “reliable” framed as the system's objective rather than an established cross-condition guarantee? Identify and rewrite any claims that exceed the observations.
2. **Novelty:** Separate system integration (perception-guided planning, frozen PPO, low-level correction) from potentially novel selective actuator-adaptation decisions. Identify relevant prior art and missing controls rather than asserting novelty without comparison.
3. **Causal information-flow audit:** Verify that the real obstacle geom position, true hidden gain, gain restoration and oracle labels remain evaluation-only, and that simulator localization/camera extrinsics are clearly disclosed.
4. **Metrics and controls:** Critically assess the mismatch between velocity-loss improvement (0.0855 to 0.0383) and forward-progress decrease (3.62 m to 2.44 m). Suggest a truly progress-matched route comparison without changing or rerunning prior protected data.
5. **Confounded sensing:** Explain why a scaled effort channel can be observationally indistinguishable from a gain change, and why the frozen heuristic fails in this scenario. Recommend a defensible abstention or independent-evidence approach, without pretending it is already validated.
6. **Integrated G3 interpretation:** Both correction-off and heuristic reached goal; the 48-step difference at one seed is illustrative. Confirm the fair shared-seed/scene/fault/perception contract and the limitations from one development pair.
7. **Report completeness:** Identify the smallest missing tables, plots, algorithm pseudocode, source/checkpoint manifest, hardware/software-version facts, failure examples, citations, reproducibility and AI-use disclosure needed for a high-quality Stanford course submission.
8. **Unexecuted work:** Flag every instance where a proposed 72-episode obstacle study, learned VLA, hardware robot, statistical confidence, or contact safety certificate is described as completed. Keep the study's scope consistent and objective in the problem statement.

## Requested output from the reviewer

Return a **major/minor issues table**, a **claim-to-evidence matrix**, and **exact proposed changes to the primary course report** (section, original language, replacement language). Distinguish blocking issues for submission from optional post-submission research. **Do not create hypothetical experiment results**. If further simulator tests are essential, state precisely which hypothesis, paired baseline, seeds, cost, data, and release gate they would require.

The author and ChatGPT can then implement the accepted corrections, regenerate the final paper, independently check the source numbers and visuals, and freeze a submission version. The goal is to **review and finish the current project**, not start another competing study.
