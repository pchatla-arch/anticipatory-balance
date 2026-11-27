# Reliable Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Adaptive Execution


**Draft course research report — evidence cutoff November 26, 2026 (Pacific)**  
**Prasuna Chatla**  
Stanford University · AA228/CS238  
*Status: S0/S1a actuator GPU evidence verified; healthy G2c navigation and one paired G3 actuator-fault navigation development trial completed and independently audited. Full multi-seed obstacle study remains unexecuted.*

**Research focus:** A frozen humanoid locomotion policy follows a depth-perceived obstacle route while an online actuator-execution layer compensates for hidden servo-gain changes. The reported contribution is system integration plus a limited empirical evaluation, not a newly trained VLA or a general safety guarantee.

## Abstract

A frozen humanoid walking controller can lose velocity-tracking accuracy when actuator response differs from the nominal simulator model. Online adaptation can compensate by estimating servo gain from causal command/position/effort telemetry and applying a bounded target correction. However, the same residual measurements may also be explained by effort-sensor miscalibration or passive losses, creating an identification problem. This study evaluates a frozen Unitree G1 PPO walker in MuJoCo/JAX with hidden proportional-servo gain faults, delayed/noisy telemetry, healthy controls and an unannounced return to nominal gain. The already completed S0 smoke and S1a long-horizon development experiments comprise 12 and 32 paired GPU episodes, respectively. Under a 30% gain reduction, the heuristic-gated adapter reduced mean post-onset velocity tracking loss from 0.0855 to 0.0383 (55.2%) but also reduced mean forward progress from 3.62 m to 2.44 m (32.6%). After gain restoration it withdrew within 5–6 control ticks in both observed episodes; an ungated estimated inverse failed to withdraw in both. Under a healthy plant with effort telemetry scaled by 0.8, the heuristic made unnecessary corrections and increased velocity loss. Results show a clear selectivity/progress/observability trade-off but do not establish general safety or task-level locomotion superiority. A separate head-depth navigation study established real G1 cylinder contacts and one healthy nominal goal-reaching detour with static depth-map memory and zero recorded cylinder-contact events; a subsequently executed single-seed G3 fault-navigation pilot reached the goal both with correction disabled and with the frozen heuristic. This establishes integration feasibility but does not support general task-level superiority.

## 1. Problem definition

Humanoid locomotion controllers frequently act through low-level position servo interfaces with behavior that changes under aging, calibration drift, heating, or unmodeled gains. A frozen walking policy may be robust to moderate variation yet fail to maintain commanded velocity or useful progress under persistent actuation mismatch. A reusable execution adapter must compensate for plant changes without exposing injected fault labels to the practical controller, upsetting healthy locomotion, violating target/slew bounds, or exceeding the robot's real-time control budget.

The central question is **when an estimate of actuator gain provides enough evidence to justify a corrective command**. An incorrect estimate can introduce harmful target offsets in an otherwise healthy plant, particularly when the effort sensor itself has shifted scale. This is distinct from the higher-level navigation problem: goal following and obstacle avoidance provide a task context but must not be used to conceal actuation failures.

## 2. System and information boundaries

The robot is Unitree G1 in MuJoCo Playground with a **frozen PPO nominal walking policy**. The nominal policy uses existing proprioception/joystick observations to issue position targets; no PPO retraining is part of the measured comparisons.

Let $u_t^{\mathrm{nom}}$ be the commanded actuator target, $q_t$ the joint position and $K_{p,j}$ its nominal servo stiffness. In an idealized proportional component,

$$
\tau_{t,j}^{\mathrm{plant}}\approx g_{t,j}K_{p,j}(u_{t,j}-q_{t,j}) + \tau_{t,j}^{\mathrm{other}},
$$

where hidden $g_{t,j}$ changes during the registered fault schedule. The actual evaluation uses the simulator's position-dependent actuator bias, force limits, contact dynamics and unchanged joint damping; the equation is a local motivational model, not a globally exact dynamics cancellation.

The practical adapter observes delayed source-aligned held command, measured joint state and modeled effort packets. It does **not** receive the hidden gain, fault-onset time, restoration time, or evaluation-only oracle state. The oracle comparison is explicitly privileged.

An estimated inverse initially requests

$$
\delta_{t,j}^{\mathrm{raw}} =
\left(\frac{1}{\widehat g_{t,j}}-1\right)
\big(u_{t,j}^{\mathrm{nom}}-q_{t,j}\big).
$$

The actual applied target is **bounded** by amplitude (0.20 rad), target slew (2.0 rad/s) and the simulator actuator control range. The heuristic gate further screens estimator excitation, packet age, residual RMS and a dead band before releasing offsets. A correction remains limited by healthy-operation false-intervention and fault-restoration requirements.

A critical identifiability issue arises if an effort sensor is scaled by unknown $s_j$:

$$
\widetilde\tau_{t,j}\approx s_jg_{t,j}K_{p,j}(u_{t,j}-q_{t,j}).
$$

A single effort channel often identifies only the **product** $s_jg_{t,j}$; it cannot in general distinguish actuator-gain degradation from sensor-scale shift without independent information or sufficiently informative excitation. A selective controller must sometimes abstain.

## 3. Experiments and protocol

The S0 smoke validated the pinned G1 checkpoint, CUDA JAX closed-loop execution, control-time loop, GitHub write path and independent raw CSV auditor (12 episodes, 160 control ticks). The subsequent S1a development screen ran **4 scenario families × 4 methods × 2 paired scenarios = 32 episodes** at 400 control steps (50 Hz), with a fixed walk–turn–stop–resume command schedule.

**Scenario families:** healthy; 0.70 servo gain after a hidden onset; 0.70 gain with unannounced restoration; healthy servo gain with effort telemetry multiplied by 0.8. The rollout also includes delay, dropout and measurement noise in the declared telemetry channel.

**Methods:** no target correction; privileged true-gain oracle; estimated inverse; and the frozen heuristic gate. The scenario IDs and sensor noise draws are paired across methods within each family; the gain and restoration labels remain hidden from the practical controllers. Physics was evaluated using the pinned MuJoCo 3.16.0 and JAX 0.12.0.1 on a Secure RTX A6000, with the verified nominal PPO checkpoint tree hash.

**Metrics:** post-onset capped planar-velocity squared tracking loss, forward progress projected onto body heading during the exposure window, mean absolute corrective offset while the physical plant is healthy, gain-estimation error (evaluation only), physical falls, correction-bound violations and restoration withdrawal. A corrected offset is withdrawn once every actuator is below the declared 0.01-rad threshold. The reporting denominator distinguishes scheduled restoration, restoration reached and successful withdrawal.

The source/data identifiers are retained in the evidence section. All results shown here are development experiments, not an untouched held-out sample. The separate original WP2.2 scientific attempt is not replayed or merged into these results.

## 4. Measured GPU results

Each cell below contains **two paired scenarios**; values are means computed from raw `episodes.csv`, not new measurements.

| Plant/sensor condition | Controller | Tracking loss ↓ | Forward progress (m) | Healthy-plant correction (rad) |
| --- | --- | ---: | ---: | ---: |
| Gain 0.70 | None | 0.085517 | 3.621946 | 0 |
| | Oracle | 0.028719 | 2.220856 | 0 |
| | Estimated inverse | 0.031378 | 2.228384 | 0.083154 |
| | Heuristic gate | 0.038328 | 2.440928 | 0.000116 |
| Gain 0.70 then restore | None | 0.061005 | 2.649040 | 0 |
| | Oracle | 0.030879 | 2.104550 | 0.000673 |
| | Estimated inverse | 0.031503 | 2.114995 | 0.083257 |
| | Heuristic gate | 0.036317 | 2.258562 | 0.002404 |
| Healthy plant | None | 0.022096 | 1.827288 | 0 |
| | Oracle | 0.021619 | 1.838226 | 0 |
| | Estimated inverse | 0.021339 | 1.803680 | 0.075119 |
| | Heuristic gate | 0.022658 | 1.838078 | 0.000321 |
| Sensor scale 0.8, healthy plant | None | 0.026801 | 1.746148 | 0 |
| | Oracle | 0.026801 | 1.746148 | 0 |
| | Estimated inverse | 0.031381 | 1.641541 | 0.107939 |
| | Heuristic gate | 0.031044 | 1.660157 | 0.043697 |

**Fault compensation and task progress.** The heuristic reduces gain-fault loss by about 55.2% versus correction-off, yet gives up approximately 32.6% of recorded forward progress. Because this benchmark measures a local velocity error, not verified arrival at a fixed station or a progress-matched success criterion, **lower tracking loss must not be reported as improved task completion**. Notably, the privileged oracle also reduces measured progress.

**Restoration behavior.** Both restoration episodes reached the unannounced healthy return. The heuristic withdrew below 0.01 rad in 5 and 6 steps (median 5.5, or 110 ms); the ungated estimated inverse withdrew in 0/2 and kept applying offsets. An inactive no-correction arm is not an adaptive withdrawal baseline.

**Sensor confounding.** A 0.8 torque-scale perturbation at healthy actuator gain increased the heuristic's mean tracking loss from 0.026801 (none) to 0.031044 while causing 0.043697 rad mean unnecessary correction. The inverse method's unnecessary correction reached 0.107939 rad. This is a negative result indicating that effort-channel model mismatch can trigger the adaptation logic itself.

**Safety and statistical scope.** The 32 measured S1a episodes recorded no falls and no reported control-bound violations. Two paired episodes per cell are insufficient to estimate rare-failure rates, generalize to unseen gait regimes or claim physical-robot safety. Bootstrap intervals obtained from two points should not be treated as meaningful population confidence intervals.

## 5. Implementation and evidence audit

The run includes an independent recomputation of every mean loss directly from raw episode rows, checks complete family/method cells, exact scenario ID pairing and full reported row count, and publishes each result directory to a dedicated GitHub branch with remote SHA readback. A separate provider query confirmed the GPU pod stopped after final publication. Partial development runs are retained and labeled, never silently combined with later attempts.

- S0 source and result: [12-row smoke](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_course_dev_20261127T024308Z_104a34).
- S1a source commit `89b6d0dd31020f00032330c76684c727b1f697a0`; [32-row dataset and raw traces](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_course_dev_20261127T032240Z_5fc7dd); publication commit `896000b68aa9c7ea6efcd33ea33b4973ed629bb0`.
- S1b independent descriptive readout: [S1b_raw_data_diagnosis.md](S1b_raw_data_diagnosis.md).
- Full execution/status record: [EXPERIMENT_TRACKER.md](EXPERIMENT_TRACKER.md).

## 6. Physical obstacle navigation follow-on — healthy nominal gate completed

The current branch contains optional static-cylinder scene insertion and explicit contact pairs for the G1 torso, pelvis and thighs. The stage sequence is: (G2a) compile the **real G1 MJCF** and prove contacts under evaluation-only forced overlap; (G2b) render and calibrate actual G1 head-camera metric depth and detect the stationary obstacle without privileged geom state; (G2c) navigate the healthy frozen G1 around it, measuring contact and genuine route completion; (G3) use identical navigation/perception with hidden gain changes, then compare recovery against correction-off. An original 72-episode obstacle evaluation registration exists but is *not yet executed*.

The physical cylinder was independently compiled and contact-tested on the real G1 model, and a nominal depth-guided pilot produced authentic simulator footage. The pilot failed to reach the goal and therefore **does not prove successful avoidance or complete navigation**. It also tested no actuator adaptation under obstacle navigation. Sections 3–4 remain solely the independent actuator-adaptation benchmarks.

### G2b/G2c: Initial real depth-guided G1 route diagnostic (November 27 UTC)

The first contact-enabled obstacle pilot was executed on a real MuJoCo/JAX Unitree G1 simulator in one GPU-backed development rollout (source `1fd2d776cd06b270c1b099c6ce8d474d55ec4931`). A stationary 0.30-m-radius cylinder was inserted at (2.8 m, 0 m) using four explicit robot-obstacle contact pairs. The physical model compile/contact proof passed independently in G2a. The head camera generated **96 × 96 metric depth frames**, with the convention calibrated on **4,183 detected ground-support pixels**; the cylinder was detected from depth without giving the online planner the object's true MuJoCo position. Simulator odometry and camera extrinsics remain disclosed privileged navigation aids, not a perception-learning claim.

The pilot executed **800 control steps (16 simulated seconds at 50 Hz)**, capturing **160 actual RGB frames** at 10 Hz and yielding a verified H.264 MP4. The obstacle was detected in 22 of 160 depth frames; most later blank frames occurred as the robot turned laterally, so 22/160 is **not** a valid recognition recall fraction. A stable detection was confirmed by control tick 5, and the last detection occurred at tick 105. A conservative waypoint detour was commanded for 102 sampled frames; from tick 515 onward, the navigator issued a stop because its previously tracked obstacle was near-field but no longer recently observed. The robot reached approximately (2.11 m, 1.57 m) rather than its goal (5 m, 0 m), and **did not complete the route**. The simulator recorded **zero explicit cylinder-contact sensor activations** over 800 control ticks and zero sampled obstacle contact frames. These data establish genuine rendering, perception-to-control integration and a protective stopping failure mode, **not** successful obstacle avoidance or navigation under an actuator fault.

The previous attempts also recorded reproducible implementation problems (missing EGL loader and batch-observation indexing). Both were corrected before the completed diagnostic; no intermediate failure was presented as a scientific result. Raw status, frame-level odometry/steering/contact records, depth calibration and actual simulator video were published under [G2b/G2c diagnostic evidence](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_g2b_nominal_20261127), verified result commit `eff258a3e11129fbf31a64388229acd5ac8becb2`. **The result remains incomplete for the course's full navigation goal.** A subsequent tracked-static-obstacle memory rule might address the camera-field-of-view loss, but has not yet been evaluated on GPU.

### G2c2: Safe static-map persistence enabled a complete healthy G1 route

To investigate the G2b stale-perception stop without using privileged obstacle coordinates, a second G2c2 nominal navigation diagnostic retained the last confirmed **depth-estimated** location of the known stationary cylinder, subject to a new geometric condition: the already-planned remaining waypoint segments must clear the obstacle disk inflated by the robot proxy radius, the planner's clearance and an additional margin. The rule is considered only after the first detour waypoint is passed, with fresh head-camera frames still required. If the remaining route cannot be certified, the robot must stop. This is an **experimental memory heuristic for a stationary, single-obstacle scene**, not a general uncertainty-aware safety guarantee for moving or unseen obstacles.

One real JAX CUDA G1 rollout, source `6d75add0f2b6ceb601f90a5376818edcce10cfee`, ran **1,186 control ticks** on the healthy frozen walking policy with the same cylinder and (5 m, 0 m) target. The robot reached the goal at position approximately **(5.013 m, 0.185 m)**. The experiment recorded **zero static-cylinder contact sensor activations** and zero sampled obstacle contact frames. A separate evaluator-only analysis of 238 sampled route frames found minimum pelvis-to-cylinder *center* distance of 1.434 m, though this distance is not a full-body swept-volume clearance guarantee. The depth detector saw the cylinder in 22 frames; command categories were 161 ordinary detour, 75 static-memory detour, one slow approach, and one goal-reached. The 238-frame simulator RGB MP4 was encoded at 10 fps, independently decoded, and published with the complete frame-level record as [G2c2 physical nominal-route evidence](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g1_g2c_static_memory_20261127), result commit `521d00438f5efdf2e1682367baceffe56e83c923`. The corresponding A6000 pod was verified **EXITED**.

Together, G2b and G2c2 show that a depth-confirmed stationary obstacle can be retained as a temporary world-map hypothesis when it leaves the camera's field of view, allowing the **single tested** nominal controller to finish the course without recorded collisions. They do **not** demonstrate robustness across seeds, varied obstacle geometry, moving hazards, unobserved occupancy, task-level fault adaptation, physical G1 hardware or a trained vision-language-action policy. The central S1a actuator findings above remain independent from this healthy-navigation demonstration. **A one-seed paired gain-change navigation pilot was executed subsequently and is reported below; the registered 72-episode study remains unexecuted.**

### G3: Paired fault-conditioned G1 depth navigation (development, one seed)

Following G2c, the same frozen PPO, stationary contact-enabled cylinder, metric head-depth camera, calibrated depth detection, and static-memory waypoint navigator were evaluated under a **hidden 30% leg-servo gain reduction** at control tick 200 and unannounced return to nominal gain at tick 900. Both arms used seed 1 and the same 1,200-step limit. The first disabled actuator correction; the second applied the previously frozen causal heuristic, using delayed, noisy, source-timestamped effort and joint-state telemetry with no access to injected gain labels or the cylinder's true position. A 300-step causal-integration smoke preceded the paired trial and recorded 291 released delayed telemetry packets.

| Paired G3 outcome | Correction disabled | Frozen heuristic gate |
| --- | ---: | ---: |
| Goal reached | Yes | Yes |
| Control steps to goal | 1,159 | 1,111 |
| Final distance to goal (m) | 0.1945 | 0.1917 |
| Explicit cylinder-contact frames | 0 | 0 |
| Real recorded H.264 frames | 232 | 223 |
| Nonzero adaptive correction control ticks | 0 | 703 |
| Released delayed telemetry packets | N/A | 1,090 |

The heuristic completed the route **48 control ticks, or 0.96 simulated seconds, earlier** than correction disabled. Both controllers achieved the same goal without a recorded collision. The heuristic's final mean estimated leg gain was approximately 0.937 after the return-to-healthy interval; this aggregate estimate is not by itself evidence of complete corrective-offset withdrawal in navigation, because the run did not retain a full per-tick withdrawal trace. The sensor-scale false-adaptation issue observed in S1a remains unsolved.

The experiment was independently audited from actual frame-level pose/obstacle/contact records and both H.264 files, enforcing identical seed, source revision, fault onset/return schedule, and scene/camera parameters. The [raw G3 paired development results](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1) include two videos and a separate [paired audit](https://github.com/pchatla-arch/anticipatory-balance/blob/foresight/experiment-results/results/g3_paired_fault_20261127_seed1_7ea45d1/paired_independent_audit.json). Frozen source revision: `7ea45d1522e3d70ee950b98eb4aee1fe35013fd5`; verified evidence commit: `55dfa2d2a15790f1caa13b9460faf268081225ab`. The provider reported the GPU pod `EXITED` after publication.

**Interpretation:** G3 confirms that online correction, causal effort telemetry, true head-depth obstacle perception, contact-enabled MuJoCo physics, and the frozen G1 locomotion policy can operate together in an end-to-end development run. This is **n=1 matched seed** and the correction-disabled arm also succeeded; no robust, held-out, hardware, trained-VLA, or general task-level performance claim is justified. The original 72-row obstacle evaluation remains unexecuted.

## 7. Limitations and next research direction

The primary limitation is observability. A healthy plant with a scaled effort sensor can impersonate degraded proportional servo response. A promising research direction is a **hypothesis-aware abstention gate** that compares gain-fault and sensor-fault explanations, requests safe excitation where necessary and limits correction when hypotheses remain ambiguous. Such a gate needs a separate preregistered study and is **not a measured improvement here**. Other limitations include reliance on model-derived effort packets, only two paired seeds per cell, short simulation horizons, zero physical hardware, and no measured obstacle contact/navigation outcome. A learned VLA, learned depth-estimation network, a combined actuator-fault-plus-obstacle experiment and robust learned PPO controller are outside the measured scope.

## 8. Conclusion

Frozen-policy actuator correction was demonstrated in a real G1 simulator with audited GPU data. Heuristic gating improved the chosen velocity-loss metric under a 30% gain reduction and withdrew promptly after hidden restoration, but produced lower measured progress and remained vulnerable to effort-sensor calibration errors. The evidence therefore supports a **bounded, selective execution-layer feasibility finding with unresolved utility and identifiability trade-offs**, not a general robustness or safety claim.

## AI-use disclosure — draft for author verification

AI assistance supported technical discussion, architecture and test review, coding/debugging, data-check scripts and drafting. Simulator runs, output audit and interpretation should be independently verified and all assistance disclosed in accordance with the course policy. The author must check that this statement accurately describes the actual use before submission.

## References — source verification required before final PDF

- J. Schulman et al., *Proximal Policy Optimization Algorithms*, arXiv:1707.06347 (2017).
- E. Todorov, T. Erez and Y. Tassa, *MuJoCo: A Physics Engine for Model-Based Control*, IROS (2012).
- MuJoCo Playground and its pinned Unitree G1 locomotion implementation (source revision and license in the repository).
- Frozen 2026 experiment protocols, scripts, run logs and source commits linked above.

*Report remains a reproducible research draft, not a finalized submission PDF or held-out evaluation. Genuine healthy and paired fault-conditioned G1 simulation videos were verified; hardware deployment, a trained VLA, broad robustness and the registered 72-row obstacle experiment have not been demonstrated.*
