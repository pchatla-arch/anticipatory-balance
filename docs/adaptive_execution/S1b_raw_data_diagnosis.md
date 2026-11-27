# S1b: independent analysis of the completed S1a GPU screen

**Status:** CPU re-analysis of existing S1a raw episode records; **no additional experiment**.  
**Source:** `foresight/experiment-results/results/g1_course_dev_20261127T032240Z_5fc7dd/bench_dev/episodes.csv`.  
**Simulator:** Unitree G1, pinned MuJoCo/JAX, 400 ticks at 50 Hz; 2 paired scenario IDs per family.  
**Source commit:** `89b6d0dd31020f00032330c76684c727b1f697a0`.  
**Evidence:** published result commit `896000b68aa9c7ea6efcd33ea33b4973ed629bb0`.

This report independently groups the raw CSV on (family, method), pairs on scenario_id within each family, and calculates means for velocity tracking loss, forward progress and healthy-plant correction. It does not re-use published summary cells as its inputs. There are only two distinct scenario IDs per family, so all contrasts are development observations rather than inferential proof.

## Paired mean results

| Family | Method | Loss (lower is better) | Forward progress (m) | Harmful healthy offset (rad) |
| --- | --- | ---: | ---: | ---: |
| gain_070 | none | 0.085517 | 3.621946 | 0.000000 |
| gain_070 | oracle | 0.028719 | 2.220856 | 0.000000 |
| gain_070 | estimated_inverse | 0.031378 | 2.228384 | 0.083154 |
| gain_070 | heuristic_gate | 0.038328 | 2.440928 | 0.000116 |
| gain_070_return | none | 0.061005 | 2.649040 | 0.000000 |
| gain_070_return | oracle | 0.030879 | 2.104550 | 0.000673 |
| gain_070_return | estimated_inverse | 0.031503 | 2.114995 | 0.083257 |
| gain_070_return | heuristic_gate | 0.036317 | 2.258562 | 0.002404 |
| healthy | none | 0.022096 | 1.827288 | 0.000000 |
| healthy | oracle | 0.021619 | 1.838226 | 0.000000 |
| healthy | estimated_inverse | 0.021339 | 1.803680 | 0.075119 |
| healthy | heuristic_gate | 0.022658 | 1.838078 | 0.000321 |
| sensor_scale_080 | none | 0.026801 | 1.746148 | 0.000000 |
| sensor_scale_080 | oracle | 0.026801 | 1.746148 | 0.000000 |
| sensor_scale_080 | estimated_inverse | 0.031381 | 1.641541 | 0.107939 |
| sensor_scale_080 | heuristic_gate | 0.031044 | 1.660157 | 0.043697 |

## Conclusions permitted by the raw data

1. **Gain fault:** heuristic gate reduces the recorded velocity-loss mean by **55.18%** relative to no correction (0.085517 to 0.038328). However, mean forward progress decreases **32.61%** (3.62195 to 2.44093 m). The two metrics disagree; controller efficacy is not established at a matched task or distance budget. The oracle similarly trades loss for distance, strengthening the possibility that the error metric rewards different speed/profile outcomes. Do not present loss improvement alone as locomotion or navigation success.
2. **Return to healthy:** heuristic's recorded correction withdrawal ticks are **6 and 5** (2/2 withdrew; median 5.5 control ticks = 110 ms). Estimated inverse has no successful withdrawal (2/2 never withdrew), despite a somewhat smaller tracking loss. The none arm has zero correction by definition; it is not evidence of successful adaptive withdrawal.
3. **Effort miscalibration:** in healthy gain with sensor scale 0.8, heuristic's velocity loss rises from 0.026801 to 0.031044 versus no correction (~15.8% worse), with mean harmful correction 0.043697 rad. Inverse is similarly worse and has 0.107939 rad harmful correction. The measurement channel cannot be assumed to uniquely distinguish a true servo-gain fault from a scale error; this is an identifiability risk.
4. **Healthy baseline:** heuristic produces near-zero correction (0.000321 rad) but a small higher loss than none (0.022658 vs 0.022096). Inverse acts strongly despite healthy gain (0.075119 rad), motivating an abstention mechanism.
5. **Falls:** none of 32 episodes fell. This does *not* prove safety (tiny sample, short simulated horizon, no physical obstacle).

## Recommended next method (NOT evaluated yet)

**Hypothesis-aware confidence gate**, not an unregistered claim: compare likelihoods of alternative explanations for the torque residual, e.g. H_gain (gain changed) and H_scale (effort sensor scale changed). Only release bounded residual correction if the likelihood ratio favors a true actuation fault and gain-estimation uncertainty is sufficiently small. If both explanations remain compatible with the observed packets, abstain or request safe excitation. Without an independent source of actuator effort or sufficiently informative joint dynamics, the hypotheses may be unidentifiable, so abstention must remain a first-class outcome.

This is a direction for future controlled ablation; it is not shown to work in S1a.

## Highest-value next tests

**S2a:** compile an actual static cylinder into G1 MJCF; verify collision pairs and head camera using real simulator data. Do not drive the robot until the collider/contact gate passes.

**S2b:** calibrate real head-depth rendering on the stationary cylinder using nonprivileged detection, then count missed detections and false alarms. Privileged true cylinder position is evaluation-only.

**S2c:** attempt a healthy frozen-walker detour with actual contacts and save real trajectory/video, even if it fails. Use the same forward/yaw limits and report failure cases; only after healthy locomotion is verified should paired actuator-fault navigation be attempted.

**Paper:** position the contribution as a real-GPU empirical study of selective actuator correction under delayed/noisy effort telemetry, with an identified loss-versus-progress trade-off and an important nuisance-calibration failure. Distinguish conjectures, executable code, synthetic tests, GPU physics and unexecuted experiments.

## Methodological caveats

The sample size is **n=2 per cell** and the data are development split, not independent held-out evaluation. Report raw paired episodes and use descriptive means. Bootstrap intervals calculated from two samples are not reliable uncertainty estimates. The forward progress is the benchmark's torso-heading projection over the exposed horizon, not direct goal completion, path efficiency, minimum clearance, contact force, or station arrival. The original 72-episode obstacle protocol is not executed.
