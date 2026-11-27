# Reliable Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Adaptive Execution
**Prasuna Chatla — Stanford University**  
AA228/CS238 empirical paper working draft, 26 November 2026  
**Status: auxiliary actuator-adaptation analysis based on S0/S1a GPU data. For the current integrated navigation and G3 evidence, see the primary Course_Report_20261127_Draft.md.**

## Abstract

A frozen reinforcement-learned humanoid walking policy may suffer tracking errors when position-controlled actuators drift from their nominal servo gains. A causal adaptation module using delayed, noisy target, joint-position and effort packets can estimate changes and request bounded target offsets without retraining the walking policy. This study compares correction disabled, a privileged inverse-gain oracle, an ungated estimated inverse, and a confidence-like heuristic gate in a simulated Unitree G1. The full 32-episode development screen covers healthy operation, 30% gain reduction, unannounced gain restoration and a false effort-scale sensor fault. On the gain-reduction family, the heuristic reduces planar-velocity tracking loss from 0.08552 to 0.03833 (55.2%), but also reduces heading-projected progress from 3.62 m to 2.44 m (32.6%). It withdraws correction after return to normal in both tested episodes, whereas the ungated inverse persists in both. Under a nuisance effort-scale error, the heuristic introduces unnecessary correction and performs worse than no adaptation. These results demonstrate feasibility and important identifiability and metric-design limitations; they do not demonstrate generalizable safety, physical robot performance or obstacle avoidance.

## Problem and formulation

The frozen PPO controller produces nominal position targets q*_t for a commanded velocity c_t. Hidden actuator effectiveness g_{i,t} changes the position-servo relation while the policy and contact model remain fixed. A simplified proportional relation is

$$
\tau_{i,t}\approx g_{i,t}k_i(q^\star_{i,t}-q_{i,t}).
$$

The estimator receives source-timestamped packets with held target, measured joint position and modeled effort; it does not receive the injected gain or fault-onset time. With regression feature \phi_t=q^\star_t-q_t and normalized effort y_t=\tau_t/k, recursive least squares maintains a bounded gain estimate:

$$
K_t=\frac{P_t\phi_t}{\lambda+\phi_t^2P_t},\quad
\hat g_{t+1}=\mathrm{clip}(\hat g_t+K_t(y_t-\hat g_t\phi_t),g_{\min},g_{\max}).
$$

The current position is causally approximated by propagating the newest packet using measured velocity and its age. Estimated inverse correction requests

$$
\Delta q^\star_t=(1/\hat g_t-1)(q^\star_t-\hat q_t).
$$

The heuristic permits this only after enough valid packets, sufficiently low residual RMS, adequate excitation, a sufficiently nonnominal estimated gain, and fresh telemetry. Every method shares the same amplitude, slew and simulator control-range constraints. The oracle uses the injected gain as an evaluation-only privileged reference.

## Experimental evidence

The frozen nominal G1 PPO checkpoint has tree SHA-256 14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de. Experiments use MuJoCo 3.16.0, JAX 0.12.0.1, a CUDA RTX A6000, and the frozen heuristic configuration source d756c37d0564837a74132c30f8e366320a6bb1a5. S0 completed 12 GPU development episodes at 160 control steps. S1a completed 32 GPU development episodes across 16 experimental cells at 400 50-Hz control steps. Each method within a family uses the same two scenario IDs and telemetry model. Independent audit verified cell completeness and the raw mean velocity losses.

### S1a: 30% leg-servo gain reduction

| Method | Velocity tracking loss ↓ | Forward progress (m) ↑ | Falls |
|---|---:|---:|---:|
| Correction disabled | 0.085517 | 3.621946 | 0/2 |
| Privileged inverse oracle | 0.028719 | 2.220856 | 0/2 |
| Ungated estimated inverse | 0.031378 | 2.228384 | 0/2 |
| Frozen heuristic gate | 0.038328 | 2.440928 | 0/2 |

The heuristic reduces tracking loss by 55.2% relative to no correction while reducing forward progress by 32.6%. The privileged oracle also trades progress for lower loss. The measures do not establish improved task completion; progress is heading-projected displacement over the post-onset interval, not arrival at a destination. Command-exposure, gait-speed and progress-matched controls are necessary before claiming a task-level benefit.

### S1a: return-to-healthy experiment

| Method | Tracking loss ↓ | Progress (m) | Withdrawal of offset |
|---|---:|---:|---|
| Disabled | 0.061005 | 2.649040 | Not applicable |
| Oracle | 0.030879 | 2.104550 | 2/2, median 3 ticks |
| Ungated inverse | 0.031503 | 2.114995 | 0/2 |
| Heuristic gate | 0.036317 | 2.258562 | 2/2, median 5.5 ticks |

A 5.5-tick median at 50 Hz corresponds to 110 ms. The denominator matters: there were two observed restoration events per method, not an independently powered population sample. An ungated inverse tracked velocity slightly better but failed to return its offset to zero in either observed episode.

### S1a: sensor-scale confound

With physically healthy actuators but effort measurements scaled to 80%, correction-disabled loss was 0.026801, the heuristic loss was 0.031044 and the inverse loss was 0.031381. Heuristic healthy-plant harmful correction was 0.043697 rad; ungated inverse correction was 0.107939 rad. This is a concrete negative result. A gain change and torque-sensor scale error are not necessarily identifiable using one noisy effort channel, and threshold gating is not sufficient in this scenario.

## Implications and method extension

The empirical study does not establish that uncertainty gating is universally safe; it shows why **separating useful actuator correction from misleading sensor evidence** is essential. A hypothesis-aware adaptation gate could compare the evidence for true gain change with alternatives such as torque-scale error, and abstain when the hypotheses are observationally equivalent. This extension has **not been trained, tuned or evaluated** and must not be claimed as an experimental finding.

The next controlled evaluation should pair task progress and velocity loss for each scenario, report false intervention on healthy and nuisance families, and explicitly measure withdrawal success (scheduled, reached, withdrew). Any new tuning should be tested on separate seeds or disclosed as development-only.

## Physical-obstacle demonstration boundary

A stationary cylinder has been added to the real G1 MuJoCo model. An evaluation-only forced-overlap experiment verified four explicit physical contact pairs; zero cylinder contacts existed in the start pose. Head-camera rendered depth, nonprivileged detection and a conservative waypoint planner are separate prototype components. A head-camera pilot is underway; it must pass actual rendering, confidence and stale-frame constraints, collision measurement, route completion and real-frame video encoding before any physical obstacle-avoidance success is reported. The simulator root pose is an explicitly privileged localization assumption, not real-world autonomous localization. No VLA training or hardware deployment has occurred.

## Reproducibility and limitations

S0 and S1a source/results live on the dedicated foresight/experiment-results branch in directories g1_course_dev_20261127T024308Z_104a34 and g1_course_dev_20261127T032240Z_5fc7dd, respectively. The S1a source commit is 89b6d0dd31020f00032330c76684c727b1f697a0 and published results commit is 896000b68aa9c7ea6efcd33ea33b4973ed629bb0. All 32 S1a records reported no falls, but only two matched seeds per cell, restricted gain faults, and short simulation horizons were tested. The development data do not establish reliable confidence intervals, generalization, safety or hardware transfer.

## AI assistance disclosure (draft)

AI assistance supported algorithm exploration, code review, regression-test design, experiment troubleshooting and drafting. Reported numerical findings originate from the versioned simulator and raw episode records. Final scientific claims, disclosure and submission remain the responsibility of the author.
