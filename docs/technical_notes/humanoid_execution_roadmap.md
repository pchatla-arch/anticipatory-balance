# Online Actuator Adaptation for Reliable Humanoid Locomotion

Engineering roadmap, reviewed November 15, 2026. The implementation baseline is
[`aac0699`](https://github.com/pchatla-arch/anticipatory-balance/commit/aac06996b8bab2980234c4c5fb63a7c78025806f).
The current known-gain feasibility protocol is unchanged. This note describes
subsequent work; it does not announce an online-adaptation or GPU-recovery result.

## Problem

A learned humanoid locomotion controller can lose task performance when the
relationship between commanded joint targets and physical actuation changes.
An execution system must detect identifiable changes, apply corrections within
declared limits, and preserve useful commanded motion while exposing uncertainty
and insufficient corrective authority.

The target capability is reliable execution of a locomotion work cycle:
walking to a station, turning, stopping, and resuming after an actuator-response
change. The practical controller receives a specified telemetry interface and
does not receive injected fault parameters or fault-onset labels.

## Public requirements motivating the scope

Tesla's public [Optimus mission][1] includes balance, navigation, perception and
physical interaction. Its current [Whole Body Controls posting][2] emphasizes
physics-informed policy development, dynamics modeling, simulation transfer,
evaluation and deployment. Its [inference role][3] emphasizes integration of neural
models with timing-constrained robot software. These public descriptions motivate
the requirements below; they do not establish Tesla's internal architecture or
endorsement of this proposed module.

| Requirement | Project evidence |
|---|---|
| Robust learned motion | Held-out actuator faults, recovery and healthy-regression measurements |
| Useful execution | Completion of a commanded route with turn, stop and restart phases |
| Model-based reasoning | Explicit fault physics, observation assumptions and identification limits |
| RL contribution | Actuator-randomized PPO comparison; a residual only if it closes a measured gap |
| Runtime integration | Defined interfaces, timestamps, resource bounds and measured batch-one latency |
| Repeatable evaluation | Versioned checkpoints, paired episodes, retained failures and frozen test splits |

This is one locomotion capability within a larger humanoid system. The first
deliverable does not establish general-purpose autonomy, manipulation, visual
navigation, hardware transfer or operation on Optimus hardware.

## Current evidence boundary

| Component | Status at this review |
|---|---|
| Frozen G1 PPO walker and normalization | Implemented and restored locally |
| Proportional-servo gain fault | Implemented, with factors 0.85/0.70/0.55 across twelve leg actuators |
| Bounded controller using the true gain | Implemented |
| Eight-second forward-walking screen and confirmation protocol | Implemented |
| CPU regression and short batched integration checks | Passed; see the [gate note](adaptive_execution.md) |
| GPU recovery measurements | No published adaptive-gate result found at this review |
| Online estimator and confidence logic | Proposed |
| Commanded route, turn, stop and resume benchmark | Proposed |
| Actuator-specific robust-PPO training comparison | Proposed |
| Lag, vision, learned residual and hardware deployment | Separate future work |

The existing training [randomizer](../../foresight/envs/g1_impact/randomize.py)
varies contact friction, joint friction loss, armature, masses and reference
positions. It does not randomize `actuator_gainprm` and `actuator_biasprm`.
Consequently, a meaningful robust-PPO comparison must explicitly include the
tested actuator changes; generic domain randomization is not sufficient evidence
that this baseline has been covered.

## Milestone 0: retain the known-gain feasibility gate

The current gate asks whether moderate response loss degrades the frozen walker
and whether bounded privileged correction can recover performance. It is a
compensability check. It uses paired screening and independent confirmation,
retains failures, and distinguishes numerical invalidity from physical failure.

The fault changes proportional servo response, with joint damping and existing
force limits unchanged. It is not a calibrated model of motor aging, heating,
battery discharge or loss of maximum torque. Passing it supports the next
identification experiment for this model family.

The existing [pod command and budget](adaptive_execution.md#running-on-the-pod)
remain the first action. No additional training or new fault family is added to
that run by this roadmap.

## Milestone 1: hide the parameters and identify the change

Define the telemetry contract before selecting an estimator.

| Allowed practical inputs | Evaluation-only information |
|---|---|
| Issued joint targets and recent command history | Injected gain and any future lag parameter |
| Measured joint position and velocity | Fault-onset time and fault family labels |
| Measurement timestamps and age | Simulator-perfect contact forces unless explicitly granted to every practical method |
| Explicitly specified, simulated effort telemetry if included | Privileged future state and controller outcome |

Start with the gain family already shown compensable. A small classical
estimator is the reference implementation. Gain is not automatically identifiable
from motion: contact loads, friction, inertia, measurement errors and insufficient
excitation can produce similar observations.

Use three telemetry conditions: ideal effort measurements for diagnosis;
declared noisy/delayed effort measurements for the main telemetry experiment;
and no direct effort measurement as an identifiability ablation. A simulator's
perfect actuator-force channel can make identification artificially easy. Noise
settings are simulation assumptions until measured hardware data supports them.

Report estimated versus injected parameters, convergence delay, confidence
coverage, false adaptation during healthy/contact-only episodes, and downstream
task recovery. Estimation accuracy alone is insufficient if the correction
worsens walking.

Confidence may use excitation, residual consistency and parameter uncertainty.
It controls corrective authority. Reducing correction returns authority to a
possibly degraded nominal policy; that behavior is not a fall-prevention or
stability guarantee.

## Milestone 2: demonstrate a commanded locomotion cycle

The first task extension is an indoor station-to-station route on flat ground.
A proposed sequence is: walk one straight segment, turn, approach a second
station, stop briefly and resume. Route lengths, stop tolerances and timeout are
calibrated using healthy operation, then fixed before fault comparisons.

The initial route controller may use simulator pose. Describe that condition as
commanded route following with privileged localization. It does not demonstrate
vision-based autonomous navigation. Healthy route completion is a prerequisite;
stopping and restarting must be measured rather than assumed from walking reward.

Implementation needs a command schedule that remains authoritative throughout
the episode. The inherited joystick environment can resample commands after 500
steps, so a longer route needs an explicit command-interface change and test.
The existing eight-second gate ends before that resampling boundary.

Measure work-cycle completion, position/heading error at stations, upright stop
duration, recovery time, distance/time lost, failures, healthy regression,
correction magnitude and clipping. Report phase-specific behavior so good
straight-line walking cannot hide failed turns or stops.

After the original gain family passes, introduce asymmetric joint-group changes
and gradual drift as separately declared test conditions. Lag is a later model
family with a causal compensation limit; a pure delay is not equivalent to a
first-order response filter.

A carrying task requires a separate design. Payload changes alter inertia and
center of mass, and a grasp adds contact constraints. A gain inverse cannot be
assumed to solve those problems. The healthy carrying behavior and disturbance
model would need their own feasibility checks.

## Milestone 3: compare adaptation with learning robustness

There are two distinct comparison questions.

### Reuse the same frozen policy

| Method | Parameter knowledge | Purpose |
|---|---|---|
| Uncorrected frozen PPO | No fault parameters | Reference |
| Known-parameter bounded correction | True parameters | Privileged diagnostic |
| Classical estimated correction | Allowed telemetry only | Online adaptation |
| Confidence-limited estimated correction | Same allowed telemetry | Value of uncertainty handling |
| Optional learned residual | Same allowed telemetry | Test a named classical-model limitation |

Use the same frozen actor and normalization, matched scenarios, and comparable
correction limits for practical adapter methods. The oracle's information advantage
must remain explicit.

### Compare with retraining the policy

Train a PPO baseline with actuator-specific dynamics randomization, including
within-episode changes if those are tested. This is the first additional policy
training experiment. A recurrent or history-conditioned PPO alternative becomes
useful if the result depends on identification from history.

This comparison asks whether an adapter is useful relative to learning robustness
directly. The retrained actor is deliberately not frozen to the original weights.
Give direct policies a comparable physically allowable target envelope, rather
than limiting them to the adapter's small residual range. Report the different
action interfaces and effective authority explicitly.

Match evaluation scenarios and permitted deployment information. Account for
training steps, tuning trials and compute; report adaptation compute separately.
Keep training/validation/test faults distinct. Follow an initial positive result
with additional independently trained policies before claiming seed robustness.

A learned residual is justified only after a concrete limitation appears, such
as an interaction or state dependence not handled adequately by the classical
model. Its comparison must show a useful task-level gain without concealing
healthy regressions or increased corrective authority.

## Runtime deliverable

The runtime connects the policy, estimator and correction module through a
documented observation/action interface. Preserve observation ordering, previous
action semantics and normalization across exports. Record model/config versions,
sequence numbers, timestamps and command limits.

A small C++ implementation of the estimator/corrector and a replay harness are
appropriate deployment artifacts. Actor export is a separate parity check;
the current JAX checkpoint should remain the numerical reference. A faster model
format or lower precision is useful only if its closed-loop behavior remains
acceptable. Tesla's public model-deployment role makes this a relevant engineering
requirement, not a reason to presume a particular export tool will work.[4]

Measure batch-one policy + estimator + correction latency, including input/output
transfer where relevant. Record p50/p95/p99, deadline misses against the 20 ms
control interval, memory, warmup/compilation separately, and the exact machine.
Synchronize asynchronous accelerator work before timing. Batched simulation
throughput is a separate metric.

Test missing, stale and reordered telemetry through declared supervisory
behavior. For timing-fault experiments, simulated physics must continue while
commands are delayed; pausing the entire simulation does not reproduce control
latency. Fall outcomes remain part of those tests.

Deterministic replay, scenario generation and automated regression reports align
with the public validation-infrastructure requirements.[5] A simulated replay
system is software-in-the-loop evidence, not a hardware-in-the-loop result.

## Optional perception extension

Rendered depth can eventually supply free-space estimates to a local route
controller. Its causal role must be visible: a changed observation changes the
motion command, and the lower controller realizes that command under actuator
uncertainty.

Compare oracle geometry, actual rendered depth, and delayed/degraded depth using
the same locomotion controller and fault scenarios. Measure collisions, route
completion, perception-induced failures and observation age. Logging camera
frames while simulator geometry controls the route does not establish a
closed-loop vision contribution.

This is a separate work package. It would not by itself demonstrate the
large-scale multimodal model training in the public Vision & Foundation Models
role.[6] Learned visual representations should be added where their effect can
be measured, rather than inferred from a camera in the system diagram.

## Completion artifacts and claim boundaries

A complete core deliverable should include a representative task demonstration,
versioned code and checkpoints, permitted-input contracts, held-out evaluation,
failure analysis, runtime measurements and the classical-versus-learned comparison.
Representative trajectory recording is needed for later rendering; the current
gate's episode tables alone are not a motion video.

Only after measured recovery with hidden parameters is it appropriate to claim
online adaptation. Hardware transfer requires hardware evidence. A single
locomotion work cycle supports a focused execution claim, not general-purpose
humanoid autonomy. Alignment with a company's public engineering needs does not
establish research novelty.

## Public sources reviewed November 15, 2026

[1]: https://www.tesla.com/SI
[2]: https://www.tesla.com/careers/search/job/applied-reinforcement-learning-engineer-whole-body-controls-optimus-276000
[3]: https://www.tesla.com/careers/search/job/si-infrastructure-engineer-network-deployment-inference-optimus-245229
[4]: https://www.tesla.com/careers/search/job/si-infrastructure-engineer-model-optimization-deployment-optimus-247659
[5]: https://www.tesla.com/careers/search/job/software-engineer-validation-infrastructure-tesla-ai--278007
[6]: https://www.tesla.com/careers/search/job/si-engineer-vision-foundation-models-optimus-233307

- [Tesla Optimus mission and software foundations][1].
- [Applied Reinforcement Learning Engineer, Whole Body Controls, Optimus, req. 276000][2].
- [Network Deployment & Inference, Optimus, req. 245229][3].
- [Model Optimization & Deployment, Optimus, req. 247659][4].
- [Validation Infrastructure, Tesla SI, req. 278007][5].
- [Vision & Foundation Models, Optimus, req. 233307][6].

Official job pages and their indexed descriptions were reviewed. The technical
roadmap is a project recommendation; job availability and requirements can change.
