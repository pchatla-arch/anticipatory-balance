# Online actuator adaptation for a frozen humanoid walking policy

Status: the first known-gain feasibility experiment is implemented. Local CPU
checks are recorded below; GPU performance and recovery have not yet been
measured. This portfolio direction has its own timeline, independent of the
Foresight course submission.

## What “adaptive execution” means

The walking policy requests motion. The robot's actuators determine how that
request is realized. If the actuator response changes, the same policy output
can produce a different physical result.

An execution adapter sits between the frozen policy and its actuator commands.
The eventual adaptive version estimates changes from telemetry and adjusts its
bounded correction online. “Execution” refers to realizing motor commands; it
does not refer to running Python jobs or scheduling GPU kernels.

| Component | Job | First feasibility gate | Later adaptive version |
|---|---|---|---|
| Frozen PPO actor | Generate nominal joint position targets for walking | Existing `nominal_s1`; weights and normalization unchanged | Same frozen actor |
| Actuator model | Explain the response from targets to actuation | Injected proportional gain is known to the oracle | Gain, then possibly lag, estimated from telemetry |
| Bounded corrector | Modify outgoing position targets | Classical known-gain correction | Same corrector driven by estimates |
| Confidence logic | Limit correction when identification is unreliable | Not implemented or claimed | Uses residuals, excitation, and estimation uncertainty |
| Learned residual | Address a demonstrated limitation of the classical controller | Not implemented | Added only after a controlled comparison identifies a useful gap |

The first gate is **privileged compensation**, not online adaptation. It asks
whether a useful bounded correction exists. Replacing the true parameters with
online estimates is what makes the eventual controller adaptive.

## Relationship to Foresight and the portfolio

Foresight's high-level controller decides when to switch walking and bracing
skills under uncertain impact forecasts. This experiment concerns the actuator
interface underneath a single walking skill. It reuses the G1 environment,
checkpoint, installation, and evidence infrastructure. It does not establish
that the Foresight brace policy is good, or fix the outstanding Session H hold
evaluation.

The portfolio question is concrete:

> Can an unchanged learned walking policy retain useful performance when its
> actuator response changes, using an identifiable, bounded adaptation module?

This provides work in system identification, dynamics, feedback control, frozen
RL deployment, and reproducible evaluation. Vision is not part of the first
gate. A later perception module would need a specific predictive role, such as
anticipating a change in contact conditions, with its own measured benefit.
Adding depth merely to broaden the title would not strengthen this experiment.

Online motor adaptation is established work. [RMA (RSS 2021)](https://ashish-kmr.github.io/rma-legged-robots/)
uses a base policy with a history-based adaptation module for quadrupeds.
[Stability-Aligned Residual Adaptation](https://arxiv.org/abs/2603.07775)
is also closely related to bounded correction around frozen RL policies.
No first-of-its-kind claim is made for this architecture. The intended evidence
is an engineered system with explicit assumptions, useful recovery, and clear
answers about what estimation and learning contribute.

## Exact first fault: proportional-servo gain loss

The pinned Playground G1 actor produces 29 position-target offsets:

\[
r_k=q_{\mathrm{default}}+0.5a_k.
\]

It updates every 20 ms; MuJoCo advances ten 2 ms physics steps per update.
See the pinned [G1 joystick source](https://github.com/google-deepmind/mujoco_playground/blob/9de993826d1a486f18aec4e8d3179e93ff64ed41/rss2025/mujoco_playground/_src/locomotion/g1/joystick.py)
and [physics stepping helper](https://github.com/google-deepmind/mujoco_playground/blob/9de993826d1a486f18aec4e8d3179e93ff64ed41/rss2025/mujoco_playground/_src/mjx_env.py).

For selected unit-gear position servos, the proportional contribution is

\[
\tau=gK_p(u-q).
\]

The implementation scales **both** `actuator_gainprm[:, 0]` and
`actuator_biasprm[:, 1]` by the same gain. Scaling only the first changes the
equilibrium and would be a different fault. Joint damping and existing torque
limits are unchanged. This models a proportional response mismatch, not a loss
of maximum torque capacity, a broken motor, or a calibrated electrical model.
The parameterization follows MuJoCo's [position actuator definition](https://mujoco.readthedocs.io/en/stable/XMLreference.html#actuator-position).

The first gate selects the twelve hip, knee and ankle actuators using joint
names, not a hardcoded action slice. Gains are 0.85, 0.70, and 0.55. The fault
begins after an initially healthy walking interval. No projectile is launched,
external pushes are disabled, and the original policy observation convention
is preserved.

## Known-gain correction and its limits

At the current measured joint position, a known-gain inverse requests

\[
u=q+\frac{r-q}{g},\qquad
\Delta u=\left(\frac1g-1\right)(r-q).
\]

Illustrative calculation, not a measured result: with `q=0.4 rad`, `r=0.5 rad`,
`Kp=75 Nm/rad`, the healthy proportional torque is 7.5 Nm. At `g=0.8`, the same
target gives 6 Nm. A target of 0.525 rad restores 7.5 Nm at that instant.

The implemented correction has a 0.20 rad amplitude limit and a requested
correction rate limit of 2 rad/s. The actuator's existing control-range clipping
still applies. That clipping can change the *effective* correction more quickly
than its requested rate limit; these are separately recorded quantities, not a
claim that the entire outgoing target trajectory is slew limited.

The correction is computed at 50 Hz and held over the physics substeps. Joint
position changes during that hold, so the formula does not continuously recreate
healthy stiffness. Bounds, sampling, contact dynamics and force saturation all
matter. A simple inverse equation is a wiring check; successful bounded walking
recovery on fresh scenarios is the actual feasibility evidence.

Lag is deliberately deferred. A first-order command filter and a pure delay are
different models, and pure delay cannot be canceled exactly by causal inversion.
Neither is implemented or claimed by `adaptive-servo-gain-gate-v1`.

## Experiment and predeclared decision rule

The gate uses eight-second episodes, forward commands of 0.4 or 0.6 m/s, and
fault onset between 1.5 and 2.5 seconds. It first evaluates 32 matched scenarios
under eight conditions: healthy, healthy with identity adapter, and each of the
three gains with/without the privileged correction.

The least severe passing screening gain is then evaluated on 128 **new** matched
scenarios. Confirmation includes healthy, identity, faulty, and corrected arms.
All screening conditions remain in the evidence, including failures. The small
`smoke` profile uses eight scenarios and never declares a scientific pass.

Primary cost is the mean squared planar local-velocity tracking error over the
fixed post-fault horizon, capped at 4 `(m/s)^2`. A terminal transition and every
remaining step receive this maximum penalty. Consequently, falling early cannot
improve a tracking average by removing difficult timesteps. Post-fault forward
progress, failure rate, pre-fault failures, correction bounds, clipping, and raw
actuator force are reported separately. Raw `actuator_force` is not an audit of
joint-level torque limits.

The engineering gate requires:

1. At least 90% healthy survival and a healthy-adapter identity check.
2. Material degradation: a positive cost gap satisfying at least the larger of
   0.03 absolute cost or 25% of healthy cost, or at least a 15 percentage-point
   increase in failure rate with positive cost gap.
3. At least 20% survival under the selected fault, so the test is not entirely
   catastrophic.
4. The bounded oracle recovers at least half the excess cost and does not
   increase the observed failure rate.
5. On confirmation, the paired bootstrap 95% lower confidence bound on cost
   reduction is positive.
6. Valid numerical state, matched pre-fault failure outcomes, and compliance
   with the requested correction limits.

These are predeclared project decisions, not universal standards or stability
guarantees. The confidence interval is over evaluation episodes for one trained
walker, not variation across independently trained policies. Failure to find a
passing gain rejects this tested setup; it does not prove that all adaptation
methods or all robot platforms are unsuitable.

Nonfinite states are explicit failures. Inactive rollout lanes receive safe
reset inputs before further vectorized simulation. Their later simulated values
cannot change the retained outcome. Statistics use explicit selection rather
than multiplication by masks; `NaN * False` would otherwise still be `NaN`.

## Running on the pod

On the dedicated GPU pod:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/portfolio/adaptive-execution-gate/scripts/start_adaptive_gate.sh)
```

The script checks GitHub access as `pchatla-arch`, restores published checkpoints,
uses an isolated `/workspace/adaptive-execution-gate` checkout, and runs in tmux
`adaptive-gate`. A working existing venv is checked without upgrades. An empty
pod uses the repository's pinned setup. No PPO training or rendering is needed.

View its live output with:

```bash
tmux attach -t adaptive-gate
```

Compact evidence is published to `foresight/experiment-results` under a unique
`results/adaptive_gate_*` directory. `run.json` reports operational completion;
`summary.json` separately reports scientific `pass`, `fail`, or `inconclusive`.
A scientifically failed but correctly completed experiment is useful evidence.

The default attempt budget is 120 minutes **including setup**, with ten minutes
reserved for publication and stop checks. At the stated $2.50/hour rate, two
hours cost $5. This is an attempt budget, not a guaranteed billing limit:
pre-existing idle time, failed publication, or unavailable stop controls can
leave billing running. The script requests Stop only after verifying this run's
upload and checking that the GPU is idle. Check the RunPod dashboard after any
failure. `NO_STOP=1` disables the stop request.

## Validation before paid GPU time

The existing Python environment and checkpoint were used without installations.
Checks cover gain/position-bias consistency, instantaneous torque compensation,
requested correction limits, actuator control clipping, fault onset, policy
action history, restoring the original model after errors, pairing, selection,
numerical failures, and launcher publication/deadline behavior.

An actual G1 CPU physics check compares the new healthy wrapper with the original
environment using the same reset and action. Joint positions, velocities, actor
observation, action history and RNG are bitwise equal after a complete control
step. This is a local regression check, not proof of GPU recovery.

A separate real G1 CPU check executes a two-lane, three-control-step scan with
the published frozen actor: one lane has gain 0.70 without correction and the
other uses the privileged correction. Both lanes remain finite; the correction
path activates and respects its amplitude and requested-rate bounds. This short
check validates the batched integration, not the walking-recovery hypothesis.

Run the lightweight tests with the existing environment:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p 'test_adaptive*.py' -v
```

The optional real-physics identity test additionally uses
`FORESIGHT_RUN_CPU_PHYSICS=1` and requires the normal Playground/Menagerie assets.

GPU/MJX-Warp execution and the recovery hypothesis remain to be tested on the
pod. No positive adaptation or recovery result is claimed in this note.

## If the gate passes

The next stage hides the injected parameter from the controller. The estimator
can use only a specified telemetry interface: commanded targets, joint state,
and any explicitly modeled available effort measurements. Contact forces,
excitation, inertia and measurement noise can confound actuator identification;
the estimator must expose that uncertainty instead of always returning a
confident gain.

Then compare no correction, known-parameter correction, estimated-parameter
correction, and confidence-limited correction on held-out changes. Report
adaptation time, healthy regression, failures, tracking cost, intervention size,
and cases where correction should be withheld. The privileged controller remains
a diagnostic reference, not a deployable baseline.

Only after this comparison should a learned residual be considered. It must
improve a named deficiency of the classical version under the same information,
action bounds and evaluation scenarios. If it does not, the useful result is
that learning was unnecessary for that tested mismatch.

Implementation: [actuator environment](../../foresight/adaptive/actuator_env.py),
[gate analysis](../../foresight/adaptive/gate_analysis.py),
[evaluator](../../scripts/eval_adaptive_gate.py), and
[pod worker](../../scripts/adaptive_gate_pod.py).
