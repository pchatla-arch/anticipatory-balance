# Foresight: Session F implementation and experiment plan

Reviewed November 14, 2026. Evidence is pinned to Session D/E revision
[`58445c74a02172a94d2e2db36e2b1a8449849c9e`][source]. The existing release
diagnostic is at [`359bbfe9718bf970f7ebdeb4402a54185d811d91`][release].
This document and its companion audit were prepared on CPU. No new robot
experiment or online lookahead result is reported here.

## Decision

Prioritize **A: lead-conditioned response valuation and one-observation
lookahead**, and **B: online rendered-depth evaluation**. Add episode-level
statistics immediately. Attempt **C: matched belief/recurrent PPO gates** after
the sensing and switching contracts are stable, then replicate across training
seeds. Defer a second robot.

Session E motivates these experiments; it does not establish that lookahead
repairs latency. Lead conditioning, release persistence, waiting, and new
information are separate interventions. A useful result may be that better
late-brace valuation matters while waiting rarely helps. Keep that outcome
admissible before running the experiment.

The strongest existing change remains retention: at belief preparation weight
0.1 and 50 Hz/zero configured delay, recorded failure on scheduled hits changed
from 63.65% to 30.46%, while mean braced duration increased from 0.296 to 1.593 s.
This is not evidence attributable to lookahead, and the old duration metric
needs termination masking. See [the Session E review](SESSION_E_REVIEW.md).

## A1. What the existing E6b data actually support

The audit recovered **512 unique scenario groups**, not 16,384 independent
trials: each scenario appears across eight lead settings, two modes, and
projectile/no-projectile conditions. There are no nominal lead 0 or 0.05 s
conditions, and no physical near-miss trajectories. The lead value 9 denotes
bracing from reset; it is not a continuous nine-second lead.

A CPU logistic prediction check used all scheduled BRACE/projectile scenarios
at six leads from 0.1 to 1.0 s, without conditioning on contact. It trained on
one 256-scenario batch and evaluated on the other, then reversed the split.
Every scenario and all its lead variants stayed in one fold. The resulting
3,072 predictions represent 512 independent scenarios.

| Candidate response predictors | Out-of-fold Brier score | Out-of-fold log loss |
|---|---:|---:|
| log momentum | 0.176126 | 0.523354 |
| log momentum and log speed | 0.169637 | 0.507032 |
| log momentum, lead, lead squared, momentum/lead interaction | 0.124299 | 0.389174 |
| Above plus log speed and speed/lead interaction | 0.113999 | 0.362951 |

Lower values are better. Both fold directions favored lead and speed. This is
exploratory prediction evidence under the oracle's fixed protocol, not a
measured improvement in online control. E6b is now development data; reserve
new scenario seeds for final confirmation. Reproduction files are in
[`analysis/session_f/`](../analysis/session_f/README.md), and the complete
[response audit](SESSION_F_RESPONSE_AUDIT.md) records support and limitations.

Three issues prevent directly substituting this fit into the controller:

1. **Release behavior differs.** E6b releases at ground-truth nominal arrival
   plus 0.5 s. Session E retains BRACE until detection timeout; the proposed
   deadline controller uses estimated arrival, minimum commitment, and a cap.
   A response fit describes a complete intervention, including release.
2. **Contact is affected by the action.** Contact counts change from 503/512
   at 0.5 s lead to 205/512 at 1.0 s. Keeping only contacts drops the avoidance
   caused by earlier switching and changes the population being compared.
   Use all scheduled scenarios for total action value; report contact rates
   separately so avoidance is not described as stronger impact absorption.
3. **The lead clocks differ.** Oracle lead refers to nominal arrival; the
   filter's TTC is future entry into a 0.45 m radius. Neither is necessarily
   physical first contact. At nominal 0.1 s lead, the median reconstructed
   switch-to-contact-step-start interval is only 0.04 s, with contact occurring
   within that 0.02 s step. Log actual onset and all time definitions directly.

The two nominally equivalent CONTINUE mode copies also disagree on 4/512
outcomes. Use one fixed baseline and check repeatability before interpreting
small effects. The old generator uses a 2–4 m range, while the frontier uses
3–5 m. The fit targets failure during the recorded six-second episode; some
late contacts lack complete 2.5 s follow-up.

### Minimum new response data

First freeze the release protocol using the existing small diagnostic. Then
collect paired intervention data under that protocol, adding lead 0 and 0.05 s
and any post-nominal-arrival opportunities the controller may use. Add physical
near misses and no-launch controls. Record decision-time inputs, launch
geometry, onset, estimated deadline, actual release and reason, first contact,
first failure, and censoring. Match the deployment geometry.

Model failure, preparation duration, and tracking cost for the same bounded
intervention. Preserve joint uncertainty in hit, speed, momentum, and lead.
Do not substitute mean lead into a nonlinear response model or use future
contact/posture measurements as deployable input. If timing-before-failure is
needed for WAIT, collect it; eventual failure probability alone does not
specify whether failure occurs before the next delivery.

## A2. The smallest defensible lookahead

Call this **one-observation belief-space lookahead**. Ordinary QMDP assumes
full observability after one action, so the explicit observation branch is an
additional belief-space calculation. A QMDP leaf approximation can be named
separately if one is actually implemented. See Ross et al. (2008), sections
2.2.4–2.2.5 and 3, [primary paper][pomdp].

For the initial analytical prototype, use one common decision trigger, one
absolute horizon, and a single wait opportunity. At the trigger choose
CONTINUE through that horizon, one fully specified bounded BRACE intervention,
or WAIT until the next scheduled delivery and then choose CONTINUE/BRACE.
The future choice still occurs if the scheduled frame is empty. This is a
two-stage prototype; repeated replanning needs its own continuation value and
cannot silently reset the wait opportunity.

Let Q_W be the value of waiting and selecting the future action after the
observation; let Q_W0 be the value of the identical physical wait followed by
selection without that measurement. Include pre-delivery failure, the reduced
brace lead, actual preparation cost, and the same absolute horizon in both.

    information value = Q_W - Q_W0
    total wait advantage = Q_W - max(Q_CONTINUE_now, Q_BRACE_now)

Positive information value does not imply positive total wait advantage.
Compare the same permissible brace durations in both delayed alternatives;
otherwise a changing release rule becomes another intervention.

Implementation requirements:

- Integrate action utility over each posterior before choosing its best action.
  Choosing the best action separately for every hidden state grants oracle
  information. Use independent or cross-fitted evaluation samples when
  optimizing noisy Monte Carlo estimates.
- Use the actual next delivery time and its capture timestamp. A delayed frame
  may already have been captured. Do not add one frame period to latency, or
  read the simulator's pending true-state buffers to obtain its content.
- Include an explicit empty scheduled frame, capture-time visibility, and
  uncertainty. A tick without a scheduled frame is a different event. Keep
  filter updates identical across planning ablations.
- Retain mass uncertainty: position observations alone do not reveal mass
  when candidate masses have the same trajectory.
- Include survival before the future decision and count each failure/cost once.
  Disable waiting beyond the fixed horizon. Measure planner runtime and
  unresolved Monte Carlo uncertainty.

The [detailed formulation](SESSION_F_VOI_DESIGN.md) includes delayed-frame and
survival equations. Twelve standard-library analytical tests pass, covering
timing, empty observations, information value, harmful delay, and oracle
leakage. These are formulation checks; the online JAX controller has not been
implemented or GPU-validated by this audit.

### Required comparison ladder

| Comparison | Intended attribution |
|---|---|
| Fixed-lead myopic versus lead-conditioned myopic | Response-model improvement |
| Lead-conditioned myopic versus blind delayed option | Timing and preparation-cost scheduling |
| Blind delayed option versus observation-branch option | Additional information |
| Uninformative versus informative prospective observation | Negative control for information gain |

Keep reactive and tuned TTC as practical anchors. Use the same release rule,
sensor/filter, checkpoint pair, decision opportunities, paired external
scenarios, and horizon within each attribution. Tune cost weights/thresholds
on development seeds, then freeze them before final evaluation. A fixed-lead
privileged policy is a reference, not an optimal performance bound.

## B. Online depth is the next perception milestone

E8 renders during rollout but runs its NumPy detector afterwards. Its reported
1,037 environment steps/s therefore excludes detection. The camera named
`head_cam` is torso-mounted, has 90-degree FOV, and uses known robot/camera pose
and a known 0.15 m sphere radius. The 202 recorded misses all occur at
0.123–0.626 m, consistent with the detector's 0.6 m exclusion. The 458 labelled
false alarms remain close to the actual sphere, predominantly prelaunch or
boundary-label cases; they do not establish a clutter false-positive rate.
Sources: [detector][depthcode], [camera][camera], [E8 results][depthresults].

Start with a small online evaluation using the existing frozen skills:

    rendered depth -> geometric detection -> timestamped queue -> belief -> gate

Ground-truth object position, identity, launch status, and visibility belong
only in evaluator metrics. Sensor packets should carry validity, detection,
capture time, delivery time, and capture pose if needed. Queue actual detections,
preserve missing frames, and backproject with capture pose. Keep the known-radius
simple-scene assumptions explicit. Report first useful detection before contact,
coverage, false preparation, failure, progress, and brace duration over all
scheduled scenarios; do not remove invisible threats.

Run 32–64 scenarios first to establish full-loop correctness and cost. Measure
compile separately from warmed rendering, transfer, detection, filter, decision,
and full rollout. Wall-clock batch processing time is not the configured
simulated sensor latency. Only then size a larger paired evaluation. A successful
online result supports the title's depth-perception claim; offline E8 does not
yet establish it.

## C. Correct the explicit-versus-implicit comparison

E14 is a **feedforward low-level PPO actor**, with 108 inputs and 29 actions.
It receives the latest detection and age, not recurrent history. Its critic
also gets privileged object position/velocity. It is a useful separate
end-to-end baseline, but not a recurrent implicit-belief gate.
Sources: [network configuration][awareconfig], [environment][awareenv].

For H2, compare three learned gates over the same frozen WALK/BRACE pair:

| Gate | Actor information |
|---|---|
| Latest-detection MLP | Latest packet, age/mask, allowed robot state |
| Recurrent PPO | History of the same packets and allowed robot state |
| Belief-summary PPO | Filter summary plus the same allowed robot state |

Match actions, decision cadence, release constraints, reward, critic privilege,
scenario distribution, training interaction budget, and tuning/evaluation splits.
Reset recurrent state correctly and train with sequences. Report the explicit
physics prior supplied by the filter. Evaluate trained and held-out latencies.
Keep low-level E14 comparisons descriptive because their action space and
training task differ.

## D. Statistical reporting starts now

For fixed checkpoints, paired scenario bootstrap intervals can be computed
without retraining skills. Save every episode with stable scenario IDs,
checkpoint/configuration hashes, seeds, first-termination records, and source
revision. Resample scenario IDs jointly across compared methods.

Independent gate-training seeds measure gate-learning variability. Independent
walker/brace pairs measure skill-training variability; their response models
must be refitted or explicitly tested for transfer. More evaluation batches do
not create new trained-policy seeds. Report every seed, and preserve shared
scenario pairing when combining scenario and policy uncertainty. Three policy
seeds are useful replication but do not guarantee precise population inference.

## Execution order and budget control

| Stage | Evidence required before expanding GPU work |
|---|---|
| F0: release diagnostic | Correct paired traces, onset/release/termination accounting, measured tradeoff |
| F1: matched response pilot | Correct timing labels, adequate late-lead support, grouped held-out prediction |
| F2: lookahead pilot | Valid observation branches and costs; no privileged inputs; matched ablations |
| F3: online depth pilot | Real rendered packets control actions; complete-loop timing and visibility accounting |
| G1: matched PPO gates | Recurrent training works; identical information/action/release contracts |
| G2: replication | Frozen tuning choices; independent seeds; paired per-episode inference |

CPU preparation for these stages can overlap. Start with small timed pilots and
explicit run limits, then estimate larger runs from the measured complete loop.
The proposal's one-hour and 1.5-hour allocations are hypotheses, not measured
budgets. Recorded E14 training took 36.14 min plus 66.48 s JIT, not 20–25 min
([training record][awaretraining]).

At a hypothetical 1,000 environment steps/s, 200M steps takes 55.56 hours, not
a week. The measured E8 rollout rate cannot predict CNN/PPO training throughput
because detection and optimization are excluded. Keeping vision-PPO out of the
critical path remains sensible. A second robot needs valid skills, handover,
impact calibration, and sensing; retraining on another robot is replication
across embodiments unless a specific transfer test is defined.

Keep WAIT as a temporal decision over CONTINUE for the initial prototype. HEDGE
needs an additional defensible physical action; LOOK needs active sensing.
Neither is necessary to test whether an extra observation changes the brace
decision. Defer POMCP because the bounded lookahead addresses the immediate
hypothesis with less implementation work, not because two actions make the
POMDP trivial.

## Current readiness

Completed here: pinned source/data review, grouped CPU response audit, precise
lookahead design, twelve analytical CPU checks, and a revised one-command pod
launcher with verified result publication and guarded stop requests. The release
diagnostic remains the next prepared pod task; see [POD_RUNBOOK.md](POD_RUNBOOK.md).
New Session F production JAX,
online depth integration, matched response collection, and recurrent PPO are
future implementation work. Do not launch a long Session F training suite on
the basis of this design document.

[source]: https://github.com/pchatla-arch/anticipatory-balance/tree/58445c74a02172a94d2e2db36e2b1a8449849c9e
[release]: https://github.com/pchatla-arch/anticipatory-balance/commit/359bbfe9718bf970f7ebdeb4402a54185d811d91
[pomdp]: https://www.cs.cmu.edu/~sross1/publications/Ross-OnlinePOMDP-JAIR.pdf
[depthcode]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/scripts/depth_detector.py
[camera]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/foresight/envs/g1_impact/scene.py
[depthresults]: https://github.com/pchatla-arch/anticipatory-balance/tree/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E8_depth_detector
[awareconfig]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/checkpoints/aware_s1/000202342400/ppo_network_config.json
[awareenv]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/foresight/envs/g1_impact/impact_joystick.py
[awaretraining]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/aware_s1/summary.json
