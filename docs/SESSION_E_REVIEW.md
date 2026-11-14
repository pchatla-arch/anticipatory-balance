# Foresight: Session E results review

Reviewed November 14, 2026, against `session-d` at
[`58445c74a02172a94d2e2db36e2b1a8449849c9e`](https://github.com/pchatla-arch/anticipatory-balance/tree/58445c74a02172a94d2e2db36e2b1a8449849c9e).
The published results arrived in `c3d12db5`; the subsequent commit updates the
paper and README. This review used source and output inspection plus CPU
calculations. No GPU experiments were rerun.

## Assessment

Session E substantially strengthens the project. The old-release ablation now
quantifies a large failure reduction from preserving BRACE after its entry
condition expires. Both modular decision rules achieve approximately 30% failure
at zero configured latency, compared with approximately 69% for reactive walking.
The object-aware PPO policy also completed training and evaluation. These are
useful results. The paper should distinguish them from claims of correct passage
release, optimal POMDP decisions, causal error attribution, or a skill ceiling.

## 1. The strongest result: persistence changes the outcome

The cleanest comparison is E13 old release versus E15 at the same belief
preparation weight, c_prep=0.1, 50 Hz detections, zero configured delay:

| Metric | Old release | Session E retention |
|---|---:|---:|
| Failure on scheduled hits | 63.65% | 30.46% |
| False preparation on scheduled near-misses | 50.08% | 49.92% |
| Ever braced on scheduled hits | 98.60% | 98.60% |
| Mean braced duration, all episodes | 0.296 s | 1.593 s |

This is a 33.20 percentage-point observed failure reduction (52.15% relative).
Trigger incidence barely changes, while bracing lasts about 5.38 times longer.
The large benefit is therefore associated with retaining the intervention after
it starts. The same release change also helps TTC: at theta=0.3, failure changes
from 52.85% to 29.69%.

This is not a comparison at equal duration cost. The recorded false-preparation
metric counts whether BRACE ever occurs on a near-miss; it does not charge a
longer brace more than a shorter one. The braced-duration metric also includes
scan steps after first termination in this historical evaluator. Precise
cost comparisons should use the newer termination-masked records.

Sources: [old-release output][old], [E15 50 Hz output][p1],
[metric definitions and aggregation][eval].

## 2. Actual release is still controlled by detection timeout

Session E removes `want` from the stay-braced condition. It still tries to
recognize passage using `isfinite(b.t_hit) & (b.t_hit < -release_after)`.
The PF only counts samples whose entry time is positive and returns their
conditional average, or positive infinity. That passage predicate cannot fire
under the configured model. After the minimum commitment, release is driven by
more than 50 steps without a detection. The `hold=0.5` parameter enters the
entry-cost expression but does not implement a post-impact hold timer.

The results therefore evaluate a timeout-based retention rule. They do not
evaluate the explicit latched-deadline candidate on `foresight/experiments-v2`.
Keep the timeout rule as an empirical baseline when testing a deadline: the
deadline is easier to audit, but its physical performance remains to be measured.
Do not replace the paper description with a claimed physical-passage mechanism.

Sources: [rules][rules], [future-entry statistic][pf].

## 3. The frontier is promising, with no uniform method winner

At 50 Hz and zero configured delay, all from the same E15 summary:

| Controller | Failure on hits | False preparation | Mean braced time | Progress |
|---|---:|---:|---:|---:|
| Reactive | 68.74% | 0% | 0 s | 1.428 m |
| TTC theta=0.2 | 36.22% | 14.45% | 1.229 s | 1.484 m |
| Belief c_prep=0.2 | 30.81% | 20.19% | 1.351 s | 1.414 m |
| TTC theta=0.3 | 29.69% | 25.85% | 1.327 s | 1.464 m |
| Belief c_prep=0.1 | 30.46% | 49.92% | 1.593 s | 1.315 m |
| Privileged fixed 0.5 s lead | 35.83% | 0% | 0.700 s | 1.458 m |

The c_prep=0.2 belief point is more selective on near-miss episodes than TTC
theta=0.3, but is not cheaper in mean braced time. At a 25% false-preparation
ceiling, the best tested belief point has lower failure than the best feasible
TTC point. This is an exploratory comparison on a coarse grid: TTC theta=0.3
misses that ceiling by less than one percentage point, and denser thresholds
could alter the conclusion. At the paper's 50% ceiling, TTC has the lowest
observed failure where both methods have a feasible tested point.

Best tested scheduled-hit failure under false preparation <=50%:

| Rate | Delay | TTC | Belief |
|---|---:|---:|---:|
| 10 Hz | 0 ms | 31.43% | 33.87% |
| 10 Hz | 100 ms | 40.97% | 45.93% |
| 10 Hz | 300 ms | No feasible tested point | 63.57% |
| 50 Hz | 0 ms | 29.69% | 30.46% |
| 50 Hz | 100 ms | 36.38% | 42.24% |
| 50 Hz | 300 ms | 48.57% | 54.43% |

Each condition reports 2,048 total rollouts; hit and near-miss rates use their
respective subsets. These are unweighted means of batch-conditional rates.
Operating points are selected using the evaluation outcomes, without a held-out
tuning split. The sub-percentage-point TTC/belief difference at 50 Hz/0 ms is
not a demonstrated significant difference. No paired discordance data are saved.

Sources: [E15 50 Hz][p1], [E15 10 Hz][p5], [sweep][sweep].

## 4. The ladder cannot identify a sensing share or a skill ceiling

The fixed 0.5 s oracle is a privileged timing reference, not an optimal controller
or upper bound. Other controllers exceed its fall performance while using
different start times and substantially longer persistence. The recorded ladder
contains improvements greater than 100% of reactive-minus-oracle and negative
terminal shares. Those are arithmetic differences, not a causal error budget.

The idealized-sensing condition simultaneously removes noise and random misses,
widens FOV from 120 to 360 degrees, and extends range from 6 to 20 m. It also
changes how long detections remain fresh under the timeout-release rule:

| Selected point | Standard 50 Hz | Idealized sensing |
|---|---:|---:|
| TTC failure | 29.69% | 28.64% |
| TTC mean braced time | 1.327 s | 2.799 s |
| TTC progress | 1.464 m | 0.968 m |
| Belief failure | 30.46% | 31.79% |
| Belief mean braced time | 1.593 s | 3.053 s |
| Belief progress | 1.315 m | 0.849 m |

Aggregate behavior is consistent with longer persistence when the sensor sees
more of the trajectory; it does not directly trace each release. The experiment
does not isolate sensing noise, and the remaining failures cannot be assigned
solely to the skill. Call it an information-and-policy sensitivity ladder.

Sources: [perfect-sensing output][perfect], [ladder][ladder], [rules][rules].

## 5. Ablation conclusions need narrower wording

- Minimum commitments of 0.1, 0.3, and 0.5 s show little change in this timeout
  regime. All are shorter than the 1 s detection timeout; this does not establish
  that commitment is generally irrelevant.
- The priors change both mean and variance: U(4,16) has mean 10 kg, U(8,8.01)
  mean 8.005 kg, and U(1,32) mean 16.5 kg. They do not isolate uncertainty
  representation. At fixed c_prep=0.2, the wide prior raises failure from 30.81%
  to 61.49%; retuning c_prep=.05 recovers a competitive point. Describe this as
  prior sensitivity with retuning, not unconditional robustness.
- Changing tau_brace tests a start gate, not physical readiness. Lowering it to
  0.10 s allows later interventions; raising it to 0.35 s suppresses them. A
  lead-time-conditioned benefit model is a useful next hypothesis, not yet a
  validated improvement.
- Changing r_contact changes both the hit event and entry time. Its lower
  false-preparation operating points are useful sensitivity evidence.
- TTC does not use tau_brace or the mass prior in its request, yet those runs
  show small differences at the same threshold. Repeatability needs checking
  before ranking sub-percentage-point ablation gains.

Sources: [Session E commands][run], [prior-point results][point],
[wide-prior results][wide], [E15 reference][p1].

## 6. Object-aware PPO is a completed, useful baseline

| Actor observation setting | Failure on hits | Contact on hits | Progress |
|---|---:|---:|---:|
| 10 Hz / configured 100 ms | 44.46% | 95.40% | 1.902 m |
| 50 Hz / configured 0 ms | 40.97% | 95.19% | 1.956 m |

The 50 Hz override genuinely configures the actor's internal detector. The
actor acts at 50 Hz in both settings. Its input is a latest-detection summary
(valid flag, relative position, age), without explicit object history, velocity,
or recurrent state; the critic gets privileged position/velocity. This is a
feedforward latest-observation baseline, not equal temporal information to the PF.

Static call order adds one control tick between the aware detector's update and
the actor consuming that observation: configured 0/100 ms corresponds to a fresh
feature age of 20/120 ms. This is a code-derived alignment issue, not a measured
latency experiment. Each aware evaluation also regenerates projectile trajectories
against its own policy and sensor setting. The paper already acknowledges the
aware-versus-modular trajectory difference; it also limits comparison between
the two aware settings. Reuse one external trajectory bank for a paired study.

The aware policy has no explicit BRACE mode, so its zero false-preparation and
braced-time counters cannot establish absence of learned slowing or postural
preparation. Its reported zero progress loss subtracts its sole row from itself.
Report absolute progress, and remove the unobserved claim that it never stops.

Training completed 202,342,400 steps, final reward 2.7429, best 2.7852, in 36.1
training minutes plus 66.5 s JIT on an RTX A6000. The paper's 25 minutes and final
reward 2.8 need correction. A post-training rendering traceback follows completed
training; evaluation and the published checkpoint show that training was usable.
The aware randomizer correctly uses 4–16 kg, unlike the original frozen skills'
1–8 kg mass randomization. The observations still use the assumed synthetic
sigma=.03+.02*range, 120-degree FOV model, not E8-fitted rendered observations.

Sources: [aware 10 Hz][aware10], [aware 50 Hz][aware50],
[training record][training], [actor/environment code][env], [evaluator][eval].

## 7. Next work and reporting

Keep these results. The immediate GPU task is the existing small paired release
trace diagnostic from `359bbfe`, not another full Session E run. Compare timeout
and deadline contracts on their failure/progress/duration tradeoff. Then freeze
the contract, use held-out scenario seeds and denser thresholds near the desired
preparation budget, and save per-episode outcomes and pre-termination costs.

Session E deliberately disables calibration dumps and saves only aggregates in
these evaluation directories. It cannot support retrospective paired confidence
intervals or detailed timing claims. The newer runner's episode and event traces
address that gap; its GPU integration remains pending unless separately reported.

Paper wording should describe a major improvement from revised BRACE persistence,
a latency-dependent failure/preparation tradeoff, and a useful learned baseline.
Remove claims that sensing or belief errors have been ruled out, that less than
10% of an identifiable gap is due to perception, that the remaining failures are
a proven skill ceiling, or that the evaluated observation model is measured depth.
Most of these corrections require no additional GPU time.

[old]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E13_release_old/summary.json
[p1]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E15_rule_fixed_p1/summary.json
[p5]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E15_rule_fixed_p5/summary.json
[sweep]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E15_rate_sweep/summary.json
[perfect]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E12_perfect/summary.json
[ladder]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E12_oracle_gap/summary.json
[point]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E13_prior_point/summary.json
[wide]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E13_prior_wide/summary.json
[rules]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/foresight/decision/rules.py
[pf]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/foresight/belief/particle_filter.py
[eval]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/scripts/eval_decision_frontier.py
[run]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/scripts/run_session_e.sh
[aware10]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E14_aware_ppo/summary.json
[aware50]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/E14_aware_ppo_50hz/summary.json
[training]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/results/aware_s1/summary.json
[env]: https://github.com/pchatla-arch/anticipatory-balance/blob/58445c74a02172a94d2e2db36e2b1a8449849c9e/foresight/envs/g1_impact/impact_joystick.py
