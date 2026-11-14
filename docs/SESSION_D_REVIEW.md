# Foresight: review of published Sessions B–D

Reviewed on 2026-11-14. Evidence snapshot: [`session-d` at `8d0302ef1261c3f0df37a08e92c39c6686eca3a8`](https://github.com/pchatla-arch/anticipatory-balance/tree/8d0302ef1261c3f0df37a08e92c39c6686eca3a8).
This review distinguishes completed experiments, subsequent code, and proposed
work. No GPU runs were performed during the review.

## Assessment

The project has a viable experimental core: a trained walking/BRACE pair, a
nontrivial disturbance regime, useful oracle switching results, large paired
decision sweeps, and an initial rendered-depth detector. The published results
do not yet show that the belief rule improves on a tuned TTC baseline, or that
the evaluated decision loop consumes rendered depth. A short release diagnostic
has higher value than repeating the broad observation-rate sweep.

## 1. BRACE v2 establishes that anticipation can help

In E6b, with 512 episodes per condition, the paired BRACE-family baseline falls
in 359/512 episodes (70.12%). Switching to BRACE v2 at a privileged 0.5 s lead,
then holding through arrival plus 0.5 s, reduces this to 211/512 (41.21%).
Contact remains high: 99.80% versus 98.24%. This supports a useful intervention
at that timing and scenario distribution. It does not establish general skill
robustness, an optimal timing rule, or performance under uncertain sensing.

The no-projectile controls have zero observed falls in 512 episodes per tested
BRACE lead time. At 0.5 s lead, mean progress drops from 2.553 to 2.066 m. That
is evidence of a real preparation cost in this protocol. Zero observed failures
is not a proof of universal switching stability.

Longer leads need separate interpretation: at a 1 s lead, contact falls to
40.04%; from episode start it falls to 13.48%. Their lower failure rates combine
changes in exposure/contact with impact tolerance. Report scheduled-hit outcomes
and actual contact rates together, without claiming identical physical impacts.

Source: [E6b summary](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/results/E6b_oracle_v2/summary.json).

## 2. E11 is useful negative evidence for the earlier rule

The actual rates are 10, **25**, and 50 Hz. Two control ticks at dt=0.02 s is
25 Hz; the 20 Hz script comments are stale. Each condition uses 2,048 rollouts
(two batches of 1,024) with one frozen skill pair. The two new rate sweeps have
147,456 condition-rollouts in total, but reuse paired scenarios and are not
that many independent scenarios or training seeds.

The following selects the lowest scheduled-hit fall rate in the tested grid
under a false-preparation budget of at most 50%:

| Detector | Latency | TTC fall rate | Belief fall rate |
|---|---:|---:|---:|
| 10 Hz | 0 ms | 54.80% | 59.80% |
| 25 Hz | 0 ms | 54.03% | 59.89% |
| 50 Hz | 0 ms | 53.06% | 63.31% |
| 10 Hz | 100 ms | 60.04% | 61.07% |
| 25 Hz | 100 ms | 57.37% | 61.96% |
| 50 Hz | 100 ms | 55.28% | 61.84% |
| 10 Hz | 300 ms | No feasible tested point | 66.65% |
| 25 Hz | 300 ms | 62.04% | 65.94% |
| 50 Hz | 300 ms | 62.88% | 64.14% |

This is a common maximum budget, not matched realized preparation cost. For
example, at 10 Hz/0 ms the selected TTC and belief false-preparation rates are
37.44% and 17.48%. Parameters were selected using the outcomes being reported;
these are descriptive frontiers, not held-out policy comparisons. There is no
observed crossover favoring the belief rule under this budget and grid, but
these data do not establish universal TTC dominance.

Reactive 68.80% and oracle 35.48% baselines are reused from E10. The new rate
runs omit reactive rollouts, so their computed progress-loss fields are NaN.
Historical metrics average batch-conditional rates rather than pooling integer
counts. No paired confidence intervals are supplied; small differences should
not be described as significant. The new diagnostic saves full episode records
and counts for future paired analysis.

Sources: [sweep summary](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/results/E11_rate_sweep/summary.json),
[selection code](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/scripts/plot_rate_sweep.py).

## 3. Release logic must be evaluated separately

Results commit `d53bbf84...` precedes the release change `1faf09d2...`. Sessions
C–D therefore evaluate the old rule. Its entry condition requires estimated
TTC > 0.22 s, and its stay condition also requires the entry request. Once
minimum commitment expires, crossing that estimated-TTC threshold can release
BRACE before impact. This is a source-level defect; its quantitative contribution
to the reported gap has not been measured.

The later fix removes that request dependency, but still tests whether the
conditional future-hit TTC is negative. `PF.stats` only includes samples with
positive entry time, returning infinity otherwise. The negative-TTC test cannot
fire. The updated rule instead waits for detector staleness, potentially holding
through the rest of the episode. It also allows stale beliefs to trigger again.

The candidate `deadline` protocol in this branch latches a bounded release time
at the first eligible request and allows one commitment per one-projectile
episode. Historical `legacy` and `stale_only` modes are retained for comparison.
See [the run instructions](EXPERIMENTS_V2.md) for the contract and 64-scenario
diagnostic. A CPU-tested contract is not yet GPU or scientific validation.

Sources: [Session D rules](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/foresight/decision/rules.py),
[future-hit statistic](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/foresight/belief/particle_filter.py).

## 4. E8 demonstrates depth localization, with label limitations

The detector operates on rendered depth, exact robot/camera pose, a ground/height
mask, and an assumed 0.15 m sphere radius. It does not receive the object's true
position as detector input. Ground truth is used to score detections.

There are 12,800 frame-environments from 128 episodes. Under the script's
visibility definition, 3,451/3,653 are detected (94.47%). The reported 458/9,147
false alarms (5.01%) are all within 0.205 m of the real sphere center: 200 are
before launch, 255 have a launched center outside the image, and three have a
center beyond the range cutoff. These are detections outside the evaluation's
eligibility mask, not established spurious-object detections. Partial visibility
and surface-versus-center boundaries plausibly explain many; inspect frames
before assigning exact causes.

All 202 misses are at ranges 0.123–0.626 m. The detector masks near-body surface
points within 0.6 m, so copying 5.53% as an independent dropout probability is
inappropriate. The 1–6 m accepted detections have approximately 2–3 cm per-axis
RMS error, including bias. The fit is exploratory, not a validated Gaussian
likelihood; its negative slope was clipped without a constrained refit.

The camera is attached to `torso_link` near head height. The reported 65.56%
out-of-image fraction covers launched frames, including later frames; it is not
the fraction of incoming threats never visible before impact. The 40.15 ms batch
render timing excludes the offline detector and may include queued GPU work;
it is not a measured complete sensor-to-decision latency.

Most importantly, E11 still calls the ground-truth-based parametric detector,
with FOV120 degrees and noise sigma=.03+.02*range. It neither consumes rendered
frames nor loads the E8 fitted settings. Describe E8 as a separate vision pilot.
The parametric detector also used current launch validity on delayed positions;
this branch now buffers validity at capture time. The release diagnostic uses
zero latency to avoid confounding its comparison with that change.

Sources: [E8 data](https://github.com/pchatla-arch/anticipatory-balance/tree/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/results/E8_depth_detector),
[depth detector](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/scripts/depth_detector.py),
[controller](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/foresight/decision/controller.py).

## 5. E9 is a diagnostic of score/outcome disagreement

At zero latency, the first-batch ECE improves from .2223 at 10 Hz to .1813 at
50 Hz; Brier improves from .2625 to .2153. At 300 ms, the fraction of eventual
contact hits never crossing p_hit=.5 before contact is 73.3% at 10 Hz and 43.0%
at 50 Hz. These numbers motivate studying timing, but do not isolate filter error.

The predicted event is future entry into a 0.45 m pelvis cylinder under constant
relative motion. The label is eventual physical contact under an acting BRACE
controller. The geometry, time horizon, and future policy differ. Future entry
can already be in the past before physical contact occurs. Align these targets
before claiming a calibrated contact posterior or assigning the oracle gap to
perception.

The plots use only the first batch per condition, although JSON/CSV retain both
25/50 Hz batches. Repeated timepoints are correlated within episodes; Wilson
intervals treating them as independent do not provide episode-level uncertainty.
Threshold-crossing medians exclude episodes that never cross and must be paired
with the never-crossed fraction. Recompute uncertainty by episode after aligning
the target and horizon.

Sources: [calibration CSV](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/results/E9_calibration/calibration.csv),
[analysis code](https://github.com/pchatla-arch/anticipatory-balance/blob/8d0302ef1261c3f0df37a08e92c39c6686eca3a8/scripts/calibration.py).

## 6. Remaining experiments and reporting

Session E scripts and an object-aware PPO implementation exist, but no E12/E13/E14
result directories are committed in the reviewed snapshot. Treat them as pending.
The aware actor receives a latest detection, age, and validity rather than object
velocity/history; privileged true motion goes to its critic. It is a useful
feedforward baseline, not yet an information-matched temporal baseline. Its
current evaluator generates projectile reference trajectories using the policy
passed as nominal; persist and reuse one shared scenario bank before comparing
it with the modular controller. Check the one-tick observation alignment too.

Next: run the short release diagnostic; inspect predicted-versus-physical timing;
then freeze the chosen contract and tune on separate seeds. Only then evaluate a
small held-out rate/latency grid, integrate rendered detections, and add the aware
baseline on identical external trajectories. Sequential oracle ablations should
be called descriptive differences, not uniquely attributable causal shares.

Suggested project-update wording:

> Oracle switching with BRACE v2 substantially reduces failures in the selected
> disturbance regime while imposing a measurable locomotion cost. The first
> decision sweeps do not establish an advantage for the belief rule over tuned
> TTC thresholds. Review identified release-logic and probability-target issues
> that are now being isolated with paired diagnostics. A separate rendered-depth
> pilot demonstrates object localization; integration into the decision loop
> remains ongoing.
