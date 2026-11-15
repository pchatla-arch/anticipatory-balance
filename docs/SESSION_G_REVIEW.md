# Session G review and the focused Session H follow-up

Evidence reviewed: scientific branch `session-d` at
`84c161a6d3782a93385d4a258ecc48487e5bfe57`, including Session G results committed
in `65d753f`. The calculations below pool the saved episode arrays rather than
averaging conditional rates across batches. Machine-readable calculations are in
[`analysis/session_g_review/metrics.json`](../analysis/session_g_review/metrics.json).

## What the final seed-pair results show

Both E18 and E19 contain 2,048 scenarios, with 1,433 intended hits and 615 intended
misses. The table reports failure on intended hits at zero added latency. Rates
can differ slightly from the old summary CSVs because those average batch rates.

| Controller | Pair 1: nominal_s1 / brace2_s1 | Pair 2: nominal_s2 / brace2_s2 |
| --- | ---: | ---: |
| Reactive | 983/1433 = 68.60% | 968/1433 = 67.55% |
| Oracle, 0.5 s lead | 510/1433 = 35.59% | 672/1433 = 46.89% |
| TTC, 0.3 s | 538/1433 = 37.54% | 703/1433 = 49.06% |
| TTC, 0.4 s | 520/1433 = 36.29% | 692/1433 = 48.29% |
| Belief lead, preparation cost 0.05 | 519/1433 = 36.22% | 715/1433 = 49.90% |
| Belief VOI, preparation cost 0.05 | 532/1433 = 37.12% | 721/1433 = 50.31% |

Sources: `results/E18_rule_family_p5/episodes.npz` and
`results/E19_seed2_p5/episodes.npz`. The earlier quoted seed-1 result near 0.31
belongs to an earlier implementation/setting, not these final matched G rows.

The paired oracle improvement over reactive walking is 33.01 percentage points
for pair 1 (approximate normal 95% interval 30.44–35.58) and 20.66 points for pair 2
(18.27–23.04). The discordant counts are respectively 491 reactive-only versus
18 oracle-only falls, and 331 versus 35. These intervals quantify scenario
variation within each pair; they do not estimate uncertainty across training
seeds.

The defensible conclusion is that the effect depends materially on the trained
policy pair. Both the walker and BRACE change, and each pair's projectile
reference trajectory is generated from its walker. This does not isolate a
weaker BRACE policy as the cause. Two pairs also provide a limited view of
training-seed variability. E19 uses the pair-1 response surrogate, so response
model transfer is an additional possible contributor to belief-rule performance.

The subsequent [controller audit](technical_notes/controllers_and_freezing.md) also identifies
a training-distribution caveat: the skill registry records that the seed-1 checkpoints
predate the per-skill mass-randomizer fix. BRACE seed 1 was trained with 1–8 kg
randomization despite a configuration declaring 4–16 kg; the later training code uses
the per-skill range. These two pairs therefore do not isolate random-seed variance
under a matched training distribution.

Do not claim an unchanged strict method ranking. At zero delay, belief lead
minus TTC 0.4 is −0.07 points for pair 1 (95% interval −1.32 to +1.18) and +1.61
points for pair 2 (−0.03 to +3.24). The point ordering changes, while both intervals
include zero.

## Held-out decisions: a tradeoff, not a demonstrated VOI improvement

E18h uses fresh scenario seeds and fixed controller parameters. At 100 ms added
latency and belief preparation cost 0.05:

| Controller | Hit failure | Mean brace time, all episodes |
| --- | ---: | ---: |
| TTC, 0.3 s | 41.36% | 0.624 s |
| Belief lead | 46.55% | 0.480 s |
| Blind wait | 47.41% | 0.453 s |
| Observation lookahead | 48.12% | 0.451 s |

The paired observation-lookahead minus blind-wait difference is +0.71 points
(95% interval −0.05 to +1.47). Against belief lead it is +1.56 points (+0.69 to
+2.44), accompanied by less preparation time. This evidence does not demonstrate
that observing once more improves the failure rate. It supports reporting the
failure/preparation tradeoff and comparing at matched preparation cost before
claiming superiority.

Source: `results/E18h_heldout_p5/episodes.npz`. All quoted paired intervals are
descriptive normal approximations without correction for multiple comparisons.
G's archives preserve condition-array order and hit/miss labels; pairing is
inferred from the evaluator's shared reset keys and scenario generation. They
do not retain explicit scenario IDs or all per-episode input parameters.

## Release duration changed materially

At zero delay in pair 1, TTC 0.3 averages 0.607 s of bracing, TTC 0.4 averages
0.706 s, belief lead at cost 0.05 averages 0.741 s, and VOI at the same cost
averages 0.695 s. For TTC 0.3, 1,657 of 1,660 ever-braced episodes record the final
release reason as `passed`; three record staleness. The corrected passed
condition is therefore active.

The release criterion is based on the belief mean's **estimated closest
approach**, subject to minimum commitment and stale-observation release. It is
not a guarantee of releasing exactly 0.3 s after physical passage.

For context, earlier E16 belief lead/cost 0.05/zero delay recorded 1.459 s bracing
and 30.66% hit failures, versus G's 0.741 s and 36.22%. Other implementation
changes occurred between those sessions. The comparison motivates the H hold
ablation but cannot by itself attribute the performance change to hold duration.

## Depth false alarms require the H follow-up

E17b's approximately 21% false-alarm statistic is the fraction of **prelaunch
capture frames** with a detection. It is neither an episode rate nor the fraction
of near-miss episodes in which the robot braces. The parked projectile was in
view, which is a plausible source. The saved aggregate evidence does not prove
that every false alarm came from that object; the out-of-range parked-object
experiment tests the explanation.

At zero delay, depth TTC yields 131/367 = 35.69% hit failures, compared with
113/367 = 30.79% for the parametric source. The paired difference is +4.90 points
(95% interval +2.02 to +7.79). Visibility and center-estimation error are both
different between these arms. The depth result remains a simulation result with
known geometry and simulator pose assumptions.

Sources: `results/E17b_depth_lat0/episodes.npz`,
`results/E17b_depth_lat100/episodes.npz`, and their `table.csv` files. Historical
prelaunch frame numerators/denominators were not retained in those archives.

## Readiness remains qualified

E6c improves on the original readiness calculation, but the historical statistic
still uses post-step trajectory samples, takes ten samples at 50 Hz as 0.2 s
(their first-to-last span is 0.18 s), and reports the start of a stable window
rather than its completion. Its `ready_at_contact` calculation includes the
first contacting interval. At 0.5 s lead, `ready_frac` is 0.78125 while
`ready_at_contact` is 0.830078, reflecting the different window conventions.

Consequently, G should not be described as validating every readiness correction
in the earlier review. Historical `tau_ready_s` should retain its stated
window-start interpretation. The independently published reviewed protocol
(`dd3141a`, branch `foresight/review-rerun-v1`) uses stricter pre-action readiness
and different deadline-based release semantics. Its results must be kept
separate from G/H comparisons.

## Recommended next experiment and cost protection

Session G is already uploaded. The focused next experiment is Session H:

1. Run the depth comparison at 0 and 100 ms with the parked object out of range,
   retaining prelaunch detection counts and timestamp evidence.
2. Compare a fresh 0.3 s release reference with 0.6 and 1.0 s on the same code and
   scenarios. Preserve the G controller semantics for this ablation.
3. Retain explicit episode IDs, pooled counts, source/configuration provenance,
   release diagnostics, and paired results. Do not relabel historical results
   as products of the corrected measurement code.

The Session G launcher now avoids an unnecessary fresh-pod retraining cycle. If
E19 is already complete, it skips the seed-2 training block. Otherwise it checks
and restores the published seed-2 bundles on CPU before reuse. Missing or empty
manifests/configuration/OCDBT payloads, incomplete commit metadata, unfetched LFS
pointers, and actual restore failures stop clearly; they do not silently trigger
replacement training. If no published bundle exists, the original training path
remains available. The six focused tests use temporary fixtures and mocked
dispatch, with no GPU or PPO training.
