# Contact, failure, preparation and forward progress

## Held-out controller comparison at 100 ms

All four sensor-driven rules saw delayed parametric detections at 10 Hz and used the same frozen walking and bracing skills on 2,048 paired encounters. The reactive reference uses the same replayed encounters without sensing. Counts use **all** scheduled hits (1,407) or all scheduled misses (641), regardless of contact or failure. Progress and recorded brace time average over all 2,048. The belief-family tuning parameter is 0.05.

| Controller | Hit contacts | Hit failures | Hit contact and failure | Miss contacts | Miss failures | Misses with any brace | Brace (s) | Progress (m) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reactive walker | 1,379 | 952 | 951 | 2 | 0 | 0 | 0.000 | 1.454 |
| TTC threshold, 0.30 s | 1,323 | 582 | 581 | 2 | 0 | 387 | 0.624 | 1.491 |
| Belief, lead-conditioned | 1,323 | 655 | 654 | 2 | 0 | 171 | 0.480 | 1.502 |
| Blind wait | 1,332 | 667 | 666 | 3 | 0 | 147 | 0.453 | 1.504 |
| Observation lookahead | 1,337 | 677 | 676 | 3 | 0 | 146 | 0.451 | 1.499 |

TTC and lead-conditioned belief had the same marginal contact count on intended hits. Twenty-five encounters registered contact only under TTC and another 25 only under belief. On intended hits, 108 episodes failed only under belief and 35 only under TTC. Across all encounters, the belief-minus-TTC progress difference was +0.011 m with an approximate paired 95% interval of −0.009 to +0.030 m. On scheduled misses alone it was +0.082 m (+0.068 to +0.097 m). These two rules consumed different amounts of preparation time.

## Tables and units

| File | Coverage | Contents |
| --- | --- | --- |
| [Episode outcomes](episode_outcomes.csv) | All saved primary episode conditions in E2/E2b/E2c/E2d, E6/E6a/E6b/E6c, the two E17b sensor archives and E18/E18h/E19 | Counts by all encounters, scheduled hits, scheduled misses or no-projectile controls, with contact, failure, contact-and-failure, failure without contact, preparation and mean progress. Each available stratum has its own denominator. |
| [Earlier aggregate outcomes](historical_aggregate_outcomes.csv) | Published frontier/sensor rows for E10–E17, including E14 object-aware PPO, and the E17b reactive references | Previously reported conditional rates, progress, brace time and total N. Missing counts and outcomes are marked NA. Historical producers sometimes averaged batch rates, which precludes reconstructing exact counts from the reported percentages. |
| [Paired differences](paired_differences.csv) | Lead belief (0.05) minus TTC (0.30 s) in E18/E18h/E19 wherever both exist; rendered depth minus parametric sensing for every controller/latency in E17b | Contact, failure, preparation, recorded duration and progress. Differences in binary probabilities are proportions. Each row includes the number of encounters with higher and lower outcomes under arm A. |
| [Randomized mixture and lead belief](randomized_mixture_outcomes.csv) | Retrospective duration comparison on E18h and the E19 skill pair | Expected contact, failure, joint contact-and-failure, conditional failure, preparation, brace time and progress. The development-calibrated probability of enabling TTC is 0.766626. The mixture has expected rates rather than observed integer counts. |

Contact and failure are separate latched episode flags. The contact sensor registers contact of the object with tracked body segments. Failure records the parent termination condition or sustained excessive torso tilt under the evaluator. A hit can miss the robot. A scheduled miss can contact it. The conditional failure column uses the count of contact episodes within that row as denominator. Coincidence of the two flags does not establish the order or cause of failure. Historical contact records span the full simulated horizon, which can include contact after failure. Zero contact gives an undefined conditional rate (NA). In E2/E6, the `launch_enabled` field distinguishes no-projectile controls when the original evaluator exposed that setting. In E6, `parameter` is the privileged lead in seconds, including the archived sentinels −1 for continuing to walk and 9 for preparing from episode start.

For the decision frontier, progress sums horizontal pelvis displacement projected onto the local forward direction until the first failure or the horizon. For the E17b sensor loop, progress projects displacement onto the initial heading and stops accumulating once failure is observed. Reported distances are arithmetic means that include failure episodes. The historical brace fields count recorded decision intervals at 20 ms over the logged horizon and can include time beyond failure. In the two historical depth archives this field counts the recorded decision flag. The prospective protocol separately records the applied skill through termination. Durations, distances and conditional rates should be compared within each evaluator and prospective or historical definition.

The randomized mixture is computed by weighting the reactive and TTC episode outcomes before aggregation. For failure given contact, both the joint numerator and the contact denominator are mixed before dividing. Matched *overall* preparation time allows stratum-specific exposure to differ. On E18h the mixture and lead belief used 0.566 and 0.610 s respectively on scheduled hits and 0.287 and 0.192 s on scheduled misses. The held-out fixed-threshold comparison and this randomized analysis answer different questions. [Matched-time analysis and prospective protocol](../decision_tradeoff/README.md)

## Reproduction and uncertainty

From a checkout containing the archived inputs, run:

```bash
python scripts/foresight_outcome_tables.py
```

The script verifies the Git blob SHA-1 of each source against the [manifest](sources.sha1). All input files were verified against [the archived course branch](https://github.com/pchatla-arch/anticipatory-balance/tree/3b91140c606d474f66042e0b6b6bdc0859087cb2/results). The 627 episode-stratum rows and 386 historical aggregate rows are descriptive summaries of repeated controller conditions. Their denominators overlap; summing them would double-count encounters.

Within each historical paired archive, the evaluator fixed the reset keys, projectile trajectories and array ordering across arms. The analysis requires identical scenario-type vectors across compared arms. Historical files lack explicit scenario IDs; equality of these vectors is a partial integrity check. Different archives, skill pairs, and the E14 object-aware PPO reference are never joined as matched episodes. Paired difference intervals use mean(A−B) ± 1.96 times its within-scenario standard error for each stratum. They are pointwise, conditional on the policies and archive, and can degenerate at zero when no discordant event was observed. Comparisons with different algorithms, histories or simulated experiment settings do not provide independent sample replication. Rates observed on 512 or 2,048 simulated encounters do not measure the event rate in deployment.
