# How often should a walking humanoid prepare?

This CPU analysis asks how frequently the object must be on a scheduled hit trajectory before a more aggressive bracing policy is worth its extra preparations on scheduled near-misses. It uses the frozen G1 policy pairs and fixed decision-rule settings from the Foresight experiments. A near-miss is a scenario label. It is possible for a scheduled near-miss to make physical contact.

## Result at 100 ms of added perception delay

Suppose one brace on a scheduled near-miss costs one tenth of a fall. On the first trained pair, the following table gives the estimated loss per 100 encounters under three illustrative workloads. Smaller values are preferable. Every number uses the same held-out episode archive; the workload changes only the frequency of scheduled hits.

| Scheduled-hit frequency | Reactive walker | TTC threshold, 0.3 s | Lead-conditioned belief, cost parameter 0.05 | Lowest point estimate |
| ---: | ---: | ---: | ---: | --- |
| 5% | 3.38 | 7.80 | 4.86 | Reactive |
| 25% | 16.92 | 14.87 | 13.64 | Belief |
| 75% | 50.75 | 32.53 | 35.58 | TTC |

The 100-ms archive contains 1,407 scheduled hits and 641 scheduled near-misses. TTC had 582 failures on hits and prepared on 387 near-misses. Belief had 655 failures on hits and prepared on 171 near-misses. Both had zero observed failures on misses. At the archived mix, charging 73/216 = 0.338 fall-cost units per near-miss preparation would give these two rules the same empirical loss. At the assumed price of 0.10, their estimated losses instead cross at **39.37% scheduled hits** (paired 95% interval 32.25–49.23%). Among all three policies the two estimated switch points at this price are 11.22% (reactive to belief) and 39.37% (belief to TTC).

At 25% hits, TTC minus belief costs **+1.23** fall-cost units per 100 encounters (paired 95% interval +0.71 to +1.76). At 75% hits the same difference is **−3.05** (−4.31 to −1.80). The price and workload frequencies are illustrative choices. A measured physical or financial cost for a real robot would require separate evidence.

## What changes with delay or policy pair?

| Archived comparison | TTC vs belief crossover at price 0.10 | Paired 95% interval |
| --- | ---: | ---: |
| E18h, first skill pair, 0 ms | No interior crossing; TTC had fewer failures on hits and fewer preparations on near-misses in the sample | — |
| E18h, first skill pair, 100 ms | 39.37% | 32.25–49.23% |
| E18h, first skill pair, 300 ms | 28.45% | 26.13–31.07% |
| E19, second skill pair, 0 ms | No interior crossing; TTC had fewer failures on hits and fewer preparations on near-misses in the sample | — |
| E19, second skill pair, 100 ms | 32.99% | 26.01–42.87% |

These thresholds compare two fixed policies. The reactive walker can still have lower loss where hits are rare. Under the 100-ms E19 comparison the belief-favored region among all three policies is smaller than for E18h. E19 used a different trained walker and brace pair and the original response fit. Its scenario seeds repeat the development schedule. The E19 observations are a transfer check on this specific pair. The archive contains no E19 observation at 300 ms.

## Estimand and uncertainty

For an assumed population fraction of scheduled hits p, the expected cost per encounter for policy j is

$$
L_j(p,c)=p f_{j,h}+(1-p)\left(f_{j,m}+c b_{j,m}\right).
$$

Here f is the fraction of archived episodes that failed within the scheduled hit (h) or scheduled near-miss (m) stratum, b is the fraction of scheduled near-misses with any recorded BRACE interval, and c is the cost of one such preparation in units of one fall. The interface multiplies this loss by 100. Cost accrues once for a preparation anywhere on the miss episode. The purpose-built evaluation leaves out duration, preparation on hits and tracking progress. Those quantities matter for other objectives. The evaluation price c is distinct from the belief controller's fixed internal tuning parameter 0.05.

The calculation reweights observed conditional means. It assumes the mass, speed, geometry, detector characteristics, policy behavior and outcome distribution *within* each scheduled stratum stay as in the original run. The controllers receive no updates about the chosen hit frequency. A change in trajectories, sensing conditions, calibrated priors or hardware requires another experiment. A zero observed miss-failure rate also has sampling uncertainty, especially when projecting to rare-hit workloads.

For uncertainty the script takes 10,000 paired bootstrap resamples within the original hit and miss strata. Within each stratum it resamples the joint per-scenario TTC-minus-belief outcome differences so that fall and preparation outcomes from a scenario stay together. Frequency and price are fixed at their specified values. Intervals are pointwise, conditional on each skill pair and the saved encounter distributions. A crossing interval is published only when every bootstrap draw has a crossing strictly inside (0,1). Zero-delay comparisons have no interior point-estimate crossing. The paired bootstrap at a boundary or with no observed events cannot account for unobserved rare events. Because the archives were previously viewed, this is an exploratory sensitivity analysis.

## Reproduce on CPU

From the repository root with NumPy available, run:

```bash
python3 scripts/analyze_operating_conditions.py
```

The script checks both episode files against their Git blob hashes, verifies aligned scheduled-scenario labels, and emits [the browser data](operating_conditions.json). The JSON includes all arm/stratum event counts, the joint paired difference counts used for uncertainty intervals, and the code revision containing the inputs. Pairing follows the evaluator’s shared reset keys and array order; E18h and E19 do not retain explicit scenario IDs. The preparation measure uses the historical recorded brace count, which can include steps after termination. The registered [fresh deterministic TTC comparison](../decision_tradeoff/README.md#prospective-comparison-with-a-deterministic-threshold) uses applied intervals through termination.

[Archived E18h episodes](https://github.com/pchatla-arch/anticipatory-balance/tree/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/results/E18h_heldout_p5) · [E19 episodes](https://github.com/pchatla-arch/anticipatory-balance/tree/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/results/E19_seed2_p5) · [Session G experiment definitions](https://github.com/pchatla-arch/anticipatory-balance/blob/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/scripts/run_session_g.sh)
