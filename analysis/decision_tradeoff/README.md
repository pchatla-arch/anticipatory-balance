# Does the belief rule allocate a fixed preparation budget more selectively?

This retrospective comparison uses the recorded Session G episodes at 10 Hz detection and 100 ms added delay. The fixed belief controller uses a lead-conditioned impact model and preparation cost 0.05. The reference controller chooses at episode start whether to enable the 0.3 s time-to-contact threshold (TTC) or walk reactively throughout. Its choice is independent of the incoming object. On enabled episodes TTC uses its usual detections. The comparator tests the value of choosing *which* episodes to prepare for when both systems spend a similar average time braced.

## Calibration and evaluation

1. On E18, compute the mean recorded brace time over all 2,048 episodes for never brace (0 s), TTC (0.631865 s) and belief (0.484404 s). Set the TTC probability to 0.484404 / 0.631865 = **0.766626**. The weight is chosen only to match time, without using failures or optimizing on E18h.
2. Apply this fixed probability to each pair of archived TTC and reactive outcomes in E18h. For any metric, the mixture outcome per episode is the weighted average of these two observations. This integrates out the episode-start coin exactly for this set of scenarios. Mixture outcomes are expectations, so there is no integer mixture fall count. A deployment with finitely many coin flips would show additional variation.
3. Compare paired belief-minus-mixture outcomes within E18h. E19 repeats the calculation with a second trained skill pair and the same frozen weight. The Session G runner uses a different scenario seed for E18h, and reuses the E18 seed schedule for E19. Within a pair, projected trajectories are shared across arms. Array alignment follows the evaluator's shared reset keys and concatenation order. The script checks that scheduled hit/miss flags match across all three arms. The original archives omit explicit scenario IDs.
4. Use 10,000 paired scenario bootstrap draws, stratified by scheduled hit versus near-miss, for percentile 95% intervals. Weights, skill checkpoints and the hit/miss counts are held fixed. Intervals describe scenario sampling within each skill pair. E18h had already been viewed for previous analyses before defining this comparison, so the findings are exploratory.

| E18h controller | Failure on 1,407 intended hits | Ever braced on 641 intended near-misses | Mean recorded brace time over all 2,048 episodes |
| --- | ---: | ---: | ---: |
| Never brace | 67.66% | 0% | 0 s |
| TTC enabled every episode | 41.36% | 60.37% | 0.6240 s |
| TTC enabled with probability 0.766626 | 47.50% | 46.28% | 0.4784 s |
| Belief, lead-conditioned | 46.55% | 26.68% | 0.4795 s |

Belief minus mixture on E18h was **−19.61 percentage points** in preparation on scheduled near-misses (95% interval −23.44 to −15.72), **−0.95 points** in failure on scheduled hits (−2.41 to +0.53), and **+1.15 ms** in mean recorded brace time (−13.59 to +16.12). Denominators for these three statistics differ. Here, "near-miss" means the scheduled scenario type and a preparation is any positive recorded brace count. The historical duration field can include time after termination.

On E19 the near-miss difference was **−13.33 points** (−17.35 to −9.17). Hit failure was **1.76 points higher** for belief (+0.16 to +3.33). Recorded time was 0.5013 s for belief and 0.4928 s for the mixture. The two policy pairs change both skills and their reference trajectories, and the response fit comes from pair 1. This stress check measures transfer to these particular checkpoints. The comparison covers one fixed TTC threshold and one preparation budget. A fresh-seed deterministic TTC sweep at comparable cost would test a broader claim.

## Reproduce on CPU

From the repository root, with NumPy and Matplotlib available:

```bash
python3 scripts/analyze_foresight_tradeoff.py
```

The script checks the three input archives against their upstream Git blob identities. It writes [pooled metrics and intervals](results.json), a [vector paper figure](heldout_tradeoff.pdf) and a [PNG preview](heldout_tradeoff.png). See the archived [Session G run configuration](https://github.com/pchatla-arch/anticipatory-balance/blob/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/scripts/run_session_g.sh) and [paired evaluator](https://github.com/pchatla-arch/anticipatory-balance/blob/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/scripts/eval_decision_frontier.py).

## Prospective comparison with a deterministic threshold

Could a single, shorter TTC threshold achieve the same selectivity? E18h saved only one deterministic threshold at this delay. The next experiment calibrates nine thresholds from 0.06 to 0.30 s against the belief controller’s mean applied brace duration on 512 new encounters. It writes the selection before evaluating either controller on 2,048 further paired encounters. It counts brace time through the first terminal transition and scores failure among scheduled hits. The paired 95% interval for the difference in mean duration must lie within ±0.020 s before interpreting the risk comparison as matched. Both near-miss brace counts and failures on scheduled misses are also recorded.

To inspect the fixed configurations and seeds on any machine, in this checkout run:

```bash
python3 scripts/run_foresight_matched_duration.py --plan
```

For an approved run from a prepared Foresight GPU checkout with the project environment active, use `python scripts/run_foresight_matched_duration.py --run`. The runner writes a new `results/E20_matched_*` directory with the protocol, per-episode evidence, calibration choice and verdict or error record. This entry point uses the existing checkpoints and writes locally. Arrange the usual results transfer and pod stop before leaving a paid session. The prospective experiment is pending.
