# Foresight course paper revision history

## 2026-11-21 · Review 4 · Selective bracing and preparation duration

Renamed the paper *Selective Anticipatory Bracing for a Walking Humanoid under Perception Latency*. Added the deterministic development frontier, a fresh prespecified CPU calibration/confirmation experiment, complete archived outcome tables, explicit release logic and sensor-specific limits. The earlier randomized baseline is a retrospective sensitivity analysis. E17b is a separate closed-loop simulated depth experiment. The full corrected depth comparison and the larger, independent Warp duration comparison are pending.

The protocol was committed before simulation. Calibration selected the 0.24 s threshold with a +10.625 ms difference in mean applied preparation (TTC minus belief). On 32 independent confirmation encounters, the difference was +52.5 ms (paired 95% interval [-86.4, +191.4] ms). The threshold and belief rule had 6/19 and 6/19 scheduled-hit failures, respectively, and 11/13 and 3/13 preparations on scheduled misses. The interval extended outside the equivalence margin. These counts are descriptive; the experiment cannot establish a risk difference at equal duration. [Locked calibration choice](https://github.com/pchatla-arch/anticipatory-balance/blob/aff5b221d08f393d6e6f8b2d167ddae30c4a1bdb/analysis/deterministic_cpu_pilot/freeze.json) · [Fresh run and interval](../analysis/deterministic_cpu_pilot/)

[Point-by-point response](RESPONSE_TO_REVIEW.md) · [Paper](foresight.pdf) · [All archived outcome tables](../analysis/controller_outcomes/) · [Release rule and depth evidence](../docs/foresight/release_and_depth_audit.md)

## 2026-11-21 · Review 3 · What changes when impacts are rare?

Added an exploratory population reweighting with five archived latency/skill-pair comparisons, paired uncertainty and the offline Decision Lab. At the illustrative price of 0.10 falls per unnecessary preparation, the two sensor-driven rules cross at 39.4% scheduled hits on held-out E18h at 100 ms. Figure 4's matched-duration comparison remains a separate question. The manuscript reports the assumed price and uncertainty in the crossover. [Analysis](../analysis/operating_conditions/) · [Decision Lab](../docs/foresight/decision-lab.html)

Recorded and rendered two preselected encounters (seed 73132) on the CPU with a parametric detector and fixed world trajectories. Both policies fell on the scheduled hit. Both stayed upright and walked through the scheduled near-miss. The clip illustrates timing and the limits of a brace. Its four trajectories serve as examples. [Replay and event record](../docs/foresight/replay/)

Added an empty-pod E20 launcher and local tests for the publish-and-verify path. The prospective 512-scenario calibration and up to 2,048 fresh confirmation scenarios still require an authorized GPU run. [Fresh-pod entry](../scripts/start_foresight_e20.sh)

## 2026-11-21 · Review 2: what does the belief buy at the same preparation time?

The first review established the individual controllers' operating points. This revision adds an exploratory experiment using three archived episode sets, a figure with paired uncertainty, and a prospective comparison that would address the strongest alternative explanation.

- **Decision and comparison.** At 10 Hz and 100 ms delay, set the probability of enabling the 0.3 s TTC policy by matching its expected recorded time in BRACE to the lead-conditioned belief rule on E18. Score that fixed mixture on E18h. Estimate its expectation from both saved policy outcomes in each scenario. E18h had already been examined when this analysis was designed.
- **Result.** Across 1,407 held-out scheduled hits the difference in failure probability (belief minus mixture) was −0.95 percentage points, with a paired 95% interval of [−2.41, +0.53]. Across 641 scheduled near-misses, preparation decreased by 19.61 points [15.72, 23.44]. The mean time difference across all encounters was +1.15 ms [−13.59, +16.12].
- **Transfer.** The second policy pair had 13.33 points less preparation on near-misses but 1.76 points more failure on scheduled hits. The two policy pairs change the frozen skills and share the first pair's response fit. Historical durations may include steps after termination.
- **Paper and repository.** The six-page author-review manuscript updates the abstract, comparison protocol, interpretation, limitations and supporting literature. Figure 4 is reproducible from the archived episode arrays. The repository overview now leads with the results readers can reproduce. [Paper](foresight.pdf) · [analysis and figure](../analysis/decision_tradeoff/).
- **Prospective work.** A prepared-checkout runner selects one deterministic TTC threshold on 512 calibration scenarios using duration only. It freezes the choice for 2,048 new paired scenarios and measures applied time through termination. The full GPU experiment is pending. [Protocol and commands](../analysis/decision_tradeoff/#prospective-comparison-with-a-deterministic-threshold).

## 2026-11-21 · Review 1 for the December report

Prasuna Chatla retains the final wording, claims, AI-use disclosure and submission decision. This review updates the LaTeX report and its compiled PDF. It uses archived Foresight experiments through Session G and the partial Session H record. No new course experiment was required for these corrections.

| Evidence | Editorial decision |
| --- | --- |
| E18/E19. Each skill pair has 1,433 scheduled hits among 2,048 scenarios. | Report each pair with its own reactive reference. The walker and brace both change. The seed-1 training range and reuse of the first pair's response fit limit attribution. |
| E18h. There are 1,407 scheduled hits in the held-out set. The TTC, lead-conditioned belief, blind-wait and observation-lookahead rules were compared on fresh scenario seeds with fixed settings. | Give failure and mean brace time together. The paired lookahead-minus-blind-wait failure interval crosses zero at 100 ms. |
| E17b. At zero added delay, depth TTC had 131/367 scheduled-hit failures and the matching parametric TTC had 113/367. | Keep the measured depth comparison distinct from the earlier pilot. Identify the parked-object confound and the frame versus episode denominators. |
| Session H. The two depth smoke stages passed. The hold smoke failed and none of the full depth or hold stages started. | Reserve the corrected full depth comparison and longer-hold conclusions for future evidence. |

### Source record

- [Original course manuscript and PDF](https://github.com/pchatla-arch/anticipatory-balance/tree/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/paper). The seven-page draft was recovered from the head of archived PR #17.
- [Session G audit, including the denominators, paired intervals and readiness conventions](https://github.com/pchatla-arch/anticipatory-balance/blob/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/docs/SESSION_G_REVIEW.md) and [machine-readable calculations](https://github.com/pchatla-arch/anticipatory-balance/blob/e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f/analysis/session_g_review/metrics.json).
- [Partial Session H stage report](https://github.com/pchatla-arch/anticipatory-balance/blob/1d9ee6d3493f58db22954bd0d5c5fe7944826405/results/session_h_20261115T211850Z_1f77c1/report.md).

### Submission handoff

Prasuna should check the claims against the tables, express the interpretation in her own words and complete the end-of-report disclosure with the name of each AI tool used and its actual role. Submit one 4–6-page conference-style PDF to Gradescope by Friday, December 4 at 5 p.m. Pacific. The public instructions list the write-up as the required file for the paper option. A group includes a contribution section after the conclusions. Each student subsequently submits two anonymous one-page peer-review PDFs by Friday, December 11 at 5 p.m. Pacific. [Project instructions](https://aa228.stanford.edu/final-project/) · [course policy](https://aa228.stanford.edu/#llm-usage-policy) · [schedule](https://docs.google.com/spreadsheets/d/1ZvG0SdesUP1hs7TOZUa2ctn9lcbQE4sKsYcXoWXrJxI/edit?usp=sharing).
