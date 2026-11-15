# Foresight paper v3: review against committed results

## Verdict and provenance

The project has made substantial progress. E16 adds a lead-conditioned response model and an actual observation-branch lookahead. E17 connects rendered depth to the belief, skill selector and physics online. Those are implemented experiments, and the earlier objection that depth was only an offline detector study no longer applies to E17.

The paper is a promising course-project report, but it should not be submitted unchanged. Its strongest results survive this review; several claims about readiness, oracle limits, exact inference, the value of information and statistical certainty need narrower wording or targeted verification. The immediate priority is to make the claims match the experiments, then resolve a small number of controller and observation contracts before buying more GPU time.

Reviewed scientific source: [session-d at a4eeeb131f04d200f4726a8db8ae9e6e8f32f601][source]. The uploaded Foresight_paper_v3.pdf is byte-identical to the committed paper/foresight.pdf: 1,186,704 bytes, Git blob 67902b8bc5d89a3baebcc932ba7660c011e32f78. The manuscript is seven PDF pages. Numerical work in this review used saved summaries, CSVs and CPU calculations; it did not rerun MuJoCo, JAX policies or GPU experiments. E6b, E12, E13, E14 and E15 result blobs remain unchanged from the preceding reviewed source; E16 and E17 are the new evidence.

The tables below retain the repository's stored aggregation convention, generally an unweighted mean of batch-specific rates. They are not newly simulated results. Full-precision extracts and calculations are saved in [review_metrics.json](../analysis/paper_v3/review_metrics.json).

## 1. What the new work supports

1. **Preparation is useful in this simulation regime.** E6b's 0.5 s privileged switch reduces failure conditional on contact from about 0.703 to 0.419; its zero-command control does not provide a comparable improvement. Say “shared replayed projectile trajectories,” not “identical physical impacts,” since policy-dependent robot posture and position change contact.
2. **The release rule mattered materially.** The saved old-release ablation and E15 support the diagnosis that using the initiation condition to decide whether to stay braced was harmful. Under the current implementation, selected zero-added-latency points reduce failure on scheduled hits from about 0.69 to 0.30–0.34. This supports the effect of the implemented change, while the remaining release issue below limits its interpretation.
3. **Lead-dependent valuation improves the myopic decision rule in several latency settings.** E16 demonstrates a useful direction: value a partially formed brace instead of rejecting every action inside a hard readiness threshold. Its contribution is clearer than a blanket claim that every belief-based method beats the tuned TTC threshold.
4. **Online rendered-depth anticipation now runs.** E17 renders current state, extracts detections, updates the belief, chooses the frozen skill and advances physics using that skill. The detector does not use true object position or velocity to calculate its centre, although launch validity still uses simulator metadata.
5. **The object-aware PPO policy provides a useful learned comparison.** E14 supplies an additional failure/progress tradeoff. It is a feedforward object-aware policy, not a recurrent implicit-belief baseline, and its evaluation is not a paired comparison on the same external trajectories.

Sources: [E6b summary][e6b], [E15 10 Hz][e15p5], [E15 50 Hz][e15p1], [E16 10 Hz][e16p5], [E16 50 Hz][e16p1], [online depth evaluator][depthloop], [E14 summaries][e14].

The saved E6b episode data permit a stronger paired statement without another GPU run: on 512 scenarios, the 0.5 s BRACE intervention reduces scheduled-trial failure from 359/512 (70.1%) to 211/512 (41.2%). There are 159 CONTINUE-only failures and 11 BRACE-only failures. The estimated reduction is 28.9 percentage points, with an approximate 95% normal interval calculated from paired differences of 24.6–33.2 percentage points. This is within-checkpoint, scenario-level uncertainty; it does not estimate variability across skill-training seeds. It supports the central benefit much more strongly than the unresolved one-to-three-point policy comparisons.

## 2. Highest-priority manuscript corrections

### A. Table 1 does not measure “ready at impact”

The readiness function scans from the first switch through the entire remaining rollout. It does not stop at first contact and does not require the qualifying window to remain inside BRACE. The saved ready fraction is simply the fraction with a finite result. Moreover, the returned time is the **start** of the first retrospectively successful 0.2 s window, not the time when a complete sustained window has elapsed. At 50 Hz, the final sample in that window is 0.18 s after its first sample. The height condition is one-sided, pelvis height at most target plus 3 cm, rather than absolute distance within 3 cm.

Therefore Table 1's 4%, 5%, 12% and 96% must not be labeled “ready at impact,” and the 0.21–0.22 s statistic is not a validated causal minimum preparation time. These errors also affect the abstract, Section 5.2, the explanation of the hard timing gate, and the conclusion.

**Immediate correction:** label the existing statistic as “fraction ever meeting the sustained posture criterion after switching” and describe the reported time as the retrospectively identified window start. Describe 0.22 s as the configured gate derived from that exploratory statistic. To retain a readiness-at-impact claim, recompute a pre-contact, BRACE-active metric from full trajectories if retained; the scalar episode CSV cannot recover it exactly. Otherwise run only the small readiness diagnostic after its definition is fixed.

Source: [readiness function and aggregation][readiness].

### B. The oracle does not establish a 42% skill floor

The full E6b data contain later-lead rows that contradict the repeated plateau claim:

| Nominal lead | Failures / scheduled scenarios | Contacts / scheduled scenarios | Failure conditional on contact |
|---|---:|---:|---:|
| 0.5 s | 211 / 512 = 41.2% | 503 / 512 = 98.2% | 41.9% |
| 0.75 s | 97 / 512 = 18.9% | 412 / 512 = 80.5% | 23.5% |
| 1.0 s | 49 / 512 = 9.6% | 205 / 512 = 40.0% | 23.9% |

The long-lead contact subsets change substantially, so these rows do not isolate greater impact absorption either. They show why neither a plateau nor a universal physical limit can be inferred from the selected 0.5 s point. The fixed-lead oracle is a **privileged timing baseline**, not an optimization over all admissible action timings and holds.

**Immediate correction:** remove “the curve flattens at 0.42,” “whatever the lead,” and “the skill is now the floor.” Replace with: “Residual failures may reflect both skill capability and decision/hold design; the current fixed-lead oracle does not isolate the physical limit.” A stepping recovery skill remains sensible future work, but its necessity is not established by this plateau argument.

Source: [E6b summary][e6b].

### C. The current hold is still not the stated “until passed or stale” contract

The rule releases on a finite expected contact time below minus 0.3 s. However, belief statistics construct expected contact time only from samples whose entry time is positive; otherwise they return infinity. The negative-time release branch is unreachable with the default configuration.

After the minimum commitment, online release therefore depends on detection staleness, not the intended passage event. A stale belief can also request a new brace because the initiation expression lacks a stale guard. The response-model training oracle holds until nominal arrival plus 0.5 s, while the online action cost charges expected arrival plus 0.5 s. Neither matches the implemented stale-driven hold in general.

This does not erase the measured old-versus-new improvement. It changes what that improvement demonstrates and affects both the response-value interpretation and the “perfect sensing” comparison, because visibility and dropout also change how long BRACE is held.

**Immediate correction:** document the implemented hold honestly. **Next code gate:** choose a single explicit hold contract, test initiation/retention/release independently, and ensure the oracle labels, cost model and evaluated actions use that contract. Retain the published E15–E17 results under their original version; label any rerun as a new protocol.

Sources: [rule release][release], [positive-future contact statistics][stats], [oracle hold][oraclehold].

### D. E16 is an approximate lookahead, with a misspecified future sensor under latency

There is a real observation branch. The implementation updates sixteen hypothetical future posteriors, averages action costs within each posterior, then takes the better action; it does not choose actions with privileged sampled mass knowledge. That is a meaningful implementation achievement.

The forecast nevertheless assumes a successful, zero-latency position observation after a fixed full detector period. It ignores the next delivery phase, capture age of that next delivered observation, visibility and the no-detection branch. The rule is evaluated every 20 ms, so “wait a full detector period” at every tick does not always mean “wait for the next observation.” The object can also contact during the wait; the code checks mean TTC rather than integrating that event over the belief. Its waiting prefix charges tracking cost while continuing to walk, which should be reconciled with the stated reward definition.

The Gaussian filter is also approximate as implemented. Initialization of a delayed observation increases position variance by lag-squared times velocity variance but omits the required position–velocity cross-covariance. Velocity process noise is added every control step, while the delayed measurement update treats the backward constant-velocity relation as exact with independent observation noise. Multi-step lookahead uses a simplified process covariance. The omitted initialization cross-covariance is 0.9 at 100 ms and 2.7 at 300 ms per axis for the configured velocity prior variance of 9.

**Immediate correction:** use “approximate Gaussian kinematic belief” and “one-observation lookahead under a simplified future-sensor model.” Remove claims of exact delayed inference and the causal conclusion that one more observation intrinsically has little value. Small gains under this approximation do not identify the true value of information.

**Next code gate:** use actual capture/delivery timestamps and include the empty-observation branch. Compare the same lead model with (i) brace now, (ii) wait with no new information, and (iii) wait with the observation branch. Keep hold, response model, priors and tuning split fixed. This separates the effect of lead valuation, delay and information.

Sources: [lookahead model][lookahead], [belief initialization/update][filter], [action cost and rule][costs], [controller cadence][controller].

### E. Describe the response fit's actual support

The fitter uses only contacting BRACE episodes at leads 0.1, 0.2, 0.3 and 0.5 s: 2,023 rows from 510 distinct scenario groups. It excludes 0.75 s, 1.0 s and from-reset bracing, and caps the modeled lead at 0.5 s. Thus “all of the oracle leads” is incorrect.

The statement that the four displayed Table 1 means are reproduced within 0.03 **is true in-sample**. The fit has not thereby been validated out of sample or for other holds, speeds at equal momentum, or very late actions below the observed minimum lead. The controller also passes one mean TTC to the response model rather than retaining the joint distribution of momentum and lead.

The CPU review reproduced the published lead coefficient, approximately −4.281685 per second. An exploratory scenario-disjoint two-batch check of this exact narrow-support model gave Brier score 0.13379 and log loss 0.41566; both fold coefficients remained negative. This is useful supporting development evidence, not independent policy validation. The model's capped prediction at long leads remains about 0.41 on contacting rows while their observed contact-conditioned rates are about 0.24; those rows are outside its fitted support and have changed contact selection.

The timing labels also differ: E6b's lead is time before nominal arrival, while online TTC is time until entry into a 0.45 m decision region. Reconcile those definitions before treating the online response prediction as calibrated at its supplied lead.

**Immediate correction:** “We fit a contact-conditioned logistic surrogate to oracle leads 0.1–0.5 s; it matches the four training-bin means within 0.03.”

Source: [response fitter][fit].

## 3. Correct reading of the E16 results

The following are best observed failure rates under the paper's false-preparation cap of 50%; parenthesized values are the actual false-preparation rate at each selected point. Selection uses the evaluated sweep, so these are development-frontier points, not held-out chosen-policy estimates.

| Detector / added latency | TTC | Original myopic belief | Lead model | Lookahead |
|---|---:|---:|---:|---:|
| 10 Hz / 0 ms | .314 (.376) | .339 (.416) | .307 (.478) | .316 (.396) |
| 10 Hz / 100 ms | .410 (.426) | .459 (.468) | .438 (.346) | .402 (.495) |
| 10 Hz / 300 ms | no feasible tested point | .636 (.149) | .599 (.218) | .607 (.194) |
| 50 Hz / 100 ms | .364 (.410) | .422 (.489) | .385 (.465) | .391 (.369) |
| 50 Hz / 300 ms | .486 (.434) | .544 (.356) | .500 (.450) | .527 (.341) |

At 10 Hz/100 ms, the selected lookahead point has a similar observed failure rate to TTC, 40.2% versus 41.0%, but **more** false preparations, 49.5% versus 42.6%. Section 6's “matches it while spending far fewer false preparations” is false for these selected points. Neither point dominates the other on both axes, and no paired uncertainty is available for this comparison.

“Recovers about a third of the gap at every setting” is also not a reproducible description. Applying (myopic failure − lead failure) / (myopic failure − TTC failure) to the selected same-cap points gives 43.9%, 63.2% and 76.2% at 10 Hz/100 ms, 50 Hz/100 ms and 50 Hz/300 ms. At 10 Hz/300 ms there is no admissible TTC point, so that fraction is undefined. The different false-preparation expenditures and development-set selection make a single percentage-of-gap slogan less informative than the actual frontier.

**Recommended wording:** “Lead conditioning improves several latency-dependent frontier points. The approximate lookahead reaches a similar observed failure rate to the tuned threshold at 10 Hz/100 ms within the 50% false-preparation cap, while the methods retain different failure/preparation tradeoffs. We do not establish a consistent advantage from information lookahead alone.”

Sources: [E15 10 Hz][e15p5], [E15 50 Hz][e15p1], [E16 10 Hz][e16p5], [E16 50 Hz][e16p1].

## 4. Correct reading of the online depth results

E17 is the strongest new systems result. Its evidence supports a working rendered-depth decision loop with competitive observed failure rates in a restricted launch regime. The exact results are:

| Added latency | Rule | Parametric: failure / false preparation | Depth: failure / false preparation | Relative reduction in false preparation |
|---|---|---:|---:|---:|
| 0 ms | TTC | .243 / .229 | .237 / .110 | 52% |
| 0 ms | Myopic belief | .317 / .338 | .319 / .236 | 30% |
| 0 ms | Lead model | .311 / .255 | .317 / .166 | 35% |
| 0 ms | Lookahead | .322 / .207 | .314 / .138 | 33% |
| 100 ms | TTC | .284 / .454 | .265 / .393 | 14% |
| 100 ms | Myopic belief | .379 / .324 | .368 / .233 | 28% |
| 100 ms | Lead model | .357 / .206 | .325 / .109 | 47% |
| 100 ms | Lookahead | .371 / .170 | .336 / .095 | 44% |

Sources: [E17 zero added latency][depth0], [E17 100 ms][depth100].

At zero added latency, every failure-rate difference is smaller than one percentage point. At 100 ms, depth's observed failure rate is lower by approximately 1.1–3.6 percentage points. False-preparation reductions range from 14% to 52%; “halves false preparations” does not describe all cells. “No loss” should be replaced by a statement about observed point estimates, not statistical equivalence.

The comparison still needs these qualifications:

- **Privileged validity:** the depth queue uses pixel-derived centre estimates but ANDs detector validity with the simulator's launch flag. Exact camera/robot pose and known sphere radius are also assumed. State these simulation assumptions.
- **Timing mismatch:** depth stores launch validity at capture time; the parametric detector applies the delivery-time launch flag to a buffered old position. At 100 ms, it can accept a position captured before launch that depth rejects. Correct capture-time validity before attributing the latency comparison to the observation source.
- **Different sensor models:** the nominal 90-degree setting is shared, but parametric visibility is a pelvis-centred horizontal cone; depth uses the actual camera frustum and close-range, height and pixel-count masks. Parametric sensing adds IID 5.5% dropout; depth has geometric misses without injected IID dropout. These are comparisons of implemented sensors, not an isolated pixels-versus-noise ablation.
- **Changed estimator:** E8 calibrated a median of candidate points; E17 uses their arithmetic mean. The online estimator's error should be measured rather than inheriting the median estimator's calibration without verification. “No object model” should say “no learned detector, with known sphere radius and geometric masks.”
- **Timing units:** the 45–51 ms number measures a synchronized render-and-detect batch of 256 worlds. It excludes subsequent decision, actor and physics work. It is not one-camera latency, and it is not added to simulated observation age. Report batch throughput separately from the explicit 0/100 ms simulated delays.

The 512 scenarios comprise 367 intended hits and 145 labeled misses under the committed runner's seeds and draw sequence. The code saves only aggregate results, not paired episode outcomes or per-batch rows. Small differences cannot be assigned reproducible paired confidence intervals from these committed artifacts. Contact on intended hits is 87–93% for the decision methods versus about 97.6% reactive, so include contact rate and failure conditional on contact alongside scheduled-hit failure.

Sources: [depth extraction, queue and outputs][depthloop], [parametric capture/delivery logic][detector], [E8 detector][e8detector].

## 5. Other corrections that can mostly be made without GPU work

| Manuscript statement | Evidence-based correction |
|---|---|
| Decision layer runs at 10 Hz | The controller is called every 20 ms; 10 Hz is the detector cadence in those conditions. |
| Evaluated near misses use offsets 0.4–0.8 m | The evaluation override fixes the offset at +0.6 m. Distinguish environment/training defaults from evaluated scenarios. |
| E13, E14 and E15 non-perfect detector runs use measured range-independent noise | Their saved configurations retain sigma0=.03, sigma1=.02. E17 uses sigma1=0, while E12's perfect-sensing condition sets both noise coefficients to zero. Describe each historical artifact by its saved configuration. |
| The oracle ladder removes one limitation at a time | Perfect sensing jointly removes noise/dropout and changes FOV/range; changed visibility also changes stale-driven hold duration. Treat it as a combined observation ablation. |
| A broad mass prior proves uncertainty helps, while a point prior proves overconfidence harms | Point 8 kg, uniform 4–16 kg and uniform 1–32 kg differ in mean as well as spread; cost tuning also changes. Compare equal-mean priors before attributing effects to uncertainty alone. |
| Commitment length makes no difference | Only the tested 0.1–0.5 s settings under the current longer stale-driven hold showed little change. This is not general commitment irrelevance. |
| Aware PPO training took 25 minutes and finished at reward 2.8 | The saved run reports 2,168.51 s, about 36.1 minutes, training time; final reward 2.7429, best 2.7852. |
| Aware PPO gains progress by never stopping | External BRACE usage is zero by construction; that does not establish what joint-level behavior it learned. Report progress without an unobserved behavioral explanation. |
| One third of launched objects are never in the image | E8's 65.56% is a launched-frame outside-image fraction, including post-pass frames. The saved CSV has 15/128 episodes with no launched frame whose centre is in view. Neither statistic measures pre-impact threat coverage. |
| E8's 5% false alarms describe spurious objects | All 458 nominal false alarms are near the actual sphere: 200 prelaunch, 255 centre outside image, 3 centre outside range. Define the labeling event explicitly. |
| Rendering inside PPO training runs at about 1,000 steps/s | The cited measurement is a rendering rollout benchmark, not a pixel-PPO training benchmark. Do not turn it into an established training-cost claim. |
| Differences of .03 are real because N=2,048 | Use the hit/miss denominators, paired outcome differences and uncertainty; account for sweep selection and the single skill-seed pair. E17 has only 512 total scenarios. |

Sources: [controller][controller], [scenario override][scenario], [E12 perfect-sensing result][perfect], [E14 training summary][awaretrain], [E14 configuration][awareconfig], [E8 observations][e8data], and saved E13/E15 summaries at the reviewed commit.

## 6. Revision and experiment order

### First: strengthen the paper with the evidence already available

1. Correct readiness labels, remove the physical-floor claim, and distinguish the fixed oracle baseline from an optimum.
2. Replace blanket “one third,” “halves,” “no loss” and “exact” claims with the actual conditions and numbers above.
3. Add a compact protocol table: detector cadence, added latency, hold/release version, noise, FOV, launch azimuth, near-miss offset, checkpoint IDs, and metric denominators. Keep E15, E16 and E17 scenario regimes distinct.
4. Generate manuscript tables directly from full-precision JSON. Table 3 currently includes double-rounding differences at 100 ms: .264614 rounds to .26, .324645 to .32, and .094762 to .09 at two decimal places.
5. Shorten the abstract to one problem, one architecture, three principal findings and one limitation. Move the debugging chronology and secondary ablations out of the main narrative.
6. Improve Figure 5: its six panels are too small in a single column. Use a full-width figure or select the few panels supporting the argument. Figure 1's cell labels also need enlargement. The inspected PDF pages have no obvious overlapping content, but readability is a stronger concern than layout breakage.

### Second: complete CPU contracts before another paid run

- Specify separate initiation, minimum commitment and release semantics; test positive-TTC, passed-object, stale-observation and restart cases.
- Fix capture-time validity and record both capture and delivery timestamps.
- Choose a consistent delayed-filter formulation, test initialization covariance and propagate process covariance consistently.
- Make lookahead timing/no-observation assumptions explicit and add the blind-wait comparison using the same response model.
- Ensure each experiment records source SHA, model/checkpoint hashes, all overrides, scenario IDs and per-episode outcomes. Do not treat “summary.json exists” as sufficient provenance for a cached run.

### Third: spend GPU time on a small confirmatory set

Run a small gate for the corrected hold/timestamp/readiness contracts, then only the affected decision and depth comparisons. Keep thresholds/costs chosen on development scenarios fixed for a fresh test set, retain episode-level pairing, and report progress plus brace duration as well as false preparation. Repeat across independent skill seeds after the protocol is stable. Measure wall time on the first small batch before quoting a compute budget.

The highest-value scientific ablation is lead-only versus blind wait versus observation lookahead with matched execution and observation contracts. The highest-value systems check is the online depth comparison with matched capture validity and calibration of the detector actually used. A second humanoid, terrain locomotion or a new stepping skill would add scope before these claims are secure.

## 7. Recommended central claim

Foresight studies when a walking humanoid should pay the locomotion cost of preparing for an uncertain external impact. Two frozen PPO skills separate low-level behavior from the anticipation decision. Paired simulation experiments show that preparation can reduce failure, that the initiation/retention contract has a large effect on outcomes, and that lead-conditioned response models improve several latency-dependent decision tradeoffs. An online rendered-depth detector also drives the decision loop successfully in a restricted visual regime. The current results do not establish that belief lookahead consistently dominates a tuned TTC threshold, nor that remaining failures identify a physical skill limit. Exact timing, observation assumptions and held-out replication are the next steps needed to strengthen those conclusions.

[source]: https://github.com/pchatla-arch/anticipatory-balance/tree/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601
[e6b]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E6b_oracle_v2/summary.json
[e15p5]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E15_rule_fixed_p5/summary.json
[e15p1]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E15_rule_fixed_p1/summary.json
[e16p5]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E16_lookahead_p5/summary.json
[e16p1]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E16_lookahead_p1/summary.json
[e14]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E14_aware_ppo/summary.json
[readiness]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/scripts/eval_oracle_brace.py#L50-L67
[release]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/decision/rules.py#L136-L154
[stats]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/belief/particle_filter.py#L125-L153
[oraclehold]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/eval/rollout.py#L171-L178
[lookahead]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/belief/particle_filter.py#L156-L186
[filter]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/belief/particle_filter.py#L70-L113
[costs]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/decision/rules.py#L85-L118
[controller]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/decision/controller.py#L35-L79
[fit]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/scripts/fit_response_model.py#L32-L58
[depth0]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E17_depth_closed_loop/summary.json
[depth100]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E17_depth_closed_loop_lat100/summary.json
[depthloop]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/scripts/eval_depth_closed_loop.py#L142-L220
[detector]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/perception/detections.py#L54-L74
[e8detector]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/scripts/depth_detector.py#L65-L76
[scenario]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/foresight/eval/rollout.py#L132-L136
[perfect]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E12_perfect/summary.json
[awaretrain]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/aware_s1/summary.json
[awareconfig]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/aware_s1/config.json
[e8data]: https://github.com/pchatla-arch/anticipatory-balance/blob/a4eeeb131f04d200f4726a8db8ae9e6e8f32f601/results/E8_depth_detector/detections.csv
