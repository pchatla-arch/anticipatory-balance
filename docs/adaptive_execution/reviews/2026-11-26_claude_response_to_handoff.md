# Claude's response to the project plan and review handoff

**Date:** November 26, 2026, 22:40 PT · **Responds to:** `PROJECT_PLAN_AND_REVIEW_HANDOFF.md` at `98b55e6`
**Author of this note:** Claude (independent reviewer; builder of the telemetry models, scenario families, estimator, heuristic gate and evaluation harness). Prasuna decides everything below.

## 1. Verdict

The project can be completed credibly by December 4 **as an actuator-adaptation study with a navigation demonstration**, from evidence that mostly exists today. What cannot be completed credibly in the time left is a *reliability* result for navigation under faults: that needs seeds, instrumentation and a pre-declared metric that do not exist yet. The gap between the two is the title.

What already supports a paper:

- A compensable fault: a 30% leg-servo gain loss degrades command-following; the true-gain correction restores it (oracle rows in every bench).
- A method with a measured cost structure: the gated point estimate recovers most of the oracle's improvement; the ungated estimate recovers slightly more at ten to fifty times the harmful intervention on a healthy plant.
- A measured identifiability limit: a 0.8 effort-sensor scale is indistinguishable from a weak motor from these packets; both practical methods act on a healthy plant (negative result, kept).
- Withdrawal after an unannounced return to health, with denominators.
- A value-of-timely-information result for the set-based rule (authority versus sample age, replay; G1 memo).
- One end-to-end navigation pair under a fault, with videos, showing the same command-following effect inside the loop.

## 2. Ranked blocking issues for the submission

| # | Issue | Why it blocks | Resolution | Owner, by when |
|---|---|---|---|---|
| B1 | The paper's method (the frozen heuristic, `d756c37`) has no closed-loop evidence at n ≥ 8. The 256-episode CPU bench used the untuned gate and the pre-velocity-channel sensor model; S1a has two episodes per cell. | "Analysis and results" needs a quantitative comparison with uncertainty; n = 2 cannot carry the main table. | **Running now (CPU, started 22:11 PT):** `results/bench_dev_cpu_v1_frozen/{declared,declared_fast,exact}`: 9 families × 4 methods × 8 paired episodes × 300 steps, frozen heuristic, current sensor model, traces saved, AE-012 registration record. CPU physics is bitwise deterministic on this stack, so pairing is exact. About 30 min per preset. | Claude, tonight |
| B2 | The report's abstract, §4 and §8 still read lower `progress_m` as a cost ("reduced forward progress 32.6%"). The handoff accepts that this is reversed; the text has not changed. | A grader will read the abstract. Conclusions must follow from results. | Replace with the stop-phase table and a speed-versus-command figure (edits in §6 below). | ChatGPT edits text; Claude supplies figure; Nov 27 |
| B3 | Title: "Reliable Humanoid Navigation Under Actuator Uncertainty…" promises a reliability result; the navigation evidence is one seed, one cylinder, with the memory rule patched after the first run failed. | "Problem clearly stated" and "conclusions follow" both suffer if the title claims more than the evidence. | Either drop "Reliable" (two options in §5), or run the instrumented batch (B4, §4) *and* still keep "reliable" out of the title and in the objective sentence. | Prasuna decides, Nov 27 |
| B4 | G3 logs no per-tick gain estimate, offset, body-frame velocity, contact or fall state. | Without it the navigation result cannot show what the adapter did, and no new batch is worth running. | Add the fields listed in §4 to `eval_g3_depth_actuator.py`; CPU smoke; Claude reviews the diff. | ChatGPT, Nov 27; Claude review same day |
| B5 | Own words and disclosure. The course requires the submitted prose to be the author's; the current draft is assistant-written and its AI disclosure is generic. The submitted proposal was a different topic ("Commit, Hedge, or Look"). | Course policy; also the "contributions" section is required. | Prasuna writes the 4–6 pages from the Primer and the results documents; disclosure names each tool and its role (text in §6); one sentence on the topic change from the proposal. | Prasuna, Nov 29–Dec 2 |
| B6 | GPU ledger has none of the Nov 26 pod starts; posted charges ≈ $2.3 of the $4.44 cap with recent pods unposted. | Budget rule; reproducibility section needs compute facts. | Add the nine rows to `gpu_ledger.md` from the PR #51 comments, reconcile against the billing page, and state the total in the paper. | ChatGPT rows, Prasuna reconcile; Nov 27 |

Closed by this note: the "17 versus 4 tuning evaluations" discrepancy. `89b6d0d` was a checkpoint commit with 4 entries; the frozen commit `d756c37` has 17 entries and 17 `eval_*` directories, and S1a's `frozen_heuristic.json` records `evaluations: 17`.

Not blocking, and should be dropped from the paper: the 72-row registration (state the scope reduction once, as the handoff proposes); the language interface and any VLA mention (unit-tested stubs, no model ran); the set-membership candidate in closed loop (its replay authority result stands on its own as a figure).

## 3. Claim-to-evidence matrix

| Claim as currently worded | Evidence | Verdict |
|---|---|---|
| "Reliable humanoid navigation under actuator uncertainty" (title) | G3: one seed, both arms reach the goal, 0 contacts; rule patched after G2b failed | **Not supported** as a result; supportable only as the stated objective |
| Heuristic reduces post-onset velocity loss 55% at gain 0.70 (S1a) | 32 rows, n = 2/cell, reproduced | Supported as a development observation; needs B1 for the main table |
| Heuristic "reduces forward progress 32.6%" | NPZ traces: uncorrected walker moves 0.33–0.35 m/s during the commanded stop; healthy 0.06; oracle 0.07–0.12; gate 0.05–0.27 | **Reversed**: the extra distance is a failure to follow commands; correction restores command-following |
| Heuristic withdrew in 2/2 return episodes (5–6 ticks); inverse 0/2 | S1a rows | Supported at n = 2; the 256-bench's old gate withdrew in 1/8, so the frozen gate's withdrawal must be re-measured at n = 8 (B1 run includes `gain_070_return`) |
| Sensor-scale 0.8 triggers harmful correction (gate 0.044 rad, inverse 0.108 rad) | S1a rows; same effect in the 256-bench | Supported; identifiability limit, negative result |
| Heuristic "completed the route 48 ticks earlier" | G3 pair; healthy reference 1,185 ticks (slower than both) | Descriptive only; not a benefit metric in this task |
| Gated walker follows navigator speed commands under fault (+3% vs +32% uncorrected) | 5-tick samples, post-hoc | Consistent with S1a; needs per-tick logging and seeds to be a result |
| Perception never receives the true cylinder position; adapter never receives true gain | Code paths in `obstacle_route.py`, `single_causal_controller.py`; `hidden_gain_evaluation_only` only in frames | Supported from source review; keep the disclosure of simulator localization and extrinsics |
| Depth detector detected the cylinder in 22 frames | Frames 0–105 only, then out of view | Supported; must not be reported as recall |

## 4. The one experiment worth a GPU session, and its protocol

Only if Prasuna wants the navigation demonstration to become a (small) result. Otherwise the paper reports G3 as one pair and this section is future work.

- **Arms (4):** healthy + none; healthy + gate (false-intervention check); fault + none; fault + gate. Fault: gain 0.70 on the leg group, onset tick 200, return tick 900, as in G3.
- **Seeds:** 5, fixed before launch, disjoint from seed 1 (propose 11, 12, 13, 14, 15); 1,200-tick cap; same scene, goal, detector, navigator, frozen walker, frozen heuristic `d756c37`; source commit pinned after B4 lands.
- **Primary metric, declared now:** mean squared body-frame forward-velocity tracking error over ticks 200–900, computed per tick from the logged command and body-frame velocity (the loop analogue of the bench's `postfault_loss`, same cap). Secondary: the same error over ticks 900–end (withdrawal), goal reached / timeout / fall / contact, path length, nonzero-offset ticks on the healthy arms (false intervention), withdrawal tick after 900, packet age and loss.
- **Per-tick fields to log (B4):** tick, command (vx, yaw), body-frame velocity, planar position and yaw, hidden gain (evaluation-only column), ĝ per leg joint, requested and effective offset per leg joint, packets released/lost and age, contact flag, fall/termination flag, navigator reason.
- **Early termination rule:** a fall or timeout scores the primary metric over the ticks actually run and is reported as a failure; it is never dropped.
- **Cost:** 20 episodes × ~5 min ≈ 100 min on an A6000 at $0.69/h ≈ $1.15 plus setup; one manifest, one attempt, 150-minute ceiling, publish before stop. Fits the $4.44 cap with the ledger reconciled first.
- **Analysis:** paired differences per seed, all five shown; 5-seed intervals reported as such. No changes to estimator, gate, memory rule or thresholds after seeing results.

I will write the manifest and the auditor; ChatGPT runs the pod (Prasuna's standing split); I audit the output the same day.

## 5. Title

The paper is about a decision under uncertainty: a hidden actuator gain, observed only through delayed, noisy packets, and a controller that must decide how much to correct and when to hold back. Two options that keep ChatGPT's structure and match the evidence:

1. *Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Bounded Online Gain Correction* (drop "Reliable"; "reliable" becomes the first sentence of the problem statement).
2. *Acting on a Weakening Motor: Gated Online Gain Correction for a Frozen Humanoid Walker, with a Depth-Guided Navigation Demonstration* (names what was measured and what was demonstrated).

## 6. Exact proposed edits to `Course_Report_20261127_Draft.md`

| Section | Original | Replacement |
|---|---|---|
| Abstract | "reduced mean post-onset velocity tracking loss from 0.0855 to 0.0383 (55.2%) but also reduced mean forward progress from 3.62 m to 2.44 m (32.6%)." | "reduced mean post-onset velocity tracking loss from 0.0855 to 0.0383 and restored command-following: during the commanded stop the uncorrected walker kept moving at 0.33–0.35 m/s, while the corrected walker slowed to 0.05–0.27 m/s (oracle 0.07–0.12 m/s; healthy walker 0.06 m/s)." |
| Abstract | "Results show a clear selectivity/progress/observability trade-off" | "Results show a recovery-versus-harm trade-off governed by what the telemetry can identify" |
| §3 Metrics | "forward progress projected onto body heading during the exposure window" | "forward displacement projected onto body heading during the exposure window (reported, but not a task metric: the schedule contains a commanded stop)" |
| §4, "Fault compensation and task progress" | whole paragraph | "**Command-following.** The schedule commands walk, turn, stop and resume. Table X gives the achieved forward speed per segment: the uncorrected faulted walker overshoots every command after onset and does not stop when commanded (0.33–0.35 m/s); the oracle and the gate restore the healthy walker's behaviour. The extra displacement of the uncorrected arm (3.62 m vs 2.19 m for the healthy walker at the same 0.6 m/s command) is this overshoot." + the stop-phase table from the handoff §4 |
| §4 | "Notably, the privileged oracle also reduces measured progress." | delete |
| §4 G3 table | row "Final distance to goal (m)" | delete (threshold crossing); add "Achieved vs commanded speed, ticks 200–900 (5-tick samples, post hoc)" with the audit's values and the caveat |
| §4 G3 text | "The heuristic completed the route 48 control ticks, or 0.96 simulated seconds, earlier" | keep, then add: "The healthy walker on the same scene took 1,185 ticks, so arrival time is not a benefit metric in this task; the difference is descriptive, one seed." |
| §4 G3 text | "The heuristic's final mean estimated leg gain was approximately 0.937 after the return-to-healthy interval" | "The final across-leg gain estimate after restoration was 0.937; per-tick estimates were not retained, so the estimator's behaviour during the fault is not observable in this run." |
| §8 Conclusion | "but produced lower measured progress and remained vulnerable to effort-sensor calibration errors" | "restored command-following including the commanded stop, and remained vulnerable to effort-sensor calibration errors" |
| AI-use disclosure | generic paragraph | "Claude (Anthropic) built the telemetry models, scenario families, RLS estimator and heuristic gate, the tuning run, the evaluation harness and CPU benchmarks, the figures, and reviewed every result file. ChatGPT (OpenAI) built the set-membership candidate and its replay study, the depth detector, waypoint navigator and static-map rule, the G3 integration and pod launchers, operated the GPU pod, and wrote draft reports. Both reviewed each other's code and results. All decisions, merges, the experimental protocol choices and the prose of this paper are the author's own." (Prasuna to verify and rewrite in her words) |
| Intro or Contributions | — | one sentence: the topic changed from the submitted proposal (value-of-information gating under delayed perception) to this one on Nov 21; course staff were informed on <date>. |
| §3 or appendix | — | compute facts: pods, GPU type, hours, cost from the reconciled ledger; CPU benchmark on 2 cores. |

## 7. Plan to December 4 (PT)

| Date | Claude | ChatGPT | Prasuna |
|---|---|---|---|
| Nov 26 night | CPU bench v1 (B1) running; commit results; figure drafts: speed vs command per segment, loss by family × method × telemetry, recovery vs harm | B2 text edits; B4 instrumentation; ledger rows (B6) | Title (B3); go/no-go on §4 batch |
| Nov 27 | Audit bench v1; write §4 manifest and auditor; review B4 diff | CPU smoke of instrumented G3; manifest comment on issue #31 | Reconcile billing; confirm 3 or 4 units; email course staff on the topic change if not done |
| Nov 28 | Audit §4 batch same day; final figures and tables from result files | Run §4 batch (if authorized), publish before stop | — |
| Nov 29–30 | Primer updated with final numbers; deck numbers synced | Report draft updated with audited numbers | Write the paper prose (4–6 pages, course template) from the Primer and results docs; Nov 30: formally retire the Foresight fallback |
| Dec 1–2 | Adversarial read; every number checked against a result file | Adversarial read; reference check | Revise |
| Dec 3 | Final figure check | Final PDF render check | Final edits, disclosure, contributions |
| Dec 4 | — | — | Submit on Gradescope |

If the §4 batch is not authorized or fails, nothing above moves: G3 stays one pair, reported as such, and the paper stands on the bench.

## 8. Decisions requested from Prasuna

1. Title: option 1, option 2, or keep as agreed (with "reliable" then reading as a claim the paper must defend).
2. §4 batch: authorize (~$1.15 plus setup, one attempt) or not.
3. Confirm the split: ChatGPT operates the pod; Claude writes protocols and audits; neither changes a method after results are seen.
4. Units (3 or 4) and whether course staff have been told about the topic change.
