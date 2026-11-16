# Adaptive execution document history

This is an append-only explanation of controlled revisions. The machine register
is document_versions.json; human decisions are recorded separately in
decision_log.md. A version identifies document content, not an experiment's
success. Current working artifacts use stable paths; historical copies use full
commit permalinks. Do not invent an earlier version for an unversioned file.

## OAA-PLAN 0.1.0

**Date:** November 15, 2026  
**State:** Proposed amendment; Claude cross-review and Prasuna merge pending.  
**Previous baseline:** Unversioned execution plan at
[b9103ec0a67724f598ccb91bdcd00221464e31fd](https://github.com/pchatla-arch/anticipatory-balance/blob/b9103ec0a67724f598ccb91bdcd00221464e31fd/docs/adaptive_execution/execution_plan.md),
merged by Prasuna in [PR #16](https://github.com/pchatla-arch/anticipatory-balance/pull/16).
This is the first controlled plan version, not a retroactive renumbering.

**Reason.** Prasuna requested document version control, detailed update history
and a co-work plan with Prasuna as gatekeeper.

**Detailed changes.**

- Replace the external-document mirror description with canonical GitHub source.
  Define an implementation owner, cross-reviewer and human decision authority.
- Record proposal 0.3 selection and the completed CPU telemetry/interface work;
  keep cross-review, live integration, GPU recovery and scientific claims pending.
- Apply independent authorship to the primary heuristic comparator; retain
  cross-review of secondary baselines, including ChatGPT's robust-PPO work.
- Add the missing trace generator/recorder and explicit ownership of the shared
  RLS estimator, inverse candidate and reset behavior. Both primary methods
  consume the same common physical constraints.
- Correct the test dependency: validation and freeze package -> G2 approval ->
  registered held-out evaluation -> G3 claim review.
- Replace the requirement that every confound cause false adaptation with
  declared sensor/intervention semantics and measured outcomes. Preserve
  legitimate negative controls.
- Split the same-effort GRU comparison from the optional no-effort study; the
  latter needs its own valid measurement/uncertainty contract.
- Distinguish CPU/JAX logic checks from Warp/GPU integration, compile/smoke
  execution and timing. Do not duplicate paid runs merely for peer review.
- Make G1 targets and G2 statistical registration explicit pending deliverables,
  to be accepted before confirmation/testing. Post-test favorable subsets are
  exploratory until new confirmatory evidence exists.
- Locate the authority script at
  [71204bac4f2c38389f4a85a256409ba3c43959ec](https://github.com/pchatla-arch/anticipatory-balance/blob/71204bac4f2c38389f4a85a256409ba3c43959ec/docs/adaptive_execution/reviews/check_authority.py).
  It contains an illustrative acceleration allowance and per-joint 99th
  percentile speeds from the same rollout. The reported percentages remain
  unreplicated sign-screen rates, not full feasible authority.
- Retain the original date targets as dependency-dependent planning windows,
  keep Foresight's Dec 4 submission separate, and add an actionable 72-hour
  handoff.
- Add source/PDF version identity, a build manifest, detailed history, decision
  ledger and a standard-library consistency checker.

**Experimental impact.** No change to Stage A's controller, launcher, checkpoint,
eligibility criteria, metrics or budget. No new GPU job, package installation,
training result or recovery claim is introduced. Later gates require their own
approved configuration and evidence.

**Verification.** Focused source review resolved the circular gate dependency and
other listed inconsistencies. PDF visual inspection and exact build-input/output
hashes are recorded in document_qa.json and execution_plan.build.json. The
consistency checker rejects stale files, unsupported approval declarations and
changed artifacts without a version increment.

**Remaining work.** Claude reviews the exact amendment, Prasuna decides its merge,
and each package supplies its own code/replay/run evidence. This record does not
claim that the two assistants have a direct session channel.

**Release identity.** The current PR's commit and registered SHA-256 values
identify this working release. Its immutable commit is added to the historical
register in the next release, avoiding a self-referential commit hash.

## OAA-PROP 0.3.1

**Date:** November 15, 2026  
**Published revision:**
[1614b65c65ae89090b76741fab8716225b50c1f2](https://github.com/pchatla-arch/anticipatory-balance/commit/1614b65c65ae89090b76741fab8716225b50c1f2).  
**Previous baseline:** OAA-PROP 0.3 at e7c7854.  
**Document:** research_candidate/proposal.md and proposal.pdf, 17 pages.

**Reason.** Record Prasuna's selection and incorporate AE-001 through AE-003.

**Detailed changes.** Specify synchronized effort/state sampling and applied
actuator torque after joint limits. Add an opt-in native/MJX telemetry helper,
separate raw-force diagnostics and saturation masks. Clarify which sensor
channels can be confounded by friction, payload, timing or scale. Distinguish
reported error-sign screens from full feasible correction authority. Separate
no-effort identification from the initial effort-regression gate. Record that
Stage A contains episode aggregates rather than estimator traces.

**Evidence.** Ten CPU tests passed. A native G1 interface check covered all 29
actuators at gains 1.0 and 0.7 with preserved source time and caller data.
Exact commands, versions and hashes are in
[telemetry_cpu_validation.json](research_candidate/telemetry_cpu_validation.json).
The scalar analytical checks are unchanged.

**Limits.** No walking recovery, full-G1 online estimator, Warp parity or GPU
sampling overhead was measured. The true-gain gate remained unchanged.

**Review/decision.** Proposal family 0.3 was selected by Prasuna. Publication and
the later plan merge do not establish a separate completed Claude review or
approval of every technical claim in revision 0.3.1.

**Subsequent evidence clarification.** The authority script, absent from the
then-reviewed branch trees, was later found in the historical commit linked
above under OAA-PLAN 0.1.0. Preserve the original as-of-review note; the new plan
records the updated source availability. No numerical result is promoted to
independently reproduced evidence.

## OAA-PROP 0.3

**Date:** November 15, 2026  
**Published revision:**
[e7c7854038ba040e8586660f77bac8bec81af26f](https://github.com/pchatla-arch/anticipatory-balance/commit/e7c7854038ba040e8586660f77bac8bec81af26f).  
**Document:** research_candidate/proposal.md and proposal.pdf, 16 pages.

**Reason.** Strengthen Online Actuator Adaptation into a specific, falsifiable
research method while preserving the frozen-policy execution architecture.

**Detailed changes.** Define timestamped gain sets, measurement-age and
command-hold uncertainty, clipping-aware feasible correction and a closed-form
projection. Derive a conditional same-state actuator-force comparison. Make a
tuned heuristic the primary comparator, separate learned estimation from RL,
and define evidence that could reject the claimed benefit. Include return to
healthy actuation, held-out splits, policy-seed uncertainty, runtime and a
conditional paper path.

**Evidence.** Scalar reference: ten finite cases, two invalid-input checks and
71 analytical sweep points. No humanoid recovery was measured.

**Review/decision.** Published for review; later selected by Prasuna as the
proposal family. Superseded as the current technical text by revision 0.3.1.

## OAA-DESIGN 0.2

**Date:** November 15, 2026  
**Published revision:**
[d44883d2345f76709b6163e4b6b21808b456b957](https://github.com/pchatla-arch/anticipatory-balance/commit/d44883d2345f76709b6163e4b6b21808b456b957).  
**Document:** paper.md and paper.pdf, 24 pages.

**Reason.** Expand the Fleet Robust Execution specification into an engineering
design and experimental protocol.

**Detailed changes.** Document actuator physics, causal telemetry assumptions,
bounded correction, supervision, learning comparisons, held-out evaluation and
C++ runtime requirements. Correct gain-versus-lag language, the correction
slew definition, fallback claims and the evidence boundary.

**Evidence.** CPU-checked feasibility implementation was the source baseline;
online identification, GPU recovery and deployment remained proposed.

**Review/decision.** The independent design review at
[5c0043a5a3f494c37f67b28e7e50760493ab8eb7](https://github.com/pchatla-arch/anticipatory-balance/blob/5c0043a5a3f494c37f67b28e7e50760493ab8eb7/docs/adaptive_execution/reviews/review_v0.2.md)
identified AE-001 through AE-003 and other findings. Retained as historical
support, not the current authority for corrected telemetry or the selected
research method.

## Rules for the next update

1. Change the document's own version and status when its controlled content
   changes. Update source and PDF together where a PDF is registered.
2. Add a new detailed entry; retain earlier entries and append corrections.
3. Rebuild and visually inspect changed PDFs. Register actual hashes and build
   provenance. Legacy artifacts remain labeled byte-verified; do not fabricate
   their historical build records.
4. Run the checker against the PR-base registry. Never make it silently rehash
   or bless an old PDF.
5. Record independent review and Prasuna's actual decision separately. Status
   changes alone need not change the content version, but must retain evidence.
6. After merging, future release history links to the exact merged commit.

