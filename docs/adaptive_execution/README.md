# Reliable Humanoid Navigation Under Actuator Uncertainty: Perception-Guided Planning and Adaptive Execution

**Current AA228/CS238 project title and submission framing.** The earlier online-actuator-adaptation proposals remain the underlying method and historical design record; they are not separate competing course-paper titles.

**Start here:** [Project plan, expectations, experiments, results, and Claude handoff](PROJECT_PLAN_AND_REVIEW_HANDOFF.md).

**Supporting artifacts:** [Course report draft](Course_Report_20261127_Draft.md) · [Executed experiment tracker](EXPERIMENT_TRACKER.md) · [Claude review and execution brief](CLAUDE_INDEPENDENT_REVIEW_BRIEF.md).

## Current project status and experimental evidence (November 26, 2026)

The [canonical experiment tracker](EXPERIMENT_TRACKER.md) is the single place to check completed, pending and deferred G1 experiments. [S1b negative-results analysis](S1b_raw_data_diagnosis.md) summarizes independently recomputed S1a fault/return/progress metrics. The latest [physical obstacle demo plan](G1_obstacle_recovery_demo_plan.md) specifies contact, rendered-depth, navigation, video and reporting gates.

S0 (12-row CUDA smoke) and S1a (32-row actuator comparison) are **measured and published**. G2a verified the configured physical cylinder contacts; G2b produced an authentic failed navigation pilot; G2c reached the goal in one healthy development rollout. G3 compared correction-off and the frozen heuristic under a hidden gain fault/recovery: both reached the goal, with zero recorded contacts under the configured coverage. Actual simulator videos and raw records are linked in the handoff. This one-seed comparison does not establish general reliability or superiority.

**Analysis correction:** the earlier description of shorter S1a forward distance as a performance penalty was unsupported. Saved traces show unwanted motion during commanded stops in the uncorrected arm. The [new segment audit](review_audits/2026-11-26/s1a_segment_audit.json) preserves the heuristic's inconsistent stopping and effort-sensor confound as limitations.

The original 72-episode obstacle matrix, stronger multi-seed evaluation, final report PDF, and final comparison video remain unfinished. Claude is invited to review, recommend corrections, and help with implementation, execution, and results updates; the current handoff supersedes older reviewer-role restrictions.

## Earlier actuator-adaptation proposals and design reviews

The proposals below are historical design records; they do not describe the latest experiment status.


The selected [proposal 0.3, revision 0.3.1](research_candidate/README.md) strengthens
**Online Actuator Adaptation for Reliable Humanoid Locomotion** with an explicit
uncertainty/command-hold correction rule, analytical reference, matched
comparisons and a paper plan. It includes a fair comparison with the Fleet Robust
Execution systems emphasis. The original 24-page design below is retained as
versioned supporting material.

The [selection and AE-001 through AE-003 resolution record](research_candidate/review_resolution_ae001_003.md)
defines the synchronized effort channel, separates confounds by measurement
condition, and explains what the preliminary authority screen does and does not
establish. The true-gain GPU gate remains independent of effort-based estimation.

Review draft 0.2 expands the four-page Fleet-Robust Execution specification into
a technical design and experimental protocol. The selected architecture remains
a frozen G1 PPO policy, optional calibration, online actuator estimation, bounded
position-target correction, confidence and fault supervision, and a C++ port.

## Reading order

| Document | Purpose |
| --- | --- |
| [Strengthened research proposal, PDF](research_candidate/proposal.pdf) | 17-page method, experiments, paper criteria, costs and comparison of the two proposals |
| [Research candidate review brief](research_candidate/claude_review_request.md) | Questions for Claude about correctness, research strength and proposal selection |
| [Detailed paper, PDF](paper.pdf) | Typeset 24-page review document with equations, diagrams, experimental protocol and references |
| [Detailed paper, Markdown](paper.md) | Editable source of the preceding 0.2 engineering design |
| [Review of the four-page specification](specification_review.md) | Specific corrections to physics, measurement assumptions, timing and evidence claims |
| [Independent review brief](claude_review_brief.md) | Self-contained instructions for Claude or another reviewer; requests minimal corrections within the selected architecture |

The paper includes related work, information boundaries, a conditional actuator
force-error result, classical and learned estimation, PPO comparisons, held-out
instance evaluation, runtime requirements, Foresight lessons, acceptance artifacts,
and a technical discussion guide.

## Historical evidence boundary — November 15, 2026

The reviewed implementation baseline is `aac06996b8bab2980234c4c5fb63a7c78025806f`;
the preceding roadmap baseline is `16c3296249dd6b87993b4a2e39462b877663f59d`.
The gain-fault wrapper, bounded privileged correction and feasibility protocol are
implemented and CPU-checked. GPU recovery, online identification, calibration,
confidence logic, learned adapters and the C++ runtime are not measured results
in this revision. The paper states module-specific status and required evidence.

This release adds documentation, a standalone CPU analytical reference and an opt-in telemetry helper. It does
not change the existing experiment launcher or constitute a new GPU experiment. The evidence cutoff is
November 15, 2026. Update completed-work claims only against an actual source
revision, checkpoint manifest and experiment output.

## Rebuilding the PDF

The checked-in PDF is ready to read. Rebuilding is a CPU documentation task:

```bash
python3 docs/adaptive_execution/build_paper.py
```

The builder expects Pandoc on `PATH`, Python packages `reportlab`, `matplotlib`,
`Pillow` and `pypdf`, and Liberation fonts at
`/usr/share/fonts/truetype/liberation`. It performs no package installation and
does not require the GPU research environment. It regenerates the two SVG/PNG
figures and `paper.pdf`; temporary equation assets and the build report go in
`.build/`. Review rendered pages after changes to equations or layout.

For earlier implementation detail and project context, see the
[adaptive execution note](../technical_notes/adaptive_execution.md) and
[humanoid execution roadmap](../technical_notes/humanoid_execution_roadmap.md).
