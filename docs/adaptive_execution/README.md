# Fleet Robust Execution: design and review package

Review draft 0.2 expands the four-page Fleet-Robust Execution specification into
a technical design and experimental protocol. The selected architecture remains
a frozen G1 PPO policy, optional calibration, online actuator estimation, bounded
position-target correction, confidence and fault supervision, and a C++ port.

## Reading order

| Document | Purpose |
| --- | --- |
| [Detailed paper, PDF](paper.pdf) | Typeset 24-page review document with equations, diagrams, experimental protocol and references |
| [Detailed paper, Markdown](paper.md) | Canonical editable source for technical review and proposed changes |
| [Review of the four-page specification](specification_review.md) | Specific corrections to physics, measurement assumptions, timing and evidence claims |
| [Independent review brief](claude_review_brief.md) | Self-contained instructions for Claude or another reviewer; requests minimal corrections within the selected architecture |

The paper includes related work, information boundaries, a conditional actuator
force-error result, classical and learned estimation, PPO comparisons, held-out
instance evaluation, runtime requirements, Foresight lessons, acceptance artifacts,
and a technical discussion guide.

## Evidence boundary

The reviewed implementation baseline is `aac06996b8bab2980234c4c5fb63a7c78025806f`;
the preceding roadmap baseline is `16c3296249dd6b87993b4a2e39462b877663f59d`.
The gain-fault wrapper, bounded privileged correction and feasibility protocol are
implemented and CPU-checked. GPU recovery, online identification, calibration,
confidence logic, learned adapters and the C++ runtime are not measured results
in this revision. The paper states module-specific status and required evidence.

This package changes documentation only. It does not change the existing
experiment launcher or constitute a new GPU experiment. The evidence cutoff is
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
