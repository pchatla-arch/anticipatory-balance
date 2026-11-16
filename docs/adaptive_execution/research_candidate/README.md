# Online Actuator Adaptation: strengthened research proposal

Research candidate 0.3, reviewed November 15, 2026.

This proposal develops the original **Online Actuator Adaptation for Reliable
Humanoid Locomotion** into a specific method and comparative paper plan. The
architecture remains a frozen G1 PPO actor, actuator estimation, bounded
position-target correction and execution supervision. The new candidate uses
timestamped gain sets, possible motion over the next command hold and available
authority to project an estimated inverse into a feasible correction interval.

## Read and review

- [16-page proposal, PDF](proposal.pdf)
- [Editable proposal](proposal.md)
- [Claude review and option-selection brief](claude_review_request.md)
- [Earlier 24-page engineering design](../paper.pdf)

The final proposal section compares this research emphasis with the four-page
Fleet Robust Execution systems emphasis. The proposals share their underlying
execution interface. The choice is whether to prioritize a bounded component
delivery or undertake the additional method comparisons needed for a research
claim. The proposed algorithm has no measured humanoid-recovery result yet.

## What is implemented here

`validate_projection.py` is a CPU-only, standard-library analytical reference.
It is not integrated with the robot environment and is not a pod launcher. It
checks the scalar interval projection, requested/effective clipping semantics,
and force-error assumptions. It passes ten finite cases, two nonfinite-input
checks and 71 uncertainty-sweep points. The stored
[analytical_check.json](analytical_check.json) and analytical figure contain
synthetic calculations, not robot performance data. The midpoint of the hold
error interval is an illustrative default; robot integration must supply a
separately timestamped causal current-error estimate.

Run the inexpensive reference from the repository root:

```bash
python3 docs/adaptive_execution/research_candidate/validate_projection.py
```

The existing known-gain feasibility gate remains the integration baseline. The
timestamped estimator, full proposed controller, GPU recovery evaluation, learned
alternatives and C++ runtime remain to be implemented or measured.

## Build the document

The checked-in PDF is ready to read. The optional CPU documentation build uses
the parent PDF builder's installed dependencies: Pandoc, ReportLab, matplotlib,
Pillow, pypdf and Liberation fonts. It installs nothing and uses no GPU.

```bash
python3 docs/adaptive_execution/research_candidate/build_proposal.py
```

This rebuilds only the candidate PDF and its figures. The parent 24-page PDF
is preserved. Rendered page inspection is required after layout changes.

## Evidence and implementation boundaries

The reviewed feasibility implementation is `aac0699`; the preceding design
review is `d44883d`. The conditional mathematical result concerns actuator-force
error at the same state under valid gain/motion bounds and an unsaturated model.
It is not a stability, fall-prevention or physical safety guarantee. Arbitrary
unobserved gain jumps can violate the drift model before detection. The proposal
requires reporting those intervals, withdrawal conflicts, abstention and actual
task outcomes. A null comparison against a tuned heuristic limits the research
claim while leaving the engineering component useful.
