# AE-006 Feasibility ratio can rest on a small denominator

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Minor |
| Project · area | Adaptive execution · Stage A gate |
| Found | 2026-11-15, Claude, review of design draft 0.2 (PR #8, finding R7) |
| Issue | #14 |
| Owner | ChatGPT (gate) · Claude reviews |

## Symptom
The gate advances the **least severe** gain that passes the screen. If that gain degrades
walking only slightly, the recovery fraction (faulty − oracle) / (faulty − healthy) divides by a
small number and becomes noisy.

## Evidence
- **Hypothesis.** The gate already requires "material degradation"; whether that threshold is
  large enough is a property of the screen's numbers, unknown until it runs.

## Why it did not work
A ratio criterion was used without an absolute floor on its denominator.

## Impact
Interpretation of the Stage A pass only. No code defect.

## Decision and next steps
Before selection, require explicit absolute floors on excess post-fault loss and on the
failure-rate difference; report both alongside the ratio. Can be applied at analysis time
without re-running. Stage A runs as built.
**Acceptance test:** the gate report prints the floors and the screen values against them.

## Resolution
Open.
