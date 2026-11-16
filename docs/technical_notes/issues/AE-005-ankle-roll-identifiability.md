# AE-005 Ankle-roll gain is weakly identifiable

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Minor |
| Project · area | Adaptive execution · estimator, reporting |
| Found | 2026-11-15, Claude, review of design draft 0.2 (PR #8, finding R5) |
| Issue | #13 |
| Owner | Claude |

## Symptom
Leg proportional gains (Kp) are 75 (hip, knee), 20 (ankle pitch) and **2 (ankle roll)**.

## Evidence
- **Verified** from the model: `check_effort_alignment.py` prints the leg gains.

## Why it did not work
With Kp = 2 the ankle-roll servo produces very little torque for normal tracking errors. Its
gain barely affects walking and is hard to estimate. A single per-robot summary of
identification error would be dominated by this joint.

## Impact
Per-joint identification claims, and any averaged identification metric.

## Decision and next steps
Report identifiability per joint group, with the excitation statistic for each. Exclude ankle
roll from the gain-identification claim, or report it separately with its excitation level.
**Acceptance test:** Stage B tables have one row per joint group, with the excitation level.

## Resolution
Open.
