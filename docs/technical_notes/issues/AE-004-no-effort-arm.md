# AE-004 No-effort identification arm is unspecified

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Major |
| Project · area | Adaptive execution · estimator design |
| Found | 2026-11-15, Claude, review of design draft 0.2 (PR #8, finding R4) |
| Issue | #12 |
| Owner | Claude · ChatGPT reviews |

## Symptom
Stage B lists a "no direct effort" condition but does not say how gain would be estimated in it.

## Evidence
- **Hypothesis.** Without measured torque, gain has to come from motion through inverse
  dynamics. Inverse dynamics of a walking humanoid needs the contact forces, which the
  telemetry contract withholds from the deployed layer. A physical-model estimator in this
  condition can only report "unavailable".

## Why it did not work
The three-condition ladder (exact, noisy or delayed, none) was defined before the estimator was,
so the third rung has no method attached.

## Impact
Stage B cannot run the no-effort arm as written. No existing result is affected.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Learned history estimator (Stage D) as the no-effort arm | Answers "what can learning recover where the physical model cannot run" | Needs training data and a held-out test |
| Estimate contact forces from the IMU and kinematics | Keeps a physical model | Large added scope; a second estimation problem |
| Report "unavailable" only | Honest, cheap | No result in that condition |

## Decision and next steps
Proposed: drop the no-effort arm from the Stage B gate, and make it the learned history
estimator's condition in Stage D. This also gives the learned component a role the classical
one cannot fill.
**Acceptance test:** `paper.md` lists, for each sensing condition, the estimator that runs in it.

## Resolution
Open.
