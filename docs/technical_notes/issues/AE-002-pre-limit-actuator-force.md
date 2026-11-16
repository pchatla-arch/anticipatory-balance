# AE-002 actuator_force is read before joint torque limits

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Major |
| Project · area | Adaptive execution · telemetry, evaluator diagnostics |
| Found | 2026-11-15, Claude, review of design draft 0.2 (PR #8, finding R2) |
| Issue | #10 |
| Owner | ChatGPT (telemetry module, evaluator field) · Claude reviews |

## Symptom
With a saturating command, 8 of 12 leg joints report `actuator_force` of 166–225 N·m while the
joint actually receives 88 or 139 N·m (`qfrc_actuator`). The leg joints have joint-level
actuator-force limits of 88 / 139 / 50 N·m.

## Evidence
- **Verified** on CPU: `python docs/adaptive_execution/reviews/check_effort_alignment.py`
  (second block, "saturating command").

## Why it did not work
The G1 model limits torque at the **joint** (`jnt_actfrclimited`), not at the actuator. MuJoCo's
`actuator_force` is the actuator's own output, before that joint limit; the applied torque is
in `qfrc_actuator`. A gain estimate built on `actuator_force` therefore always sees the ideal
g·Kp·(target − q), even while the joint is saturated. The estimator would report a healthy gain
at exactly the moments the correction has no headroom. The evaluator's `max_force` diagnostic
has the same meaning; its config note says it is not a torque-limit audit, but the name
invites that reading.

## Impact
- Stage B estimator and the confidence screen's torque-headroom term.
- Any "force stayed within limits" statement drawn from `max_force`.
- Stage A pass/fail: not affected.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Effort = applied joint torque `qfrc_actuator` at the actuator's DoF, plus a saturation flag at the limit | Matches what a joint-torque sensor or current-based estimate sees | Saturated samples carry no gain information and must be rejected |
| Keep `actuator_force` and compare against limits separately | Simple | Two signals that disagree; easy to misuse |

## Decision and next steps
Proposed: define effort as applied joint torque and reject samples at the limit, raising a
headroom flag. Rename the evaluator field to `max_commanded_actuator_force`, or switch it to
applied torque.
**Acceptance test:** under a saturating command the telemetry module marks those samples
saturated and the estimator ignores them.

## Resolution
Open.
