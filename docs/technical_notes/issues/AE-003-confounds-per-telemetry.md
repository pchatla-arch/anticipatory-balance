# AE-003 Stage B confounds cannot fail under effort telemetry

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Major (the central fault-versus-confound claim) |
| Project · area | Adaptive execution · experimental design |
| Found | 2026-11-15, Claude, review of design draft 0.2 (PR #8, finding R3) |
| Issue | #11 |
| Owner | Claude (Stage B) · ChatGPT reviews |

## Symptom
Stage B's planned confounds for "is it a real actuator fault?" are payload, push, friction and
contact. Under synchronized applied-torque telemetry, payload, push and contact cannot produce a
false gain estimate at all.

## Evidence
- **Hypothesis**, from the actuator model. With aligned applied torque (AE-001, AE-002 fixed),
  τ = g·Kp·(target − q) holds whatever forces act on the body. Payload, push and contact change
  the motion, so they change target − q, but not the relation between that and τ.
- Check: a pilot per telemetry condition on a healthy robot; the false-adaptation rate should be
  near zero for these confounds with effort, and non-trivial without it.

## Why it did not work
The confound list was chosen from the motion side, "what else could make tracking worse",
not from the measurement side, "what else could make the force-versus-error relation look
different". The draft carried this over from the original specification (Claude's), where the
contact-freezing rule rested on the same intuition.

## Impact
As written, the Stage B false-adaptation control in the main telemetry condition cannot fail,
so passing it would show nothing. Stage A is unaffected.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Confound set per telemetry condition | Each control can fail; the claim becomes precise | More conditions to run |
| Keep one confound set | Simpler | Main-condition control is vacuous |

## Decision and next steps
Proposed confound sets:
- **Effort-based condition:** joint friction or damping included in a torque-sensor reading
  (a friction change biases ĝ by −τ_fric/(Kp·(target − q))), sensor scale and bias, saturation.
- **Motion-based condition:** payload, push, contact.

The research question is stated per condition in `paper.md`.
**Acceptance test:** in the pilot, at least one confound per condition produces a measurable
false-adaptation rate for the uncapped estimated inverse, so the confidence screen has
something to prevent.

## Resolution
Open.
