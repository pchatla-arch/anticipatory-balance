# AE-001 Effort sample paired with the wrong joint position

| Field | Value |
| --- | --- |
| Status | Open |
| Severity | Blocking for Stage B (identification); no effect on Stage A |
| Project · area | Adaptive execution · telemetry |
| Found | 2026-11-15, Claude, review of design draft 0.2 (PR #8, finding R1) |
| Issue | #9 |
| Owner | ChatGPT (telemetry module) · Claude reviews |

## Symptom
On a **healthy** robot (true gain 1), estimating the gain as actuator_force / (Kp·(target − q))
with the joint position returned after the step gives a median error of 0.33% and up to
**7.2%** on one leg joint, from a single 2 ms physics substep. The first fault under test is a
15% gain loss (0.85), so this artefact is about a third of the effect being measured.

## Evidence
- **Verified** on CPU, G1 knees-bent keyframe, random targets ±0.1 rad.
- Reproduce: `python docs/adaptive_execution/reviews/check_effort_alignment.py` (prints the
  error for the pre-step and post-step pairing: 0.0000 vs 0.0033 median, 0.0720 max).

## Why it did not work
MuJoCo's step computes actuator force from the current joint position, *then* integrates.
The force field left in `data` after a step therefore belongs to the position **before** the
final integration, while `data.qpos` is the position **after** it. With ten 2 ms substeps per
20 ms control step, only the last substep's force survives, and it is one substep out of step
with the returned position. The error grows with joint speed. It was not caught because the
Stage A oracle uses the true gain and never forms this ratio; only an estimator does.

## Impact
- Stage B and every estimator built on simulator effort: biased gain estimates and false
  "faults" on healthy robots.
- Stage A results: **not affected**.
- The design paper's statement that exact simulated effort makes gain "almost algebraically
  observable" holds only once the pairing is fixed.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Record q before the last substep inside the physics loop | Exact pairing, no extra compute | Touches the step loop |
| Recompute effort at the returned state with one `mjx.forward` | Leaves the step loop alone | One extra forward pass per control step |
| Correct q with q̇·dt | Cheap | Approximate; contact and limits break it |

## Decision and next steps
Proposed: the telemetry packet carries (q, effort) from the **same physics evaluation**, using
the `mjx.forward` option unless profiling shows its cost matters. The telemetry contract in
`paper.md` states the pairing explicitly.
**Acceptance test:** a recursive-least-squares run on a healthy robot with exact, aligned effort
gives ĝ = 1 within 1e-4 on every leg joint except ankle roll (see AE-005).

## Resolution
Open.
