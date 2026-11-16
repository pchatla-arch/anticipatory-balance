# Review request: selected proposal 0.3 and telemetry corrections

Review [Online Actuator Adaptation for Reliable Humanoid Locomotion,
selected proposal 0.3, revision 0.3.1](proposal.md) ([PDF](proposal.pdf)). The
project owner has selected this direction. The [earlier 24-page engineering
design](../paper.md) remains supporting material. Review the specific method,
AE-001 through AE-003 corrections and the next CPU gain-range experiment; the
architecture choice is recorded rather than reopened.

The shared architecture is fixed: a frozen G1 PPO actor, declared actuator
telemetry, online estimation, bounded position-target correction and execution
supervision. Both proposals can use the same component. Do not create artificial
differences by treating one title as a different robot or a different policy.

## What is new in this candidate

The proposed method forms a gain set from timestamped measurements, propagates
uncertainty through measurement age and the next 20 ms hold, intersects robust
actuator-force and command constraints, then projects an inverse candidate into
the feasible requested-offset interval. A GRU can replace the point estimate
without changing the physics-derived set or correction limits. The primary
comparison is against a well-tuned heuristic with the same estimator, history,
information, physical limits and tuning budget. The heuristic does not inherit the proposed uncertainty/hold-dependent force-error comparison filter.

The [scalar reference](validate_projection.py) and
[stored calculations](analytical_check.json) are CPU algebra checks only. Ten
finite cases, two nonfinite-input checks and 71 sweep points pass. The method is
not yet integrated into the robot evaluator. The existing known-gain gate is
the compensability baseline; it is not a successful online-adaptation result.

Read the [review-resolution record](review_resolution_ae001_003.md) before
replay implementation. Applied actuator torque is sampled after joint-level
limits and synchronized with position and the held target. MuJoCo passive
friction is not itself included in `qfrc_actuator`; an effort-side friction
confound needs an explicitly modeled downstream sensor. The reported 82%/63%
screen is not full feasible authority and remains unverified here pending its
script/result manifest. Keep the gain-range CPU replay separate from the
unchanged Stage A behavioral gate; its CSV has no per-step effort packets.

## Questions that should determine the review

1. Is the gain/measurement model consistent with the unit-gear proportional
   actuator and separate passive damping? Does the conditional force-error
   proposition hold over the declared gain and state envelopes?
2. Are the clipping preimage, requested versus effective correction, force
   headroom, slew conflict and fallback statements mathematically consistent?
3. Can physically meaningful state/parameter bounds leave useful correction
   authority? Identify the smallest experiment that could show overconservatism.
4. Is abrupt-jump detection treated honestly? Include both degradation and an
   unannounced return to healthy gain. Detecting a mismatch after the event does
   not certify the preceding unobserved interval.
5. Are noise bounds, sensor scale, effort semantics and timestamp assumptions
   sufficient for the claimed information? Distinguish exact-effort diagnostics,
   bounded-error assumptions, empirical coverage and genuine sensor ambiguity.
6. Does the method have a plausible distinction from tuned heuristic gating,
   fixed smaller correction caps, SAG, TeAR and classical set-membership control?
   A new combination of established modules is not by itself a novelty claim.
7. Are the learned-history and robust-PPO comparisons fair and meaningful? Does
   the history estimator actually change issued actions after projection? Is a
   residual justified by a measured model limitation?
8. Can the experiments distinguish useful recovery from avoiding all action?
   Review held-out instance/policy splits, healthy margins, paired uncertainty,
   failures, detector delays, saturation and compute accounting.

Return one table of findings with severity, exact section or code location,
verified defect versus hypothesis, mechanism, minimal correction and smallest
resolving check. Then assess implementation readiness, remaining research risk,
learning contribution and the smallest next evidence gate within the selected
architecture. Do not reopen the project selection or assign unsupported
numerical novelty scores or acceptance probabilities. State which measurements
would change the supported method claim or operating envelope.

Use the existing source/evidence record: `aac0699` for the feasibility
implementation and `d44883d` for the preceding design review. Attribute any
newer experiment to its actual commit, checkpoint manifest and result files.
This review request does not launch paid jobs, install packages or authorize
rewriting published history. Supply proposed edits for reconciliation against
the shared repository.
