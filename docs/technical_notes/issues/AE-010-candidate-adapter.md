# AE-010 · WP 2.2 candidate adapter

Issue [#37](https://github.com/pchatla-arch/anticipatory-balance/issues/37) records the original bench interface. On PR #35, the projection hook had no declared force or target limits, sensor error bounds or previous issued request. The WP 1.2 kernel in PR #34 requires these to project a command.

The adapter binds nominal actuator metadata and the public measurement bounds for each telemetry cell. At decision time it receives the previous issued request from the execution interface. It consumes only released packets, their per-joint validity, and the shared RLS estimate. A post-step packet labelled step t samples the state at the start of step t+1. Its age at the next decision is zero for immediate delivery.

The development configuration declares a speed envelope, gain drift, effort-margin and roundoff allowances, and a reacquisition count before evaluation. A violated envelope or abrupt unseen gain jump invalidates the conditional force comparison. A failed or unsupported projection requests withdrawal toward zero under the existing slew bound and emits its reason. Repeated inconsistencies keep the affected joint in reacquisition. Nothing here grants a gate decision, validates these envelope choices, or resolves the independent harm-metric issue AE-011.

The shared inverse uses the same RLS gain and the same causal point-position prediction as the heuristic. The hold envelope uses the measured encoder position, its physical source age, and the configured speed bound. The speed value in the dev fixture is an exploratory assumption, not a measured upper bound. Error-bound coverage and gain-set coverage must be measured separately.

For CPU replay, --trace-authority-packets together with --trace exports the pre-decision reference, last issued request, RLS channels and evaluator-only joint position. The packet arrays at t contain the post-transition releases assimilated after action t, including padded slots marked by authority_release_mask. Source truth at t describes the aligned position and applied torque at that capture step. All practical actions are computed before those truth channels are recorded.
