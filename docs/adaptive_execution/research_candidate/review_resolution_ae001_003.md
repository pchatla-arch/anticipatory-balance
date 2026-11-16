# Selected proposal 0.3: review corrections and next experiments

November 15, 2026. Selection recorded: **Online Actuator Adaptation for Reliable
Humanoid Locomotion, proposal 0.3**. Revision 0.3.1 retains the frozen G1 PPO,
estimation, bounded correction, supervisor and C++ runtime architecture.

## Reviewed evidence

- Implementation and proposal baseline: `e7c7854038ba040e8586660f77bac8bec81af26f`.
- Independent [design review, R1-R3](https://github.com/pchatla-arch/anticipatory-balance/blob/5c0043a5a3f494c37f67b28e7e50760493ab8eb7/docs/adaptive_execution/reviews/review_v0.2.md).
- Tracked issues: [AE-001](https://github.com/pchatla-arch/anticipatory-balance/issues/9),
  [AE-002](https://github.com/pchatla-arch/anticipatory-balance/issues/10),
  [AE-003](https://github.com/pchatla-arch/anticipatory-balance/issues/11).

The review is correct that AE-001/002 block trustworthy effort-based estimation,
but do not block the existing true-gain behavioral gate. That gate uses current
position and the injected gain; it does not estimate gain from torque. Its
stored raw-force maximum is explicitly a pre-joint-limit diagnostic, not a
torque-compliance result. The gate launcher, thresholds and actor interface are
unchanged in this revision.

## Corrections

| Issue | Correction in this revision | Remaining boundary |
|---|---|---|
| AE-001: effort paired with the wrong state | Opt-in telemetry performs a fresh forward evaluation on isolated data with the active physics model and held command, then copies time, effective target, q, qvel and effort together | A future recorder must preserve capture/arrival timestamps and use the model active for that sample |
| AE-002: effort read before joint limits | Packet uses `qfrc_actuator` at the joint DoF; raw `actuator_force` is separate. Both actuator- and joint-limit saturation invalidate the unsaturated regression | Exact simulator clipping status is an idealized channel; imperfect hardware status requires explicit modeling |
| AE-003: confounds do not match the effort channel | Primary effort tests use scale, bias, timing and saturation handling. Mechanical disturbances remain motion/excitation tests | Friction becomes a direct effort confound only under a declared sensor/reconstruction model that includes its loss |

The new [telemetry helper](../../../foresight/adaptive/telemetry.py) is opt-in. It
does not replace the parent walker's state or observation cache and does not add
a forward call to the existing gate. The packet exposes no injected gain or
fault-onset label. Simulator truth is used only to generate sensor physics.
Noise/delay injection, gain estimation and integration with the proposed
corrector remain separate work.

The mapping is intentionally narrow: one unit-gear proportional actuator per
hinge joint, without actuator-side gravity compensation. Reject unsupported
transmissions rather than pretending actuator force and joint torque are
interchangeable. Mechanical passive damping and dry friction are separate from
this applied-actuator contribution. For physical sensors, specify placement,
scaling, losses and error bounds independently.

These semantics follow the official MuJoCo descriptions of
[joint force clamping](https://mujoco.readthedocs.io/en/stable/modeling.html#force-limits)
and [actuation, passive forces and friction loss](https://mujoco.readthedocs.io/en/stable/computation/index.html).
Tests are in [test_adaptive_telemetry.py](../../../tests/test_adaptive_telemetry.py).
All ten tests passed on CPU, including native/MJX agreement and JIT/vmap. A
native G1 interface check also passed for all 29 actuators at gains 1.0 and
0.7, preserving source time and caller data; this was not a policy rollout.
The [validation record](telemetry_cpu_validation.json) contains commands,
versions and hashes. Use
the existing configured environment; this review requires no package upgrade:

```bash
JAX_PLATFORMS=cpu python -m unittest discover -s tests -p test_adaptive_telemetry.py -v
```

## Interpreting the reported authority screen

The project owner forwarded the following healthy-walker results. They are
**reported, not independently reproduced in this revision**. At review time,
`docs/adaptive_execution/reviews/check_authority.py` was not present on the
published review branch `5c0043a`, the selected branch `e7c7854`, or `main`
`165abfd`. Publish the script, checkpoint hash, configuration, sample counts and
result artifact before promoting these percentages to verified evidence.

| Joint group | Reported median tracking error | Measured-speed envelope excludes zero | Global-speed envelope excludes zero |
|---|---:|---:|---:|
| Hip | 0.12 rad | 86% | 84% |
| Knee | 0.23 rad | 80% | 61% |
| Ankle pitch | 0.14 rad | 83% | 45% |
| Ankle roll | 0.04 rad | 73% | 19% |
| All | 0.11 rad | 82% | 63% |

These rates screen a necessary condition for nonzero correction under a
hypothetical degraded gain. A valid gain interval containing healthy gain 1
still permits only zero effective correction under the proposed criterion.
Passing the sign screen does not establish gain identifiability, force
headroom, target or slew feasibility, useful correction magnitude, or recovery.

Measured instantaneous speed times the hold is not a guaranteed future-motion
bound. Include encoder/velocity error, age and acceleration or a supported speed
envelope. A global empirical maximum or percentile is not a hard physical bound;
future samples from a hold cannot be used as causal controller inputs. Assess
envelope coverage after faults and interventions. Report per-joint denominators
and episode identity: correlated joint-time samples are not independent trials.
The check supports continuing; it does not establish which joints causally
contribute most to recovery.

## Next CPU gain-range replay: agreed output contract

The gain-range check proposed in the review can proceed in parallel with the
known-gain GPU gate. It should produce:

1. Versioned source, checkpoint/trace hashes and explicit replay type. Forces
   synthesized on a fixed healthy trajectory are not a faulted closed-loop run.
2. Aligned issued targets, joint state, post-limit actuator effort, capture time
   and arrival time; known saturation/rejection status.
3. Declared absolute encoder and effort-error bounds, with separate bias and
   scale cases. Unknown scale and gain may remain inseparable in this channel.
4. Per-joint-group gain-set coverage, width and time to exclude gain 1; also
   report width after measurement-age and command-hold expansion.
5. Separate sign-screen rate, full feasible correction interval, attainable
   magnitude, and overlap with useful oracle corrections.
6. Saturated/rejected packets, empty gain sets, correction-slew conflicts and
   unannounced return-to-healthy behavior with delayed telemetry.

The existing Stage A `episodes.csv` contains aggregates, not these packets. A
dedicated trace generator or recorder is needed. No-effort estimation is a
later separate study, not an unchanged use of the effort regressor. Persistent
bias also means more samples need not narrow a bounded-error set indefinitely.

## GPU gate and publication boundary

Run the existing `scripts/start_adaptive_gate.sh gate`: 32 paired screening
scenarios and, if a candidate qualifies, 128 fresh confirmation scenarios. No
PPO retraining or learned estimator is involved. The runner restores committed
checkpoints, verifies account/push access, records provenance and publishes
compact results before requesting Stop. Its time budget is not a guaranteed
billing cap; unavailable stop controls or failed publication can leave a pod
running. Foresight paper work remains separate.

The optional telemetry helper has CPU tests, not measured Warp GPU overhead or
closed-loop G1 identification results. The proposed research contribution still
requires comparison with a tuned heuristic under the same information and
limits. Falsifiability is necessary for the experiment, but does not by itself
establish novelty or publishability.
