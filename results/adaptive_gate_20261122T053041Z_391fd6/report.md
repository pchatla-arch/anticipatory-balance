# Adaptive execution: known-gain feasibility gate

Scientific gate: **inconclusive**.

This tests a bounded, privileged correction for proportional-servo gain loss. It does not estimate faults online and is not evidence of learned adaptation.

| Phase | Condition | Episodes | Failure rate | Mean fixed-horizon loss | Mean post-fault progress (m) |
|---|---|---:|---:|---:|---:|
| screen | healthy | 8 | 0.000 | 0.0126 | 3.035 |
| screen | identity | 8 | 0.000 | 0.0126 | 3.034 |
| screen | gain_085_plain | 8 | 0.000 | 0.0136 | 3.297 |
| screen | gain_085_oracle | 8 | 0.000 | 0.0125 | 3.037 |
| screen | gain_070_plain | 8 | 0.250 | 0.4110 | 5.330 |
| screen | gain_070_oracle | 8 | 0.000 | 0.0137 | 3.052 |
| screen | gain_055_plain | 8 | 1.000 | 3.1487 | 1.909 |
| screen | gain_055_oracle | 8 | 0.000 | 0.0223 | 3.222 |

The same initial states, commands, onset times and noise streams are used within each phase. Confirmation uses new scenario seeds. All screened severities remain in episodes.csv.

The loss is mean squared planar local-velocity error, capped at 4 (m/s)^2, over the fixed post-fault horizon. Failure receives that maximum loss for the terminal transition and all remaining steps. Progress stops at the last finite terminal transition. Numerical failures are retained and invalidate the scientific gate.

See summary.json for paired uncertainty, selection, and threshold verdicts. A completed run with a failed scientific gate is still a useful result. No actuator lag, reduced torque limit, online estimator, vision loop, or real hardware is tested.

## Review of the November 21 GPU smoke (Pacific time)

The pod completed the Warp smoke and published all 64 episode rows from source `8440b5068d918c28928c8a2326472f6606da28e8`. Eight matched scenarios were used for each of eight conditions. The confirmation count was zero. The screen failed the identity control, selected no candidate, and the scientific verdict remains **inconclusive**. The machine-generated record above and the original evidence are available in [the upload](https://github.com/pchatla-arch/anticipatory-balance/tree/df8d5ee99036b243d62653d1c025ea2898ade16f/results/adaptive_gate_20261122T053041Z_391fd6).

Joining healthy and gain-one adapter rows by scenario ID gives the following maximum absolute differences.

| Recorded quantity | Maximum paired difference | Scenario |
| --- | ---: | --- |
| Peak reported actuator force, before joint-level clipping | 0.06475067 N·m | `seed73011_env00003` |
| Progress after onset | 0.00618553 m | `seed73011_env00001` |
| Mean post-onset tracking loss | 0.00038848 (m/s)² | `seed73011_env00001` |

All sixteen healthy and identity episodes had zero commanded correction and slew and finished without a fall. The current screen takes the largest absolute difference across quantities with different units and checks it against `1e-6`. The force difference sets the reported maximum. Progress and loss each independently exceed that tolerance. The run did not audit post-limit joint torque. [Episode rows](episodes.csv) · [control flags, thresholds and confidence intervals](summary.json).

At gain 0.70, the plain arm fell in 2/8 scenarios and the known-gain arm in 0/8. Mean loss fell from 0.41098 to 0.01366 against healthy 0.01261. The 0.85 condition produced little degradation, and every plain 0.55 episode fell. The 0.70 result motivates investigating repeatability at unity gain. These are screening observations with privileged knowledge of the gain.

**Handoff to Claude and Prasuna.** Before a larger paid run, compare healthy/healthy, identity/identity and healthy/identity on the same reset, commands and compiled Warp path. Capture the first divergent applied command or physics state and its control step. Archive the per-channel differences with units and the code SHA. If identical commands reproduce the drift, characterize repeatability separately before proposing new tolerances. If commands diverge, trace the adapter or replay path. Prasuna decides whether the full gate should advance after this check.


## Follow-up on the repeatability attempt

The next pinned pod run restored both policies on CUDA and stopped before the first rollout. The evaluator at `b215b9c` imports an unavailable `DeterminismType` at line 79. On a native-built 29-actuator G1 with the same package versions and the Warp backend on CPU, that import produced the same exception. The model also lacks the `deterministic` option read at line 357 when determinism is disabled, and at lines 171–172 of the default-mode companion checker. Setting the flag to off alone leaves that second failure in place. [GPU attempt](https://github.com/pchatla-arch/anticipatory-balance/tree/7693ebf4c4f43d968b8b30808cfb64abf57b7fd2/results/adaptive_gate_20261122T061338Z_d901f2) · [CPU reproduction](../../docs/adaptive_execution/reviews/warp_determinism_api_cpu_20261122.json) · [reproduction script](../../docs/adaptive_execution/reviews/probe_warp_determinism_api.py)

This evidence establishes an API incompatibility. It adds no new walking episodes. The source of the earlier identical-input variance and the outcome of the scientific gate remain open. An explicit test on the installed backend should precede the next paid smoke. Prasuna decides whether to authorize confirmation after the identity and healthy-repeat checks pass.
