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
