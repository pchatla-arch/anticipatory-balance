# Adaptive execution: known-gain feasibility gate

Scientific gate: **inconclusive**.

Installed MuJoCo-Warp lacks DeterminismType.ALL, opt._impl.deterministic. Default-backend engineering smoke only.

This tests a bounded, privileged correction for proportional-servo gain loss. It does not estimate faults online and is not evidence of learned adaptation.

| Phase | Condition | Episodes | Failure rate | Mean fixed-horizon loss | Mean post-fault progress (m) |
|---|---|---:|---:|---:|---:|
| screen | healthy | 8 | 0.000 | 0.0126 | 3.036 |
| screen | identity | 8 | 0.000 | 0.0126 | 3.036 |
| screen | healthy_repeat | 8 | 0.000 | 0.0126 | 3.036 |
| screen | gain_085_plain | 8 | 0.000 | 0.0136 | 3.295 |
| screen | gain_085_oracle | 8 | 0.000 | 0.0125 | 3.036 |
| screen | gain_070_plain | 8 | 0.125 | 0.4543 | 5.517 |
| screen | gain_070_oracle | 8 | 0.000 | 0.0137 | 3.048 |
| screen | gain_055_plain | 8 | 1.000 | 3.1508 | 1.893 |
| screen | gain_055_oracle | 8 | 0.000 | 0.0223 | 3.226 |

screen: the healthy repeat reproduced the healthy arm bitwise (largest metric difference 0); identity largest difference 0.

The same initial states, commands, onset times and noise streams are used within each phase. Confirmation uses new scenario seeds. All screened severities remain in episodes.csv.

The loss is mean squared planar local-velocity error, capped at 4 (m/s)^2, over the fixed post-fault horizon. Failure receives that maximum loss for the terminal transition and all remaining steps. Progress stops at the last finite terminal transition. Numerical failures are retained and invalidate the scientific gate.

See summary.json for paired uncertainty, selection, and threshold verdicts. A completed run with a failed scientific gate is still a useful result. No actuator lag, reduced torque limit, online estimator, vision loop, or real hardware is tested.
