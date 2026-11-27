# Adaptive execution bench: dev split

Source 89b6d0dd31020f00032330c76684c727b1f697a0, physics jax, 400 control steps, schedule walk_turn_stop_resume, 2 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | estimated_inverse | 2 | 0.000 | 0.0314 | 0.0832 | 0.072 | — |
| gain_070 | declared | heuristic_gate | 2 | 0.000 | 0.0383 | 0.0001 | 0.073 | — |
| gain_070 | declared | none | 2 | 0.000 | 0.0855 | 0.0000 | 0.300 | — |
| gain_070 | declared | oracle | 2 | 0.000 | 0.0287 | 0.0000 | 0.300 | — |
| gain_070_return | declared | estimated_inverse | 2 | 0.000 | 0.0315 | 0.0833 | 0.079 | — |
| gain_070_return | declared | heuristic_gate | 2 | 0.000 | 0.0363 | 0.0024 | 0.076 | 5.5 |
| gain_070_return | declared | none | 2 | 0.000 | 0.0610 | 0.0000 | 0.300 | 0.0 |
| gain_070_return | declared | oracle | 2 | 0.000 | 0.0309 | 0.0007 | 0.300 | 3.0 |
| healthy | declared | estimated_inverse | 2 | 0.000 | 0.0213 | 0.0751 | 0.000 | — |
| healthy | declared | heuristic_gate | 2 | 0.000 | 0.0227 | 0.0003 | 0.000 | — |
| healthy | declared | none | 2 | 0.000 | 0.0221 | 0.0000 | 0.000 | — |
| healthy | declared | oracle | 2 | 0.000 | 0.0216 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared | estimated_inverse | 2 | 0.000 | 0.0314 | 0.1079 | 0.000 | — |
| sensor_scale_080 | declared | heuristic_gate | 2 | 0.000 | 0.0310 | 0.0437 | 0.000 | — |
| sensor_scale_080 | declared | none | 2 | 0.000 | 0.0268 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared | oracle | 2 | 0.000 | 0.0268 | 0.0000 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|
| gain_070 | declared | estimated_inverse | -0.0069 [-0.0157, +0.0018] | +0.0830 [+0.0830, +0.0831] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | none | +0.0472 [+0.0371, +0.0573] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | oracle | -0.0096 [-0.0181, -0.0011] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | estimated_inverse | -0.0048 [-0.0054, -0.0042] | +0.0809 [+0.0735, +0.0882] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | none | +0.0247 [+0.0148, +0.0346] | -0.0024 [-0.0026, -0.0022] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | oracle | -0.0054 [-0.0058, -0.0051] | -0.0017 [-0.0019, -0.0016] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | estimated_inverse | -0.0013 [-0.0019, -0.0008] | +0.0748 [+0.0724, +0.0772] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | none | -0.0006 [-0.0007, -0.0004] | -0.0003 [-0.0004, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | oracle | -0.0010 [-0.0023, +0.0002] | -0.0003 [-0.0004, -0.0003] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | estimated_inverse | +0.0003 [-0.0007, +0.0014] | +0.0642 [+0.0641, +0.0644] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | none | -0.0042 [-0.0069, -0.0016] | -0.0437 [-0.0453, -0.0421] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | oracle | -0.0042 [-0.0069, -0.0016] | -0.0437 [-0.0453, -0.0421] | +0.0000 [+0.0000, +0.0000] |

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
