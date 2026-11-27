# Adaptive execution bench: dev split

Source aec7f18c11deb9de98a4ead15f988d602878165f, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| damping_x3 | exact | estimated_inverse | 8 | 0.000 | 0.0343 | 0.0000 | 0.000 | — |
| damping_x3 | exact | heuristic_gate | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| damping_x3 | exact | none | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| damping_x3 | exact | oracle | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| gain_055 | exact | estimated_inverse | 8 | 0.000 | 0.0412 | 0.0000 | 0.019 | — |
| gain_055 | exact | heuristic_gate | 8 | 0.375 | 1.0770 | 0.0000 | 0.024 | — |
| gain_055 | exact | none | 8 | 1.000 | 2.6355 | 0.0000 | 0.450 | — |
| gain_055 | exact | oracle | 8 | 0.000 | 0.0413 | 0.0000 | 0.450 | — |
| gain_070 | exact | estimated_inverse | 8 | 0.000 | 0.0293 | 0.0000 | 0.013 | — |
| gain_070 | exact | heuristic_gate | 8 | 0.000 | 0.0325 | 0.0000 | 0.014 | — |
| gain_070 | exact | none | 8 | 0.000 | 0.0734 | 0.0000 | 0.300 | — |
| gain_070 | exact | oracle | 8 | 0.000 | 0.0283 | 0.0000 | 0.300 | — |
| gain_070_return | exact | estimated_inverse | 8 | 0.000 | 0.0256 | 0.0081 | 0.023 | 31 |
| gain_070_return | exact | heuristic_gate | 8 | 0.000 | 0.0266 | 0.0039 | 0.024 | 9 |
| gain_070_return | exact | none | 8 | 0.000 | 0.0502 | 0.0000 | 0.300 | 0 |
| gain_070_return | exact | oracle | 8 | 0.000 | 0.0261 | 0.0009 | 0.300 | 2 |
| gain_085 | exact | estimated_inverse | 8 | 0.000 | 0.0267 | 0.0000 | 0.007 | — |
| gain_085 | exact | heuristic_gate | 8 | 0.000 | 0.0266 | 0.0000 | 0.008 | — |
| gain_085 | exact | none | 8 | 0.000 | 0.0290 | 0.0000 | 0.150 | — |
| gain_085 | exact | oracle | 8 | 0.000 | 0.0269 | 0.0000 | 0.150 | — |
| healthy | exact | estimated_inverse | 8 | 0.000 | 0.0227 | 0.0000 | 0.000 | — |
| healthy | exact | heuristic_gate | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| healthy | exact | none | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| healthy | exact | oracle | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| sensor_bias | exact | estimated_inverse | 8 | 0.000 | 0.0257 | 0.1607 | 0.000 | — |
| sensor_bias | exact | heuristic_gate | 8 | 0.000 | 0.0266 | 0.0145 | 0.000 | — |
| sensor_bias | exact | none | 8 | 0.000 | 0.0271 | 0.0000 | 0.000 | — |
| sensor_bias | exact | oracle | 8 | 0.000 | 0.0271 | 0.0000 | 0.000 | — |
| sensor_scale_080 | exact | estimated_inverse | 8 | 0.000 | 0.0318 | 0.0817 | 0.000 | — |
| sensor_scale_080 | exact | heuristic_gate | 8 | 0.000 | 0.0311 | 0.0787 | 0.000 | — |
| sensor_scale_080 | exact | none | 8 | 0.000 | 0.0309 | 0.0000 | 0.000 | — |
| sensor_scale_080 | exact | oracle | 8 | 0.000 | 0.0309 | 0.0000 | 0.000 | — |
| stale_telemetry | exact | estimated_inverse | 8 | 0.000 | 0.0305 | 0.0000 | 0.027 | — |
| stale_telemetry | exact | heuristic_gate | 8 | 0.000 | 0.0570 | 0.0000 | 0.027 | — |
| stale_telemetry | exact | none | 8 | 0.000 | 0.0570 | 0.0000 | 0.300 | — |
| stale_telemetry | exact | oracle | 8 | 0.000 | 0.0300 | 0.0000 | 0.300 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|
| damping_x3 | exact | estimated_inverse | -0.0003 [-0.0006, -0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| damping_x3 | exact | none | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| damping_x3 | exact | oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_055 | exact | estimated_inverse | -1.0357 [-1.9011, -0.3654] | +0.0000 [+0.0000, +0.0000] | -0.3750 [-0.7500, -0.1250] |
| gain_055 | exact | none | +1.5585 [+0.8169, +2.1975] | +0.0000 [+0.0000, +0.0000] | +0.6250 [+0.2500, +0.8750] |
| gain_055 | exact | oracle | -1.0357 [-1.8992, -0.3664] | +0.0000 [+0.0000, +0.0000] | -0.3750 [-0.7500, -0.1250] |
| gain_070 | exact | estimated_inverse | -0.0032 [-0.0059, -0.0010] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | exact | none | +0.0409 [+0.0232, +0.0609] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | exact | oracle | -0.0042 [-0.0064, -0.0022] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | exact | estimated_inverse | -0.0011 [-0.0024, +0.0002] | +0.0042 [+0.0037, +0.0047] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | exact | none | +0.0236 [+0.0161, +0.0347] | -0.0039 [-0.0042, -0.0036] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | exact | oracle | -0.0005 [-0.0020, +0.0006] | -0.0030 [-0.0033, -0.0028] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | exact | estimated_inverse | +0.0001 [-0.0006, +0.0006] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | exact | none | +0.0024 [+0.0001, +0.0046] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | exact | oracle | +0.0003 [-0.0006, +0.0010] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| healthy | exact | estimated_inverse | +0.0003 [-0.0006, +0.0013] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| healthy | exact | none | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| healthy | exact | oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | exact | estimated_inverse | -0.0010 [-0.0017, -0.0002] | +0.1462 [+0.1444, +0.1478] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | exact | none | +0.0005 [-0.0002, +0.0012] | -0.0145 [-0.0157, -0.0130] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | exact | oracle | +0.0005 [-0.0002, +0.0012] | -0.0145 [-0.0157, -0.0130] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | exact | estimated_inverse | +0.0007 [-0.0011, +0.0024] | +0.0030 [+0.0018, +0.0041] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | exact | none | -0.0003 [-0.0026, +0.0020] | -0.0787 [-0.0812, -0.0760] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | exact | oracle | -0.0003 [-0.0026, +0.0020] | -0.0787 [-0.0812, -0.0760] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | exact | estimated_inverse | -0.0265 [-0.0377, -0.0175] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | exact | none | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | exact | oracle | -0.0270 [-0.0397, -0.0171] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
