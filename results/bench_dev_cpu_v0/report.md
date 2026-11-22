# Adaptive execution bench: dev split

Source a45b10c0820e3eb45d0edfbbe9c1cf1cc6f261d8, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | estimated_inverse | 8 | 0.000 | 0.0303 | 0.0610 | 0.081 | — |
| gain_070 | declared | heuristic_gate | 8 | 0.000 | 0.0310 | 0.0021 | 0.083 | — |
| gain_070 | declared | none | 8 | 0.000 | 0.0511 | 0.0000 | 0.300 | — |
| gain_070 | declared | oracle | 8 | 0.000 | 0.0276 | 0.0000 | 0.300 | — |
| gain_070 | exact | estimated_inverse | 8 | 0.000 | 0.0296 | 0.0000 | 0.051 | — |
| gain_070 | exact | heuristic_gate | 8 | 0.000 | 0.0309 | 0.0000 | 0.050 | — |
| gain_070 | exact | none | 8 | 0.000 | 0.0511 | 0.0000 | 0.300 | — |
| gain_070 | exact | oracle | 8 | 0.000 | 0.0276 | 0.0000 | 0.300 | — |
| gain_070_return | declared | estimated_inverse | 8 | 0.000 | 0.0239 | 0.0671 | 0.109 | — |
| gain_070_return | declared | heuristic_gate | 8 | 0.000 | 0.0252 | 0.0275 | 0.107 | 37 |
| gain_070_return | declared | none | 8 | 0.000 | 0.0396 | 0.0000 | 0.300 | 0 |
| gain_070_return | declared | oracle | 8 | 0.000 | 0.0240 | 0.0009 | 0.300 | 2 |
| gain_070_return | exact | estimated_inverse | 8 | 0.000 | 0.0237 | 0.0265 | 0.078 | — |
| gain_070_return | exact | heuristic_gate | 8 | 0.000 | 0.0244 | 0.0233 | 0.078 | — |
| gain_070_return | exact | none | 8 | 0.000 | 0.0396 | 0.0000 | 0.300 | 0 |
| gain_070_return | exact | oracle | 8 | 0.000 | 0.0240 | 0.0009 | 0.300 | 2 |
| healthy | declared | estimated_inverse | 8 | 0.000 | 0.0259 | 0.0692 | 0.033 | — |
| healthy | declared | heuristic_gate | 8 | 0.000 | 0.0263 | 0.0022 | 0.035 | — |
| healthy | declared | none | 8 | 0.000 | 0.0254 | 0.0000 | 0.000 | — |
| healthy | declared | oracle | 8 | 0.000 | 0.0254 | 0.0000 | 0.000 | — |
| healthy | exact | estimated_inverse | 8 | 0.000 | 0.0254 | 0.0000 | 0.000 | — |
| healthy | exact | heuristic_gate | 8 | 0.000 | 0.0254 | 0.0000 | 0.000 | — |
| healthy | exact | none | 8 | 0.000 | 0.0254 | 0.0000 | 0.000 | — |
| healthy | exact | oracle | 8 | 0.000 | 0.0254 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared | estimated_inverse | 8 | 0.000 | 0.0288 | 0.0985 | 0.205 | — |
| sensor_scale_080 | declared | heuristic_gate | 8 | 0.000 | 0.0290 | 0.0833 | 0.213 | — |
| sensor_scale_080 | declared | none | 8 | 0.000 | 0.0275 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared | oracle | 8 | 0.000 | 0.0275 | 0.0000 | 0.000 | — |
| sensor_scale_080 | exact | estimated_inverse | 8 | 0.000 | 0.0279 | 0.0832 | 0.200 | — |
| sensor_scale_080 | exact | heuristic_gate | 8 | 0.000 | 0.0275 | 0.0798 | 0.200 | — |
| sensor_scale_080 | exact | none | 8 | 0.000 | 0.0275 | 0.0000 | 0.000 | — |
| sensor_scale_080 | exact | oracle | 8 | 0.000 | 0.0275 | 0.0000 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|
| gain_070 | declared | estimated_inverse | -0.0008 [-0.0019, +0.0004] | +0.0589 [+0.0533, +0.0651] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | none | +0.0201 [+0.0157, +0.0251] | -0.0021 [-0.0028, -0.0013] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | oracle | -0.0034 [-0.0057, -0.0019] | -0.0021 [-0.0028, -0.0013] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | exact | estimated_inverse | -0.0013 [-0.0020, -0.0008] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | exact | none | +0.0202 [+0.0156, +0.0258] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | exact | oracle | -0.0033 [-0.0049, -0.0022] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | estimated_inverse | -0.0013 [-0.0017, -0.0009] | +0.0396 [+0.0356, +0.0435] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | none | +0.0144 [+0.0093, +0.0195] | -0.0275 [-0.0295, -0.0252] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | oracle | -0.0012 [-0.0021, -0.0004] | -0.0266 [-0.0286, -0.0244] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | exact | estimated_inverse | -0.0006 [-0.0015, +0.0003] | +0.0032 [+0.0023, +0.0040] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | exact | none | +0.0153 [+0.0098, +0.0206] | -0.0233 [-0.0252, -0.0213] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | exact | oracle | -0.0004 [-0.0016, +0.0009] | -0.0224 [-0.0242, -0.0205] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | estimated_inverse | -0.0004 [-0.0009, +0.0000] | +0.0671 [+0.0595, +0.0745] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | none | -0.0009 [-0.0015, -0.0002] | -0.0022 [-0.0030, -0.0014] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | oracle | -0.0009 [-0.0015, -0.0002] | -0.0022 [-0.0030, -0.0014] | +0.0000 [+0.0000, +0.0000] |
| healthy | exact | estimated_inverse | -0.0000 [-0.0001, +0.0001] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| healthy | exact | none | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| healthy | exact | oracle | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | estimated_inverse | -0.0003 [-0.0006, +0.0002] | +0.0152 [+0.0096, +0.0225] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | none | -0.0015 [-0.0035, +0.0003] | -0.0833 [-0.0870, -0.0796] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | oracle | -0.0015 [-0.0035, +0.0003] | -0.0833 [-0.0870, -0.0796] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | exact | estimated_inverse | +0.0005 [-0.0002, +0.0012] | +0.0035 [+0.0021, +0.0051] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | exact | none | +0.0000 [-0.0013, +0.0015] | -0.0798 [-0.0835, -0.0762] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | exact | oracle | +0.0000 [-0.0013, +0.0015] | -0.0798 [-0.0835, -0.0762] | +0.0000 [+0.0000, +0.0000] |

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
