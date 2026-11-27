# Adaptive execution bench: dev split

Source aec7f18c11deb9de98a4ead15f988d602878165f, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| damping_x3 | declared_fast | estimated_inverse | 8 | 0.000 | 0.0341 | 0.0711 | 0.000 | — |
| damping_x3 | declared_fast | heuristic_gate | 8 | 0.000 | 0.0340 | 0.0003 | 0.000 | — |
| damping_x3 | declared_fast | none | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| damping_x3 | declared_fast | oracle | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| gain_055 | declared_fast | estimated_inverse | 8 | 0.000 | 0.0436 | 0.0648 | 0.073 | — |
| gain_055 | declared_fast | heuristic_gate | 8 | 0.500 | 1.2521 | 0.0009 | 0.075 | — |
| gain_055 | declared_fast | none | 8 | 1.000 | 2.6355 | 0.0000 | 0.450 | — |
| gain_055 | declared_fast | oracle | 8 | 0.000 | 0.0413 | 0.0000 | 0.450 | — |
| gain_070 | declared_fast | estimated_inverse | 8 | 0.000 | 0.0291 | 0.0712 | 0.077 | — |
| gain_070 | declared_fast | heuristic_gate | 8 | 0.000 | 0.0308 | 0.0003 | 0.077 | — |
| gain_070 | declared_fast | none | 8 | 0.000 | 0.0734 | 0.0000 | 0.300 | — |
| gain_070 | declared_fast | oracle | 8 | 0.000 | 0.0283 | 0.0000 | 0.300 | — |
| gain_070_return | declared_fast | estimated_inverse | 8 | 0.000 | 0.0260 | 0.0713 | 0.082 | 96 |
| gain_070_return | declared_fast | heuristic_gate | 8 | 0.000 | 0.0283 | 0.0037 | 0.083 | 7 |
| gain_070_return | declared_fast | none | 8 | 0.000 | 0.0502 | 0.0000 | 0.300 | 0 |
| gain_070_return | declared_fast | oracle | 8 | 0.000 | 0.0261 | 0.0009 | 0.300 | 2 |
| gain_085 | declared_fast | estimated_inverse | 8 | 0.000 | 0.0269 | 0.0731 | 0.071 | — |
| gain_085 | declared_fast | heuristic_gate | 8 | 0.000 | 0.0274 | 0.0005 | 0.076 | — |
| gain_085 | declared_fast | none | 8 | 0.000 | 0.0290 | 0.0000 | 0.150 | — |
| gain_085 | declared_fast | oracle | 8 | 0.000 | 0.0269 | 0.0000 | 0.150 | — |
| healthy | declared_fast | estimated_inverse | 8 | 0.000 | 0.0225 | 0.0642 | 0.000 | — |
| healthy | declared_fast | heuristic_gate | 8 | 0.000 | 0.0224 | 0.0005 | 0.000 | — |
| healthy | declared_fast | none | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| healthy | declared_fast | oracle | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| sensor_bias | declared_fast | estimated_inverse | 8 | 0.000 | 0.0264 | 0.1607 | 0.000 | — |
| sensor_bias | declared_fast | heuristic_gate | 8 | 0.000 | 0.0267 | 0.0129 | 0.000 | — |
| sensor_bias | declared_fast | none | 8 | 0.000 | 0.0271 | 0.0000 | 0.000 | — |
| sensor_bias | declared_fast | oracle | 8 | 0.000 | 0.0271 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared_fast | estimated_inverse | 8 | 0.000 | 0.0313 | 0.0980 | 0.000 | — |
| sensor_scale_080 | declared_fast | heuristic_gate | 8 | 0.000 | 0.0346 | 0.0667 | 0.000 | — |
| sensor_scale_080 | declared_fast | none | 8 | 0.000 | 0.0309 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared_fast | oracle | 8 | 0.000 | 0.0309 | 0.0000 | 0.000 | — |
| stale_telemetry | declared_fast | estimated_inverse | 8 | 0.000 | 0.0317 | 0.0988 | 0.082 | — |
| stale_telemetry | declared_fast | heuristic_gate | 8 | 0.000 | 0.0570 | 0.0000 | 0.084 | — |
| stale_telemetry | declared_fast | none | 8 | 0.000 | 0.0570 | 0.0000 | 0.300 | — |
| stale_telemetry | declared_fast | oracle | 8 | 0.000 | 0.0300 | 0.0000 | 0.300 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|
| damping_x3 | declared_fast | estimated_inverse | +0.0001 [-0.0009, +0.0010] | +0.0708 [+0.0679, +0.0738] | +0.0000 [+0.0000, +0.0000] |
| damping_x3 | declared_fast | none | +0.0006 [-0.0001, +0.0012] | -0.0003 [-0.0005, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| damping_x3 | declared_fast | oracle | +0.0006 [-0.0001, +0.0012] | -0.0003 [-0.0005, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| gain_055 | declared_fast | estimated_inverse | -1.2085 [-2.0317, -0.4057] | +0.0639 [+0.0602, +0.0683] | -0.5000 [-0.8750, -0.1250] |
| gain_055 | declared_fast | none | +1.3834 [+0.6797, +2.0658] | -0.0009 [-0.0015, -0.0004] | +0.5000 [+0.1250, +0.8750] |
| gain_055 | declared_fast | oracle | -1.2108 [-2.0326, -0.4093] | -0.0009 [-0.0015, -0.0004] | -0.5000 [-0.8750, -0.1250] |
| gain_070 | declared_fast | estimated_inverse | -0.0017 [-0.0039, +0.0004] | +0.0709 [+0.0660, +0.0744] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared_fast | none | +0.0426 [+0.0237, +0.0639] | -0.0003 [-0.0006, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared_fast | oracle | -0.0026 [-0.0044, -0.0010] | -0.0003 [-0.0006, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared_fast | estimated_inverse | -0.0022 [-0.0040, -0.0005] | +0.0675 [+0.0643, +0.0706] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared_fast | none | +0.0220 [+0.0148, +0.0326] | -0.0037 [-0.0042, -0.0034] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared_fast | oracle | -0.0022 [-0.0042, -0.0002] | -0.0029 [-0.0032, -0.0025] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | declared_fast | estimated_inverse | -0.0005 [-0.0014, +0.0005] | +0.0727 [+0.0657, +0.0794] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | declared_fast | none | +0.0016 [-0.0006, +0.0036] | -0.0005 [-0.0009, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | declared_fast | oracle | -0.0005 [-0.0014, +0.0005] | -0.0005 [-0.0009, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared_fast | estimated_inverse | +0.0001 [-0.0008, +0.0011] | +0.0637 [+0.0618, +0.0659] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared_fast | none | +0.0000 [-0.0011, +0.0010] | -0.0005 [-0.0006, -0.0004] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared_fast | oracle | +0.0000 [-0.0011, +0.0010] | -0.0005 [-0.0006, -0.0004] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | declared_fast | estimated_inverse | -0.0003 [-0.0018, +0.0015] | +0.1477 [+0.1454, +0.1499] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | declared_fast | none | +0.0005 [-0.0003, +0.0015] | -0.0129 [-0.0143, -0.0115] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | declared_fast | oracle | +0.0005 [-0.0003, +0.0015] | -0.0129 [-0.0143, -0.0115] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared_fast | estimated_inverse | -0.0033 [-0.0066, -0.0004] | +0.0313 [+0.0283, +0.0344] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared_fast | none | -0.0037 [-0.0073, -0.0010] | -0.0667 [-0.0682, -0.0651] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared_fast | oracle | -0.0037 [-0.0073, -0.0010] | -0.0667 [-0.0682, -0.0651] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | declared_fast | estimated_inverse | -0.0253 [-0.0371, -0.0150] | +0.0988 [+0.0909, +0.1073] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | declared_fast | none | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | declared_fast | oracle | -0.0270 [-0.0397, -0.0171] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
