# Adaptive execution bench: dev split

Source d756c37d0564837a74132c30f8e366320a6bb1a5, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| damping_x3 | declared | estimated_inverse | 8 | 0.000 | 0.0344 | 0.0814 | 0.000 | — |
| damping_x3 | declared | heuristic_gate | 8 | 0.000 | 0.0342 | 0.0002 | 0.000 | — |
| damping_x3 | declared | none | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| damping_x3 | declared | oracle | 8 | 0.000 | 0.0346 | 0.0000 | 0.000 | — |
| gain_055 | declared | estimated_inverse | 8 | 0.000 | 0.0361 | 0.0776 | 0.075 | — |
| gain_055 | declared | heuristic_gate | 8 | 0.750 | 1.8547 | 0.0008 | 0.086 | — |
| gain_055 | declared | none | 8 | 1.000 | 2.6355 | 0.0000 | 0.450 | — |
| gain_055 | declared | oracle | 8 | 0.000 | 0.0413 | 0.0000 | 0.450 | — |
| gain_070 | declared | estimated_inverse | 8 | 0.000 | 0.0295 | 0.0823 | 0.077 | — |
| gain_070 | declared | heuristic_gate | 8 | 0.000 | 0.0322 | 0.0002 | 0.079 | — |
| gain_070 | declared | none | 8 | 0.000 | 0.0734 | 0.0000 | 0.300 | — |
| gain_070 | declared | oracle | 8 | 0.000 | 0.0283 | 0.0000 | 0.300 | — |
| gain_070_return | declared | estimated_inverse | 8 | 0.000 | 0.0264 | 0.0807 | 0.087 | — |
| gain_070_return | declared | heuristic_gate | 8 | 0.000 | 0.0299 | 0.0037 | 0.088 | 6 |
| gain_070_return | declared | none | 8 | 0.000 | 0.0502 | 0.0000 | 0.300 | 0 |
| gain_070_return | declared | oracle | 8 | 0.000 | 0.0261 | 0.0009 | 0.300 | 2 |
| gain_085 | declared | estimated_inverse | 8 | 0.000 | 0.0269 | 0.0842 | 0.072 | — |
| gain_085 | declared | heuristic_gate | 8 | 0.000 | 0.0277 | 0.0004 | 0.077 | — |
| gain_085 | declared | none | 8 | 0.000 | 0.0290 | 0.0000 | 0.150 | — |
| gain_085 | declared | oracle | 8 | 0.000 | 0.0269 | 0.0000 | 0.150 | — |
| healthy | declared | estimated_inverse | 8 | 0.000 | 0.0219 | 0.0764 | 0.000 | — |
| healthy | declared | heuristic_gate | 8 | 0.000 | 0.0228 | 0.0004 | 0.000 | — |
| healthy | declared | none | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| healthy | declared | oracle | 8 | 0.000 | 0.0224 | 0.0000 | 0.000 | — |
| sensor_bias | declared | estimated_inverse | 8 | 0.000 | 0.0263 | 0.1565 | 0.000 | — |
| sensor_bias | declared | heuristic_gate | 8 | 0.000 | 0.0277 | 0.0082 | 0.000 | — |
| sensor_bias | declared | none | 8 | 0.000 | 0.0271 | 0.0000 | 0.000 | — |
| sensor_bias | declared | oracle | 8 | 0.000 | 0.0271 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared | estimated_inverse | 8 | 0.000 | 0.0322 | 0.1072 | 0.000 | — |
| sensor_scale_080 | declared | heuristic_gate | 8 | 0.000 | 0.0326 | 0.0466 | 0.000 | — |
| sensor_scale_080 | declared | none | 8 | 0.000 | 0.0309 | 0.0000 | 0.000 | — |
| sensor_scale_080 | declared | oracle | 8 | 0.000 | 0.0309 | 0.0000 | 0.000 | — |
| stale_telemetry | declared | estimated_inverse | 8 | 0.000 | 0.0317 | 0.0988 | 0.082 | — |
| stale_telemetry | declared | heuristic_gate | 8 | 0.000 | 0.0570 | 0.0000 | 0.084 | — |
| stale_telemetry | declared | none | 8 | 0.000 | 0.0570 | 0.0000 | 0.300 | — |
| stale_telemetry | declared | oracle | 8 | 0.000 | 0.0300 | 0.0000 | 0.300 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|
| damping_x3 | declared | estimated_inverse | +0.0002 [-0.0005, +0.0009] | +0.0812 [+0.0774, +0.0850] | +0.0000 [+0.0000, +0.0000] |
| damping_x3 | declared | none | +0.0003 [-0.0002, +0.0008] | -0.0002 [-0.0003, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| damping_x3 | declared | oracle | +0.0003 [-0.0002, +0.0008] | -0.0002 [-0.0003, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| gain_055 | declared | estimated_inverse | -1.8186 [-2.5072, -1.0146] | +0.0768 [+0.0722, +0.0812] | -0.7500 [-1.0000, -0.3750] |
| gain_055 | declared | none | +0.7808 [+0.2264, +1.4649] | -0.0008 [-0.0013, -0.0004] | +0.2500 [+0.0000, +0.6250] |
| gain_055 | declared | oracle | -1.8135 [-2.5002, -1.0100] | -0.0008 [-0.0013, -0.0004] | -0.7500 [-1.0000, -0.3750] |
| gain_070 | declared | estimated_inverse | -0.0026 [-0.0075, +0.0013] | +0.0820 [+0.0776, +0.0878] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | none | +0.0412 [+0.0235, +0.0625] | -0.0002 [-0.0004, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | oracle | -0.0039 [-0.0084, -0.0007] | -0.0002 [-0.0004, -0.0001] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | estimated_inverse | -0.0034 [-0.0054, -0.0016] | +0.0771 [+0.0750, +0.0794] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | none | +0.0203 [+0.0137, +0.0309] | -0.0037 [-0.0041, -0.0032] | +0.0000 [+0.0000, +0.0000] |
| gain_070_return | declared | oracle | -0.0038 [-0.0066, -0.0013] | -0.0028 [-0.0033, -0.0024] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | declared | estimated_inverse | -0.0008 [-0.0020, +0.0004] | +0.0838 [+0.0752, +0.0924] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | declared | none | +0.0013 [-0.0003, +0.0027] | -0.0004 [-0.0007, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| gain_085 | declared | oracle | -0.0008 [-0.0015, -0.0000] | -0.0004 [-0.0007, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | estimated_inverse | -0.0009 [-0.0026, +0.0004] | +0.0760 [+0.0738, +0.0783] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | none | -0.0004 [-0.0021, +0.0010] | -0.0004 [-0.0005, -0.0004] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | oracle | -0.0004 [-0.0021, +0.0010] | -0.0004 [-0.0005, -0.0004] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | declared | estimated_inverse | -0.0014 [-0.0039, +0.0008] | +0.1483 [+0.1460, +0.1507] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | declared | none | -0.0006 [-0.0011, -0.0000] | -0.0082 [-0.0089, -0.0075] | +0.0000 [+0.0000, +0.0000] |
| sensor_bias | declared | oracle | -0.0006 [-0.0011, -0.0000] | -0.0082 [-0.0089, -0.0075] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | estimated_inverse | -0.0005 [-0.0022, +0.0008] | +0.0606 [+0.0568, +0.0651] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | none | -0.0018 [-0.0044, +0.0002] | -0.0466 [-0.0483, -0.0447] | +0.0000 [+0.0000, +0.0000] |
| sensor_scale_080 | declared | oracle | -0.0018 [-0.0044, +0.0002] | -0.0466 [-0.0483, -0.0447] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | declared | estimated_inverse | -0.0253 [-0.0371, -0.0150] | +0.0988 [+0.0909, +0.1073] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | declared | none | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| stale_telemetry | declared | oracle | -0.0270 [-0.0397, -0.0171] | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
