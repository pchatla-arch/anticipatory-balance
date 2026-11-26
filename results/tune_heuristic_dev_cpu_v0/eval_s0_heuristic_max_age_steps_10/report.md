# Adaptive execution bench: dev split

Source a02c9f5af8604355f6719ff79e5eb007a837b60a, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | heuristic_gate | 8 | 0.000 | 0.0304 | 0.0007 | 0.079 | — |
| gain_070_return | declared | heuristic_gate | 8 | 0.000 | 0.0277 | 0.0060 | 0.088 | 8 |
| healthy | declared | heuristic_gate | 8 | 0.000 | 0.0225 | 0.0011 | 0.000 | — |
| sensor_scale_080 | declared | heuristic_gate | 8 | 0.000 | 0.0356 | 0.0728 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
