# Adaptive execution bench: dev split

Source c8111ed4d74cf0030b82dd4fcb319b9a39b20083, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | heuristic_gate | 8 | 0.000 | 0.0298 | 0.0010 | 0.078 | — |
| gain_070_return | declared | heuristic_gate | 8 | 0.000 | 0.0258 | 0.0101 | 0.090 | 26 |
| healthy | declared | heuristic_gate | 8 | 0.000 | 0.0220 | 0.0014 | 0.000 | — |
| sensor_scale_080 | declared | heuristic_gate | 8 | 0.000 | 0.0324 | 0.0841 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
