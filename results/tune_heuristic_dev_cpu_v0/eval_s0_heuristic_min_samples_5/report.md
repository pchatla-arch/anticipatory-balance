# Adaptive execution bench: dev split

Source 577241f2333c4040d20e8a22de845c3ec4090789, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | heuristic_gate | 8 | 0.000 | 0.0332 | 0.0004 | 0.079 | — |
| gain_070_return | declared | heuristic_gate | 8 | 0.000 | 0.0300 | 0.0037 | 0.088 | 5 |
| healthy | declared | heuristic_gate | 8 | 0.000 | 0.0227 | 0.0006 | 0.000 | — |
| sensor_scale_080 | declared | heuristic_gate | 8 | 0.000 | 0.0342 | 0.0476 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
