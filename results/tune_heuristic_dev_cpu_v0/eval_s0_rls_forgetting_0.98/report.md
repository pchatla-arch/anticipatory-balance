# Adaptive execution bench: dev split

Source 8a8d931657ed3b14035c10fa0ab4069793b85659, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | heuristic_gate | 8 | 0.000 | 0.0343 | 0.0020 | 0.089 | — |
| gain_070_return | declared | heuristic_gate | 8 | 0.000 | 0.0267 | 0.0269 | 0.115 | — |
| healthy | declared | heuristic_gate | 8 | 0.000 | 0.0218 | 0.0011 | 0.000 | — |
| sensor_scale_080 | declared | heuristic_gate | 8 | 0.000 | 0.0332 | 0.0843 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
