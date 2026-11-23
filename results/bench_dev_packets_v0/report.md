# Adaptive execution bench: dev split

Source dd8d3db80896a41c122aaf4368d6c3452461ad79, physics jax, 300 control steps, schedule walk_turn_stop_resume, 8 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_055 | declared | none | 8 | 1.000 | 2.5267 | 0.0000 | 0.450 | — |
| gain_055 | exact | none | 8 | 1.000 | 2.5267 | 0.0000 | 0.450 | — |
| gain_070 | declared | none | 8 | 0.000 | 0.0740 | 0.0000 | 0.300 | — |
| gain_070 | exact | none | 8 | 0.000 | 0.0740 | 0.0000 | 0.300 | — |
| gain_085 | declared | none | 8 | 0.000 | 0.0292 | 0.0000 | 0.150 | — |
| gain_085 | exact | none | 8 | 0.000 | 0.0292 | 0.0000 | 0.150 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
