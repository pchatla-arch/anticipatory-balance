# Adaptive execution bench: dev split

Source 43ca493bb5e7ca5e7397b1059a1eb922514cca50, physics jax, 160 control steps, schedule walk_turn_stop_resume, 2 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| gain_070 | declared | heuristic_gate | 2 | 0.000 | 0.0375 | 0.0001 | 0.106 | — |
| gain_070 | declared | none | 2 | 0.000 | 0.0688 | 0.0000 | 0.300 | — |
| gain_070 | declared | oracle | 2 | 0.000 | 0.0211 | 0.0000 | 0.300 | — |
| healthy | declared | heuristic_gate | 2 | 0.000 | 0.0132 | 0.0003 | 0.000 | — |
| healthy | declared | none | 2 | 0.000 | 0.0137 | 0.0000 | 0.000 | — |
| healthy | declared | oracle | 2 | 0.000 | 0.0149 | 0.0000 | 0.000 | — |

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|
| gain_070 | declared | none | +0.0314 [+0.0238, +0.0390] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| gain_070 | declared | oracle | -0.0164 [-0.0188, -0.0139] | -0.0001 [-0.0002, -0.0000] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | none | +0.0005 [+0.0005, +0.0005] | -0.0003 [-0.0004, -0.0002] | +0.0000 [+0.0000, +0.0000] |
| healthy | declared | oracle | +0.0017 [+0.0011, +0.0023] | -0.0003 [-0.0004, -0.0002] | +0.0000 [+0.0000, +0.0000] |

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
