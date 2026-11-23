# Adaptive execution bench: dev split

Source aa51278507b4354cf6cb3a164ffb44b3f9642aa4, physics jax, 8 control steps, schedule walk_turn_stop_resume, 1 episodes per cell.

| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) | Mean source age, hip / knee (ms) | Withdrew / reached / scheduled |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| healthy | declared | candidate_velocity | 1 | 0.000 | 0.0000 | 0.0000 | 0.000 | — | 50.00 / 50.00 | — |

Source age is the elapsed time from the newest valid released measurement to the decision, averaged across live, source-available joint ticks separately for hips and knees. Each hold file gives available and total joint ticks. Withdrawal is the median among finite episodes that reached the threshold. Counts show withdrawals, finite episodes observed at the scheduled return, and all episodes scheduled to return.

Paired differences are method minus `heuristic_gate` on identical scenarios and sensor draws (95% bootstrap intervals).

| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |
|---|---|---|---|---|---|

Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. Methods never receive the injected gain, onset or return step; the oracle does.
