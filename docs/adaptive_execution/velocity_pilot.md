# WP 2.2 velocity-envelope development pilot

Run this CPU experiment after Prasuna records the development protocol decision. From the repository, fetch the draft into a separate checkout. Use the existing project Python environment with nominal s1 and its pinned RSS 2025 Playground checkout.

```sh
git fetch origin oaa/wp2.2-velocity-envelope-pilot
git worktree add --detach ../oaa-velocity-pilot FETCH_HEAD
cd ../oaa-velocity-pilot
```

The run covers 40 cells and 96,000 simulated control transitions on CPU. Each cell writes compressed per-step traces. Runtime and disk use depend on the host. The small wiring run includes JAX compilation and cannot be scaled linearly to estimate the full run.

The acceleration bound came from development screening. Run the wiring check at any time. It exercises one candidate on a healthy episode using separate seeds.

```sh
python scripts/run_velocity_pilot.py wiring --out results/velocity_pilot_wiring
```

The scientific run requires Claude's final `results/tune_heuristic_dev_cpu_v0/heuristic.json` committed on `oaa/claude-bench` and Prasuna’s G1 decision. Once both are ready, pin the published revision and use it for both commands.

```sh
git fetch origin oaa/claude-bench
heuristic_rev=$(git rev-parse FETCH_HEAD)
python scripts/run_velocity_pilot.py plan --heuristic-rev "$heuristic_rev" --out results/velocity_pilot_plan
python scripts/run_velocity_pilot.py run --heuristic-rev "$heuristic_rev" --out results/velocity_pilot_dev
```

A plan produced without `--heuristic-rev` records the pending freeze. The scientific run archives the exact Git blob as `frozen_heuristic.json` and records its revision, path, SHA-256, development tuning provenance, and fully resolved RLS and heuristic parameters before physics. All methods share these settings; only the heuristic uses its gate thresholds. The named sensing presets are Claude's `DECLARED` and `DECLARED_FAST`. The controller consumes `bound_velocity_rad_s` and the shared physical sensor injects `velocity_error_rad_s` once.

Each command expects a fresh output directory. The plan records the development seed map, checkpoint, source tree, sensor models, fixed budget and candidate parameters before physics. The eight-tick wiring check uses a separate seed and one healthy episode. The development run uses one attempt, eight paired episodes per cell, four scenario families, two transport conditions, five methods and 300 control ticks. Earliest delivery permits a post-transition packet at the next decision. The declared channel keeps the original one-to-three-tick transport delay and in-order queue. Both retain the same source-indexed draws for position, torque, velocity error and dropout. All five methods receive the same channel within a cell. The symmetric velocity enclosure uses a 400 rad/s² acceleration assumption and ±0.5 rad/s velocity error. The original comparison uses the 10 rad/s speed enclosure.

Read `episodes.csv` for per-episode fixed-horizon loss, falls, intervention on a physically healthy plant, and time to withdraw after gain restoration. Compare the velocity and speed candidates in `paired_vs_speed.json`, and the frozen tuned heuristic in `summary.json`. The summary table reports the mean elapsed age of the most recent valid released sample at each decision for hip and knee joint ticks. On return-to-healthy episodes it reports the conditional median time to withdraw and counts of episodes that withdrew, reached the scheduled return with finite observations, and were scheduled to return. Each `hold_*.json` gives the observed and source-available hip and knee counts, and the actual source-age histogram. For both candidates it additionally gives complete sampled-hold coverage, using each sample’s own time from the source, including the decision boundary and every MJX substep. The `trace_*.npz` files include control reasons and ordered packet releases. Use the live mask for physical trace analyses. The accuracy claim here is limited to sampled simulation substeps, with intervention governed by the original projection and common limits. Validation, test, gate sign-off and GPU runs are independent of this pilot.
