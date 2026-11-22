# WP 2.2 velocity-envelope development pilot

Run this CPU experiment after Prasuna records the development protocol decision. From the repository, fetch the draft into a separate checkout. Use the existing project Python environment with nominal s1 and its pinned RSS 2025 Playground checkout.

```sh
git fetch origin oaa/wp2.2-velocity-envelope-pilot
git worktree add --detach ../oaa-velocity-pilot FETCH_HEAD
cd ../oaa-velocity-pilot
```

The run covers 40 cells and 96,000 simulated control transitions on CPU. Each cell writes compressed per-step traces. Runtime and disk use depend on the host. The small wiring run includes JAX compilation and cannot be scaled linearly to estimate the full run.
 The proposed acceleration bound came from screening on earlier development trajectories. The reference heuristic uses Claude’s implementation and its default thresholds. Its tuning is a separate work package.

```sh
python scripts/run_velocity_pilot.py plan --out results/velocity_pilot_plan
python scripts/run_velocity_pilot.py wiring --out results/velocity_pilot_wiring
python scripts/run_velocity_pilot.py run --out results/velocity_pilot_dev
```

Each command expects a fresh output directory. The plan records the development seed map, checkpoint, source tree, sensor models, fixed budget and candidate parameters before physics. The eight-tick wiring check uses a separate seed and one healthy episode. The development run uses one attempt, eight paired episodes per cell, four scenario families, two transport conditions, five methods and 300 control ticks. Earliest delivery permits a post-transition packet at the next decision. The declared channel keeps the original one-to-three-tick transport delay and in-order queue. Both retain the same source-indexed draws for position, torque, velocity error and dropout. All five methods receive the same channel within a cell. The symmetric velocity enclosure uses a 400 rad/s² acceleration assumption and ±0.5 rad/s velocity error. The original comparison uses the 10 rad/s speed enclosure.

Read `episodes.csv` for per-episode fixed-horizon loss, falls, intervention on a physically healthy plant, and time to withdraw after gain restoration. Compare both candidates in `paired_vs_speed.json`, and the default heuristic in `summary.json`. Each `hold_*.json` gives the observed and source-available hip and knee counts, and the actual source-age histogram. For both candidates it additionally gives complete sampled-hold coverage, using each sample’s own time from the source, including the decision boundary and every MJX substep. The `trace_*.npz` files include control reasons and ordered packet releases. Use the live mask for physical trace analyses. The accuracy claim here is limited to sampled simulation substeps, with intervention governed by the original projection and common limits. Validation, test, gate sign-off and GPU runs are independent of this pilot.
