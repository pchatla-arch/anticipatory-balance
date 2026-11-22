# AE-011 Healthy-offset score dropped post-onset samples on families without an actuator change

| Field | Value |
| --- | --- |
| Status | Resolved (fix on `oaa/claude-bench`, PR #35) |
| Severity | Major (biased a tuning and noninferiority metric; no claim relied on it yet) |
| Project · area | Adaptive execution · evaluation harness (WP 2.3) |
| Found | 2026-11-22, ChatGPT's review of PR #35 at c5397f8 |
| Issue | #38 |
| Owner | Claude fixes · ChatGPT reviews · Prasuna merges |

## Symptom
`harmful_offset_rad` (mean |effective correction| while the plant is healthy) used
`plant_healthy = (t < onset) | (t >= return)` for every family. On the healthy, sensor-scale,
sensor-bias and damping families no actuator changes, so the plant is healthy at every step;
the mask nevertheless excluded the scheduled window, which is exactly where the damping
confound acts and where the sensor confounds have had time to mislead the estimator.
`mean_applied_offset_rad` also divided by `exposed + healthy_steps`, double-counting steps
that are both after onset and healthy.

## Evidence
Verified by reading `foresight/adaptive/bench.py` (ChatGPT) and reproduced in the closed-loop
test added with the fix: on a gain-1.0 family `healthy_steps` must equal the number of observed
steps. In `results/bench_dev_cpu_v0` the healthy-family harmful offsets therefore averaged
only the pre-onset and post-return steps (about 60% of the episode).

## Why it did not work
The mask was written for the gain families and reused for the confound families without
checking that "scheduled window" and "plant healthy" coincide only when the gain changes.

## Impact
`harmful_offset_rad` and `mean_applied_offset_rad` on the four no-change families in the dev
report are biased; direction: healthy-family values are roughly representative (the estimator
misbehaves before onset as much as after) but the damping family's confound window was
excluded entirely. No tuning, test or paper number used these values yet. The tuning run
started with the old code was stopped and restarted.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Derive `plant_healthy` from the true per-actuator gain (all gains 1 ⇒ healthy everywhere) | Exact; one line | None |
| Per-family flag | Explicit | A second source of truth beside the scenario gains |

## Decision and next steps
`plant_healthy = (t < onset) | (t >= return) | all(gain_true == 1)`; `mean_applied_offset_rad`
divides by the observed step count. Test: `ClosedLoopTests` asserts `healthy_steps ==
observed steps` on a gain-1 family. Dev report regenerated with the fix before any tuning.

## Resolution
Fixed in commit (see PR #35), 2026-11-22.
