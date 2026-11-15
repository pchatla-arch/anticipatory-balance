# Session H: reviewed follow-up to completed Session G

Session G results are already published at `65d753f`; Session H starts from
`84c161a6d3782a93385d4a258ecc48487e5bfe57`, which moved the parked projectile to
`[-80, 0, 0.9]`. H investigates that depth-sensor correction and the effect of
changing the release threshold. It preserves Session G's controller, filter,
observation model, response fit, and trained policies.

## Start on a fresh or restarted pod

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/session-d/scripts/start.sh) h
```

If prompted, complete GitHub browser/device authentication. The default path
checks repository push permission before starting the experiment worker. No
token belongs in chat or in the remote URL. Existing working credentials can be
used with `FORESIGHT_SKIP_GITHUB_LOGIN=1`, which explicitly bypasses that check.

The launcher creates an isolated checkout at `/workspace/foresight-session-h`
and tmux session `foresight-h`. It reuses a compatible existing Playground venv;
otherwise it installs the pinned Python 3.12 environment. Existing unrelated
environments are not upgraded. All policies and the response model are restored
from GitHub. **Session H trains no policies and refits no response model.**

```bash
tmux attach -t foresight-h
```

Detach with Ctrl-B, then D. Log:
`/workspace/foresight-session-h/logs/fresh_pod.log`.

## Time and completion

The experiment budget is **75 minutes after setup**. At $2.50/hour that interval
costs $3.125; setup, login, reporting and upload add time. Fresh Python setup has
a separate 25-minute limit, and initial system-tool installation has separate
timeouts. This is not a total billing cap or a guarantee that every stage fits.
Change the experiment limit with `FORESIGHT_MAX_MINUTES` if needed.

The first three stages are small operational gates. A stage failure or timeout
halts subsequent stages, writes a partial report, and attempts to publish all
completed evidence and logs. Negative performance is valid evidence and never
constitutes a gate failure.

Results go automatically to the dedicated **`foresight/experiment-results`**
branch, under `results/session_h_<timestamp>_<id>/`. The publisher verifies the
remote commit before permitting automatic pod stop. With working RunPod stop
controls, the wrapper requests **Stop**, never Terminate. `NO_STOP=1` keeps the
pod running intentionally. Missing stop controls or a failed upload leave the
pod running and print the manual action; check the RunPod dashboard.

To retry publication without repeating GPU work:

```bash
bash /workspace/foresight-session-h/scripts/start_session_h.sh retry-publish RUN_ID
```

For only the small gates, use `start.sh h smoke`; the budget defaults to 20
minutes. Full H already includes these gates. Each invocation creates new output
directories; historical summary files are never treated as current completion.

## Experiment plan

Every stage uses the seed-1 walking/BRACE-v2 pair, 300 control steps at 50 Hz,
and the existing `results/response_model.json`. Commands, settings, source
revision, checkpoint hashes and response hashes are saved.

| Stage | Episodes per condition | Conditions | Purpose |
| --- | ---: | ---: | --- |
| Depth gate, 0 ms | 16 | 9 | Render/controller integration and evidence checks |
| Depth gate, 100 ms | 16 | 9 | Delayed image-delivery path |
| Hold gate | 32 | 9 | Release parameter, paired outcomes and accounting |
| E18b release 0.3 s | 2,048 | 9 | Fresh same-code reference |
| E18b release 0.6 s | 2,048 | 9 | Longer-release comparison |
| E18b release 1.0 s | 2,048 | 9 | Longer-release comparison |
| E17c depth, 0 ms | 512 | 9 | Parked object outside the detector range |
| E17c depth, 100 ms | 512 | 9 | Same scene correction with delay |

The hold arms share seeds 200/201, scenarios, thresholds, costs, and all settings
except `release_after`. Conditions are one reactive reference, TTC thresholds
0.3/0.4 s, and belief-lead preparation costs 0.05/0.1 at delays 0/100 ms. These
repeat G's development scenarios for a controlled exploratory ablation; they
are not a newly untouched test set.

Depth uses seeds 300/301, as in E17b: reactive plus four methods (TTC 0.3, lead,
blind wait, observation lookahead at cost 0.05) under parametric and rendered
depth observations. Gate seeds are separate. The primary comparison is the
current same-code paired evidence; historical G numbers have different
accounting and are labeled accordingly.

## Meaning of the hold setting

`release_after` is compared with the continually updated belief mean's closest
approach. A strict threshold at 20 ms ticks, the 300 ms minimum commitment, and
the staleness release affect actual duration. Another brace may begin later.
It is **not** a guaranteed hold of exactly that duration after physical passage.
The new tables record release reasons, repeated braces and horizon truncation.

The response fit and preparation-cost hold remain fixed at their Session G
settings. Longer execution holds therefore test sensitivity and response-model
transfer, not optimally refitted controllers. The oracle's separate nominal
arrival plus 0.5 s hold is not part of H's nine-condition release comparison.

This is separate from `foresight/review-rerun-v1` (`dd3141a`), which changes the
action contract and delayed-belief machinery. Combining those changes with a
hold ablation would make attribution ambiguous, so H retains G's behavior.

## Reporting corrections

- Rates are pooled from episode numerators/denominators, not averaged across
  batches with different hit counts.
- Explicit scenario IDs, seed/index, mass/speed, contact/failure events and
  full-precision CSVs accompany numeric-only NPZ archives.
- Primary preparation time counts intervals through the first terminal
  transition. Post-termination requests remain in separately labeled raw or
  legacy fields for comparison with G's original metric.
- Depth records prelaunch capture-frame and detection counts, both before
  termination and under the raw G counting convention. Prelaunch false
  detections and false preparation on near-miss episodes are distinct measures.
- Configuration/source/artifact hashes and completion markers support checks
  of counts, pairing, release setting, event times and pooled metrics.
- `report.md`, `analysis.json` and `hold_comparisons.csv` compare longer holds
  against the fresh 0.3 s arm with paired effects and approximate 95% intervals.
  These intervals describe scenario variation for one frozen policy pair.

The image detector still assumes an uncluttered simulated sphere, known radius,
and exact robot/camera pose. Command-derived velocity and G's future-observation
approximation are retained and documented. Rendering time is measured
separately from simulated observation delay. No hardware claim is made.

## Validation before the pod

The combined suite passed **82 tests, no skips**, using real JAX/JAXLIB 0.12.0.1
on CPU for the controller and detector tests. The suite also checks the launcher,
publication receipts, partial failure handling, evidence arithmetic, and
conditional pod-stop behavior.

The actual frontier evaluator restored both saved seed-1 policies and completed
two environments with three physical steps, writing CSV, numeric NPZ, strict
JSON, paired summaries and a plot. Both published seed-2 policies also restored
through the new G checkpoint-reuse validator. The actual environment has the
projectile parked at `[-80, 0, 0.9]`.

Records are under `analysis/session_h_review/`. These are CPU integration checks;
the GPU depth-rendering experiments and their scientific outcomes remain for
the pod run.

## Session G reuse and interpretation

`run_session_g.sh` now skips the seed-2 training block when E19 already exists.
Otherwise, a published bundle is validated and restored on CPU before reuse.
An incomplete existing bundle stops clearly; a missing bundle retains the
original training path. There is no reason to retrain or rerun completed G just
to review it. The corrected numerical review is in `docs/SESSION_G_REVIEW.md`.

Two trained pairs reveal sensitivity, but do not isolate whether the walker,
BRACE policy, reference trajectory, or transferred response model caused it.
The current data also do not establish an observation-lookahead advantage.
Readiness has remaining post-step/window-definition limitations described in
the review; H does not claim to resolve them.
