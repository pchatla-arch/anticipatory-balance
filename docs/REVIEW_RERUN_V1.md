# Foresight: reviewed experiment rerun

This branch builds on scientific branch `session-d`, commit
`cc17b7b1066a5694a4c9e8c5e4dddf0a8f5ecdcf` (Session G). It retains the published
walking and BRACE v2 checkpoints and adds a new, explicitly versioned experiment
protocol. Historical Session E/F/G numbers are not relabeled as results of this
protocol. New scientific conclusions must wait for the GPU run and its audits.

## One command on the pod

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/review-rerun-v1/scripts/start_review.sh) review
```

Run this in the pod terminal. A fresh pod may request GitHub browser/device
authentication before starting experiments; complete that prompt so the result
publisher can use your account. The default launcher verifies repository push
permission before starting the worker. Do not paste credentials into chat or a remote
URL. Existing working Git authentication can be selected with
`FORESIGHT_SKIP_GITHUB_LOGIN=1`; that explicit override bypasses the login and
permission check and trusts your existing Git setup.

The command creates the isolated checkout `/workspace/foresight-review-runner`,
reuses a compatible existing Playground venv or installs the pinned fresh
environment, restores both published policies, and starts tmux session
`foresight-review`. It does **not** train PPO policies. A busy GPU or tracked
local changes in the runner checkout stop the launcher before new work starts.

```bash
tmux attach -t foresight-review
```

Detach with **Ctrl-B, then D**. The full log is
`/workspace/foresight-review-runner/logs/fresh_pod.log`. Run-specific evidence is
under `results/review_v1_<timestamp>_<id>/` in that checkout.

### Time and stopping

The full experiment process budget is **90 minutes after setup**, including CPU
tests and GPU preflight. Pinned Python setup has a separate **25-minute** limit;
initial system-tool installation, downloads, login, reporting, and publication
add time. This is an execution budget, not a guarantee that every stage will
finish or that billing stops at 90 minutes. A timed-out stage halts subsequent
stages and publishes the completed evidence plus failure logs.

The wrapper requests **Stop**, never Terminate, after the current run's upload is
verified. Automatic stopping also needs working RunPod controls/credentials.
If upload verification or stop controls fail, the pod remains running and the
launcher prints the manual action. Check the RunPod dashboard before leaving it
unattended. `NO_STOP=1` keeps the pod running intentionally. GPU compute billing
and persistent storage are separate concerns.

The smaller operational run is:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/review-rerun-v1/scripts/start_review.sh) smoke
```

It defaults to 30 minutes after setup and omits held-out tests. Full mode already
contains the small gates; running smoke separately is optional and repeats them.
For a repeatable revision, set `FORESIGHT_CODE_REF` to this branch's full commit
SHA. To change the budget, set `FORESIGHT_MAX_MINUTES` before the command.

## Questions and experiments

All stages use a 20 ms control timestep, 360 steps (7.2 seconds), the same frozen
`nominal_s1`/`brace2_s1` pair, and copied launch trajectories across conditions.
Each episode has an explicit scenario ID. Scenario seeds for gate, development,
and final test sets are disjoint. Different methods, rates, and latencies within
a set deliberately share scenarios. The exact commands are saved in `plan.json`.

| Stage | Full-run scenarios per condition | Purpose |
| --- | ---: | --- |
| Oracle gate | 32 | Reference timing, switching, launch/no-launch controls, and readiness records |
| Oracle development | 256, in two batches | Fit a new response surrogate at leads 0.1, 0.2, 0.3, 0.5, and 0.75 s, with a CONTINUE reference |
| Response fit | Development data only | Fit failure versus log momentum and preparation lead; grouped prediction checks |
| Decision gate | 32 | Compile and audit the six controller families at 0 and 100 ms delay |
| Depth gates | 16 at each delay | Compile and audit the online image path at 0 and 100 ms delay |
| Decision tests | 512 per condition | Compare fixed controllers at 10/50 Hz and 0/100/300 ms simulated delay |
| Depth tests | 512 per condition | Compare depth, parametric, and visibility-matched parametric observations at 0/100 ms delay |

Decision families are reactive walking, reference-entry oracle, TTC threshold,
lead-aware belief, a blind-wait rule, and one-observation lookahead. TTC is fixed
at 0.3 s and the belief preparation cost at 0.1. These are **fixed-controller
comparisons**, not an optimized Pareto frontier or a claim that the threshold is
the best possible baseline. The reactive and oracle conditions are shared across
latencies, giving 14 conditions per full parametric stage, not 18. Depth has one
shared reactive baseline plus four methods times three observation sources.

Operational gates check complete tables, timestamps, paired IDs, pooled
numerators/denominators, provenance, commitment bounds, and reference-event
coverage. They do **not** require any method to beat another. Negative results
are valid results. Missing reference events and censored outcomes are retained
or counted explicitly; the fitter's exclusions are saved.

## Corrections relative to the review

### Action duration and timing

The primary protocol commits once per episode. The deadline is locked when the
decision is made from the predicted entry time plus a 0.5 s hold, rounded to
control ticks, subject to the 0.3 s minimum commitment and a 4 s maximum hold.
Later observations do not silently shorten the chosen intervention. The oracle
uses the corresponding fixed deadline from its privileged reference event.
Preparation cost uses the same executed-duration convention. This intentionally
differs from Session G's adaptive closest-approach release; it must be reported
as a new protocol, not a numerically equivalent patch. Historical release logic
is available only through explicit legacy settings.

The oracle event is the first reference-trajectory entry into the 0.45 m
horizontal collision proxy, with the vertical gate checked and within-step
constant-velocity interpolation. It is a **reference-entry oracle**, not exact
knowledge of physical contact under every counterfactual policy. The projected
event can move when the robot responds. Missing events are listed, and the
operational gate requires at least 90% coverage among intended hit scenarios.

### Belief and value of another observation

Capture validity, capture time, and delivery time travel together. A delayed
frame captured before launch cannot become valid merely because launch has
occurred by delivery. The corrected filter uses the constant-velocity,
zero-process-noise model appropriate to the pre-impact projectile, including
the delayed position/velocity cross-covariance and a Joseph covariance update.
Nonzero process noise requires an explicit alternative inference protocol.

Lookahead uses the actual next delivery from the current cadence phase and
configured latency. It includes no-detection and early-entry branches. Valid
hypothetical observations use the same analytic delayed Gaussian update as the
online filter, followed by new posterior samples; they do not falsely identify
a full six-dimensional state by concentrating weights on one prior sample.
No-detection updates use a moment-matched visibility/dropout model. Mass remains
unobserved and is averaged before choosing an action. The implementation is a
one-observation approximation, not an exact POMDP solver. Its future visibility
forecast is a simplified pelvis-centered cone, including on the depth path.

### New response fit

The old contact-conditioned, nominal-arrival fit is not silently reused. A
run-specific model is fitted after new development oracle rollouts. It includes
scheduled launched reference-entry scenarios, **including noncontacts and
failures before the planned switch**. Labels use the scheduled discretized
intervention, preventing selection only of episodes surviving long enough to
execute BRACE. Censored and missing-reference rows are excluded explicitly and
counted. Grouped checks hold out an entire scenario batch, not alternate leads
from the same scenario. Policy test data never fit the response model.

The reduced response is logistic in log momentum and lead. It does not resolve
all speed, posture, or direction effects. Lead values above observed support
are clipped; values below the shortest training lead are extrapolation. The
stored model contains its support, source hashes, exclusions, and limitations.

### Rendered depth

The actual depth path consumes rendered images, uses the per-axis median
surface estimator plus known-radius correction, and does not gate detections
with the ground-truth launch flag. The projectile is parked outside the sensor
range before launch. False detections before launch are measured. Robot/camera
pose, camera calibration, and object radius remain simulation assumptions.

Three sources distinguish observation mechanisms:

- `depth`: image detections and image-derived centers.
- `parametric`: geometry/FOV/dropout plus noisy ground-truth centers.
- `parametric_matched`: the actual image validity with a noisy ground-truth
  center. This is an explicitly privileged diagnostic to separate missed
  detections from center-estimation error, not a deployable vision method.

The likelihood is fixed before held-out tests; calibration and range-dependent
errors on those tests are diagnostics, not fitting data. Rendering wall time is
recorded separately from the controlled simulated perception delay. This is a
simulation study and makes no hardware or real-camera claim.

### Outcome and uncertainty evidence

Readiness requires active BRACE throughout a stable pre-contact window. At 50 Hz,
0.2 s stability spans eleven samples, not ten. Readiness at contact is checked
at the pre-action state immediately before the interval where contact is first
registered. First readiness, stable-window start, and readiness at contact are
different fields. Post-impact posture is not counted as anticipatory readiness.

Rates pool episode counts, rather than averaging conditional rates across
batches. CSV tables preserve IDs, scenario parameters, contact and failure
events, preparation intervals, progress, and censoring. Paired comparisons
include discordant counts and approximate normal 95% intervals. These quantify
scenario variation for **one frozen skill pair**; they do not measure variation
across training seeds. Common-contact analyses are labeled descriptive because
contact itself can depend on the intervention.

## Results and review workflow

The runner writes `run.json`, `plan.json`, `tests.json`, `preflight.json`, stage
tables, stage `audit.json` files, paired comparisons, numeric trace archives,
`response_model.json`, `report.md`, and `analysis.json`. When relevant data are
available it also writes three figures: fixed controller comparisons, depth
comparisons, and readiness. Partial reports explicitly name failed and unstarted
stages. A local ZIP is retained in `logs/`.

Compact evidence is pushed to **`foresight/experiment-results`**, in a separate
results tree. The publisher preserves the working checkout, redacts text logs,
allows only named evidence files, and verifies the remote commit. It never
force-pushes. Numeric NPZ arrays are validated without unpickling. Publication
failure does not change an experiment failure into success, or discard completed
data. To retry upload without repeating GPU work:

```bash
bash /workspace/foresight-review-runner/scripts/start_review.sh retry-publish RUN_ID
```

After the run, review the uploaded report and audits before modifying the paper.
Keep the same fixed settings for held-out comparisons. Further parameter tuning
needs a development split and another untouched test set. Multiple training
seeds remain follow-up work if the corrected effect is useful.

## Validation before pod execution

Local validation uses an isolated Python 3.12 environment with actual JAX/JAXLIB
0.12.0.1 CPU, Brax 0.14.2, MuJoCo/MJX 3.16.0, and the pinned Playground source.
It includes compiled and vectorized belief/controller tests, sensor timing,
evidence arithmetic and reports, response fitting, publication to isolated local
Git repositories, and conditional-stop behavior. Both published policies have
been restored and a real G1 reset/action/physics step has executed on CPU.
The machine-readable test record is under `analysis/review_rerun_v1/`.

The combined CPU suite passed 136 tests with no skips. After the final launcher
corrections, all 14 launcher tests passed again, including script references and
the pre-experiment GitHub push-permission check. A separate real batched CPU test
ran two environments for three physical steps and verified trajectory alignment
and the requested brace schedule. See the saved validation records for scope.

Use `start_review.sh` for this protocol. Historical session runners are preserved
for provenance; execute them at their original revisions if reproducing their
historical settings and response fits.

These checks do not establish that GPU Warp rendering or the complete experiment
fits the budget. The first pod gates exercise those paths, and failures preserve
their evidence for correction before another paid run.
