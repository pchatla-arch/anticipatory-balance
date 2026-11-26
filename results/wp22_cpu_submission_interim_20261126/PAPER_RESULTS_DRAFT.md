# Results draft — Online Actuator Adaptation (CPU development evidence)

**Draft for AA228/CS238 · Nov 26, 2026 · Do not describe as held-out or G2 results**

## Experimental setup

An existing CPU JAX/MuJoCo development benchmark at source commit `a45b10c0820e3eb45d0edfbbe9c1cf1cc6f261d8` evaluates a frozen Unitree G1 walker under four scenarios: actuator proportional-servo gain reduced to 0.70, reduced gain followed by unannounced restoration, healthy actuation, and an effort-sensor scale confound at 0.80. It compares no correction, privileged oracle, estimator-driven inverse correction, and estimator-plus-heuristic gating. Each method is evaluated with exact or declared delayed telemetry (1–3 control steps, 2% packet dropout) using 8 paired scenarios per cell, 300 control steps per episode. The grid consists of **32 cells and 256 episode records**.

This is an **earlier baseline development benchmark**, run before the 17-evaluation frozen heuristic. It does not evaluate the proposed velocity-informed hold envelope. The later WP 2.2 pilot, whose registered protocol has 40 cells / 320 rows / 96,000 transitions, is **not published and not yet independently verified**.

## Recomputed baseline results (declared telemetry)

| Family | Correction off loss | Earlier heuristic loss | Privileged oracle loss | Heuristic relative loss reduction | Earlier heuristic harmful offset | Ungated inverse harmful offset |
|---|---:|---:|---:|---:|---:|---:|
| Gain 0.70 | 0.05113 | 0.03101 | 0.02757 | 39.34% | 0.00208 rad | 0.06100 rad |
| Gain 0.70 → healthy | 0.03962 | 0.02522 | 0.02401 | 36.34% | 0.02749 rad | 0.06707 rad |

Eight matched scenarios per row. All 32 cell means were independently recomputed from the archived `episodes.csv`, with **maximum absolute discrepancy 0** relative to the published summary for checked mean loss, harmful offset, progress and fall rate. No falls or nonfinite episodes were observed in these 256 development records. The fact that the no-correction arm has more progress than the corrected arm in some cells shows that postfault tracking loss is not the only relevant behavior metric; both must be reported.

### Paired exploratory analysis

Paired heuristic-minus-none mean loss difference for gain 0.70 under declared telemetry: **−0.02011**, 95% paired bootstrap interval **[−0.02519, −0.01548]**. For the gain-return family: **−0.01440**, interval **[−0.01941, −0.00932]**. These intervals use 10,000 deterministic seed-228 resamples of eight matched episode differences per family; this is a post-hoc development analysis, not a generalization or held-out guarantee. Full paired tables are in `bench_cpu_paired_bootstrap.json`.

### Important negative finding: withdrawal after restoration

In the earlier heuristic baseline under declared transport, only **1 of 8** return-to-healthy episodes recorded a withdrawal; the reported conditional median is 37 control steps **for that one withdrawal**, not for all eight. Seven did not withdraw within the observed post-return window. A median alone is misleading; report the numerator and exposure denominator. This motivates the more conservative velocity-aware authority and release policy, but does not prove it solves the problem.

## Frozen heuristic tuning (separate source and stage)

Claude's completed development-grid at commit `d756c37d0564837a74132c30f8e366320a6bb1a5` includes 17 objective evaluations on the development split. The selected configuration is forgetting factor 0.90, dead band 0.10, maximum residual RMS 0.02, maximum age 3, minimum samples 10, and minimum excitation 0.02. The composite objective `loss + 0.5 harm + 0.1 falls` decreased from **0.05186746** to **0.04278539**, **17.51%**. Healthy interference decreased from 0.04267326 to 0.02349465 (**44.94%**), while loss rose slightly from 0.03053083 to 0.03103806 (**+1.66%**). No falls occurred within the recorded tuning evaluations.

**Limitation:** coordinate-descent development tuning is not evidence of out-of-sample improvement; the same development families inform the pending pilot. Later velocity-envelope superiority must be demonstrated by the separately frozen and audited 40-cell experiment.

## Preflight engineering evidence (not a scientific trial)

One distinct 8-tick healthy CPU wiring episode on reviewed WP 2.2 source `aa51278507b4354cf6cb3a164ffb44b3f9642aa4` was finite and had no fall. Source-available sample age averaged **50.00 ms** for hip and knee joints. The archived independent reduction matched source-age histograms and sampled hold coverage: 24/24 observed hip joint-ticks and 8/8 observed knee joint-ticks covered. No gain restoration was scheduled, so withdrawal was **not applicable**, not 0 ms. The archived preflight passed 22 tests and 10 subtests. This only verifies wiring and instrumentation.

## Scope and paper interpretation

A suitable supported contribution at this stage is a **CPU development empirical study of gain estimation, gated correction, telemetry confounding, and return-to-healthy withdrawal** with a transparent incomplete follow-on velocity-envelope experiment. Avoid claims of robust actuator adaptation across distributions, closed-loop superiority of the velocity envelope, real-time GPU or hardware performance, held-out generalization, or humanoid deployment. The source and raw data are linked from `README.md`.

## Remaining experiment to strengthen this section

Recover the original PR #46 scientific attempt using its exclusive lock and `scientific_attempt.json`. If confirmed unstarted and released, execute *one* CPU-only 40-cell development pilot with PR #46 source pinned to `4b20e3e` and Claude's frozen heuristic pinned to `d756c37`. Then publish all 320 paired rows, 40-cell hold-age and coverage records, raw NPZ traces, results and SHA digests; audit AE-014 denominators; obtain Claude's independent critique. If not possible by Nov 30, submit the accurate development-baseline results and explicit missing follow-on experiment rather than inferring unobserved results.

