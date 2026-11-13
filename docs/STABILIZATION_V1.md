# Foresight: first stabilization and switching diagnostic

For the current GitHub/pod workflow, use [POD_SMOKE_TEST.md](POD_SMOKE_TEST.md).
The stabilization changes are already integrated on `foresight/stabilization-v1`;
the archived patch does not need to be applied again. The convenience runner
automatically publishes compact results to `foresight/pod-results`.

Project title: **Foresight: Latency-Aware Anticipatory Bracing from Depth Perception**.

This proposed patch is based on repository commit
`3409b26fb999acf9b9d79c1f712e8902e382cfee`. It prepares the first diagnostic after
the Session A review. It is not a completed belief controller or a retrained BRACE
skill. It requires no new packages and makes no installation, training, commit,
or push calls. Runtime validation on the existing GPU stack is still required.

## Purpose

Distinguish ordinary stopping under the walking policy, switching into the BRACE
policy, and the effect of a projectile. Keep paired reset states, sampled inputs,
and incoming world-space launch trajectories identical across comparison arms.

| Controller | Projectile enabled | Projectile launch suppressed |
|---|---|---|
| CONTINUE | Reactive reference | Locomotion control |
| Nominal actor with scheduled zero command | Stopping under disturbance | Stopping control |
| Nominal to BRACE to nominal | Skill switch under disturbance | Skill-switch control |

A nominal reference rollout fixes each launch position and velocity before the
six comparison arms run. These are still the original approach scenarios; this
patch does not retarget every policy to force a collision. Contact coverage can
remain low. That is acceptable for the first transition diagnostic and must be
reported. Shared external trajectories do not imply identical contact outcomes.

The no-projectile control suppresses launch without changing the nominal arrival
or brace schedule. The parked object remains in the model; the diagnostic fails
if a launch-disabled arm records a projectile contact.

## Changes

- Refresh the command in both G1 observation arrays before selecting the actor.
  The adapter uses the inspected rss2025 G1 one-frame observation contract:
  command occupies indices 9:12 and privileged state begins with actor state.
  It does not resample sensor noise or change observation dimensions.
- Add opt-in fixed world-space launch vectors and a launch-enable mask. Defaults
  preserve the prior training path and scenario random draws.
- Preserve legacy full-horizon progress in `progress_m`, while new comparisons
  use `progress_before_failure_m` with an explicit first-failure convention.
- Classify fixed-window post-contact survival as observed success, observed
  failure, or unknown/censored. Missing contact and failure before contact do not
  become impact-recovery successes. Early observed failures are conclusive even
  if the remaining horizon is shorter than the requested recovery window.
- Record separate first-failure flags for inversion, self contact, nonfinite
  physics, and sustained tilt. Multiple flags may be true.
- Define diagnostic readiness using pelvis height, uprightness, bilateral foot
  support, lateral stance width, active brace, and sustained sampled dwell. A
  first readiness event is not proof of impact safety or readiness at contact.
- Save readable scenario and outcome tables, source/package metadata, and raw
  compressed traces before invariant checks. Refuse to overwrite a results
  directory. Numerical or input-pairing failures produce `harness_failure`.

## Local validation

From the repository root, using its existing Python environment:

```bash
python -m unittest discover -s tests -p 'test_eval_*.py' -v
python scripts/eval_switch_gate.py --help
```

These tests are simulator-free: they require NumPy and the standard library.
They check event boundaries and censoring, stale command replacement, readiness,
paired scenario accounting, result preservation, and nonfinite-data handling.
They do not establish JAX tracing, Warp execution, or policy quality.

## First pod run

Use the already active rss2025 environment and repository root. The checkpoint
paths below come from the committed Session A summaries; the loader rejects
missing checkpoints rather than silently continuing with a random policy.

```bash
python scripts/eval_switch_gate.py \
  --nominal logs/nominal_s1/G1Impact-20261113-063725/checkpoints/000202342400 \
  --brace logs/brace_s1/G1ImpactBrace-20261113-065807/checkpoints/000202342400 \
  --impl warp --episodes 32 --batches 1 --lead 0.5 --hold 0.5 --steps 300 \
  --out results/switch_gate_smoke_v1
```

This is 32 shared scenarios, six comparison arms (192 comparison rollouts),
plus 32 reference rollouts. It is a smoke diagnostic, not a statistically
validated safety or improvement claim. A .5-second lead plus .5-second hold
uses nominal arrival time and is not the final 300 ms commitment controller.

Return `summary.json`, `manifest.json`, `scenarios.csv`, and `episodes.csv` from
the output directory. Keep the `batch_*.npz` archives for failure inspection;
they can be loaded with `numpy.load(path, allow_pickle=False)`. Package versions
and the local Playground source revision are recorded in the manifest.

If the run fails, preserve the directory and return the traceback together with
its `summary.json` and `manifest.json`. Choose a new output name for a rerun.

## Decision after the diagnostic

| Observation | Next action |
|---|---|
| Harness invariant or numerical failure | Correct the implementation/runtime issue before interpreting policy quality. |
| CONTINUE fails without a projectile | Inspect the restored walker, reset states, and control interface. |
| Zero-command nominal fails without a projectile | Investigate stopping behavior and command transitions. |
| BRACE adds failures over zero-command nominal without a projectile | Inspect switch traces, then train on walking-to-brace transitions if warranted. |
| Transitions are stable without impact but bracing harms impact outcomes | Improve the brace posture and disturbance training; recalibrate the impact task. |
| No new failures in the small sample | Increase paired evaluation and inspect readiness; this is not yet evidence that bracing helps. |

The script reports `switch_failure` if any paired no-projectile trial newly
fails under BRACE compared with the zero-command nominal actor. This is an
inspection flag, not a significance test. Otherwise it reports `inconclusive`;
there is deliberately no automatic declaration that the BRACE skill is valid.

## Boundaries of this iteration

The patch does not yet solve target/contact calibration, ensure 2.5 seconds of
follow-up after every actual contact, implement walking-state BRACE training,
or build depth perception and the belief layer. Short follow-up is now exposed
as censored. Those next tasks depend on this diagnostic. Existing published
results remain historical; they are not regenerated or relabeled by applying
the patch.

The first accepted research milestone is a usable preparation behavior: stable
transitions and a paired lead-time region where preparation improves the
failure-versus-interruption tradeoff. Only then should the belief decision
comparison be interpreted as a test of anticipation.
