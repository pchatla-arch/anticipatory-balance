# WP 2.2 CPU evidence and Dec 4 submission recovery

**As of 2026-11-26 · development split only · no registered test · no new GPU/pod authorization**

This directory preserves **measured** CPU-development tuning results and the earlier G1 CPU-wiring preflight. It is **not** the 40-cell closed-loop velocity-envelope scientific experiment. Never present the 8-tick preflight as 320 scientific episodes or tuning improvement as a held-out result.

## Published data

- [evidence.json](evidence.json): 17 source-referenced tuning evaluations, frozen parameters, quantitative reductions, preflight and unresolved experiment state.
- [tuning_metrics.csv](tuning_metrics.csv): 17 actual objective/loss/harm/fall measurements, suitable for the tuning plot.
- Underlying primary files: `oaa/claude-bench@d756c37d0564837a74132c30f8e366320a6bb1a5:results/tune_heuristic_dev_cpu_v0/{tuning_log.json,heuristic.json}`.
- Preflight independent audit: `oaa/wp2.2-velocity-envelope-pilot@4b20e3e737b790bdb853faf6fc9c099c59f48316:results/velocity_pilot_wp22_preflight/wiring_audit.json`.

**Reproducible facts:** start objective 0.0518674595374614; best objective 0.042785387802869085 (17.51% lower); start healthy intervention harm 0.04267326110973954; best 0.023494650311768055 (44.92% lower); locomotion loss increases from 0.03053082898259163 to 0.031038062646985054; zero falls in the recorded tuning evaluations. These are development-selection measurements, not confidence intervals or an unbiased generalization estimate. Tuning used the same development families; the selected configuration is a coordinate-descent result.

## One-attempt experiment recovery — do this BEFORE any physics

The original scientific operator reserved exactly one attempt on [PR #46 comment 7087861171](https://github.com/pchatla-arch/anticipatory-balance/pull/46#issuecomment-7087861171). The original prepared workspace path was `/workspace/scratch/a92174288b11/pilot_runtime`, which is **not accessible to this chat**. Missing GitHub results do **not** establish no attempt started. The current user decision on [issue #31](https://github.com/pchatla-arch/anticipatory-balance/issues/31) is CPU-only, no further pod sessions, no held-out test.

On the **original host only**, inspect these files before touching the launcher:

```sh
ROOT=/workspace/scratch/a92174288b11/pilot_runtime
ls -la "$ROOT"
for f in pilot_status.json last_freeze_check.txt scientific_attempt.json publication.json; do
  echo "--- $f"; test ! -f "$ROOT/$f" || cat "$ROOT/$f"
done
ps -ef | grep -E 'launch_when_frozen|complete_result|publish_result|run_velocity_pilot' | grep -v grep || true
```

**If `physics_complete` already exists:** retain all original data and use that workspace's documented `complete_result.py` and `publish_result.js` with its verified content cache and lock. The original scripts must be inspected for CLI parameters; do not invent arguments. Read back the published Git blobs and independently verify raw episodes and traces.

**If `scientific_attempt.json` exists:** do not replay physics, even if it failed. Preserve raw evidence and report the failure or incomplete attempt.

**If no scientific attempt exists AND the original operator is verified gone / claim released:** only then establish one replacement owner, publish provenance, and invoke the existing documented single-attempt launcher, with exclusive file lock. Do not start an independent runner while the original claimant may remain alive.

A checkout of the exact reviewed pilot source can be prepared *without* using scientific seeds:

```sh
git fetch origin oaa/wp2.2-velocity-envelope-pilot oaa/claude-bench
git checkout --detach 4b20e3e737b790bdb853faf6fc9c099c59f48316
git show d756c37d0564837a74132c30f8e366320a6bb1a5:results/tune_heuristic_dev_cpu_v0/heuristic.json >/tmp/wp22_frozen_heuristic.json
python scripts/run_velocity_pilot.py plan --heuristic-rev d756c37d0564837a74132c30f8e366320a6bb1a5 --out results/velocity_pilot_plan_FREE_UNIQUE_PATH
```

The driver asserts the pinned nominal-s1 checkpoint hash `14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de`, CPU platform, clean tracked source, and frozen comparator provenance. A plan is **not a scientific run**. Do **not** run `run` directly as a second experiment; follow the original coordinated launcher.

## Scientific acceptance before claiming results

Fixed design: 4 families (`gain_070`, `gain_070_return`, `healthy`, `sensor_scale_080`) × 2 channels (`declared`, `declared_fast`) × 5 methods (`none`, `oracle`, `heuristic_gate`, `candidate`, `candidate_velocity`) × 8 paired episodes = **40 cells, 320 rows**, 300 control ticks = **96,000 scheduled transitions**.

On completion:
1. Archive immutable source, frozen heuristic **SHA-256**, checkpoint, seeds, telemetry models and attempt receipt before publishing.
2. Verify 320 actual `episodes.csv` rows, no duplicate scenario IDs within a cell, matching paired scenario IDs across methods, expected 40 cells, no unintended held-out families.
3. Run original `results/velocity_pilot_wp22_preflight/verify.py` against the completed result directory. It audits per-episode loss, falls, source release causality, age histograms, sampled hold coverage, withdrawal numerators/denominators, and paired statistics. Report verification failure rather than silently adjusting or rerunning.
4. Publish raw `episodes.csv`, `hold_*.json`, `trace_*.npz`, `summary.json`, `paired_vs_speed.json`, final report, provenance, and SHA256 digest ledger on `foresight/experiment-results`; verify Git reads match locally.
5. Ask Claude for an independent reproduce-and-critique review on PR #46, including uncertainty intervals and the dev-only limitation.

## Time-boxed course submission plan

- **Nov 26:** publish primary tuning CSV/JSON (this directory) and the recovery decision. No data invented.
- **Nov 27:** deadline for original-operator workspace recovery, lock verification and CPU execution decision. If inaccessible, obtain explicit claim release before creating a replacement CPU-only attempt. Prepare paper using *existing* evidence meanwhile.
- **Nov 28–29:** finish one permitted CPU scientific attempt, verify raw traces and commit it. If the attempt failed, preserve failed records and explain the gate honestly.
- **Nov 30:** independent critique and figure/table freeze. Do not backfill held-out results.
- **Dec 1–3:** finalize AA228/CS238 4–6 page paper, technical methods, results, limitations and reproducibility appendix.
- **Dec 4:** submit development-only study with accurate evidence and no unexecuted validation claims.

**Fallback if the 320-row run cannot be recovered:** report the development tuner and real eight-tick wiring evidence, plus the pre-existing G0 CPU/GPU smoke with clearly separate scope. State that the 40-cell closed-loop hypothesis was not evaluated. A scoped empirical/engineering report is more defensible than fabricated performance tables.
