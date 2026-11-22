# AE-008 Identity control fails on GPU: MuJoCo-Warp is not run-to-run reproducible by default

| Field | Value |
| --- | --- |
| Status | In progress (fix in PR #23; pod smoke needed to confirm) |
| Severity | Blocking for the scientific gate |
| Project · area | Adaptive execution · evaluator, protocol |
| Found | 2026-11-22, pod smoke `adaptive_gate_20261122T053041Z_391fd6`; diagnosed by Claude |
| Issue | #27 |
| Owner | Claude fixes · ChatGPT reviews (protocol owner) · Prasuna merges |

## Symptom
The first smoke run that executed on the GPU completed all eight arms (exit 0, 0 invalid
episodes, 0 bound violations), but the screen analysis reports `status: fail` with the reason
"Healthy-model adapter did not reproduce the healthy control", `identity_max_difference =
0.0648`. In the gate profile this verdict would have failed the scientific gate regardless of
the fault results.

Per episode, the identity arm (gain 1, correction enabled) differs from the healthy arm by
1e-4 to 4e-4 in post-fault loss, up to 6 mm in progress over 3 m, and up to 0.065 N·m in peak
actuator force. Mean loss 0.0126 in both; progress 3.035 vs 3.034 m.

## Evidence
- Verified: with gain exactly 1 the correction is exactly zero. `desired = (1/g − 1)(r − q)` is
  `0.0 × finite = ±0.0`, `jp.any(delta != 0)` is False, so `executed_action == action` and the
  physics receives bitwise identical inputs. The two arms run the same compiled executable
  (the second call took 2.5 s, no recompilation) with the oracle flag as a traced input.
- Verified: on CPU with the JAX backend the two arms are bitwise identical (smoke runs of
  2026-11-15 and 2026-11-16, `identity` check passed with difference 0).
- Verified: MuJoCo-Warp accumulates with `wp.atomic_add` in `constraint.py` (40 sites),
  `smooth.py` (31), `passive.py` (30), `solver.py` (17) and others. Warp's documentation names
  floating-point atomics on CUDA as the main source of non-determinism, and its default mode is
  `NOT_GUARANTEED`. The pod runs Warp 1.18.0; MuJoCo-Warp's own `opt.deterministic` mask
  (`CONTACTS | CONSTRAINT | ATOMICS`) defaults to 0 (off).
- Verified on CPU with the Warp backend (`scripts/check_warp_reproducibility.py`, 8 walkers,
  160 control steps, real policy, commit of PR #23): with the default (non-deterministic)
  mask, healthy, identity and healthy_repeat are bitwise identical in every recorded metric.
  The Warp backend executed sequentially therefore reproduces, and the identity adapter runs
  the same code path; the GPU discrepancy is execution-order noise. Results in
  `results/warp_cpu_check/report.json` and issue #27 (the deterministic-mask run is reported
  there too).
- Hypothesis until the next pod smoke: with `opt.deterministic = ALL` and
  `--xla_gpu_deterministic_ops=true`, healthy, identity and healthy_repeat agree bitwise on the GPU.

## Why it did not work
The gate's identity control was specified as bitwise equality (tolerance 1e-6) and validated
only on a CPU backend, where execution is sequential and reproducible. On CUDA the order in
which threads apply floating-point atomic additions varies between launches; addition is not
associative, so the last bit varies, and a walking controller amplifies last-bit differences
over 400 steps into millimetres and tenths of a newton-metre. The control therefore measures
backend scheduling noise, not the adapter.

## Impact
Blocks the gate profile: `identity_ok` false makes every candidate fail. Does not affect the
fault results themselves (the 0.55 fault floors the walker in 8/8 episodes, the oracle
recovers 8/8), but it would have hidden them behind a failed gate. Also affects every paired
comparison's noise floor on the GPU, which the protocol had assumed to be zero.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Enable MuJoCo-Warp `opt.deterministic = ALL` and XLA deterministic ops | Restores the bitwise identity control as designed; no protocol change | Slower kernels (unmeasured on GPU); needs a pod smoke to confirm |
| Replace the bitwise identity check with a tolerance calibrated on a healthy repeat | Backend-agnostic | Weakens the control from exact to statistical; protocol decision for ChatGPT |
| Compare the arms statistically only (paired CIs) | Already part of confirmation | Leaves the identity control undefined |

## Decision and next steps
Both of the first two, kept separate: (1) the evaluator requests deterministic execution for
every pod run (`--deterministic auto`), applying `opt._impl.deterministic = 7` to the Warp
model and `--xla_gpu_deterministic_ops=true` before JAX starts; (2) a `healthy_repeat` control
arm re-executes the healthy arm with identical inputs, and the analysis reports
`run_to_run_reproducible` and `repeat_max_difference`. The identity criterion is unchanged.
If the repeat still differs on the GPU, the reason text says so explicitly, and the tolerance
question goes to ChatGPT as the protocol owner. Acceptance test: pod smoke on the fixed branch
shows `identity: true` and `run_to_run_reproducible: true` in `summary.json`.

## Resolution
(Open. Fill in after the next pod smoke.)
