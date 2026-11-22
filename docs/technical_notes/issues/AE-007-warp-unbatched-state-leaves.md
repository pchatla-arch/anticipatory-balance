# AE-007 Stage A gate crashed on GPU: MuJoCo-Warp state has unbatched leaves

| Field | Value |
| --- | --- |
| Status | Resolved (fix in PR #23, pending merge) |
| Severity | Blocking |
| Project · area | Adaptive execution · evaluator |
| Found | 2026-11-16, Prasuna's first two pod runs; diagnosed by Claude from console.log |
| Issue | #22 |
| Owner | Claude fixes · ChatGPT reviews · Prasuna merges |

## Symptom
Both Stage A gate runs on 2026-11-16 (`adaptive_gate_20261116T064015Z_4fae65`, source 1614b65;
`adaptive_gate_20261116T070551Z_220613`, source f597023) stopped 3–5 minutes after launch, before
any episode, with

    ValueError: Incompatible shapes for broadcasting: shapes=[(32,), (0,), (0,)]

raised inside `choose_rows` (scripts/eval_adaptive_gate.py, formerly line 93). Both runs uploaded
their logs to `foresight/experiment-results`; no scientific result exists.

## Evidence
- Verified. The MuJoCo-Warp backend's `Data` carries 23 leaves under `data._impl` that are not
  batched along the environment axis: a shared contact pool of 131072 rows, `(1,)` counters and
  zero-size buffers. The JAX backend's state has none. Tracing the real rollout with `impl='warp'`
  on CPU (`jax.eval_shape`) reproduces the pod error exactly at batch 8 and 32.
- Reproduce (fails on the original code, passes on the fix):
  `FORESIGHT_RUN_CPU_PHYSICS=1 pytest tests/test_adaptive_warp_rows.py`.

## Why it did not work
`choose_rows` applied a per-environment `jp.where` to every leaf of the state tree; a `(0,)` leaf
cannot broadcast against a `(32,)` mask. The pre-pod CPU check ran the smoke profile with
`--impl jax`, because the evaluator refused Warp on CPU, so the GPU backend's state structure
was never exercised. The tests also used the JAX backend.

## Impact
Two short GPU attempts lost (about 5 minutes of GPU time each); Stage A had not run. No claim
or result is affected because none existed.

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| Select rows only on leaves batched on the environment axis | Minimal; unbatched leaves are shared by construction | Must guard against an unbatched leaf whose size equals the batch size |
| Convert Warp state to the JAX layout before masking | Backend-agnostic | Expensive, and MJX does not expose it |
| Drop the row selection and let finished lanes keep simulating | Simplest | Reintroduces the NaN-propagation hazard the selection prevents |

## Decision and next steps
`select_batched_rows` selects row-wise only on leaves whose leading dimension equals the batch
size and takes unbatched leaves from the new state; a shape change in an unbatched leaf raises.
None of the unbatched leading dimensions (0, 1, 131072) equals a batch size in use (8, 32, 128).
Process change: the pre-pod CPU checklist now exercises the Warp backend (trace test above, and
`--impl warp --allow-cpu` execution, see AE-008), and every pod campaign starts with `smoke`.

## Resolution
Fix in PR #23 (commit 0c2a9d9). Pod smoke `adaptive_gate_20261122T053041Z_391fd6` (source
8440b50, run branch `claude/gate-run`) completed all eight arms on the GPU with the fix:
operational success, 0 invalid episodes. See AE-008 for what that run revealed next.
