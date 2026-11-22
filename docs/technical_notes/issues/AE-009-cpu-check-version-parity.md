# AE-009 CPU checks ran on MuJoCo 4.0.1 while the pod pins 3.16.0

| Field | Value |
| --- | --- |
| Status | In progress (guard and mirror environment in place; pod rerun pending) |
| Severity | Blocking (one pod run lost; invalidates the "CPU-verified" label of earlier checks) |
| Project · area | Adaptive execution · process, launcher, evaluator |
| Found | 2026-11-22 06:13 UTC, Prasuna's smoke `adaptive_gate_20261122T061338Z_d901f2`; cause identified by ChatGPT (PR #29) and Claude |
| Issue | #32 |
| Owner | Claude (process failure and fix) · ChatGPT reviews · Prasuna merges |

## Symptom
The smoke run on `claude/gate-run` (b215b9c) failed 62 s after launch, before any episode:

    ImportError: cannot import name 'DeterminismType' from
    'mujoco.mjx.third_party.mujoco_warp._src.types'

The AE-008 fix had been reported as "CPU-tested on the same JAX 0.12, MuJoCo 4.0.1 and brax
0.14.2 as the pod". The pod does not run MuJoCo 4.0.1.

## Evidence
- Verified. The pod installs Playground `rss2025` with `uv sync --frozen`; its `uv.lock` pins
  `mujoco==3.16.0`, `mujoco-mjx==3.16.0`, `warp-lang==1.18.0` (console log of
  `adaptive_gate_20261122T053041Z_391fd6` lists every installed wheel; `preflight.json` of
  that run: mujoco 3.16.0, mujoco-mjx 3.16.0, warp-lang 1.18.0, orbax 0.12.8).
- Verified. Claude's sandbox interpreter (`/home/claude/pgvenv`) has mujoco 4.0.1,
  mujoco-mjx 4.0.1, warp-lang 1.19.0, orbax 0.12.9. The "pod versions" quoted on 2026-11-21
  were read from that interpreter's own CPU-smoke `preflight.json`, not from a pod record.
- Verified. In the 3.16.0 vendored MuJoCo-Warp there is no `DeterminismType` and no
  `opt.deterministic` model option (the only "deterministic" in its sources is a docstring
  in `island.py`); it has 178 `wp.atomic_add` sites. The option used by AE-008 exists in
  4.0.1.
- Verified. A venv built from the pod's lock (`uv sync --frozen --extra test`, CPU) matches
  all 261 pod-installed wheels that it contains, 0 version mismatches
  (`scripts/check_env_parity.py`). On it, the AE-008 unit test reproduces the pod's
  ImportError, and PR #29's capability-detecting code passes 95 tests plus the physics
  tests (Warp trace at batch 8/32; full-physics env test, 81 s).

## Why it did not work
Two interpreters were treated as one. Every CPU check since Session D used `pgvenv`, which
had drifted to a newer MuJoCo than the pod's lock, and no step compared installed versions
against the lock. The AE-007 fix happened to be version-independent, which hid the drift;
the AE-008 fix used an API that is version-dependent, and the drift surfaced. The claim of
version parity was made from the wrong file.

## Impact
One pod attempt (about 1 minute of GPU time). More importantly, the "CPU first" rule was not
actually satisfied for any pod entry point until now: the checks ran, but on different
software. No scientific result is affected (none exists yet). The AE-008 determinism fix is
not available on the pod's stack; the identity-control question is reopened (options below).

## Options considered
| Option | Pros | Cons |
| --- | --- | --- |
| A. Bump the pod to mujoco/mjx 4.0.1 + warp 1.19.0 and use MuJoCo-Warp's own `opt.deterministic = ALL` | Bitwise identity control as designed; the determinism knobs were built by the physics authors with per-kernel record bounds | Physics version change: re-baseline Stage A (cheap now, nothing scientific exists); a bootstrap/lock change on the pod; untested GPU behaviour of 4.0.1 + 1.19.0 there |
| B. Keep 3.16.0; replace the bitwise identity criterion with a tolerance calibrated on the `healthy_repeat` probe | No environment change; honest about a GPU noise floor that paired statistics already absorb | Protocol change (ChatGPT owns it); the control becomes statistical, not exact |
| C. Keep 3.16.0; set Warp's global `wp.config.deterministic = RUN_TO_RUN` before import | Compiles and runs on CPU with the 3.16.0 kernels; 34 of 40 recently compiled modules received the deterministic rewrite, the rest hold integer counters only | Record capacity is "the code-generated lower bound" unless raised; data-dependent loops in MuJoCo-Warp can overflow it, and inside CUDA graph capture an overflow truncates contributions silently. Needs a GPU validation against default physics before any use. A detour for the Dec 4 schedule |

## Decision and next steps
- Process (done): `scripts/check_env_parity.py` compares the interpreter's versions with the
  pod's `uv.lock` and exits nonzero on any difference; it is the first line of every CPU
  checklist and its JSON is stored with the evidence. All CPU checks now run in the
  pod-mirror venv (built with `uv sync --frozen --extra test` from the same lock).
- Pod code (ChatGPT, PR #29): capability detection; auto-smoke falls back to a labelled
  default-backend diagnostic run; auto-gate refuses to run nondeterministic physics
  silently; `on` fails before any rollout.
- Identity control: Claude recommends **B** for the course schedule, with **A** revisited at
  G4 if a bitwise control is wanted for the release. ChatGPT proposes the criterion; Prasuna
  decides. A smoke on PR #29's code supplies the noise floor either way.
- Acceptance for this note: a smoke run on the pod whose `preflight.json` versions match the
  mirror venv, with the `healthy_repeat` arm reported.

## Resolution
(Open.)
