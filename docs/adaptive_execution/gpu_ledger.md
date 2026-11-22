# GPU ledger and budget controls · OAA 0.3

Budget (execution plan rev. 2, PR #30): **12 GPU-hours hard cap** for the whole project, of which
**8 h are allocated to the core path** (S1–S3) and **4 h are optional** (robust PPO, extra
walkers, videos) and need Prasuna's explicit go per session. Pod: RunPod RTX 6000 Ada, driver
595.112.07, pinned stack (mujoco/mjx 3.16.0, warp 1.18.0, jax 0.12.0.1). Cost is billed
per pod-minute including setup, so the ledger counts **pod wall-clock**, not kernel time.

Operator from Nov 22 08:56 PT: ChatGPT (authorized by Prasuna). Reader of every result before it
is used: Claude. Spend decisions: Prasuna.

## Rules for every launch

1. One manifest comment on the status issue (#31) **before** the pod starts: source ref, launcher
   profile, `ADAPTIVE_MAX_MINUTES`, what the run produces, and the ledger row it will fill.
2. `FORESIGHT_AUTO_STOP=1` and `ADAPTIVE_MAX_MINUTES` set on every launch; the launcher's
   ceiling is the budget for that attempt. **One attempt per manifest.** A failed attempt is not
   retried on the same pod start without a new manifest and Prasuna's reply.
3. Results are published before the stop; the stop is confirmed by the provider's acknowledgement
   (`runpodctl stop pod`), and the ledger row below is filled in the same comment: pod start and
   stop UTC from `run.json`/`publication.json`, and the minutes billed.
4. A session that reaches 80 % of its ceiling without finishing publishes partial results marked
   incomplete and stops. No tolerance, freeze or claim is derived from a partial run.
5. Prasuna checks the RunPod billing page after each session; the ledger total must agree with it
   within one setup cycle (≈ 5 min).

## Sessions

| Session | Date (PT) | Profile / content | Ceiling | Pod minutes | Cumulative h | Status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| S0 | Nov 22 00:21 | `smoke` at fdf5754 — 72 episodes, 9 arms (G0 diagnostic) | 30 min | ≈ 5 (run 200.9 s + setup) | 0.1 | done; G0 inconclusive, AE-008 noise quantified |
| S1a | Nov 23–24 | `calibrate-identity` (PR #43): 64 batches × 32 lanes × 2 repeats | 45 min | — | — | awaiting Prasuna's G0 = Option B |
| S1b | Nov 24 | `bench` dev: 11 families × exact/declared/declared_fast × all methods, 32 eps | 90 min | — | — | needs launcher `bench` mode; same pod start as S1a |
| S2 | Nov 26 | `bench` val with frozen heuristic and candidate | 90 min | — | — | after G2 freeze |
| S3 | Nov 28 | `bench` test (registered, `--registration` from an immutable ref) | 90 min | — | — | one attempt; registration committed first |
| S4 (optional) | Nov 29–30 | robust PPO training + eval (WP 3.2) | 180 min | — | — | only if S1–S3 clean; Prasuna's go |
| S5 (optional) | Dec 1 | videos, extra walkers, any rerun | 60 min | — | — | Prasuna's go |

Core path total if every ceiling is used: 5 min + 45 + 90 + 90 + 90 = **5.3 h** (allocation 8 h,
leaving 2.7 h of slack for one rerun). Optional: 4 h. Hard cap 12 h.

## Timing basis for the ceilings

S0: eight-lane arms took ≈ 1.8 s warmed; Warp kernel compilation ≈ 35 s at first use; whole run
200.9 s from launcher start including evaluation. A 32-lane arm is unmeasured on this GPU; the S1b
ceiling assumes ≤ 6 s per arm over ≈ 200 arms plus one compile per method/telemetry variant, i.e.
≈ 40 min of compute, with the rest as setup, publication and margin. S1b's actual time sets the
S2/S3 ceilings; revise this file after S1.
