# G1 GPU next experiments — staged validation and release plan
**Prepared:** November 26, 2026  
**Current scope:** PR #51, frozen G1 PPO, online actuator adaptation. No new GPU run launched by this document.

## Live hardware choices (verify at launch)

Preferred: one Secure RTX A6000 (48 GB VRAM) at <= **$0.69/hour**, CUDA host **13.2**. Alternate: Secure RTX 6000 Ada at <= $1.09/hour or Community RTX 6000 Ada at <= $0.74/hour, if actually available. All must have exactly one GPU. Never accept unsupported cloud/price or duplicate project pod.

**Authorization/accounting:** the originally approved $4.44 compute allocation is cumulative across all related runs, not a renewable per-pod allowance. The current Python launcher computes a cap against its own provider start time and **does not** deduct prior charges. Before new billable allocation, verify the actual account billing and remaining user-approved spending. Network/storage charges may differ. A pod-side watchdog and hourly external monitor are **not** a guaranteed independent provider cap. Confirm provider EXITED.

## S0: GPU physics smoke (the ONLY enabled paid experiment in this revision)

The one-command `scripts/g1_detached_run.py` is a short interactive enrollment that creates a true detached worker, worker-death monitor, GitHub interim snapshots, and provider stop requests. The child `run_g1_pod_one_command.py` automatically chooses `G1_BENCH_PROFILE=smoke` (the ONLY accepted profile).

**Fixed S0 development design:**
- Frozen PPO `checkpoints/nominal_s1/000202342400` with tree SHA-256 `14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de`.
- Baseline heuristic frozen at `d756c37d0564837a74132c30f8e366320a6bb1a5`.
- Scenario families `gain_070`, `healthy`; 3 methods `none`, `oracle`, `heuristic_gate`; two paired scenarios each; **12 episode rows total**.
- JAX physics, actual GPU device, `declared` telemetry; 160 control ticks at 50 Hz (3.2 simulated seconds). Gain-onset window 75–125 ticks, leaving an observed post-onset interval; this is a *smoke*, not full locomotion/recovery evidence.
- Same seed IDs and packet draws per family across methods, compared with an independent auditor that refuses missing/duplicate cells and summary disagreement.
- Check frozen checkpoint hash, source SHA, dependency versions, synthetic depth/language tests, supervisor safety guards and write-permission preflight **before** paid physics.
- Save raw episode CSV, per-method NPZ traces, config, logs and result audit. GitHub interim checkpointing at completed cell boundaries and final verified publication under `foresight/experiment-results`.
- Stop the pod after final publication or any failure, with a separate monitor watching worker death. Maximum **one provider-hour** from actual pod `startedAt`, including setup; 20 minutes reserved for publication/stop. Do **not** treat a successful smoke as real head-camera obstacle avoidance or a trained VLA.

**S0 stop/go:** pass only if all 12 rows exist, their family-method pairing and raw mean losses match the summary, published GitHub commit SHA matches remote, and provider is verified EXITED. If compilation/test/checkpoint/auth fails, publish the failure receipt where possible, stop and diagnose off-pod. Do not silently rerun scientific episodes.

## S1: Longer paired development benchmark — NOT ENABLED

After S0 is independently audited, consider a separately approved development registration with healthy, gain-0.70, return-to-healthy and sensor-scale confound; correction-off, oracle, estimated inverse, frozen heuristic; 8 paired scenarios and 300 control ticks per cell = 128 rows. This was attempted before, with only 32 preserved rows. Do **not** treat those 32 as new S0 observations. The full runner is intentionally disabled on the S0 revision to prevent an accidental expensive rerun.

S1 scientific requirement: original seeds and family definitions, matched paired methods and hidden gain labels, releases/withdrawal denominators, simulated contacts/falls and healthy false-intervention. Inspect S0 actual wall time and residual budget *before* scheduling.

## S2: Navigation + perception real physics — NOT ENABLED

Physical contact-enabled stationary cylinder, rendered calibrated G1 head depth, obstacle estimate (no true geom leakage to planner), conservative detour, and nominal frozen PPO walking. Validate a deliberate collision-control case and healthy obstacle bypass, measuring camera latency, false alarms, clearance and contacts. **The current repo includes unit-tested depth/route primitives but not a validated MuJoCo integrated runner.** Do not present synthetic unit tests as a real obstacle demonstration or fabricate a video.

## S3: Language-conditioned integrated demo / later learned VLA — NOT ENABLED

Only after S2, pair healthy vs hidden gain fault/return and correction-off vs adaptation with identical navigation stack. A bounded language-goal parser exists but it is not a trained VLA. The eventual 72-episode course demo (2 scenes × 2 gain schedules × 3 methods × 6 paired seeds) requires its independent physics runner, real H.264 footage, raw trace auditor and preregistered scene controls.

## Approved S0 invocation

On a freshly deployed pod with the approved GPU/cloud/rate, clone PR #51 and **checkout its current verified source commit**, then run `python3 scripts/g1_detached_run.py`. It prompts for RunPod ID, API key and GitHub token privately; never paste tokens into shell command text or chat. If pod-wide listing is denied by API-key scope, explicitly confirm in RunPod that no other G1 project pod exists.

After enrollment, follow `/workspace/g1_run_outputs/detached_session.log` and `detached_monitor.log`; inspect GitHub results branch for snapshots; verify provider `EXITED`. If full scientific success occurs, do not immediately launch S1 without inspecting logs and cumulative actual spend.

## Evidence

- Previously rescued 32/128 real dev records: result commit `fc8e2904333fb8d5f9f8904a69f20c678e9571f3`.
- CPU/OS tests: 81 cases passed at https://github.com/pchatla-arch/anticipatory-balance/actions/runs/45926606584; smoke-registry/timing tests added separately and require their new CI run.
- No real G1 obstacle/video/VLA evidence as of this registration.
