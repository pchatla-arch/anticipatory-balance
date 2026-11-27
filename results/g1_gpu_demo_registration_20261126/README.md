# Frozen G1 experiment queue — 2026-11-26

The **72-episode development experiment has been registered, but has NOT run**. [Registration](registered_experiment_queue.json) contains every scenario (2 layouts × 2 fault schedules × 3 methods × 6 paired seeds, 1200 control ticks max). The registration fixes comparison groups, input-leakage controls, output schema, budget, and the order of experiment gates. Registration source: `f360a7c61e6e6c5763882a0cc95a5b4029a83251`. Later commits to the PR add orchestration/preflight only; physics must record its actual source revision separately. Frozen baseline: `d756c37d0564837a74132c30f8e366320a6bb1a5`. Walker checkpoint-tree hash: `14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de`.

## What can be executed immediately

The branch includes `scripts/g1_demo_preflight.sh`. Run it from a pinned checkout with the right MuJoCo/JAX environment and checkpoint mounted (set `G1_CHECKPOINT_DIR`). This tests the pure-CPU depth geometry, language task stack and route logic, checks a clean Git source tree, prints dependency versions and verifies checkpoint contents. **It will NOT run MuJoCo physics or launch a GPU.**

A separately provided CPU-only registration/audit package contains `g1_demo_registry.py` and nine passing local regression tests. It creates a deterministic manifest and checks paired CSV episode data. The manifest's exact published index is in `registered_experiment_queue.json`; the package's Python JSON may have different whitespace and metadata representation. Regenerate and pin the exact execution manifest before starting physics.

## Execution order, hard gates and honest labels

| Stage | Required output | Failure/stop policy |
|---|---|---|
| P0 Authorization, GPU+watchdog | One Community RTX 6000 Ada CUDA 13.2 <=$0.74/hr; no competing pod; verified hard stop and publisher | No allocation if unavailable/unsafe |
| P1 Python/checkpoint | Passing repository tests, expected nominal-s1 checkpoint checksum and compiled environment | No physics launch |
| P2 Depth/contact scene | Physically contact-enabled cylinder plus rendered head depth, calibration, detector counts, oracle-only contact evaluation | Preserve failure; no claims of functional perception |
| P3 Healthy navigation | Actual G1 trajectory, root pose, collisions, falls, video frames and route clearance | Don't proceed to faulted full matrix if failed |
| P4 Registered adaptation trials | 72 scenario records: 24 independent paired scenarios × 3 arms, maximum 1200 steps | No seed selection, no silent rerun, retain partial count |
| P5 Raw-data audit | Recomputed paired metrics, timestamps, collision/fall/withdrawal denominators, confidence intervals | Publish discrepancy, do not silently fix |
| P6 Delivery + stop | GitHub verified blobs/MP4 and RunPod stop receipt with status EXITED | Stop before cost cap even if publication fails |

**Not yet ready:** P2 physical contact-enabled scene and actual depth-to-policy integration, PR #51 real G1 evaluator and MP4 publisher, hard provider-independent automatic shutdown. Those are implementation tasks, not working experiments. Merely provisioning a GPU cannot complete them. No claim of end-to-end readiness should be made until these gates are demonstrated.

## Scope and integrity

Do NOT consume held-out data or replay protected PR #46 WP2.2 attempt. Do not treat a deterministic text goal parser as an executed VLA. The primary research is actuator uncertainty. The obstacle navigation demo uses simulated robot localization, which must be disclosed. All 72 results must include failures and raw simulator evidence, not just a curated video.

**Authorized limit:** one Community RTX 6000 Ada 48GB GPU, at most 6 compute hours and $4.44 total at <=$0.74/hr; include all compute time in ledger. Reserve a minimum 30 minutes for publication/stop, preferably stop earlier with a safety margin. Provider stop and EXITED readback are mandatory. No service-side billing guarantee is implied by an hourly ChatGPT automation alone.
