# G1 detached RunPod execution and recovery — November 26, 2026

**Status:** Reliability hardening committed. Offline regression tests are added, but the new detached launcher has **not yet been executed on a newly deployed pod**. Do not claim a complete 128-episode science run or G1 obstacle/VLA demonstration from these changes.

## Why this revision exists

The November 26 A6000 pod `9be02r81opb7nj` ran 32/128 development episode records, but the foreground worker then disappeared and left the GPU running idle. A manual inspection rescued all 32 rows plus NPZ traces to `foresight/experiment-results` (verified commit `fc8e2904333fb8d5f9f8904a69f20c678e9571f3`) and stopped the pod. There was no observed cgroup OOM kill. Root cause remains unknown; terminal disconnect/SIGHUP is a hypothesis.

## What has changed

1. **Detached scientific process:** `scripts/g1_detached_run.py` asks securely for RunPod pod ID and the RunPod/GitHub tokens once, verifies the authorized GPU and one project pod, then launches `scripts/run_g1_pod_one_command.py` in an independent OS session with `start_new_session=True`. The terminal returns immediately, leaving the experiment running.
2. **Detached rescue monitor:** Another independent in-pod process watches the worker PID and its Linux start-tick identity; it regards zombies and missing PIDs as failures and requests provider stop if the worker is unexpectedly gone. If the worker completes normally and the provider is already EXITED, no extra stop is issued.
3. **Incremental GitHub backups:** The monitor checks for new verified episode rows every 10 seconds. After at least 8 complete, unique records and a minimum 120-second interval, it backs up episode CSV, metadata, logs and stable NPZ traces (older than 15 seconds) to a unique immutable `results/g1_interim/<run-id>/<timestamp>` directory on `foresight/experiment-results`. It commits, pushes without force, and checks the remote SHA. These are *partial*, never final scientific results.
4. **Unexpected death recovery:** After up to 20 seconds for the worker's normal final publication, the monitor attempts a final partial snapshot, then requests RunPod provider shutdown even when GitHub publication fails. The exact registered attempt is not silently rerun.
5. **Authentication and launch hard gates:** Pod price/CUDA, source, checkpoint, real GitHub write access and previous ownership checks remain required. Credentials remain private files (0600), not command-line arguments. A short-lived signed-in interactive enrollment receipt lets the detached child avoid asking for the pod-list 403 confirmation without a terminal.
6. **Concurrence:** The GitHub askpass helper is now replaced atomically to prevent the interim and final publisher from reading a half-written helper.
7. **Regression suite:** `tests/test_g1_detached_run.py` contains nine offline tests for process detection, row counts/duplicate rejection, selecting the active run, provider-stop behavior, and no-token refusal. These are included in `g1_demo_preflight.sh`. **Tests have been committed but not yet executed in the full checkout on this revision.**

## One command for the NEXT approved, empty RunPod pod

Use only after checking the provider GPU cost and ensuring the remaining cumulative project budget permits a new session. The experimental pipeline currently runs the existing 128-episode JAX actuator-development benchmark, NOT the proposed 72-episode real-obstacle/video benchmark.

```bash
bash -lc '
set -euo pipefail
cd /workspace
if [ ! -d anticipatory-balance/.git ]; then
  git clone --branch oaa/g1-obstacle-demo-prototype-20261126 \
    https://github.com/pchatla-arch/anticipatory-balance.git
fi
cd anticipatory-balance
git fetch origin oaa/g1-obstacle-demo-prototype-20261126
git checkout --detach <PIN_THE_NEW_PR51_SHA>
bash scripts/g1_pod_group_cleanup.sh
exec python3 scripts/g1_detached_run.py
'
```

Replace the placeholder with the exact final commit recorded after this document is committed. Do not reuse `956e032`, `b243154` or any earlier launcher revision.

## How to inspect progress without stopping the worker

```bash
tail -f /workspace/g1_run_outputs/detached_session.log
tail -f /workspace/g1_run_outputs/detached_monitor.log
```

Read checkpoint evidence on the GitHub results branch. The publisher has no permission to alter the protected WP2.2 science attempt on PR #46.

## Important limits and costs

The deadline is computed from the provider's actual `startedAt`, not the local script start, and requests stop with a publication reserve. This is an **in-container monitor**, not a provider-enforced or separately hosted shutdown guarantee. If the whole container is killed, the network fails, the provider API denies stop, or the monitor is terminated, the automatic stop may fail. Confirm `EXITED` in RunPod and use the existing external monitoring task as a backup.

All previous stopped pods are part of the user's actual cumulative spending, which the launcher does not automatically reconcile against the earlier authorized $4.44 total. Before new billable work, review charges/remaining authorization.

**Publication note:** Interim backup commits are immutable snapshots, and the final publisher uses a separate `results/g1_course_dev_<runid>` path. They must not be interpreted as independent experimental repetitions or held-out evaluations.
