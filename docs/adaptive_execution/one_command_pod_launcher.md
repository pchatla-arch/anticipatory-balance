# One-command fresh-pod launcher — current validated scope

**File:** `scripts/run_g1_pod_one_command.py`  
**Scientific target:** real Unitree G1 JAX/MuJoCo **development benchmark** with frozen heuristic, not the separately preregistered 72-episode obstacle mission.

## Paste into a newly provisioned pod's terminal

First confirm in RunPod that it is **one NVIDIA RTX 6000 Ada 48 GB, Community Cloud, host CUDA 13.2, <=$0.74/hour**. A Secure Cloud pod is *not authorized*, even if the GPU model is correct.

```bash
bash -lc 'set -euo pipefail
cd /workspace
if [ ! -d anticipatory-balance/.git ]; then
  git clone --single-branch --branch oaa/g1-obstacle-demo-prototype-20261126 https://github.com/pchatla-arch/anticipatory-balance.git
fi
cd anticipatory-balance
git fetch origin oaa/g1-obstacle-demo-prototype-20261126
git checkout --detach ad457cb6fe7dead2cfc6fa373248d070aee3f217
exec python3 scripts/run_g1_pod_one_command.py'
```

The Python script prompts **securely in the pod terminal** for `RUNPOD_POD_ID`, `RUNPOD_API_KEY` and `GH_TOKEN` if they are not already available as environment variables. The first token is required to query the actual Pod/cloud/rate and invoke RunPod's stop API. The GitHub token requires **Contents: Read and write** on `pchatla-arch/anticipatory-balance`, to commit and push raw result files. **Never paste tokens in ChatGPT or include them in Git commands/URLs.** Use a GitHub fine-grained token restricted to this repository. This script stores secrets temporarily in a mode-600 file on the running pod; delete the pod when no longer needed.

## What it does

1. Check the *actual allocated pod* (not the broad stock list), confirmed Community/RTX 6000 Ada/CUDA 13.2, hourly price <=$0.74, and no other running project pod. Reject and request stop on unauthorized/mismatched configuration.
2. Compute deadline from provider `startedAt`, **not** from script invocation. Budget watchdog allows at most 5h20 from pod start and reserves time below the six-hour/$4.44 maximum. A detached pod-side watchdog checks the deadline and calls the provider stop API. **Important:** the watchdog is inside the pod and is *not* a provider-independent hard stop if the container or network fails. A separate hourly ChatGPT watch is a second, non-guaranteed fallback; watch RunPod billing directly.
3. Confirm noninteractive GitHub write access via a dry-run push *before* scientific work.
4. Use `scripts/bootstrap_fresh_pod.sh` to install the exact frozen Playground/Menagerie/JAX/MuJoCo Python 3.12 GPU stack, resuming an interrupted owned setup. Verify JAX GPU operation.
5. Run offline supervisor guards, the G1 route/depth/language-interface tests and checkpoint SHA-256 verification.
6. Fetch Claude's immutable frozen heuristic at `d756c37d0564837a74132c30f8e366320a6bb1a5`.
7. Run the **existing implemented** JAX/G1 actuator-adaptation benchmark on the development split: 4 scenario families × 4 methods × 8 episodes × declared telemetry = **128 real episode records**, each 300 control ticks. Save raw `episodes.csv`, `summary.json`, `report.md`, traces, logs, manifest. Independently check 128 rows, paired scenario IDs, and recomputed per-cell means.
8. Publish logs and real results to `foresight/experiment-results:results/g1_course_dev_<RUN_ID>/` using Git push (no force push; transport-only retry). Verify resulting branch commit with `git ls-remote`.
9. Mark the separately registered 72-row obstacle/depth/video trial **NOT_EXECUTED** because the physical obstacle + depth-to-policy integrated evaluator is not yet implemented/verified on PR #51. Do **not** claim obstacle detection, VLA inference or MuJoCo video.
10. Call RunPod's own stop API, make several attempts to check EXITED, and leave an urgent message if provider acknowledgement cannot be observed. The provider may terminate the command before the final readback; independent console/account verification is still necessary.

**On success of the 128-row development benchmark**, GitHub publishes genuine new development-science results, but the script exits nonzero/partial because the integrated obstacle mission is not complete. On failure it also attempts to publish failure status/logs and stop. It never replays the separately protected WP2.2 PR #46 scientific attempt and never touches held-out splits.

## Experiment integrity, publication caveats

- **There is no working 72-episode G1 obstacle runner yet.** No honest one-command script can claim to execute those missing components. This supervisor therefore does *not* present a final course demo as complete.
- If the GitHub push fails and the pod has **no persistent volume**, local results can be lost when stopped. The script retries GitHub publishing but cannot guarantee recovery after provider/network failure. For critical recordings, attach persistent storage, or use an externally verified publisher before science. Storage charges are additional and require explicit budgeting.
- The RunPod key must authorize reading and stopping the current pod. A failed provider call means the hard stop is unconfirmed; manually stop via RunPod console if needed.
- No shell process can enforce a genuinely provider-independent spending ceiling across a pod crash. To guarantee spending independently of a running container, an external provider-side limit or separate reliable host-based controller is necessary.
- This runner has **not yet been validated end-to-end on an authorized Community RTX 6000 Ada**. The source and offline guard tests are committed, but actual GPU execution and GitHub receipt must be verified before calling it operationally proven.

## Remaining separate required work

Contact-enabled static cylinder and G1 head-camera renderer -> calibrated point clouds -> landmark/detour integration -> frozen PPO + bounded actuator control -> 72 preregistered episodes -> actual video -> independent trace audit. These are the uncompleted PR #51 P2–P6 stages. Keep them separate from the actuator-only 128-row development benchmark and the protected WP2.2 experiment.
