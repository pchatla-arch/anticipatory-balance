# S1a — RunPod launch and audit checklist

**Project queue:** G1-S1A-20261126  
**Status:** CPU validated; NOT LAUNCHED. No RunPod pod has been created by this registration.  
**Prerequisite:** S0 CUDA G1 smoke 12/12 completed at results commit 9a9ed5ea3b3626d709c05d622f0e7e20ec85f876.

## Hardware and authorization

Check the currently deployed pod is precisely one Secure RTX A6000, CUDA host 13.2, and quoted <=$0.69/hour, with no other G1 study pod active. Stage limit: 90 minutes from RunPod provider startedAt; the launcher stops new work with 20 minutes remaining for GitHub publishing and shutdown. Explicit S1a attempt compute ceiling $1.05, in addition to the older project-wide spending cap $4.44 established for this period; review all old and pending charges. An in-pod watchdog is best-effort, not a provider-enforced payment cap. Verify EXITED.

## Frozen science protocol (32 paired development rows)

Four families: healthy, gain_070, gain_070_return and sensor_scale_080. Four methods: none, oracle, estimated_inverse and heuristic_gate. Two paired seeds each (4x4x2 = 32), 400 control steps at 50 Hz per episode. Full walk-turn-stop-resume schedule. Frozen Unitree G1 PPO checkpoint and baseline heuristic. Hidden injected gain and return labels are not accessible to practical methods; oracle is explicitly privileged. Raw traces and summary preserve tracking loss, distance progress, harmful offsets in healthy phases, withdrawal after return, control bound violations and falls.

The GitHub-registration file `docs/adaptive_execution/experiment_queue_G1_S1A_20261126.json` must match the launcher profile. No protected WP2.2 test science or registered 72-episode obstacle/navigation study is run here.

## Run from the next approved fresh pod terminal

Use the latest CI-approved PR #51 source SHA when running. On each pod only one attempt may be active. Set the variables that explicitly opt in to S1a (rather than defaulting to a 12-row smoke). Never paste credential values into the command, terminal transcript or ChatGPT; the launcher prompts securely.

```bash
bash -lc '
set -euo pipefail
cd /workspace
if [ ! -d anticipatory-balance/.git ]; then
  git clone --branch oaa/g1-obstacle-demo-prototype-20261126 https://github.com/pchatla-arch/anticipatory-balance.git
fi
cd anticipatory-balance
git fetch origin oaa/g1-obstacle-demo-prototype-20261126
# Replace PINNED_SHA with the latest verified PR #51 head after CI passes.
git checkout --detach PINNED_SHA
bash scripts/g1_pod_group_cleanup.sh
exec env G1_BENCH_PROFILE=s1_screen G1_QUEUE_JOB_ID=G1-S1A-20261126 G1_PROJECT_BILLING_REVIEWED=YES G1_AUTHORIZED_RUN_BUDGET_USD=1.05 python3 scripts/g1_detached_run.py
'
```

Only set G1_PROJECT_BILLING_REVIEWED=YES after checking cumulative project charges. A wrongly identified pod, missing/changed queue manifest, rejected billing acknowledgment, wrong hardware/price, missing GitHub write access, bad checkpoint, failed CPU tests or invalid paired audit must fail closed and request provider stop.

## Evidence, monitoring, go/no-go

The startup returns control after creating detached scientific worker and separate PID/deadline monitor; it is not a completion message. View `tail -40 /workspace/g1_run_outputs/detached_session.log` and `tail -40 /workspace/g1_run_outputs/detached_monitor.log` in the pod. Interim snapshots are stored on GitHub `foresight/experiment-results/results/g1_interim/`; final run under `results/g1_course_dev_<run-id>/` with a remote SHA receipt and raw 32-row audit. If the worker dies, the monitor attempts rescue and stops the pod. Inspect the provider record to confirm EXITED.

On successful completion, compare paired tracking loss *and progress*, not loss alone. Inspect return-to-healthy withdrawal denominators, correction application, healthy nuisance harm, falls and control clipping. If negative or ambiguous, publish raw evidence and do not claim successful adaptation or obstacle avoidance. No S2 real G1 depth/collision integration, video, or trained VLA exists in this runner.
