# Foresight: handoff for an agent running on the GPU pod

## Purpose and authority

Work with Prasuna Chatla on Foresight, a POMDP decision layer over frozen PPO
humanoid skills in MuJoCo Playground. The user authorizes inspecting this project,
making scoped fixes on a feature branch, running the bounded diagnostic below,
and pushing code and compact results to pchatla-arch/anticipatory-balance.
GPU time costs the user $2.50/hour. Prioritize useful execution and short reports.

This agent runs on the pod and has its own session. The ChatGPT conversation is
not available here; this file and the repository are the shared context.

## Immediate job

Establish the current pod state, preserve the current training, and complete the
stabilization switch diagnostic using the two existing trained skills. Do not
launch the full Session B sweep, BRACE v2 training, or a new 200M-step training
as part of this first job. If checkpoints are unavailable, report that specific
blocker and preserve the evidence.

1. Read project instructions and inspect the active processes, current Git
   branch/revision/status, installed rss2025 venv, GPU, mount layout, and actual
   checkpoint directories. Never print credentials or credential-bearing URLs.
2. If another training or evaluation job is active, do not start a competing GPU
   job or edit/switch its checkout. Record its stage and return a BUSY report.
   This first invocation must be tested before it is left unattended.
3. Use a separate Git worktree from origin/foresight/stabilization-v1 for diagnostic
   work. Keep the original checkout and its logs/checkpoints untouched.
   Use a new foresight/pod-agent-<UTC timestamp> branch for fixes.
4. Select one completed nominal checkpoint and one completed first-BRACE
   checkpoint from their real run directories. Numeric checkpoint directories,
   ppo_network_config.json, saved environment config, training completion, and
   actual policy restoration must agree. Do not treat config.json as weights or
   concatenate a checkpoints/0* glob into a single path. Record exact absolute
   paths, training source/config when available, and unresolved provenance.
5. Read docs/STABILIZATION_V1.md and docs/POD_SMOKE_TEST.md before execution.
   Reuse the installed environment. With explicit paths, invoke:

   FORESIGHT_VENV=/workspace/mujoco_playground/rss2025/.venv \
   FORESIGHT_NOMINAL=/exact/nominal/checkpoints/NUMERIC_STEP \
   FORESIGHT_BRACE=/exact/brace/checkpoints/NUMERIC_STEP \
   bash scripts/run_switch_gate_smoke.sh

   Replace the paths with observed paths. The runner executes 32 paired
   scenarios, six comparison arms, and 32 reference rollouts, and publishes
   compact results to foresight/pod-results using the pod's existing Git auth.
6. If a concrete harness bug appears, make the smallest evidence-based fix,
   run the relevant CPU checks, and allow at most one diagnostic retry. Stop on
   a second failure. Do not silently alter the physics, reward, scenario
   distribution, policy architecture, or acceptance criteria to obtain a pass.
7. Publish a concise report with source revision, exact checkpoint paths,
   run commands, outcome, failures, actual elapsed time, and next recommendation.
   Push source fixes only to the new feature branch, without force. For an
   upload-only failure, retry publication without rerunning the GPU experiment.

## Interpretation and known findings

- This is a diagnostic, not a performance certification. A switch_failure means
  BRACE caused a failure in a no-projectile pair where the zero-command nominal
  actor survived. Inconclusive is not proof that bracing helps.
- The original BRACE policy was trained near standing and performed poorly when
  switched from walking. The diagnostic separates skill switching, zero command,
  and the effect of a shared external projectile trajectory.
- Session B at 91affe1b575ad6ef2bff232ca8b43ae5b1a546a4 changes the training
  distribution and posture targets. Newly trained policies from that branch
  must be labelled as that condition, not exact recovery of the lost runs.
- In that Session B revision, registration creates both skills' randomizers
  from nominal default_config, so the requested heavier BRACE mass distribution
  is not actually used. Its runner also skips save_run when old committed
  results/<tag>/summary.json exists, leaving stale training summaries.
- The Session B trajectory-control API differs from the stabilization API;
  do not overlay its environment/evaluator files piecemeal.
- Shared train_skill.py historically catches any ZeroDivisionError and may
  report training success too broadly. Validate real checkpoints and logs.
  A headless guard was prepared separately but is not assumed installed here.
- Aiming at a predicted pelvis location does not guarantee equal contact across
  policies. Keep trajectories paired and report contact outcomes separately.

## Runtime and data rules

- Do not install, upgrade, downgrade, or resync the Python environment for this
  task. Do not force-push, reset local work, or overwrite old result directories.
- No duplicate GPU jobs, unlimited retries, or unrequested long sweeps.
- Keep secrets, private keys, auth caches, raw environment dumps, and credentials
  out of logs and Git. Authentication is done by the user on the pod.
- GitHub compact results do not contain policy weights. Verify persistent
  storage or a recoverable off-pod backup of checkpoints before any pod stop.
  The last supplied mount output showed /workspace on container overlay.
- Do not stop, terminate, restart, or reconfigure the pod during this first job.
  A finished script or process timeout does not stop RunPod billing.
- Report BLOCKED or BUSY honestly when access, active jobs, storage, or checkpoint
  provenance prevents execution. Do not claim remote execution from ChatGPT.

## References

- Stabilization branch:
  https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/stabilization-v1
- OpenAI Codex CLI:
  https://learn.chatgpt.com/docs/codex/cli
- Remote login:
  https://learn.chatgpt.com/docs/auth
- Non-interactive jobs:
  https://learn.chatgpt.com/docs/non-interactive-mode
- RunPod storage:
  https://docs.runpod.io/pods/storage/types
