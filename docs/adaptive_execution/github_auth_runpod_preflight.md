# G1 RunPod — GitHub authorization gate (2026-11-26)

Status: code committed and GitHub results branch exists. Actual user-token publication is untested until the next permitted pod. No GPU science is claimed.

## Important separation
A ChatGPT GitHub connection does not automatically authorize Git commands inside a fresh RunPod instance. A GitHub personal access token or equally authorized pod-local credential is needed for unattended result publication.

## Verified access before installing JAX
1. Ask for GH_TOKEN in a hidden terminal prompt when no token is supplied by the pod's private environment. Never share credentials in this chat, shell history, or the GitHub repository.
2. Store the token in a mode-0600 file under the pod's ephemeral workspace; use an isolated Git askpass environment, overriding old credential helpers.
3. Check the exact repository origin and existing foresight/experiment-results branch.
4. Make a REAL, minimal, unique Git commit under results/g1_auth_probes/ and push it to foresight/experiment-results before any JAX install or scientific physics.
5. Read back the remote SHA. Failure to push or verify blocks experiment execution and triggers a pod stop request.
6. After each finished run, publish raw result files to GitHub and verify the remote SHA again. Never rerun physics to repair a failed upload.

## Required user action
Create a fine-grained GitHub personal access token scoped to pchatla-arch/anticipatory-balance with Contents: Read and write. Ensure branch rules permit writes to foresight/experiment-results. Supply it only through the private RunPod terminal prompt or RunPod-managed secrets. Do not put it directly in a bash command or chat message.

The existing ChatGPT-connected GitHub app is independent from the terminal's GitHub credential. A token scoped to a different repository or lacking Contents write permission will be rejected immediately.

## Limits
Current launcher executes the real 128-row G1 development benchmark, not the registered 72-row obstacle navigation demonstration or video. Pod-side deadline watchdog and hourly external check are best-effort; verify provider EXITED. The authorized GPU compute maximum is $4.44 for the run, with separate storage/network charges possible. Prior pod spending is not automatically reconciled.
