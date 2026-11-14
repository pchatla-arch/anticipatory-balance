# Foresight: one-command pod workflow

## Run the prepared experiment

On the pod, from any directory:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/experiments-v2/scripts/start.sh) f
```

`f`/`f0` runs **F0**, a 64-scenario paired release diagnostic using the committed
nominal and BRACE v2 checkpoints. It compares reactive walking, a privileged
fixed-lead reference, and three release contracts for the belief gate. Each
condition gets the same external scenarios, with 50 Hz synthetic detections and
zero configured sensor delay. There is no policy retraining. The default
experiment-process budget is 20 minutes after setup; this is a limit, not a
prediction of runtime or a full-session billing cap.

For 32 scenarios, replace `f` with `smoke`. This is a smaller integration check,
not a substitute for statistically powered evaluation. The proposed Session F
lookahead/depth studies are described in [SESSION_F_PLAN.md](SESSION_F_PLAN.md)
and are not yet included in this command.

## What happens

1. The launcher creates or updates `/workspace/foresight-runner`. It reuses
   `/workspace/mujoco_playground/rss2025/.venv` if that environment exists and
   passes preflight. It preserves the separate `/workspace/anticipatory-balance`
   checkout and any running tmux session. A busy GPU prevents a second launch.
2. On an empty pod it installs missing system tools and the source-pinned
   Playground/Python/CUDA environment. An existing unknown venv is not upgraded.
   Fresh setup has a separate 25-minute bound; OS packages and downloads have
   their own limits. Fresh installation requires a suitable NVIDIA host driver.
3. If needed, GitHub offers a browser/device login before GPU work. Use that
   login on the pod; do not paste tokens into chat. Authentication through the
   ChatGPT GitHub connector does not automatically authenticate a remote pod.
4. The experiment runs in tmux `foresight-run`. A fixed source revision is
   recorded along with dependency versions, full checkpoint hashes, settings,
   per-episode outputs, and controller traces through first termination.
5. Completed or failed run records are automatically committed to
   [`foresight/experiment-results`](https://github.com/pchatla-arch/anticipatory-balance/tree/foresight/experiment-results).
   The publisher uses an independent temporary Git index and does not commit
   unrelated work or force-push. It verifies the publication against the remote
   before writing a successful local receipt. Setup failures also get a record
   and upload attempt when a checkout and writable storage are available.
6. By default, the wrapper requests this pod's **stop** after finalization if
   this invocation's upload is verified, no GPU work remains, and `runpodctl`
   and `RUNPOD_POD_ID` are available. It supports both old and new CLI syntax.
   It never requests termination/removal. Missing controls or failed upload
   leave the pod running with a clear message. User-interrupted runs also remain
   running so the interruption can be inspected.

The helper can observe that a stop request was accepted; a process inside the
pod cannot reliably report its own completed shutdown. Check the RunPod console
for **Stopped**. Stop releases compute; retained storage can still incur charges.
Container storage may be cleared by stopping. `/workspace` only has the
retention guarantees of its actual volume/mount, so the verified remote results
are the durable copy of this experiment.

To retain the pod intentionally:

```bash
NO_STOP=1 bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/experiments-v2/scripts/start.sh) f
```

## Observe progress

```bash
tmux attach -t foresight-run
```

Detach with Ctrl-B, then D. Alternatively:

```bash
tail -f /workspace/foresight-runner/logs/fresh_pod.log
```

After the worker finishes, its tmux session may no longer exist. Read
`logs/last_run.json` for experiment versus publication status, and
`logs/pod_stop_receipt.json` for the stop request when one was attempted.
An exit-zero experiment with a failed upload is not reported as overall success.

## Recover an upload without rerunning the experiment

The failure message prints the exact publication-only command. The short
wrapper also supports the saved run ID:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/experiments-v2/scripts/start.sh) retry-publish RUN_ID
```

Replace `RUN_ID` with the value printed by the runner, such as the basename of
its `results/pod_v2_...` directory. This performs no setup, simulation, or
training. It preserves the run's original source revision. A manual retry does
not stop the pod: inspect its verified upload, then use RunPod's Stop control.

If setup or evaluation fails, keep the existing environment and the saved
failure record. Review the uploaded traceback before another GPU attempt;
reinstalling or retraining is not the default recovery step.

## Reproducibility and readiness

For a fully pinned run, replace the branch in the download URL with the release
commit SHA and pass the same SHA in `FORESIGHT_CODE_REF`. The launcher records
the actual resolved revision even when the convenient branch command is used.

CPU tests cover shell routing, runner failure/timeout handling, isolated Git
publication and remote verification, and guarded pod-control requests. Pod
control tests use fake commands and cannot stop a real pod. No GPU integration
run of this new launch path has been performed in the development workspace.
The recorded validation has 96 passing operational tests and three JAX checks
skipped because JAX is unavailable here, plus twelve passing analytical lookahead
checks. See [validation.json](../analysis/session_f/validation.json).

References: [RunPod CLI pod lifecycle](https://docs.runpod.io/runpodctl/reference/runpodctl-remove-pods),
[RunPod storage lifetimes](https://www.runpod.io/blog/where-did-my-files-go-a-straight-guide-to-runpod-storage).
