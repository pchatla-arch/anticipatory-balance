# Foresight: release diagnostic after the Session D review

The broad rate pilot is suspended. Session D already evaluated 10, 25, and 50 Hz,
and review found a release error in the old rule and an unreachable passage test
in its subsequent fix. The next run compares those historical behaviors with an
explicit bounded commitment on the same scenarios. See `SESSION_D_REVIEW.md`.

This branch retains the published skills, source-pinned bootstrap, exact 20 Hz
cadence support, per-condition output, pooled counts, and result publisher. It
adds capture-time launch validity and an auditable release contract. CPU tests
are available; the pod run is the first GPU integration check for these changes.

## One command: existing or empty pod

The supported entry point is now:

```bash
bash <(curl -fsSL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/experiments-v2/scripts/start.sh) f
```

`f` currently means **F0**, the small release diagnostic described below. It does
not start new lookahead, rendered-depth, or PPO training experiments. It uses a
separate checkout at `/workspace/foresight-runner` and the existing Playground
venv when available. The original Session E checkout is not switched.

The wrapper enables automatic stop by default. A stop request requires this
invocation's verified GitHub publication, recognized RunPod controls and own pod
ID, and no active GPU processes. `NO_STOP=1` before the command disables it.
Stopping is requested through `runpodctl`; `git push` alone does not stop a pod.
The launcher never requests termination. Check RunPod for actual stopped state;
storage charges can continue after compute stops.

The complete operating guide, including failure recovery, is
[POD_RUNBOOK.md](POD_RUNBOOK.md). Code is CPU-tested; F0 remains the first GPU
integration validation of this release.

## Fresh installation details

Use `scripts/start_empty_pod.sh` from the published release. It works from any
directory and installs missing system tools, the pinned Python 3.12 environment,
and G1 assets. It fetches the repository's published nominal and BRACE v2 weights.
The entry point starts setup and experiments in tmux session `foresight-run`.

```bash
curl -fL https://raw.githubusercontent.com/pchatla-arch/anticipatory-balance/foresight/experiments-v2/scripts/start_empty_pod.sh -o /tmp/foresight-start.sh
bash /tmp/foresight-start.sh
```

For an exact release, replace the branch in the URL with its full commit SHA and
set `FORESIGHT_CODE_REF` to that same SHA when invoking the downloaded script.
Use an Ubuntu/Debian x86_64 GPU pod with host NVIDIA driver 580 or newer. The
installer checks the driver before installing packages. It does not install a
host driver. Container `/workspace` storage is supported; completed results must
be uploaded or downloaded before the container is replaced.

The pinned runtime is Playground `9de993826d1a486f18aec4e8d3179e93ff64ed41`,
its frozen `rss2025/uv.lock`, and Menagerie
`1b86ece576591213e2b666ebf59508454200ca97`. It includes JAX/CUDA12 packages
0.12.0.1, Brax 0.14.2, MuJoCo/MJX 3.16.0, and Warp 1.18.0. uv 0.13.8 is
installed locally without changing shell profiles. No separate latest-version
dependency resolution runs.

GitHub CLI offers browser/device login before GPU setup or evaluation begins.
This authorizes the new pod to upload results; the ChatGPT GitHub connection does
not authenticate the pod. Canceling login prevents launch. With existing working
Git credentials, set `FORESIGHT_SKIP_GITHUB_LOGIN=1` to skip that login.

```bash
tmux attach -t foresight-run
# Or inspect the combined setup/experiment log:
tail -f /workspace/foresight-runner/logs/fresh_pod.log
```

The environment-install step is limited to 25 minutes and can resume its owned
partial installation. Initial OS package-manager steps have separate timeouts.
The 20-minute experiment budget starts after environment setup. An existing
unrelated venv is checked and reused without package changes. An existing tmux
session is preserved. The new worker exits its tmux session when finalization
ends; it does not replace another running session. An old retained shell from an
earlier launcher must be exited manually. Process limits and pod stop are
separate checks. Direct `start_empty_pod.sh` use retains opt-in automatic stop;
the recommended `start.sh f` wrapper enables it.

## Existing environment: start evaluation

Use the existing MuJoCo Playground `rss2025` environment. The default path is
`/workspace/mujoco_playground/rss2025/.venv`; set `FORESIGHT_VENV` if yours differs.
The runner installs nothing and uses the committed `nominal_s1` and `brace2_s1`
checkpoint bundles.

```bash
bash scripts/start_experiments_v2.sh
```

The launcher creates the tmux session `foresight-v2` and prints its log path. It
does not replace an existing session. Watch with the printed `tail -f` command,
or attach with `tmux attach -t foresight-v2`. Detach with Ctrl-B, then D.

For the smoke run alone:

```bash
FORESIGHT_MODE=smoke bash scripts/start_experiments_v2.sh
```

If tmux is unavailable, run in your connected terminal:

```bash
python3 scripts/run_experiments_v2.py --mode release --max-minutes 20
```

## Execution order

1. Check for an existing experiment/GPU process, run focused CPU/JAX timing and
   commitment checks, verify a JAX GPU device, and restore both published policies.
2. Run 64 scenarios for six seconds each (32 in smoke mode), seed 400, with five
   conditions: reactive, oracle at 0.5 s lead, and belief preparation weight 0.1
   under `legacy`, `stale_only`, and `deadline` release. Detections are 50 Hz with
   zero latency. Every condition replays the same external projectile trajectories.
3. Verify complete outputs, identical scenario records, and the observed deadline
   contract. Save all scalar controller logs through the first terminal transition.
   Stop after this diagnostic; no broad sweep or skill training follows.
4. Save a review ZIP and attempt to push compact JSON/CSV/log outputs to the
   dedicated `foresight/experiment-results` branch using the pod's existing Git
   credentials. Publication does not alter the working checkout or code branch.
   A failed push prints a publication-only retry command.

The default limit is 20 minutes for experiment processes, including tests and
preflight. The evaluation uses the remaining budget; the former independent
1,000-second cap has been removed. Timeouts retain completed condition files.
Packaging and bounded publication can take additional time. Direct runner use
does not stop the pod; the recommended wrapper invokes a separate finalizer
after verified upload. Upload failures retain local data and prevent automatic
stop. `logs/last_run.json` separates experiment, archive, publication, and overall
status. The publisher verifies the remote commit and preserves the original
run's source revision even if publication is retried from a later checkout.

## Release contracts and pass criteria

`legacy` reproduces the request-dependent release used for Sessions C–D.
`stale_only` reproduces the later partial fix, whose negative-TTC passage condition
could never fire with the current belief statistic. Both retain their historical
retrigger behavior to make the comparison explicit.

`deadline` is a candidate protocol for these one-projectile episodes: accept the
first fresh request with a finite positive forecast, latch estimated arrival plus
0.5 s, respect a 0.3 s minimum commitment, and cap the duration at 3 s. The deadline
is rounded on the control grid. Later forecasts and missing detections do not move
it. The episode permits one commitment, preventing repeated starts on the same
threat. This is a defined protocol change, not an isolated causal ablation of one
line. It does not claim that estimated arrival equals actual physical contact.

A pass requires both policies to restore; all five conditions to complete;
finite headline metrics; reactive scheduled hits to produce contact; valid
detections before termination; matched scenario IDs and external trajectories;
and actual execution of at least one deadline commitment with no observed early
release, deadline drift, or rearm. Episodes ending before the deadline are marked
as observation-censored for this check. Fall-rate superiority is not a pass gate.

## Data and limitations

Each stage has `config.json`, `batches.json`, `episodes/*.csv`, `traces/*.csv`,
`summary.json`, `frontier.csv`, and `release_audit.json`. Every condition is saved before
the next begins. Per-episode records include scenario IDs and the actual shared
external projectile trajectory. Actual robot contacts can differ by policy.

Legacy batch-mean metrics are retained; pooled rates have explicit `*_pooled`
names and integer denominators. `*_until_termination` preparation metrics exclude
actions after the first terminal transition; historical metrics include them.

The 20 Hz cadence alternates three and two control ticks (60/40 ms intervals).
This experiment uses synthetic detections. Depth detector integration,
belief calibration, validated brace readiness, and additional policy seeds remain
separate work. This diagnostic cannot establish that sensing is the sole cause
of the oracle gap.
