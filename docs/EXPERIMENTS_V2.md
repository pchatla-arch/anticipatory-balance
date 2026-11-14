# Foresight: next pod experiment

This branch starts from the published Session C code and checkpoint bundles. It
adds an exact 20 Hz observation schedule on the 50 Hz control grid, per-condition
result persistence, explicit event counts, and a bounded runner. It preserves the
existing belief/rule formulation. The changes have CPU coverage; the first pod
smoke run is the GPU integration test.

## Empty pod: install and run

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

Setup runs in the background while GitHub CLI offers browser/device login. This
login authorizes the new pod to upload results; the ChatGPT GitHub connection does
not authenticate the pod. Canceling login does not stop setup. If the CLI is not
available from the OS package manager, evaluation continues and results are kept
locally. Set `FORESIGHT_SKIP_GITHUB_LOGIN=1` to skip that login deliberately.

```bash
tmux attach -t foresight-run
# Or inspect the combined setup/experiment log:
tail -f /workspace/anticipatory-balance/logs/fresh_pod.log
```

The environment-install step is limited to 25 minutes and can resume its owned
partial installation. Initial OS package-manager steps have separate timeouts.
The 45-minute experiment budget starts after environment setup. An existing
unrelated venv is checked and reused without package changes. An existing tmux
session is preserved. After a finished/failed session, exit its shell before
rerunning the entry point. These are process limits, not automatic pod shutdown.

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
python3 scripts/run_experiments_v2.py --mode pilot --max-minutes 45
```

## Execution order

1. Check for an existing experiment/GPU process, run the focused CPU/JAX timing
   checks, verify a JAX GPU device, and restore both policies.
2. Run 32 scenarios for six seconds each, with six conditions: reactive, oracle,
   TTC threshold 0.3 s, and belief preparation weight 0.2; sensing methods use
   delays of 0 and 300 ms. The detector averages 20 Hz.
3. Only after the smoke checks pass, run the pilot at 10, 20, and 50 Hz. Each rate
   uses 128 scenarios and ten conditions: reactive, oracle, TTC thresholds 0.2 and
   0.6 s, and belief preparation weights 0.05 and 0.2, at 0 and 300 ms delay.
   Scenario seed 400 is shared across rates and conditions.
4. Save a review ZIP and attempt to push compact JSON/CSV/log outputs to the
   dedicated `foresight/experiment-results` branch using the pod's existing Git
   credentials. Publication does not alter the working checkout or code branch.
   A failed push prints a publication-only retry command.

The default limit is 45 minutes for experiment processes, with a maximum of ten
minutes for the smoke and twenty minutes per pilot stage within that total.
Timeouts retain completed condition files. Packaging and publication can take
additional time. **The runner does not stop the pod or its billing.** Save results
and stop the pod after the run if no further GPU work is needed.

## What counts as a smoke pass

Both policies restore; all six conditions complete; the expected episode tables
exist; headline rates and progress are finite; reactive scheduled hits produce
contact; and sensing conditions receive valid detections before termination.
The smoke does not test statistical superiority. The one-pair pilot is exploratory.

## Data and limitations

Each stage has `config.json`, `batches.json`, `episodes/*.csv`, `summary.json`,
`frontier.csv`, and an optional plot. Every completed condition is saved before
the next begins. Per-episode records include scenario IDs and the actual shared
external projectile trajectory. Actual robot contacts can differ by policy.

Legacy batch-mean metrics are retained; pooled rates have explicit `*_pooled`
names and integer denominators. `*_until_termination` preparation metrics exclude
actions after the first terminal transition; historical metrics include them.

The 20 Hz cadence alternates three and two control ticks (60/40 ms intervals).
This experiment still uses synthetic detections. Depth detector integration,
belief calibration, validated brace readiness, and additional policy seeds remain
separate work. The current pilot cannot establish that sensing is the sole cause
of the oracle gap.
