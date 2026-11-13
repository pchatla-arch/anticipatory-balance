# Run the stabilization smoke diagnostic on the existing pod

The branch `foresight/stabilization-v1` contains the integrated stabilization
changes and a convenience runner. **GPU validation is still outstanding.** The
runner uses the existing rss2025 environment and saved policies, then publishes
compact results to the dedicated `foresight/pod-results` branch. It does not
install packages, train, render, or modify code/main. The results publisher leaves
your checked-out branch, working files, and index unchanged.

## 1. Fetch the branch

In the pod terminal:

```bash
cd /workspace/anticipatory-balance &&
git fetch origin &&
git switch foresight/stabilization-v1 &&
git pull --ff-only origin foresight/stabilization-v1 &&
bash scripts/run_switch_gate_smoke.sh
```

On first checkout, `git switch` normally creates a local branch tracking the
matching remote branch; later runs switch to the existing local branch. If Git
refuses a switch or fast-forward because of local work or divergent history,
stop and return that message. Do not force a checkout or reset files.

The archived `patches/foresight_stabilization_v1.patch` is **already integrated
on this branch**. Do not apply it again. No manual file upload to the pod is
needed.

## 2. What the runner does

It sources `/workspace/mujoco_playground/rss2025/.venv`, applies the established
Session A environment settings, runs the 31 simulator-free evaluation tests and
the CLI check, verifies that JAX sees a GPU and imports this checkout, and
restores both checkpoints before starting any evaluation rollout.

The default checkpoints, relative to the repository, are:

```text
logs/nominal_s1/G1Impact-20261113-063725/checkpoints/000202342400
logs/brace_s1/G1ImpactBrace-20261113-065807/checkpoints/000202342400
```

If your existing paths differ, provide overrides without creating a new venv:

```bash
FORESIGHT_VENV=/actual/path/to/rss2025/.venv \
FORESIGHT_NOMINAL=/actual/path/to/nominal/checkpoints/000202342400 \
FORESIGHT_BRACE=/actual/path/to/brace/checkpoints/000202342400 \
bash scripts/run_switch_gate_smoke.sh
```

Run `bash scripts/run_switch_gate_smoke.sh --help` to inspect these options
without requiring a venv or GPU. A missing checkpoint, failed test, wrong source
import, or missing GPU stops the runner before the rollout. Preserve that error;
do not reinstall packages or retrain as a workaround. Available failure logs are
still packaged and published so the failure can be reviewed.

The GPU smoke uses 32 shared scenarios, six comparison arms, and 32 reference
rollouts: **192 comparison rollouts plus 32 references**. It fixes the lead and
hold at 0.5 seconds each, the horizon at 300 control steps, and the seed at 100.
The switch uses nominal arrival time. The 2.5-second post-contact window can be
censored when the horizon is too short. This is a small diagnostic, not a
statistical validation of the skill.

## 3. Results publish automatically

The runner chooses a fresh timestamped output name and logs setup, tests, and the
GPU diagnostic. Do not create its evaluation output directory yourself. It first
closes the diagnostic log and saves a local review ZIP, then publishes selected
small outputs to the dedicated results branch. It prints the **GitHub results
URL** after a successful push. Tell ChatGPT **"done"**; it can then fetch and
review those committed results. This is a user-triggered review, not background
monitoring of the pod.

The publisher uploads available summary/manifest files, scenario/episode tables,
a sanitized log, and publication metadata. It does not upload checkpoints, raw
NPZ traces, or the ZIP. Code changes and the main branch are untouched.

Local fallback paths are also printed:

```text
Results directory: /workspace/anticipatory-balance/results/<unique-run-name>
Console log: /workspace/anticipatory-balance/logs/<unique-run-name>/run.log
Review ZIP: /workspace/anticipatory-balance/logs/<unique-run-name>/review.zip
```

Keep that **review ZIP** as a fallback. It contains the available `summary.json`,
`manifest.json`, `scenarios.csv`, `episodes.csv`, and `run.log`. An ordinary
preflight or evaluation error still creates a ZIP and attempts to publish the
files available, while preserving the diagnostic's nonzero exit status. An early
failure may produce only the log; process termination or pod loss can prevent
final packaging and publication.

If publication fails after the diagnostic completes, the runner returns nonzero,
keeps the local files, and prints an exact **publication-only retry command**.
Use that command after the connection/access issue is resolved; there is no need
to rerun the GPU diagnostic. An existing nonzero diagnostic exit code is retained
even if publication also fails.

For an intentionally offline/debug run, disable publication explicitly:

```bash
FORESIGHT_PUBLISH_RESULTS=0 bash scripts/run_switch_gate_smoke.sh
```

The offline run still prints its local review ZIP path.

A completed run has 32 scenario rows, 192 episode rows, and four raw
`batch_000_*.npz` archives. The large NPZ traces stay in the results directory for
follow-up inspection and are omitted from the compact review ZIP. The runner
does not delete or reuse earlier results.

Read `summary.status` even when the shell exits successfully:

- `harness_failure`: stop and diagnose the implementation or runtime error.
- `switch_failure`: BRACE introduced a failure in at least one no-projectile
  pair where the zero-command nominal actor survived; inspect the switch.
- `inconclusive`: the small diagnostic completed without that specific flag.
  Inspect all three no-projectile arms and the paired counts before proceeding.

The last two statuses are completed experiments; the runner returns zero when
the diagnostic and publication both complete successfully. Neither status
establishes safety or improved impact recovery. No larger sweep or retraining
starts automatically.

See [STABILIZATION_V1.md](STABILIZATION_V1.md) for the protocol and metric
definitions.
