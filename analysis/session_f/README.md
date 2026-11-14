# Session F CPU preparation

These are reproducible analysis and formulation checks, not an online controller
or a pod experiment launcher. They do not import MuJoCo, Warp, JAX, or Brax.
They do not change a policy checkpoint or production response-model file.

See [the implementation plan](../../docs/SESSION_F_PLAN.md),
[response audit](../../docs/SESSION_F_RESPONSE_AUDIT.md), and
[lookahead formulation](../../docs/SESSION_F_VOI_DESIGN.md).

## E6b exploratory prediction check

`e6b_response_subset.csv` retains the BRACE-mode, projectile-enabled rows at
nominal leads -1, 0.1, 0.2, 0.3, 0.5, 0.75, and 1 from the pinned E6b episode
table. Lead -1 is the CONTINUE baseline. Contact is not a selection criterion.
The reconstructed scenario ID identifies repetitions across interventions;
it is based on batch, mass, speed, and nominal arrival because the old data lack
an explicit reset/scenario ID. `source_manifest.json` records provenance.

With NumPy and SciPy already available, run from the repository root:

```bash
python analysis/session_f/audit_response.py
```

This validates the included data and writes `exploratory_grouped_fit.json`
beside the script. The fit uses two grouped folds, holding out one whole
256-scenario batch at a time. Standardization uses the training fold only.
The published scores describe a development analysis under the original
oracle timing/release contract. They do not establish policy improvement or
calibration under a new protocol. No package installation is required or
performed by these scripts; run the fit in an existing environment containing
its analysis dependencies.

## One-observation decision formulation

The following uses only Python's standard library:

```bash
python analysis/session_f/test_voi_contract.py
```

It checks twelve small exact examples involving delivery phase, old capture
timestamps, the finite horizon, empty observations, information value,
average-before-max, reduced brace efficacy after delay, absorbing failures,
and prefix costs. Passing these examples does not validate a deployed JAX
implementation or the response model. Pending integration checks are listed
in the lookahead formulation.
