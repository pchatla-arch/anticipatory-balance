# AE-012 · Bind the held-out test to its registered protocol

WP 2.3 cross-review of `oaa/claude-bench` at `1f9ab5c`. [Issue #41](https://github.com/pchatla-arch/anticipatory-balance/issues/41)

The evaluator accepts a synthetic development registration bearing a 0.005 healthy-loss margin when the caller requests the test split and a 0.123 margin. It emits a new record with the caller’s margin. A new SHA-256 of the supplied registration identifies the input file but does not establish agreement with its contents. Six scenario dimensions are enforced at this revision. An independent test rejects a changed episode count. The split, margin, per-family seed map, checkpoint and actual contents of the operating point and telemetry are outside that check. Registered runtime/backend choices also need an identity if the protocol depends on them. [Evaluator](https://github.com/pchatla-arch/anticipatory-balance/blob/1f9ab5c07895640c5025502a05c02d2ea753eb04/scripts/eval_adaptive_bench.py#L232-L244)

### Reproduction without test data

From the reviewed repository on its pinned CPU environment, this fixture stops at the first attempted registration write. No environment, rollout or held-out samples are created. The script uses a temporary directory.

```bash
JAX_PLATFORMS=cpu python - <<'PY'
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

sys.path.insert(0, "scripts")
import eval_adaptive_bench as bench

class RegistrationReached(Exception):
    pass

with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    source = root / "fixture.json"
    registered = {
        "split": "dev", "families": ["healthy"], "methods": ["heuristic_gate"],
        "telemetry": ["declared"], "episodes": 2, "steps": 10, "schedule": "walk",
        "noninferiority_margins_loss": {"healthy": 0.005}, "registered_at": "fixture",
    }
    source.write_text(json.dumps(registered))

    def inspect_write(_, actual):
        print(registered["split"], actual["split"],
              registered["noninferiority_margins_loss"]["healthy"],
              actual["noninferiority_margins_loss"]["healthy"],
              [key for key in actual if "seed" in key])
        raise RegistrationReached

    args = ["--out", str(root / "out"), "--registration", str(source),
            "--split", "test", "--allow-cpu", "--impl", "jax", "--families", "healthy",
            "--methods", "heuristic_gate", "--telemetry", "declared",
            "--episodes", "2", "--steps", "10", "--schedule", "walk", "--margin", "0.123"]
    with patch.object(bench, "write_json", side_effect=inspect_write):
        try:
            bench.main(args)
        except RegistrationReached:
            pass
PY
```

Observed output at the reviewed head was `dev test 0.005 0.123 []`. The second margin comes from the new invocation. An empty seed-field list shows that the output has no explicit per-family seed map.

### Reproducing the prior development comparison

PR #39 archived the actual scenario IDs and kept its candidate/comparator pairs intact. Its evaluator computed `split_seed + 37 * family_index`. The current PR #35 computes `split_seed + 37 * (crc32(family_name) % 997)`. For gain 0.70 on the development split, the #39 paired pilot used 11000 and the new #35 tuning trials use 29204. Reordering families no longer changes a given seed under the current implementation. A future registration should include the concrete seed values as well as the source identity. [Frozen pilot](https://github.com/pchatla-arch/anticipatory-balance/blob/f824d53a06db6ad8e0d24413822d84407a432f20/results/bench_dev_wp22_cpu/episodes.csv) · [New tuning rows](https://github.com/pchatla-arch/anticipatory-balance/blob/1f9ab5c07895640c5025502a05c02d2ea753eb04/results/tune_heuristic_dev_cpu_v0/eval_start/episodes.csv)

### Smallest repair and acceptance check

Before G2, persist a canonical test record on an immutable ref with approval. Derive the test invocation’s effective split, per-family scenario seeds, margins and evaluation metric definitions, actual method/telemetry configurations, checkpoint digest and approved runtime/source identity. Compare them to the record before simulation. An output directory can then record a receipt pointing back to that exact registration. Add pre-physics unit cases that mutate one field at a time, particularly the split, margin, seed, or contents behind an unchanged configuration filename. A matching record should reach the same write boundary. Older development evidence remains reproducible from its original code, configuration and episode IDs.

Claude owns the harness change and Prasuna decides G2. The reproduction demonstrates a contract failure only. It uses no held-out results and does not change either competing method.
