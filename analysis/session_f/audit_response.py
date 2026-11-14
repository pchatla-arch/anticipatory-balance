"""Reproduce the exploratory CPU-only E6b response audit.

Run with Python, NumPy, and SciPy already installed. No simulation or policy-test
data is used. The subset is verified against source_manifest.json before fitting.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import platform

import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.special import expit


HERE = Path(__file__).resolve().parent
RETAINED_LEADS = (-1.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0)
EXPECTED_SCENARIOS = 512
EXPECTED_BATCH_COUNTS = {0: 256, 1: 256}
EXPECTED_COLUMNS = (
    "scenario_id", "batch", "lead", "mass", "speed", "momentum",
    "t_hit_nominal", "contact_step", "contact", "fell", "censored",
)


def load_validated_subset():
    """Fail before any fit on altered bytes, invalid inputs, or split leakage."""
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    subset = manifest["subset"]
    payload = (HERE / subset["filename"]).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != subset["sha256"] or len(payload) != subset["bytes"]:
        raise ValueError("Subset SHA-256 or byte count does not match source_manifest.json")
    if manifest["n_scenarios"] != EXPECTED_SCENARIOS:
        raise ValueError("Manifest must describe 512 original scenario groups")
    if tuple(subset["retained_leads_s"]) != RETAINED_LEADS:
        raise ValueError("Manifest does not describe the seven required retained leads")

    reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
    if tuple(reader.fieldnames or ()) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected subset schema")
    rows = list(reader)
    if len(rows) != EXPECTED_SCENARIOS * len(RETAINED_LEADS) or len(rows) != subset["rows"]:
        raise ValueError("Expected 3584 subset rows: 512 scenarios at seven leads")
    numeric = {}
    for key in EXPECTED_COLUMNS[1:]:
        numeric[key] = np.asarray([float(row[key]) for row in rows])
        if not np.all(np.isfinite(numeric[key])):
            raise ValueError(f"Non-finite values in {key}")
    for key in ("momentum", "speed", "mass"):
        if np.any(numeric[key] <= 0):
            raise ValueError(f"Expected strictly positive {key}")
    for key in ("fell", "contact", "censored"):
        if not np.all(np.isin(numeric[key], (0.0, 1.0))):
            raise ValueError(f"Expected binary outcomes in {key}")
    if not np.all(np.isin(numeric["batch"], tuple(EXPECTED_BATCH_COUNTS))):
        raise ValueError("Expected only integer batch IDs 0 and 1")

    groups = defaultdict(list)
    for index, row in enumerate(rows):
        if not row["scenario_id"]:
            raise ValueError("Empty scenario ID")
        groups[row["scenario_id"]].append(index)
    if len(groups) != EXPECTED_SCENARIOS:
        raise ValueError("Expected exactly 512 unique scenario IDs")
    expected_leads = Counter(RETAINED_LEADS)
    batch_counts = Counter()
    source_keys = set()
    for scenario_id, indices in groups.items():
        if Counter(numeric["lead"][indices]) != expected_leads:
            raise ValueError(f"Scenario {scenario_id} must appear exactly once at every retained lead")
        for key in ("batch", "mass", "speed", "momentum", "t_hit_nominal"):
            if len(set(numeric[key][indices])) != 1:
                raise ValueError(f"Scenario {scenario_id} changes {key} across its rows")
        first = indices[0]
        batch_counts[int(numeric["batch"][first])] += 1
        source_key = tuple(numeric[key][first] for key in manifest["group_key"])
        if source_key in source_keys:
            raise ValueError("One source scenario tuple maps to multiple scenario IDs")
        source_keys.add(source_key)
    if dict(batch_counts) != EXPECTED_BATCH_COUNTS:
        raise ValueError("Expected 256 isolated scenario groups in each batch")

    scenario_ids = np.asarray([row["scenario_id"] for row in rows])
    for held in EXPECTED_BATCH_COUNTS:
        train_ids = set(scenario_ids[numeric["batch"] != held])
        held_ids = set(scenario_ids[numeric["batch"] == held])
        if train_ids & held_ids or len(train_ids) != 256 or len(held_ids) != 256:
            raise ValueError("Scenario leakage between training and held-out batch")
    validation = {
        "subset_sha256": digest,
        "subset_bytes": len(payload),
        "rows": len(rows),
        "scenario_groups": len(groups),
        "scenario_groups_per_batch": dict(batch_counts),
        "retained_leads_s": list(RETAINED_LEADS),
        "positive_finite_momentum_speed": True,
        "binary_outcomes": True,
        "one_row_per_scenario_per_lead": True,
        "batch_isolation": True,
    }
    return numeric, manifest, validation


def fit_predict(X, y, train, test):
    mean = X[train].mean(0)
    sd = X[train].std(0)
    sd[sd == 0] = 1
    A = np.column_stack([np.ones(train.sum()), (X[train] - mean) / sd])
    B = np.column_stack([np.ones(test.sum()), (X[test] - mean) / sd])
    target = y[train]
    penalty = np.r_[0., np.repeat(1e-4, X.shape[1])]

    def objective(w):
        z = A @ w
        return (np.logaddexp(0, z) - target * z).sum() + .5 * (penalty * w * w).sum()

    def gradient(w):
        return A.T @ (expit(A @ w) - target) + penalty * w

    fitted = minimize(objective, np.zeros(A.shape[1]), jac=gradient, method="BFGS", options={"gtol": 1e-7, "maxiter": 500})
    if np.linalg.norm(gradient(fitted.x), ord=np.inf) > 1e-4:
        raise RuntimeError(str(fitted.message))
    return expit(B @ fitted.x)


def metrics(target, prediction):
    p = np.clip(prediction, 1e-12, 1-1e-12)
    return {"n": len(p), "fall_rate": float(target.mean()), "predicted_rate": float(p.mean()), "brier": float(np.mean((p-target)**2)), "log_loss": float(np.mean(-target*np.log(p)-(1-target)*np.log1p(-p)))}


def run_audit():
    numeric, manifest, validation = load_validated_subset()
    batch, lead, J, v, y = (numeric[key] for key in ("batch", "lead", "momentum", "speed", "fell"))
    logJ = np.log(J)
    logv = np.log(v)
    features = {
        "logJ": np.column_stack([logJ]),
        "logJ_logv": np.column_stack([logJ, logv]),
        "logJ_lead_interactions": np.column_stack([logJ, lead, lead**2, logJ*lead]),
        "logJ_logv_lead_interactions": np.column_stack([logJ, logv, lead, lead**2, logJ*lead, logv*lead]),
    }
    result = {
        "source_commit": manifest["commit"],
        "source_blob": manifest["episodes_blob"],
        "runtime": {"python": platform.python_version(), "python_implementation": platform.python_implementation(), "numpy": np.__version__, "scipy": scipy.__version__},
        "validation": validation,
        "split": "Two directions: fit batch 0 and evaluate batch 1, then reverse. All rows for each scenario stay in one fold: six BRACE leads and a separately analyzed CONTINUE row. Original 512 scenarios, 256 per batch. Exploratory, not confirmatory policy test.",
        "selection": manifest["subset"]["selection"],
        "endpoint": "Failure during the recorded six-second episode; censored nonfalls retained. This is not complete 2.5-second post-contact survival.",
        "l2_penalty": "1e-4 on standardized non-intercept coefficients",
        "models": {},
    }
    for name, X in features.items():
        selection = (lead >= .1) & (lead <= 1)
        oof = np.full(len(y), np.nan)
        folds = []
        for held in [0, 1]:
            test = selection & (batch == held)
            train = selection & (batch != held)
            pred = fit_predict(X, y, train, test)
            oof[test] = pred
            folds.append({"held_batch": held, **metrics(y[test], pred)})
        result["models"][name] = {"pooled": metrics(y[selection], oof[selection]), "folds": folds, "by_lead": [{"lead_s": float(L), **metrics(y[lead == L], oof[lead == L])} for L in sorted(set(lead[selection]))]}
    result["continue"] = {}
    for name in ["logJ", "logJ_logv"]:
        selection = lead == -1
        oof = np.full(len(y), np.nan)
        for held in [0, 1]:
            test = selection & (batch == held)
            oof[test] = fit_predict(features[name], y, selection & (batch != held), test)
        result["continue"][name] = metrics(y[selection], oof[selection])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "exploratory_grouped_fit.json")
    args = parser.parse_args()
    result = run_audit()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"models": {key: value["pooled"] for key, value in result["models"].items()}, "continue": result["continue"]}, indent=2))


if __name__ == "__main__":
    main()
