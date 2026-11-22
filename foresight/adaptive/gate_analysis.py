"""Predeclared, paired engineering gate; no simulator or learning dependencies.

Screen selects the least severe passing gain. Confirmation uses fresh scenarios
and additionally requires a positive 95% paired-bootstrap lower bound on the
loss reduction. These are feasibility thresholds, not a stability certificate.
"""
from __future__ import annotations

import numpy as np

PROTOCOL_VERSION = "adaptive-servo-gain-gate-v1"
CANDIDATE_ORDER = ("gain_085", "gain_070", "gain_055")
THRESHOLDS = dict(healthy_survival_min=0.90, fault_survival_min=0.20,
                  absolute_loss_gap_min=0.03, relative_loss_gap_min=0.25,
                  fall_increase_min=0.15, recovery_fraction_min=0.50,
                  identity_tolerance=1e-6, bootstrap_draws=2000,
                  bootstrap_seed=228, bootstrap_confidence=0.95,
                  correction_bound_rad=0.20, slew_bound_rad_s=2.0, bound_tolerance=1e-5)
METRICS = ("postfault_loss", "progress_m", "correction_bound_fraction",
           "max_correction_rad", "max_slew_rad_s")
IDENTITY_EXTRA = ("first_nonfinite_step", "fault_exposed_steps", "max_abs_actuator_force_nm",
                  "max_abs_joint_actuation_nm", "torque_limit_excess_nm")
REQUIRED = ("phase", "condition", "scenario_id", "gain", "oracle",
            "speed_cmd_mps", "onset_step", "fell", "nonfinite",
            "termination_step", *METRICS)


def _boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1, "0", "1", "False", "True", "false", "true"):
        return value in (1, "1", "True", "true")
    raise ValueError(f"Invalid boolean: {value!r}")


CONTROLS = ("healthy", "identity", "healthy_repeat")


def _group(rows, phase):
    groups = {}
    allowed = set(CONTROLS) | {
        f"{key}_{arm}" for key in CANDIDATE_ORDER for arm in ("plain", "oracle")}
    for source in rows:
        missing = set(REQUIRED) - source.keys()
        if missing:
            raise ValueError(f"Missing fields: {sorted(missing)}")
        row = dict(source)
        name, sid = row["condition"], str(row["scenario_id"])
        if name not in allowed or row["phase"] != phase or not sid:
            raise ValueError("Unknown condition, wrong phase, or empty scenario ID")
        for key in ("fell", "nonfinite", "oracle"):
            row[key] = _boolean(row[key])
        for key in (*METRICS, "gain", "speed_cmd_mps"):
            row[key] = float(row[key])
        for key in ("onset_step", "termination_step"):
            value = float(row[key])
            if not np.isfinite(value) or value != int(value):
                raise ValueError(f"{key} must be a finite integer")
            row[key] = int(value)
        expected_gain = 1.0 if name in CONTROLS else int(name[5:8]) / 100
        if (not np.isclose(row["gain"], expected_gain, rtol=0, atol=1e-7)
                or row["oracle"] != (name == "identity" or name.endswith("_oracle"))
                or not np.isfinite(row["speed_cmd_mps"]) or row["onset_step"] < 0
                or row["termination_step"] < -1):
            raise ValueError("Condition metadata does not match its declared arm")
        if sid in groups.setdefault(name, {}):
            raise ValueError(f"Duplicate scenario {sid} in {name}")
        if np.isfinite(row["postfault_loss"]) and not 0 <= row["postfault_loss"] <= 4.000001:
            raise ValueError("Postfault loss must be in [0, 4]")
        if np.isfinite(row["correction_bound_fraction"]) and not 0 <= row["correction_bound_fraction"] <= 1:
            raise ValueError("Correction bound fraction must be in [0, 1]")
        if any(np.isfinite(row[key]) and row[key] < 0 for key in ("max_correction_rad", "max_slew_rad_s")):
            raise ValueError("Correction magnitude and slew must be nonnegative")
        prefault = 0 <= row["termination_step"] < row["onset_step"]
        if "pre_fault_failed" in row and _boolean(row["pre_fault_failed"]) != prefault:
            raise ValueError("Pre-fault flag disagrees with termination/onset steps")
        row["pre_fault_failed"] = prefault
        for key in IDENTITY_EXTRA:
            if key in row:
                row[key] = float(row[key])
        groups[name][sid] = row
    if not {"healthy", "identity"} <= groups.keys():
        raise ValueError("Healthy and identity controls are required")
    ids = sorted(groups["healthy"])
    if len(ids) < 2 or any(sorted(arm) != ids for arm in groups.values()):
        raise ValueError("Each condition needs the same two or more unique scenario IDs")
    for arm in groups.values():
        for sid in ids:
            for key in ("speed_cmd_mps", "onset_step"):
                if arm[sid][key] != groups["healthy"][sid][key]:
                    raise ValueError("Paired scenarios differ in command or onset")
    return {key: [arm[sid] for sid in ids] for key, arm in groups.items()}


def _summary(rows):
    values = np.asarray([[r[k] for k in METRICS] for r in rows], dtype=np.float64)
    invalid = np.asarray([r["nonfinite"] for r in rows]) | ~np.isfinite(values).all(axis=1)
    invalid |= np.asarray([any(not np.isfinite(r[key]) for key in IDENTITY_EXTRA if key in r) for r in rows])
    return dict(n=len(rows), fall_rate=float(np.mean([r["fell"] for r in rows])),
                invalid_episodes=int(invalid.sum()),
                control_bound_violations=sum(r["max_correction_rad"] > THRESHOLDS["correction_bound_rad"] + THRESHOLDS["bound_tolerance"]
                    or r["max_slew_rad_s"] > THRESHOLDS["slew_bound_rad_s"] + THRESHOLDS["bound_tolerance"] for r in rows),
                torque_limit_violations=sum(r.get("torque_limit_excess_nm", 0.) > THRESHOLDS["bound_tolerance"] for r in rows),
                pre_fault_failures=sum(r["pre_fault_failed"] for r in rows),
                postfault_falls=sum(r["fell"] and not r["pre_fault_failed"] for r in rows),
                mean_loss=float(values[:, 0].mean()) if np.isfinite(values[:, 0]).all() else None,
                mean_progress_m=float(values[:, 1].mean()) if np.isfinite(values[:, 1]).all() else None,
                mean_bound_fraction=float(values[:, 2].mean()) if np.isfinite(values[:, 2]).all() else None)


def _paired_ci(a, b):
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    rng = np.random.default_rng(THRESHOLDS["bootstrap_seed"])
    # Chunk resampling so the reporting step stays small even with larger batches.
    means = []
    for start in range(0, THRESHOLDS["bootstrap_draws"], 100):
        n = min(100, THRESHOLDS["bootstrap_draws"] - start)
        means.extend(delta[rng.integers(0, len(delta), (n, len(delta)))].mean(axis=1))
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def analyze_phase(rows, confirmation=False):
    """Analyze one complete phase; malformed pairing raises ValueError.

    Numerical invalidity fails the gate, even if another arm looks successful.
    A valid but unreliable healthy control makes the gate inconclusive.
    """
    phase = "confirm" if confirmation else "screen"
    groups = _group(rows, phase)
    summaries = {name: _summary(arm) for name, arm in groups.items()}
    candidates = [key for key in CANDIDATE_ORDER if f"{key}_plain" in groups or f"{key}_oracle" in groups]
    if not candidates or any(f"{key}_{arm}" not in groups for key in candidates for arm in ("plain", "oracle")):
        raise ValueError("Each included fault needs both plain and oracle arms")
    healthy = summaries["healthy"]
    numerical_ok = not any(arm["invalid_episodes"] for arm in summaries.values())
    bounds_ok = not any(arm["control_bound_violations"] for arm in summaries.values())
    torque_checked = all("torque_limit_excess_nm" in r for arm in groups.values() for r in arm)
    torque_ok = not any(arm["torque_limit_violations"] for arm in summaries.values())
    identity_keys = (*METRICS, *(key for key in IDENTITY_EXTRA if all(key in r for arm in groups.values() for r in arm)))

    def control_difference(name):
        """Largest absolute metric difference between the healthy arm and `name`, and exact equality."""
        differences = [abs(a[k] - b[k]) for a, b in zip(groups["healthy"], groups[name]) for k in identity_keys]
        largest = max(differences) if np.isfinite(differences).all() else None
        exact = (largest is not None and largest <= THRESHOLDS["identity_tolerance"]
                 and all(a[k] == b[k] for a, b in zip(groups["healthy"], groups[name])
                         for k in ("fell", "nonfinite", "termination_step")))
        return largest, exact

    identity_difference, identity_ok = control_difference("identity")
    # Optional probe: the healthy arm executed a second time with identical inputs.
    # It separates a backend that is not run-to-run reproducible (GPU floating-point
    # atomics) from a code path that actually differs under the identity adapter.
    repeat_difference, repeat_ok = control_difference("healthy_repeat") if "healthy_repeat" in groups else (None, None)
    healthy_ok = healthy["fall_rate"] <= 1 - THRESHOLDS["healthy_survival_min"] + 1e-12
    prefault_ok = all(a["pre_fault_failed"] == b["pre_fault_failed"] for arm in groups.values()
                      for a, b in zip(groups["healthy"], arm))
    result = dict(protocol_version=PROTOCOL_VERSION, phase=phase, thresholds=THRESHOLDS.copy(),
                  candidate_order=candidates, arms=summaries, numerical_ok=numerical_ok,
                  controls=dict(healthy=healthy_ok, identity=bool(identity_ok), prefault_pairing=prefault_ok,
                                bounds=bounds_ok, torque_limits_checked=torque_checked,
                                torque_limits=torque_ok if torque_checked else None,
                                identity_max_difference=identity_difference,
                                run_to_run_reproducible=repeat_ok,
                                repeat_max_difference=repeat_difference), candidates={}, reasons=[])
    for key in candidates:
        plain, oracle = summaries[f"{key}_plain"], summaries[f"{key}_oracle"]
        valid = all(s["mean_loss"] is not None and not s["invalid_episodes"] for s in (healthy, plain, oracle))
        gap = plain["mean_loss"] - healthy["mean_loss"] if valid else None
        improvement = plain["mean_loss"] - oracle["mean_loss"] if valid else None
        recovery = improvement / gap if valid and gap > 0 else None
        degradation = bool(valid and gap > 0 and (
            gap >= max(THRESHOLDS["absolute_loss_gap_min"], THRESHOLDS["relative_loss_gap_min"] * healthy["mean_loss"])
            or plain["fall_rate"] - healthy["fall_rate"] >= THRESHOLDS["fall_increase_min"]))
        ci = _paired_ci([r["postfault_loss"] for r in groups[f"{key}_plain"]],
                        [r["postfault_loss"] for r in groups[f"{key}_oracle"]]) if valid else None
        checks = dict(degradation=degradation,
                      recoverable=plain["fall_rate"] <= 1 - THRESHOLDS["fault_survival_min"] + 1e-12,
                      recovered_half_gap=recovery is not None and recovery >= THRESHOLDS["recovery_fraction_min"],
                      no_fall_regression=oracle["fall_rate"] <= plain["fall_rate"] + 1e-12,
                      positive_paired_ci=bool(ci is not None and ci[0] > 0) if confirmation else True)
        result["candidates"][key] = dict(passed=bool(numerical_ok and bounds_ok and torque_ok and healthy_ok
            and identity_ok and prefault_ok and all(checks.values())),
            checks=checks, loss_gap=gap, loss_reduction=improvement, recovery_fraction=recovery,
            paired_loss_reduction_ci95=ci, near_healthy_fall_rate=oracle["fall_rate"] <= healthy["fall_rate"] + 0.10)
    if not numerical_ok:
        result["reasons"].append("Numerically invalid episodes prevent a physical recovery claim.")
    if not bounds_ok:
        result["reasons"].append("Requested correction exceeded the 0.20 rad magnitude or 2 rad/s slew bound.")
    if not torque_ok:
        result["reasons"].append("Reported joint actuation exceeded a configured force limit.")
    if not healthy_ok:
        result["reasons"].append("Healthy walker survival is below 90%; fault test is inconclusive.")
    if not identity_ok:
        if repeat_ok is False:
            result["reasons"].append(
                "Healthy-model adapter did not reproduce the healthy control, and neither did a repeat of the "
                f"healthy arm (largest difference {repeat_difference:.3g}): the backend is not run-to-run "
                "reproducible, so the bitwise identity control cannot be evaluated on this run.")
        elif repeat_ok:
            result["reasons"].append(
                "Healthy-model adapter did not reproduce the healthy control although the backend is run-to-run "
                "reproducible: the identity adapter changes the executed code path.")
        else:
            result["reasons"].append("Healthy-model adapter did not reproduce the healthy control.")
    if not prefault_ok:
        result["reasons"].append("Arms disagree on failures before fault onset; attribution is invalid.")
    result["selected_candidate"] = choose_candidate(result)
    result["status"] = ("fail" if not all((numerical_ok, bounds_ok, torque_ok, identity_ok, prefault_ok)) else "inconclusive" if not healthy_ok
                        else "pass" if result["selected_candidate"] else "fail")
    if result["status"] == "fail" and not result["reasons"]:
        result["reasons"].append("No included gain passed all predeclared recovery checks.")
    return result


def choose_candidate(summary):
    """Return the least severe passing included gain, or None."""
    return next((key for key in CANDIDATE_ORDER if summary.get("candidates", {}).get(key, {}).get("passed")), None)
