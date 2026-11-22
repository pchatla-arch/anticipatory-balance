"""Synthetic paired trials exercise the declared adaptive-execution gate."""
import copy
import json
import unittest

from foresight.adaptive.gate_analysis import analyze_phase, choose_candidate


def trials(phase="screen", n=32, gains=("gain_085", "gain_070")):
    rows = []
    for condition in ("healthy", "identity", *(f"{g}_{a}" for g in gains for a in ("plain", "oracle"))):
        for i in range(n):
            fault = condition.startswith("gain")
            oracle = condition == "identity" or condition.endswith("_oracle")
            rows.append(dict(phase=phase, condition=condition, scenario_id=f"s{i}",
                gain=int(condition[5:8]) / 100 if fault else 1.0, oracle=oracle,
                speed_cmd_mps=0.5, onset_step=75 + i % 51, fell=False, nonfinite=False,
                termination_step=-1, postfault_loss=(0.40 if fault and not oracle else 0.15 if fault else 0.10) + 0.001 * i,
                progress_m=2.0, correction_bound_fraction=0.05 if fault and oracle else 0.0,
                max_correction_rad=0.1 if fault and oracle else 0.0,
                max_slew_rad_s=0.4 if fault and oracle else 0.0))
    return rows


class AdaptiveGateAnalysisTests(unittest.TestCase):
    def test_selects_least_severe_passing_gain_and_confirms_fresh_rows(self):
        screen = analyze_phase(trials())
        self.assertEqual(screen["status"], "pass")
        self.assertEqual(choose_candidate(screen), "gain_085")
        confirmation = analyze_phase(trials("confirm", 128, ("gain_085",)), confirmation=True)
        self.assertEqual(confirmation["status"], "pass")
        self.assertGreater(confirmation["candidates"]["gain_085"]["paired_loss_reduction_ci95"][0], 0)
        json.dumps(confirmation, allow_nan=False)

    def test_no_degradation_cannot_pass_from_tiny_improvement(self):
        rows = trials(gains=("gain_085",))
        for r in rows:
            r["postfault_loss"] = 0.1 if r["condition"] != "gain_085_plain" else 0.101
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["candidates"]["gain_085"]["checks"]["degradation"])

    def test_catastrophic_fault_fails_even_when_oracle_recovers(self):
        rows = trials(gains=("gain_085",))
        for r in rows:
            if r["condition"] == "gain_085_plain":
                r.update(fell=True, termination_step=130, postfault_loss=4.0)
        result = analyze_phase(rows)
        self.assertFalse(result["candidates"]["gain_085"]["checks"]["recoverable"])
        self.assertIsNone(result["selected_candidate"])

    def test_oracle_fall_regression_fails_despite_better_loss(self):
        rows = trials(gains=("gain_085",))
        next(r for r in rows if r["condition"] == "gain_085_oracle").update(fell=True, termination_step=399)
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["candidates"]["gain_085"]["checks"]["no_fall_regression"])

    def test_identity_mismatch_stops_selection(self):
        rows = trials()
        next(r for r in rows if r["condition"] == "identity")["progress_m"] += 0.01
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "fail")
        self.assertIsNone(choose_candidate(result))

    def test_repeat_probe_separates_backend_noise_from_code_path_differences(self):
        base = trials()
        # Without the probe the controls report no reproducibility verdict.
        plain = analyze_phase(base)
        self.assertIsNone(plain["controls"]["run_to_run_reproducible"])
        self.assertIsNone(plain["controls"]["repeat_max_difference"])

        def with_repeat(rows, jitter=0.0):
            rows = copy.deepcopy(rows)
            for r in [r for r in rows if r["condition"] == "healthy"]:
                probe = dict(r, condition="healthy_repeat")
                probe["progress_m"] += jitter
                rows.append(probe)
            return rows

        exact = analyze_phase(with_repeat(base))
        self.assertEqual(exact["status"], "pass")
        self.assertTrue(exact["controls"]["run_to_run_reproducible"])
        self.assertEqual(exact["controls"]["repeat_max_difference"], 0.0)
        self.assertNotIn("healthy_repeat", exact["candidates"])
        self.assertEqual(exact["arms"]["healthy_repeat"]["n"], 32)

        # Identity differs while the repeat is exact: the adapter changed the code path.
        rows = with_repeat(base)
        next(r for r in rows if r["condition"] == "identity")["progress_m"] += 0.01
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "fail")
        self.assertTrue(result["controls"]["run_to_run_reproducible"])
        self.assertIn("changes the executed code path", " ".join(result["reasons"]))

        # Identity and repeat both differ: the backend itself is not reproducible.
        rows = with_repeat(base, jitter=0.004)
        next(r for r in rows if r["condition"] == "identity")["progress_m"] += 0.005
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["controls"]["run_to_run_reproducible"])
        self.assertAlmostEqual(result["controls"]["repeat_max_difference"], 0.004)
        self.assertIn("not run-to-run reproducible", " ".join(result["reasons"]))
        self.assertIsNone(choose_candidate(result))
        json.dumps(result, allow_nan=False)

    def test_control_bounds_are_hard_gate_even_with_strong_recovery(self):
        for field, limit in (("max_correction_rad", 0.2), ("max_slew_rad_s", 2.0)):
            with self.subTest(field=field):
                rows = trials()
                arm = next(r for r in rows if r["condition"] == "gain_070_oracle")
                arm[field] = limit + 5e-6
                self.assertEqual(analyze_phase(rows)["status"], "pass")
                arm[field] = limit + 2e-5
                result = analyze_phase(rows)
                self.assertEqual(result["status"], "fail")
                self.assertFalse(result["controls"]["bounds"])
                self.assertIsNone(result["selected_candidate"])
                self.assertEqual(result["arms"]["gain_070_oracle"]["control_bound_violations"], 1)

    def test_joint_torque_limits_are_only_claimed_when_recorded(self):
        rows = trials()
        self.assertIsNone(analyze_phase(rows)["controls"]["torque_limits"])
        for row in rows:
            row["torque_limit_excess_nm"] = 0.0
            row["max_abs_joint_actuation_nm"] = 50.0
        self.assertTrue(analyze_phase(rows)["controls"]["torque_limits"])
        next(r for r in rows if r["condition"] == "gain_070_oracle")["torque_limit_excess_nm"] = 0.1
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "fail")
        self.assertFalse(result["controls"]["torque_limits"])

    def test_unreliable_healthy_control_is_inconclusive(self):
        rows = trials()
        for r in rows:
            if r["condition"] in ("healthy", "identity") and int(r["scenario_id"][1:]) < 4:
                r.update(fell=True, termination_step=150)
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "inconclusive")
        self.assertIsNone(result["selected_candidate"])

    def test_prefault_failures_are_retained_and_not_attributed_to_fault(self):
        rows = trials()
        for row in rows:
            if row["scenario_id"] == "s0":
                row.update(fell=True, termination_step=50, postfault_loss=4.0)
        result = analyze_phase(rows)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["arms"]["gain_085_plain"]["pre_fault_failures"], 1)
        self.assertEqual(result["arms"]["gain_085_plain"]["postfault_falls"], 0)
        rows[-1].update(fell=True, termination_step=50, postfault_loss=4.0)
        self.assertEqual(analyze_phase(rows)["status"], "fail")

    def test_any_numerical_failure_prevents_claim_without_dropping_row(self):
        for mutation in ({"nonfinite": True}, {"progress_m": float("nan")}):
            with self.subTest(mutation=mutation):
                rows = trials()
                next(r for r in rows if r["condition"] == "gain_070_oracle").update(mutation)
                result = analyze_phase(rows)
                self.assertEqual(result["status"], "fail")
                self.assertIsNone(result["selected_candidate"])
                self.assertEqual(result["arms"]["gain_070_oracle"]["invalid_episodes"], 1)
                self.assertEqual(result["arms"]["gain_070_oracle"]["n"], 32)
                json.dumps(result, allow_nan=False)

    def test_confirmation_requires_positive_paired_interval(self):
        rows = trials("confirm", 32, ("gain_085",))
        for r in rows:
            i = int(r["scenario_id"][1:])
            if r["condition"] in ("healthy", "identity"):
                r["postfault_loss"] = 1.0
            elif r["condition"] == "gain_085_plain":
                r["postfault_loss"] = 1.4
            else:
                r["postfault_loss"] = 0.0 if i < 24 else 4.0
        result = analyze_phase(rows, confirmation=True)
        candidate = result["candidates"]["gain_085"]
        self.assertTrue(candidate["checks"]["recovered_half_gap"])
        self.assertFalse(candidate["checks"]["positive_paired_ci"])
        self.assertEqual(result["status"], "fail")

    def test_pairing_and_metadata_errors_are_rejected(self):
        base = trials()
        bad = []
        rows = copy.deepcopy(base); rows.pop(); bad.append(rows)
        rows = copy.deepcopy(base); rows.append(dict(rows[0])); bad.append(rows)
        rows = copy.deepcopy(base); rows[-1]["onset_step"] += 1; bad.append(rows)
        rows = copy.deepcopy(base); rows[-1]["oracle"] = "False"; bad.append(rows)
        rows = copy.deepcopy(base); rows[-1]["phase"] = "confirm"; bad.append(rows)
        for rows in bad:
            with self.subTest(last=rows[-1]):
                with self.assertRaises(ValueError):
                    analyze_phase(rows)

    def test_csv_booleans_are_not_interpreted_by_string_truthiness(self):
        rows = trials()
        for row in rows:
            for key in ("fell", "nonfinite", "oracle"):
                row[key] = str(row[key])
        self.assertEqual(analyze_phase(rows)["status"], "pass")


if __name__ == "__main__":
    unittest.main()
