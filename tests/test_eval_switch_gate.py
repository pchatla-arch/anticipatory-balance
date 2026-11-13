"""Simulator-free checks for paired inference and diagnostic output integrity."""
import contextlib
import importlib.util
import io
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

_path = Path(__file__).resolve().parents[1] / "scripts" / "eval_switch_gate.py"
_spec = importlib.util.spec_from_file_location("switch_gate", _path)
gate = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gate)


def row(pair_id=0, fell=False, observed=False, censored=False, survived=math.nan):
  return {
      "batch": 0, "pair_id": pair_id, "fell": fell, "contact": observed or censored,
      "term_step": 4 if fell else -1, "contact_step": 2 if observed or censored else -1,
      "pre_contact_failure": False, "post_contact_eligible": observed or censored,
      "post_contact_observed": observed, "post_contact_censored": censored,
      "survived_post_contact_window": survived, "progress_before_failure_m": 1.,
      "brace_duration_through_failure_s": .2, "failure_inverted": False,
      "failure_self_contact": fell, "failure_nonfinite": False,
      "failure_sustained_tilt": False, "brace_ready": False,
  }


class PairedSummaryTests(unittest.TestCase):
  def test_pairing_uses_ids_and_keeps_rescues_separate_from_new_failures(self):
    reference = [row(0, True), row(1), row(2, True), row(3)]
    treatment = [row(3), row(1, True), row(0), row(2, True)]
    result = gate._paired(reference, treatment)
    self.assertEqual(result["n_pairs"], 4)
    for key in ("both_fail", "neither_fails", "reference_only_fails", "treatment_only_fails"):
      self.assertEqual(result[key], 1)
    self.assertEqual(result["failure_rate_difference_treatment_minus_reference"], 0.)

  def test_duplicate_or_missing_pairs_cannot_silently_enter_comparison(self):
    with self.assertRaises(RuntimeError):
      gate._paired([row(0), row(0)], [row(0), row(1)])
    with self.assertRaises(RuntimeError):
      gate._paired([row(0), row(1)], [row(0), row(2)])

  def test_censored_contact_is_not_counted_as_survival(self):
    result = gate._summarize_arm([
        row(0, observed=True, survived=1.),
        row(1, fell=True, observed=True, survived=0.),
        row(2, censored=True),
    ])
    self.assertEqual(result["n"], 3)
    self.assertEqual(result["post_contact_observed"], 2)
    self.assertEqual(result["post_contact_censored"], 1)
    self.assertEqual(result["post_contact_survival_rate_observed"], .5)

  def test_zero_contact_arm_has_unknown_contact_survival_rate(self):
    self.assertIsNone(gate._summarize_arm([row()])["post_contact_survival_rate_observed"])

  def test_no_projectile_switch_failure_is_flagged_without_claiming_significance(self):
    rows = []
    for controller in gate.CONTROLLERS:
      for condition in gate.CONDITIONS:
        for i in range(2):
          r = row(i, controller == "brace_switch" and condition == "no_projectile" and i == 1)
          r.update(controller=controller, condition=condition)
          rows.append(r)
    args = SimpleNamespace(episodes=2, batches=1, steps=300, lead=.5, hold=.5, recovery_window=2.5)
    result = gate._summary(rows, args, .02, {})
    self.assertEqual(result["status"], "switch_failure")
    self.assertEqual(result["comparison_rollouts"], 12)
    self.assertIn("not a significance test", result["status_rule"])


class IntegrityTests(unittest.TestCase):
  def test_existing_output_is_refused_before_loading_gpu_stack(self):
    with tempfile.TemporaryDirectory() as directory:
      with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
        gate._arguments(["--nominal", "unused", "--brace", "unused", "--out", directory])

  def test_unknown_results_are_null_not_zero_and_csv_schema_cannot_change(self):
    self.assertEqual(gate._json_value({"unknown": math.nan, "zero": 0.}),
                     {"unknown": None, "zero": 0.})
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "rows.csv"
      gate._save_rows(path, [{"id": 1, "outcome": math.nan}])
      before = path.read_text()
      self.assertIn("1,\n", before)
      with self.assertRaises(RuntimeError):
        gate._save_rows(path, [{"different": 2}])
      self.assertEqual(path.read_text(), before)

  def test_numerical_corruption_is_checked_through_first_failure_only(self):
    traj = {"brace_active": np.ones((3, 1), bool)}
    for key in ("nonfinite_policy_input", "nonfinite_action", "nonfinite_physics"):
      traj[key] = np.zeros((3, 1), bool)
    for key in ("pelvis_pos", "pelvis_fwd", "torso_up", "com", "feet_pos", "proj_pos", "proj_vel"):
      traj[key] = np.ones((3, 1, 3))
      traj[key][2] = np.nan
    gate._check_finite_trace(traj, np.array([1]), name="synthetic")
    traj["nonfinite_action"][1, 0] = True
    with self.assertRaises(RuntimeError):
      gate._check_finite_trace(traj, np.array([1]), name="synthetic")


if __name__ == "__main__":
  unittest.main()
