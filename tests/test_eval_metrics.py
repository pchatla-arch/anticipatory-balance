# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Adversarial, simulator-free checks for sampled evaluation bookkeeping.

Run from the repository root:
  python -m unittest discover -s tests -p 'test_eval_metrics.py' -v
"""
import unittest

import numpy as np

from foresight.eval.metrics import brace_readiness, post_contact_survival, summarize_trajectory


class PostContactSurvivalTests(unittest.TestCase):

  def test_missing_contact_precontact_failure_and_short_followup(self):
    result = post_contact_survival(
        [-1, 3, 8, 8, 8], [-1, 2, -1, 8, 9],
        num_steps=10, dt=0.1, window_s=0.3)
    np.testing.assert_equal(result["survived_post_contact_window"],
                            [np.nan, np.nan, np.nan, 0, 0])
    np.testing.assert_array_equal(result["post_contact_eligible"], [0, 0, 1, 1, 1])
    np.testing.assert_array_equal(result["pre_contact_failure"], [0, 1, 0, 0, 0])
    np.testing.assert_array_equal(result["post_contact_censored"], [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(result["post_contact_observed"], [0, 0, 0, 1, 1])
    np.testing.assert_allclose(result["post_contact_followup_s"][2:], [0.1, 0, 0.1])

  def test_inclusive_endpoint_and_later_failure(self):
    result = post_contact_survival(
        [2, 2, 6, 7], [5, 6, -1, -1], num_steps=10, dt=0.1, window_s=0.3)
    # Three intervals are .3 s, despite floating point roundoff. A failure on
    # that endpoint fails; the next-step failure does not invalidate the window.
    np.testing.assert_equal(result["survived_post_contact_window"], [0, 1, 1, np.nan])
    np.testing.assert_array_equal(result["post_contact_censored"], [0, 0, 0, 1])

  def test_last_sample_contact_is_not_extra_followup(self):
    result = post_contact_survival(
        [4, 4], [-1, 4], num_steps=5, dt=0.02, window_s=0.02)
    np.testing.assert_equal(result["survived_post_contact_window"], [np.nan, 0])
    np.testing.assert_equal(result["post_contact_followup_s"], [0, 0])

  def test_zero_duration_still_detects_failure_at_contact(self):
    result = post_contact_survival(
        [2, 2], [-1, 2], num_steps=3, dt=0.1, window_s=0)
    np.testing.assert_equal(result["survived_post_contact_window"], [1, 0])

  def test_invalid_event_sentinels_and_durations_raise(self):
    for invalid in (-2, 5, np.nan, 1.5):
      with self.subTest(term=invalid), self.assertRaises(ValueError):
        post_contact_survival([1], [invalid], num_steps=5, dt=0.1, window_s=0.3)
      with self.subTest(contact=invalid), self.assertRaises(ValueError):
        post_contact_survival([invalid], [-1], num_steps=5, dt=0.1, window_s=0.3)
    for dt, window in ((0, 0.3), (0.1, -1), (0.1, np.nan)):
      with self.subTest(dt=dt, window=window), self.assertRaises(ValueError):
        post_contact_survival([1], [-1], num_steps=5, dt=dt, window_s=window)


class TrajectorySummaryTests(unittest.TestCase):

  def test_post_failure_displacement_and_actions_cannot_improve_metrics(self):
    pos = np.zeros((5, 2, 3))
    pos[:, :, 0] = np.array([0, 1, 100, 1000, 10000])[:, None]
    forward = np.zeros_like(pos)
    forward[..., 0] = 1
    result = summarize_trajectory(pos, forward, np.ones((5, 2)), [2, 0], dt=0.1)
    np.testing.assert_equal(result["progress_before_failure_m"], [1, 0])
    np.testing.assert_allclose(result["progress_observed_s"], [0.1, 0])
    np.testing.assert_allclose(result["brace_duration_through_failure_s"], [0.3, 0.1])
    np.testing.assert_allclose(result["control_duration_through_failure_s"], [0.3, 0.1])

  def test_rotated_heading_and_missing_reset_interval(self):
    pos = np.zeros((3, 1, 3))
    pos[:, 0, 1] = [10, 11, 13]
    forward = np.zeros_like(pos)
    forward[..., 1] = 2  # Heading magnitude must not scale distance.
    result = summarize_trajectory(pos, forward, [[0], [1], [1]], [-1], dt=0.2)
    np.testing.assert_equal(result["progress_before_failure_m"], [3])
    np.testing.assert_allclose(result["progress_observed_s"], [0.4])
    np.testing.assert_allclose(result["brace_duration_through_failure_s"], [0.4])
    np.testing.assert_allclose(result["control_duration_through_failure_s"], [0.6])

  def test_invalid_post_failure_states_do_not_contaminate_valid_prefix(self):
    pos = np.zeros((4, 2, 3))
    pos[1, :, 0] = 1
    pos[2:] = np.nan
    forward = np.zeros_like(pos)
    forward[..., 0] = 1
    result = summarize_trajectory(pos, forward, np.zeros((4, 2)), [2, -1], dt=0.1)
    self.assertEqual(result["progress_before_failure_m"][0], 1)
    self.assertTrue(np.isnan(result["progress_before_failure_m"][1]))

  def test_single_poststep_sample_has_exposure_but_no_measured_displacement(self):
    result = summarize_trajectory(
        np.array([[[20, 0, 0.6]]]), np.array([[[1, 0, 0]]]), [[1]], [-1], dt=0.02)
    self.assertEqual(result["progress_before_failure_m"][0], 0)
    self.assertEqual(result["brace_duration_through_failure_s"][0], 0.02)

  def test_nan_brace_flag_is_not_silently_truthy(self):
    with self.assertRaises(ValueError):
      summarize_trajectory(np.zeros((1, 1, 3)), np.ones((1, 1, 3)),
                           [[np.nan]], [-1], dt=0.1)


class BraceReadinessTests(unittest.TestCase):

  def good_trajectory(self, steps=10, episodes=1):
    pelvis = np.zeros((steps, episodes, 3))
    pelvis[..., 2] = 0.6
    feet = np.zeros((steps, episodes, 2, 3))
    feet[:, :, 0, 1] = 0.15
    feet[:, :, 1, 1] = -0.15
    forward = np.zeros_like(pelvis)
    forward[..., 0] = 1
    return dict(pelvis_pos=pelvis, torso_up=np.ones((steps, episodes)),
                feet_pos=feet, feet_contact=np.ones((steps, episodes, 2), bool),
                brace_active=np.ones((steps, episodes), bool),
                term_step=np.full(episodes, -1), dt=0.02,
                target_pelvis_height=0.6, height_tolerance=0.025,
                min_torso_up=0.95, min_foot_separation=0.25,
                dwell_s=0.1, pelvis_fwd=forward)

  def test_five_good_samples_are_point_one_seconds_from_action_onset(self):
    inputs = self.good_trajectory()
    inputs["brace_active"][:2] = False
    result = brace_readiness(**inputs)
    self.assertEqual(result["brace_ready_step"][0], 6)
    self.assertAlmostEqual(result["brace_ready_time_s"][0], 0.14)
    self.assertAlmostEqual(result["brace_ready_latency_s"][0], 0.10)

  def test_pose_dropout_resets_dwell_and_inactive_gap_starts_new_bout(self):
    inputs = self.good_trajectory(steps=12, episodes=2)
    inputs["torso_up"][3, 0] = 0.8
    inputs["brace_active"][3, 1] = False
    result = brace_readiness(**inputs)
    np.testing.assert_equal(result["brace_ready_step"], [8, 8])
    np.testing.assert_allclose(result["brace_ready_latency_s"], [0.18, 0.10])

  def test_support_not_just_foot_positions_is_required(self):
    inputs = self.good_trajectory()
    inputs["feet_contact"][:, :, 1] = False
    result = brace_readiness(**inputs)
    self.assertFalse(result["brace_ready"][0])
    for key in ("brace_ready_step", "brace_ready_time_s", "brace_ready_latency_s"):
      self.assertTrue(np.isnan(result[key][0]))

  def test_lateral_width_is_not_fore_aft_distance(self):
    inputs = self.good_trajectory(episodes=2)
    # Both feet are .4 m apart longitudinally, but have no lateral stance width.
    inputs["feet_pos"][:, 0, :, 1] = 0
    inputs["feet_pos"][:, 0, 0, 0] = 0.2
    inputs["feet_pos"][:, 0, 1, 0] = -0.2
    # Rotate robot and stance together by 90 degrees: this should remain ready.
    inputs["feet_pos"][:, 1, :, 1] = 0
    inputs["feet_pos"][:, 1, 0, 0] = -0.15
    inputs["feet_pos"][:, 1, 1, 0] = 0.15
    inputs["pelvis_fwd"][:, 1] = [0, 1, 0]
    result = brace_readiness(**inputs)
    np.testing.assert_equal(result["brace_ready"], [False, True])

  def test_base_height_target_and_nonfinite_posture_cannot_be_bypassed(self):
    inputs = self.good_trajectory(episodes=4)
    inputs["pelvis_pos"][:, 0, 2] = 0.45  # COM-like value is not the .6 base target.
    inputs["pelvis_pos"][:, 1, 2] = np.nan
    inputs["torso_up"][:, 2] = np.nan
    inputs["feet_pos"][:, 3, 0, 1] = np.nan
    result = brace_readiness(**inputs)
    np.testing.assert_equal(result["brace_ready"], np.zeros(4, bool))
    self.assertTrue(np.isnan(result["brace_ready_latency_s"]).all())

  def test_failure_sample_cannot_complete_dwell_and_late_readiness_is_excluded(self):
    inputs = self.good_trajectory(episodes=3)
    inputs["term_step"] = np.array([4, 0, 5])
    result = brace_readiness(**inputs)
    np.testing.assert_equal(result["brace_ready"], [False, False, True])
    np.testing.assert_equal(result["brace_ready_step"], [np.nan, np.nan, 4])

  def test_short_rollout_does_not_report_zero_latency_when_never_ready(self):
    inputs = self.good_trajectory(steps=4)
    result = brace_readiness(**inputs)
    self.assertFalse(result["brace_ready"][0])
    self.assertTrue(np.isnan(result["brace_ready_latency_s"][0]))

  def test_nonintegral_dwell_rounds_up_and_zero_dwell_still_requires_a_sample(self):
    inputs = self.good_trajectory()
    inputs["dwell_s"] = 0.101
    self.assertEqual(brace_readiness(**inputs)["brace_ready_step"][0], 5)
    inputs["dwell_s"] = 0
    result = brace_readiness(**inputs)
    self.assertEqual(result["brace_ready_step"][0], 0)
    self.assertEqual(result["brace_ready_latency_s"][0], 0.02)

  def test_nan_or_nonbinary_support_flags_and_future_termination_raise(self):
    for invalid in (np.nan, 2, -1):
      inputs = self.good_trajectory()
      inputs["feet_contact"] = inputs["feet_contact"].astype(float)
      inputs["feet_contact"][0, 0, 0] = invalid
      with self.subTest(flag=invalid), self.assertRaises(ValueError):
        brace_readiness(**inputs)
    inputs = self.good_trajectory()
    inputs["term_step"] = [10]
    with self.assertRaises(ValueError):
      brace_readiness(**inputs)


if __name__ == "__main__":
  unittest.main()
