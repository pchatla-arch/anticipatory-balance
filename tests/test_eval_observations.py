"""Regression checks for stale commands at G1 policy switches."""
import unittest

import numpy as np

from foresight.eval.observations import replace_command_observation


class CommandObservationTests(unittest.TestCase):
  def test_batched_switch_updates_both_network_inputs_only(self):
    state = np.arange(2 * 103, dtype=np.float32).reshape(2, 103)
    state[:, 9:12] = [0.5, 0.0, 0.0]
    privileged = np.concatenate((state, np.ones((2, 20), np.float32)), axis=1)
    obs = {"state": state, "privileged_state": privileged, "other": object()}
    before = {k: v.copy() for k, v in obs.items() if isinstance(v, np.ndarray)}
    command = np.array([[0., 0., 0.], [0.5, 0., 0.]])
    new = replace_command_observation(obs, command, concatenate=np.concatenate)
    for key in before:
      np.testing.assert_array_equal(obs[key], before[key])
      np.testing.assert_array_equal(new[key][:, 9:12], command)
      np.testing.assert_array_equal(new[key][:, :9], before[key][:, :9])
      np.testing.assert_array_equal(new[key][:, 12:], before[key][:, 12:])
      self.assertEqual(new[key].dtype, before[key].dtype)
    self.assertIs(new["other"], obs["other"])

  def test_return_to_walking_is_visible_before_next_action(self):
    obs = {"state": np.zeros(103, dtype=np.float32)}
    walking = np.array([0.5, 0.1, -0.2], dtype=np.float32)
    out = replace_command_observation(obs, walking, concatenate=np.concatenate)
    # A policy reading the command immediately sees the resumed command.
    action_from_command = lambda observation: observation["state"][9:12].copy()
    np.testing.assert_array_equal(action_from_command(out), walking)
    np.testing.assert_array_equal(obs["state"][9:12], np.zeros(3))

  def test_rejects_mismatched_batch_instead_of_broadcasting_silently(self):
    with self.assertRaises(ValueError):
      replace_command_observation(
          {"state": np.zeros((2, 103))}, np.zeros(3), concatenate=np.concatenate)

  def test_rejects_invalid_privileged_prefix(self):
    with self.assertRaises(ValueError):
      replace_command_observation(
          {"state": np.zeros(103), "privileged_state": np.zeros(12)},
          np.zeros(3), concatenate=np.concatenate)


if __name__ == "__main__":
  unittest.main()
