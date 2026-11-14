"""Run without the simulator: python -m unittest discover -s tests -p test_detector_cadence.py."""
import importlib.util
import math
import unittest

from foresight.perception.cadence import delivery_due, validate_detector_timing


class CadenceTests(unittest.TestCase):
  def test_existing_integer_schedules_are_unchanged(self):
    for period in (1, 2, 5, 10):
      actual = [s for s in range(300) if delivery_due(s, period)]
      self.assertEqual(actual, list(range(0, 300, period)))

  def test_twenty_hz_has_alternating_grid_intervals(self):
    actual = [s for s in range(16) if delivery_due(s, 2.5)]
    self.assertEqual(actual, [0, 3, 5, 8, 10, 13, 15])

  def test_exact_counts_in_each_second(self):
    for period, expected in ((1.0, 50), (2.5, 20), (5.0, 10)):
      for second in range(6):
        actual = sum(delivery_due(s, period) for s in range(50*second, 50*(second+1)))
        self.assertEqual(actual, expected, (period, second))

  def test_invalid_periods_fail_clearly(self):
    for period in (0, -1, 0.5, math.inf, math.nan):
      with self.assertRaisesRegex(ValueError, "period"):
        validate_detector_timing(period, 0, 16)

  def test_latency_buffer_boundaries(self):
    validate_detector_timing(2.5, 0, 1)
    validate_detector_timing(2.5, 15, 16)
    validate_detector_timing(2.5, 16, 17)
    for latency, length, message in ((-1, 16, "nonnegative"),
                                      (16, 16, "latency \\+ 1"),
                                      (0, 0, "positive"),
                                      (1.5, 16, "integers")):
      with self.assertRaisesRegex(ValueError, message):
        validate_detector_timing(2.5, latency, length)


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is required for detector integration")
class DetectorIntegrationTests(unittest.TestCase):
  def test_jitted_delayed_measurements_preserve_capture_timestamp(self):
    import jax
    import jax.numpy as jp
    import numpy as np
    from foresight.perception import detections as D

    for period, latency in ((1.0, 0), (2.5, 7), (5.0, 15)):
      cfg = D.DetectorConfig(period=period, latency=latency, sigma0=0, sigma1=0,
                             miss_prob=0, buffer_len=16)
      def body(state, t):
        return D.step(cfg, state, jp.array([2.0 + t * .001, 0, .8]),
                      jp.array([0, 0, .8]), jp.array([1., 0, 0]), True,
                      jax.random.fold_in(jax.random.PRNGKey(0), t))
      _, detections = jax.jit(lambda: jax.lax.scan(body, D.init_state(cfg), jp.arange(50)))()
      valid = np.asarray(detections.valid)
      expected = np.array([bool(delivery_due(t, period)) and t >= latency for t in range(50)])
      np.testing.assert_array_equal(valid, expected)
      captured = np.arange(50)[valid] - latency
      np.testing.assert_array_equal(np.asarray(detections.capture_step)[valid], captured)
      np.testing.assert_allclose(np.asarray(detections.pos)[valid, 0], 2.0 + captured * .001,
                                 rtol=0, atol=1e-6)


if __name__ == "__main__":
  unittest.main()
