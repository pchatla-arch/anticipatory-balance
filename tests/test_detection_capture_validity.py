"""CPU history checks plus optional integration against the real JAX detector."""
import importlib.util
import unittest

from foresight.perception.cadence import delivery_due
from foresight.perception.capture import captured_launch_valid


class CaptureValidityTests(unittest.TestCase):
  def test_false_initialized_history_and_negative_capture(self):
    self.assertFalse(captured_launch_valid([False] * 4, 0))
    # A wrapped occupied slot still cannot validate a capture before step zero.
    self.assertFalse(captured_launch_valid([True] * 4, -1))

  def test_delayed_delivery_never_exposes_prelaunch_frames(self):
    for period in (1.0, 2.5, 5.0):
      for latency in (1, 3, 15):
        with self.subTest(period=period, latency=latency):
          history = [False] * 16
          launch_step = 7
          delivered = []
          for step in range(50):
            history[step % len(history)] = step >= launch_step
            cap = step - latency
            if delivery_due(step, period) and captured_launch_valid(history, cap):
              delivered.append(cap)
          self.assertTrue(delivered)
          self.assertTrue(all(cap >= launch_step for cap in delivered))
          self.assertEqual(delivered, [step - latency for step in range(50)
                                      if delivery_due(step, period) and step - latency >= launch_step])

  def test_zero_latency_matches_current_validity(self):
    history = [False] * 4
    sequence = [False, False, True, True, False, True, False, False, True]
    for step, launched in enumerate(sequence):
      history[step % len(history)] = launched
      self.assertEqual(bool(captured_launch_valid(history, step)), launched)

  def test_ring_wrap_preserves_delayed_launch_transitions(self):
    # Multiple on/off intervals exercise stale True bits and writes across wraps.
    sequence = [False, True, True, False, False, True, False] * 5
    history = [False] * 4
    latency = 3  # Longest safe delay for this buffer.
    for step, launched in enumerate(sequence):
      history[step % len(history)] = launched
      cap = step - latency
      self.assertEqual(bool(captured_launch_valid(history, cap)),
                       sequence[cap] if cap >= 0 else False)


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX is required for detector integration")
class DetectorCaptureIntegrationTests(unittest.TestCase):
  def test_jitted_detector_rejects_parked_frames_and_preserves_cadence(self):
    import jax
    import jax.numpy as jp
    import numpy as np
    from foresight.perception import detections as D

    for period, latency, length in ((1.0, 0, 1), (1.0, 3, 4), (2.5, 3, 4), (5.0, 15, 16)):
      with self.subTest(period=period, latency=latency, buffer_len=length):
        cfg = D.DetectorConfig(period=period, latency=latency, sigma0=0, sigma1=0,
                               miss_prob=0, buffer_len=length)
        state = D.init_state(cfg)
        np.testing.assert_array_equal(np.asarray(state.buf_launched), np.zeros(length, dtype=bool))
        # Both parked and active objects are in FOV/range: launch gating alone
        # must prevent delivery of the parked position after the object launches.
        launches = jp.asarray(([False] * 7 + [True] * 13 + [False] * 7 + [True] * 23))
        positions = jp.where(launches[:, None],
                             jp.stack([2.0 + jp.arange(50) * .001, jp.zeros(50), jp.full(50, .8)], axis=1),
                             jp.array([4.0, 0, .8]))
        def body(st, t):
          return D.step(cfg, st, positions[t], jp.array([0, 0, .8]),
                        jp.array([1., 0, 0]), launches[t],
                        jax.random.fold_in(jax.random.PRNGKey(0), t))
        _, detections = jax.jit(lambda: jax.lax.scan(body, state, jp.arange(50)))()
        active = np.asarray(launches)
        expected = np.array([bool(delivery_due(t, period)) and t >= latency and active[t-latency]
                             for t in range(50)])
        valid = np.asarray(detections.valid)
        np.testing.assert_array_equal(valid, expected)
        captured = np.arange(50)[valid] - latency
        np.testing.assert_array_equal(np.asarray(detections.capture_step)[valid], captured)
        np.testing.assert_allclose(np.asarray(detections.pos)[valid], np.asarray(positions)[captured],
                                   rtol=0, atol=1e-6)
        self.assertTrue(np.all(np.asarray(detections.pos)[valid, 0] < 3.0))


if __name__ == "__main__":
  unittest.main()
