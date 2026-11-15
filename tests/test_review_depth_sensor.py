"""Geometry/queue contracts plus real-JAX detector tests when JAX is available.

Run: python -m unittest discover -s tests -p test_review_depth_sensor.py -v
The optional JAX tests are CPU-compatible; no MuJoCo or GPU is required.
"""
import importlib.util
import unittest

import numpy as np

from foresight.perception import depth_geometry as G


class DepthGeometryTests(unittest.TestCase):
  def setUp(self):
    self.rays = G.pixel_directions(3, 90)
    self.camera = np.array([0., 0., 1.])
    # Camera looks along world +x; its local optical axis is -z.
    self.rotation = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    self.pelvis = np.array([0., 0., .7])

  def detect(self, depth):
    return G.detect_centre(depth, self.camera, self.rotation, self.pelvis, self.rays, .15)

  def test_axial_depth_median_and_radius(self):
    depth = np.full((3, 3), np.inf); depth[1, :] = 2.
    valid, centre, count = self.detect(depth)
    self.assertTrue(valid); self.assertEqual(count, 3)
    np.testing.assert_allclose(centre, [2.15, 0, 1], atol=1e-7)

  def test_median_retains_robust_surface_location(self):
    depth = np.full((3, 3), np.inf); depth[1, :] = [2., 2., 4.]
    valid, centre, count = self.detect(depth)
    self.assertTrue(valid); self.assertEqual(count, 3)
    np.testing.assert_allclose(centre, [2.15, 0, 1], atol=1e-7)

  def test_one_pixel_is_insufficient(self):
    depth = np.full((3, 3), np.inf); depth[1, 1] = 2.
    valid, centre, count = self.detect(depth)
    self.assertFalse(valid); self.assertEqual(count, 1)
    self.assertTrue(np.isfinite(centre).all())

  def test_no_candidate_remains_finite_and_invalid(self):
    depth = np.array([[np.nan, np.inf, -1.], [0., .01, 1000.], [np.inf, np.nan, -5.]])
    valid, centre, count = self.detect(depth)
    self.assertFalse(valid); self.assertEqual(count, 0)
    self.assertTrue(np.isfinite(centre).all())

  def test_geometric_masks_reject_close_and_far(self):
    for d in (.1, 8., 1000.):
      depth = np.full((3, 3), np.inf); depth[1, :] = d
      valid, _, _ = self.detect(depth)
      self.assertFalse(valid)

  def test_camera_pose_is_used(self):
    depth = np.full((3, 3), np.inf); depth[1, :] = 2.
    offset = np.array([1., 2., 0.])
    valid, centre, _ = G.detect_centre(depth, self.camera + offset, self.rotation,
                                      self.pelvis + offset, self.rays, .15)
    self.assertTrue(valid)
    np.testing.assert_allclose(centre, [3.15, 2., 1.], atol=1e-7)

  def test_capture_delivery_grid(self):
    for latency in (0, 1, 5):
      deliveries = [(t, G.delivery_due(t, 5, latency)[1]) for t in range(17)
                    if G.delivery_due(t, 5, latency)[0]]
      self.assertEqual(deliveries, [(c + latency, c) for c in range(0, 17, 5) if c + latency < 17])

  def test_queue_freezes_capture_validity_and_timestamp(self):
    q = G.DetectionQueue(5)
    valid = np.array([False, True]); position = np.array([[1., 0., 1.], [2., 0., 1.]])
    q.capture(0, valid, position, np.array([.03, .03]))
    valid[:] = True; position[:] = 999.
    self.assertIsNone(q.deliver(4))
    capture, received, measured, _ = q.deliver(5)
    self.assertEqual(capture, 0)
    np.testing.assert_array_equal(received, [False, True])
    np.testing.assert_array_equal(measured[:, 0], [1., 2.])

  def test_queue_rejects_silent_late_delivery(self):
    q = G.DetectionQueue(1); q.capture(0, True, [1, 2, 3], .03)
    with self.assertRaises(RuntimeError):
      q.deliver(2)

  def test_measurement_error_includes_bias_and_uses_successes(self):
    a = G.ErrorAccumulator()
    truth = np.array([[2., 0., 1.], [2., 0., 1.], [2., 0., 1.]])
    error = np.array([[.01, .02, .03], [-.01, -.02, -.03], [50., 50., 50.]])
    a.add([True, True, False], truth + error, truth, np.zeros((3, 3)), [True] * 3)
    result = a.summary(); b = next(b for b in result["bins"] if b["range_low_m"] == 2.)
    self.assertEqual(b["n_successful_launched_detections"], 2)
    np.testing.assert_allclose(b["bias_xyz_m"], [0, 0, 0], atol=1e-12)
    np.testing.assert_allclose(b["rmse_xyz_m"], [.01, .02, .03], atol=1e-12)

  def test_prelaunch_detections_are_counted_not_hidden(self):
    a = G.ErrorAccumulator()
    a.add([True], [[2, 0, 1]], [[2, 0, 1]], [[0, 0, 0]], [False])
    s = a.summary()
    self.assertEqual(s["prelaunch_detection_count"], 1)
    self.assertEqual(s["prelaunch_frame_count"], 1)
    self.assertEqual(s["prelaunch_detection_rate"], 1.)
    self.assertEqual(sum(b["n_successful_launched_detections"] for b in s["bins"]), 0)


@unittest.skipUnless(importlib.util.find_spec("jax"), "JAX unavailable; these must run in the pod CPU preflight")
class JaxDetectorTests(unittest.TestCase):
  def test_capture_time_launch_and_phase(self):
    import jax
    import jax.numpy as jnp
    from foresight.perception import detections as D
    for latency in (0, 1, 5):
      cfg = D.DetectorConfig(period=5, latency=latency, sigma0=0., sigma1=0., miss_prob=0.)
      st = D.init_state(cfg)
      step = jax.jit(lambda st, launched, rng: D.step(cfg, st, jnp.array([2., 0., 1.]),
                                                     jnp.zeros(3), jnp.array([1., 0., 0.]), launched, rng))
      seen = []
      for t in range(17):
        st, det = step(st, jnp.array(t >= 1), jax.random.PRNGKey(t))
        if bool(det.valid):
          seen.append((t, int(det.capture_step)))
      expected = [(c + latency, c) for c in (5, 10, 15) if c + latency < 17]
      self.assertEqual(seen, expected)

  def test_buffer_must_cover_delay(self):
    from foresight.perception import detections as D
    with self.assertRaises(ValueError):
      D.init_state(D.DetectorConfig(latency=16, buffer_len=16))

  def test_shared_geometry_jits_and_matches_numpy(self):
    import jax
    import jax.numpy as jnp
    depth = np.full((3, 3), np.inf); depth[1, :] = 2.
    cam = np.array([0., 0., 1.]); mat = np.array([[0., 0., -1.], [1., 0., 0.], [0., -1., 0.]])
    pel = np.array([0., 0., .7]); dirs = G.pixel_directions(3, 90)
    expected = G.detect_centre(depth, cam, mat, pel, dirs, .15)
    got = jax.jit(lambda d: G.detect_centre(d, jnp.asarray(cam), jnp.asarray(mat),
                                           jnp.asarray(pel), jnp.asarray(dirs), .15, xp=jnp))(jnp.asarray(depth))
    self.assertEqual(bool(got[0]), bool(expected[0])); self.assertEqual(int(got[2]), int(expected[2]))
    np.testing.assert_allclose(np.asarray(got[1]), expected[1], atol=1e-6)

  def test_external_detection_controller_jits_with_two_worlds(self):
    import jax
    import jax.numpy as jnp
    from foresight.belief import particle_filter as PF
    from foresight.decision import rules as RU
    from foresight.decision.controller import AnticipationController
    from foresight.perception import detections as D
    detector = D.DetectorConfig(period=5, latency=1, sigma0=.03, sigma1=0., miss_prob=0.)
    model = RU.ResponseModel(-5., 2., -6., 2., -5., 2., -3., .5, True)
    pelvis = jnp.array([[0., 0., .7], [0., 0., .7]])
    velocity = jnp.array([[.5, 0., 0.], [.5, 0., 0.]])
    heading = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    for kind in ("ttc_threshold", "belief_lead", "belief_wait", "belief_voi"):
      controller = AnticipationController(detector, PF.PFConfig(n_particles=8),
          RU.RuleConfig(kind=kind, release_mode="deadline", horizon_steps=100), model)
      carry = controller.init(2, jax.random.PRNGKey(7))
      step = jax.jit(controller.step_with_detections)
      for t in range(8):
        due, cap = G.delivery_due(t, 5, 1)
        position = jnp.array([[3. - .1 * max(cap, 0), 0., 1.],
                              [3. - .1 * max(cap, 0), 1.5, 1.]])
        carry, action, logs = step(carry, jnp.full(2, due), position,
                                   jnp.full(2, .03), jnp.full(2, cap, jnp.int32),
                                   pelvis, velocity, t, jax.random.PRNGKey(t), pelvis_fwd=heading)
        self.assertEqual(action.shape, (2,))
        self.assertEqual(carry.pf.mu.shape, (2, 6))
        self.assertEqual(logs["release_step"].shape, (2,))
        self.assertFalse(np.isnan(np.asarray(logs["cost_brace"])).any())
        self.assertFalse(np.isnan(np.asarray(logs["cost_continue"])).any())
        np.testing.assert_array_equal(np.asarray(logs["next_delivery_step"] - logs["next_capture_step"]), [1, 1])
        if t == 0:
          self.assertFalse(np.asarray(action).any())


if __name__ == "__main__":
  unittest.main()
