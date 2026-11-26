"""Pure-NumPy tests for the perception/avoidance selection (no physics claims)."""
import unittest
import numpy as np
from foresight.adaptive.obstacle_depth import (
    DepthDetection, detect_obstacle_points, ObstacleNavigator
)


def scene_points():
    """Synthetic metric backprojection: ground + 4x4 obstacle surface."""
    p = np.zeros((12, 12, 3), dtype=float)
    p[..., 0] = 4.0
    p[..., 1] = 0.0
    p[..., 2] = 0.0   # ground excluded
    p[3:7, 4:8, 0] = 2.0
    p[3:7, 4:8, 1] = np.linspace(-.12, .12, 4)[None, :]
    p[3:7, 4:8, 2] = .9
    return p


class DepthSelectionTests(unittest.TestCase):
    def test_nearest_depth_blob_without_truth(self):
        det = detect_obstacle_points(scene_points(), (0, 0), 0., step=5)
        self.assertIsNotNone(det)
        self.assertEqual(det.pixel_count, 16)
        self.assertAlmostEqual(det.center_xy[0], 2.3, places=2)

    def test_ground_is_not_obstacle(self):
        p = scene_points()
        p[..., 2] = 0.
        self.assertIsNone(detect_obstacle_points(p, (0, 0), 0., step=1))

    def test_single_pixel_noise_rejected(self):
        p = scene_points()
        p[3:7, 4:8, 2] = 0.
        p[4, 4, 2] = .7
        self.assertIsNone(detect_obstacle_points(p, (0, 0), 0., step=1))

    def test_behind_robot_rejected(self):
        p = scene_points()
        p[..., 0] = -2.
        self.assertIsNone(detect_obstacle_points(p, (0, 0), 0., step=1))

    def test_nonfinite_depth_ignored(self):
        p = scene_points()
        p[3, 4] = np.nan
        det = detect_obstacle_points(p, (0, 0), 0., step=1)
        self.assertEqual(det.pixel_count, 15)

    def test_requires_repeat_observation(self):
        nav = ObstacleNavigator((6., 0.))
        nav.ingest(DepthDetection((3, 0), 3., .3, 20, 1., 0))
        self.assertIsNone(nav.obstacle)
        nav.ingest(DepthDetection((3.1, 0), 3.1, .3, 20, 1., 5))
        self.assertIsNotNone(nav.obstacle)

    def test_stale_depth_stop(self):
        nav = ObstacleNavigator((6., 0.))
        nav.ingest(DepthDetection((3, 0), 3., .3, 20, 1., 1))
        self.assertEqual(nav.command(0., 0., 0., 25)[2], "stop_stale_depth")

    def test_near_unconfirmed_stop(self):
        nav = ObstacleNavigator((6., 0.))
        nav.ingest(DepthDetection((1.2, 0), 1.2, .3, 20, 1., 1))
        self.assertEqual(nav.command(0., 0., 0., 2)[2], "stop_unconfirmed_obstacle")

    def test_confirmed_detour(self):
        nav = ObstacleNavigator((6., 0.))
        nav.ingest(DepthDetection((3, 0), 3., .3, 20, 1., 1))
        nav.ingest(DepthDetection((3.05, 0), 3.05, .3, 20, 1., 5))
        vx, yaw, reason = nav.command(0, 0, 0, 6)
        self.assertEqual(reason, "detour")
        self.assertGreaterEqual(vx, 0.)

    def test_missed_observation_does_not_clear_accepted_obstacle(self):
        nav = ObstacleNavigator((6., 0.))
        nav.ingest(DepthDetection((3, 0), 3., .3, 20, 1., 1))
        nav.ingest(DepthDetection((3, 0), 3., .3, 20, 1., 5))
        nav.ingest(None)
        self.assertIsNotNone(nav.obstacle)


if __name__ == "__main__":
    unittest.main()
