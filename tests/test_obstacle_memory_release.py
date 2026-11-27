"""Development regressions for G2b's actual missed-FOV detour stall.

A single physical stationary cylinder is the only modeled obstacle. These are
geometric/logic unit tests, not MuJoCo walking/contact/vision guarantees.
"""
import unittest

from foresight.adaptive.obstacle_depth import DepthDetection, ObstacleNavigator


class PostDetourMemoryTests(unittest.TestCase):
    def setup_nav(self):
        nav = ObstacleNavigator((5., 0.))
        start = (0.331, -0.322)
        for t in (0, 5):
            nav.ingest(DepthDetection((2.839, -0.016), 2.5, .30, 20, 1., t))
        self.assertEqual(nav.command(*start, 0., 6)[2], "detour")
        first = nav.follower.route.waypoints[0]
        nav.ingest(None, observed_step=300)
        nav.command(*first, 0., 300)
        self.assertGreaterEqual(nav.follower.index, 1)
        return nav

    def test_exact_observed_g2b_stall_releases_on_clear_future_route(self):
        nav = self.setup_nav()
        nav.ingest(None, observed_step=515)  # fresh blank frame, not missing frame
        vx, yaw, reason = nav.command(1.963219, 1.569092, 0.28, 515)
        self.assertEqual(reason, "detour_memory")
        self.assertGreater(vx, 0.)
        self.assertLessEqual(vx, 0.4)
        self.assertLessEqual(abs(yaw), 0.5)

    def test_unsafe_remaining_path_still_stops(self):
        nav = self.setup_nav()
        nav.ingest(None, observed_step=515)
        vx, yaw, reason = nav.command(2.7, 0.4, 0.28, 515)
        self.assertEqual((vx, yaw, reason),
                         (0.0, 0.0, "stop_lost_obstacle"))

    def test_stale_camera_frame_still_stops_on_clear_route(self):
        nav = self.setup_nav()
        vx, yaw, reason = nav.command(1.963219, 1.569092, 0.28, 515)
        self.assertEqual((vx, yaw, reason),
                         (0.0, 0.0, "stop_stale_depth"))

    def test_before_first_waypoint_near_lost_object_stops(self):
        nav = ObstacleNavigator((5., 0.))
        for t in (0, 5):
            nav.ingest(DepthDetection((2.839, -0.016), 2.5, .3, 20, 1., t))
        nav.command(0.331, -0.322, 0., 6)
        self.assertEqual(nav.follower.index, 0)
        nav.ingest(None, observed_step=515)
        self.assertEqual(nav.command(1.963219, 1.569092, 0.28, 515)[2],
                         "stop_lost_obstacle")

    def test_too_close_to_stale_estimate_does_not_release(self):
        nav = self.setup_nav()
        nav.ingest(None, observed_step=515)
        self.assertEqual(nav.command(2.81, 0.05, 0.28, 515)[2],
                         "stop_lost_obstacle")


if __name__ == "__main__":
    unittest.main()
