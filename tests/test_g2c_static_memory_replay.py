"""Replay qualitative geometry from real 2026-11-27 G2b depth-G1 detour.

This tests the registered single-static-obstacle map persistence policy on CPU;
it does NOT certify actual collision avoidance, unseen obstacles or a GPU rollout.
"""
import unittest
from foresight.adaptive.obstacle_depth import DepthDetection, ObstacleNavigator


def make_confirmed_nav():
    nav = ObstacleNavigator(goal=(5.0, 0.0))
    nav.ingest(DepthDetection((2.831, 0.015), 2.5, .3, 24, 1., 0))
    nav.ingest(DepthDetection((2.839, -0.016), 2.49, .3, 24, 1., 5))
    assert nav.command(.33, -.32, .27, 6)[2] == "detour"
    return nav


class StaticMemoryReplay(unittest.TestCase):
    def test_static_detour_after_first_waypoint_is_safe_to_continue(self):
        nav = make_confirmed_nav()
        first = nav.follower.route.waypoints[0]
        nav.follower.command(*first, .3)
        self.assertGreaterEqual(nav.follower.index, 1)
        nav.ingest(DepthDetection((2.9076, .280378), 2.03, .3, 24, 1., 105))
        nav.ingest(None, observed_step=515)  # real fresh blank head frame
        vx, wz, why = nav.command(1.963219, 1.569092, .28, 515)
        self.assertEqual(why, "detour_memory")
        self.assertGreaterEqual(vx, 0.)

    def test_no_memory_before_first_waypoint(self):
        nav = make_confirmed_nav()
        nav.ingest(DepthDetection((2.9076, .280378), 2.03, .3, 24, 1., 105))
        nav.ingest(None, observed_step=515)
        self.assertEqual(nav.command(2.0, 1.5, .2, 515)[2], "stop_lost_obstacle")

    def test_collision_clearance_gates_out_memory(self):
        nav = make_confirmed_nav()
        first = nav.follower.route.waypoints[0]
        nav.follower.command(*first, .3)
        nav.ingest(None, observed_step=515)
        self.assertEqual(nav.command(2.75, .90, .2, 515)[2], "stop_lost_obstacle")

    def test_missing_fresh_camera_never_allows_memory(self):
        nav = make_confirmed_nav()
        first = nav.follower.route.waypoints[0]
        nav.follower.command(*first, .3)
        self.assertEqual(nav.command(1.96, 1.57, .28, 515)[2], "stop_stale_depth")


if __name__ == "__main__":
    unittest.main()
