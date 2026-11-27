"""Pure-Python geometric regression checks; not MuJoCo validation."""
import unittest
from foresight.adaptive.obstacle_route import (
    plan_route, target_command, WaypointFollower, _segment_distance
)


class ObstacleRouteTests(unittest.TestCase):
    def test_clear_straight_route(self):
        route = plan_route((0, 0), (5, 0), (2.5, 4), .3)
        self.assertEqual(route.waypoints, ((5, 0),))

    def test_detour_both_sides_clear(self):
        for side in (-1, 1):
            route = plan_route((0, 0), (5, 0), (2.5, 0), .3, preferred_side=side)
            pts = ((0, 0),) + route.waypoints
            self.assertEqual(len(route.waypoints), 3)
            self.assertTrue(all(_segment_distance((2.5, 0), a, b) >= route.inflated_radius
                                for a, b in zip(pts, pts[1:])))

    def test_reject_close_obstacle_start(self):
        with self.assertRaises(ValueError):
            plan_route((1.9, 0), (5, 0), (2.5, 0), .35)

    def test_reject_invalid_radius(self):
        with self.assertRaises(ValueError):
            plan_route((0, 0), (5, 0), (2.5, 0), -1)

    def test_turn_in_place_before_unsafe_forward(self):
        vx, wz, done = target_command(0, 0, 0, (0, 3))
        self.assertEqual(vx, 0)
        self.assertGreater(wz, 0)
        self.assertFalse(done)

    def test_forward(self):
        vx, wz, done = target_command(0, 0, 0, (3, 0))
        self.assertAlmostEqual(vx, .4)
        self.assertAlmostEqual(wz, 0)
        self.assertFalse(done)

    def test_stop_at_waypoint(self):
        self.assertEqual(target_command(0, 0, 0, (.01, 0)), (0., 0., True))

    def test_follower_advances(self):
        route = plan_route((0, 0), (5, 0), (2.5, 0), .3)
        follower = WaypointFollower(route)
        for i, p in enumerate(route.waypoints):
            follower.command(*p, 0)
            self.assertEqual(follower.index, i + 1)


if __name__ == "__main__":
    unittest.main()
