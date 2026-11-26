"""Known-map waypoint detour for a simulated G1 demonstration.

Planning-only component. Not a physics or collision-safety certification.
Output is (forward velocity, yaw rate), for the existing frozen walker.
"""
from dataclasses import dataclass
from math import atan2, cos, hypot, pi


def _segment_distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    n2 = dx * dx + dy * dy
    if n2 < 1e-12:
        return hypot(p[0] - a[0], p[1] - a[1])
    t = max(0.0, min(1.0, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/n2))
    return hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


@dataclass(frozen=True)
class Detour:
    waypoints: tuple
    inflated_radius: float
    obstacle: tuple
    side: int


def plan_route(start, goal, obstacle, radius, robot_radius=0.27,
               clearance=0.25, preferred_side=1):
    """Use fixed wide waypoints; reject geometry that cannot be cleared."""
    if preferred_side not in (-1, 1):
        raise ValueError("preferred_side must be +1 or -1")
    if radius <= 0 or robot_radius < 0 or clearance < 0:
        raise ValueError("invalid radii")
    R = radius + robot_radius + clearance
    sx, sy = start
    gx, gy = goal
    ox, oy = obstacle
    dx, dy = gx-sx, gy-sy
    length = hypot(dx, dy)
    if length < 1e-6:
        return Detour((tuple(goal),), R, tuple(obstacle), 0)
    ux, uy = dx/length, dy/length
    if _segment_distance(obstacle, start, goal) >= R:
        return Detour((tuple(goal),), R, tuple(obstacle), 0)
    along = (ox-sx)*ux + (oy-sy)*uy
    lateral_error = (ox-sx)*(-uy) + (oy-sy)*ux
    if (hypot(ox-sx, oy-sy) <= 2.2*R or
        hypot(ox-gx, oy-gy) <= 2.2*R or
        not (2*R < along < length-2*R)):
        raise ValueError("start or goal too close for supported detour")
    lateral = preferred_side*(abs(lateral_error)+1.6*R)
    px, py = -uy, ux
    cx, cy = sx+along*ux, sy+along*uy
    first = (cx-1.8*R*ux+lateral*px, cy-1.8*R*uy+lateral*py)
    second = (cx+1.8*R*ux+lateral*px, cy+1.8*R*uy+lateral*py)
    points = (tuple(start), first, second, tuple(goal))
    if any(_segment_distance(obstacle, a, b) < R for a, b in zip(points, points[1:])):
        raise ValueError("waypoints fail geometric clearance: do not execute")
    return Detour((first, second, tuple(goal)), R, tuple(obstacle), preferred_side)


def target_command(x, y, yaw, target, speed=0.4, yaw_gain=2.0,
                   max_yaw_rate=0.5, acceptance_radius=0.2):
    """Return (body_forward_mps, yaw_rate_rps, reached)."""
    if not (0 < speed <= 0.8 and yaw_gain > 0 and 0 < max_yaw_rate <= 1.0):
        raise ValueError("command beyond pilot limits")
    if hypot(target[0]-x, target[1]-y) <= acceptance_radius:
        return 0.0, 0.0, True
    e = (atan2(target[1]-y, target[0]-x)-yaw+pi)%(2*pi)-pi
    turn = max(-max_yaw_rate, min(max_yaw_rate, yaw_gain*e))
    forward = speed*max(0.0, cos(e)) if abs(e) <= pi/3 else 0.0
    return forward, turn, False


class WaypointFollower:
    def __init__(self, route):
        self.route = route
        self.index = 0

    def command(self, x, y, yaw):
        if self.index >= len(self.route.waypoints):
            return 0.0, 0.0, True
        v, omega, reached = target_command(x, y, yaw,
                                            self.route.waypoints[self.index])
        if reached:
            self.index += 1
            if self.index >= len(self.route.waypoints):
                return 0.0, 0.0, True
            return target_command(x, y, yaw, self.route.waypoints[self.index])
        return v, omega, False
