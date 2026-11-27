"""Perception-conditioned static obstacle detection and conservative route execution.

Uses camera depth and camera pose, NOT the simulator obstacle location. The scene
and ground-truth geom are evaluation-only. This is a bounded research prototype.
Inputs: metric world point cloud backprojected from MuJoCo's G1 head depth camera.
Navigation robot world pose is an explicitly disclosed simulator signal.

The detector handles a single static protruding obstacle; multi-obstacle scenes,
moving objects, stairs and uncertain floor geometry remain out of scope.
"""
from dataclasses import dataclass
from math import cos, hypot, sin
import numpy as np

from foresight.adaptive.obstacle_route import plan_route, WaypointFollower


@dataclass(frozen=True)
class DepthDetection:
    center_xy: tuple
    range_m: float
    radius_m: float
    pixel_count: int
    confidence: float
    observed_step: int


def _components(mask):
    """4-connected pixel components, without a scipy/opencv dependency."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    for y in range(h):
        for x in range(w):
            if visited[y, x] or not mask[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            pixels = []
            while stack:
                yy, xx = stack.pop()
                pixels.append((yy, xx))
                for y2, x2 in ((yy-1, xx), (yy+1, xx), (yy, xx-1), (yy, xx+1)):
                    if 0 <= y2 < h and 0 <= x2 < w and mask[y2, x2] and not visited[y2, x2]:
                        visited[y2, x2] = True
                        stack.append((y2, x2))
            yield pixels


def detect_obstacle_points(points_world, robot_xy, heading_rad, step,
                           radius_m=0.30, min_pixels=6,
                           min_range_m=0.75, max_range_m=5.0,
                           min_height_m=0.25, max_height_m=1.60):
    """Detect nearest supported above-ground image blob from valid world points.

    Image is depth-backprojected BEFORE this function. No ground-truth obstacle
    coordinate or simulation segmentation label is used. Assumes flat ground z=0
    and a known approximate obstacle radius. Confidence is a heuristic score,
    not calibrated probability.
    """
    pts = np.asarray(points_world, dtype=np.float64)
    if pts.ndim != 3 or pts.shape[2] != 3:
        raise ValueError("expected depth backprojection of shape (H,W,3)")
    if not np.isfinite([*robot_xy, heading_rad, radius_m]).all() or radius_m <= 0:
        raise ValueError("invalid pose or radius")
    dx = pts[..., 0] - robot_xy[0]
    dy = pts[..., 1] - robot_xy[1]
    rng = np.hypot(dx, dy)
    forward = dx * cos(heading_rad) + dy * sin(heading_rad)
    lateral = -dx * sin(heading_rad) + dy * cos(heading_rad)
    valid = (np.isfinite(pts).all(axis=-1) & (pts[..., 2] >= min_height_m)
             & (pts[..., 2] <= max_height_m)
             & (rng >= min_range_m) & (rng <= max_range_m)
             & (forward > 0.30) & (np.abs(lateral) < 2.0))
    choices = []
    for component in _components(valid):
        if len(component) < min_pixels:
            continue
        ys, xs = np.array(component).T
        q = pts[ys, xs]
        surface_xy = np.median(q[:, :2], axis=0)
        dist = hypot(surface_xy[0]-robot_xy[0], surface_xy[1]-robot_xy[1])
        if dist <= 1e-9:
            continue
        direction = (surface_xy - np.asarray(robot_xy)) / dist
        center = surface_xy + radius_m * direction
        center_dist = float(np.linalg.norm(center - np.asarray(robot_xy)))
        lateral_spread = float(np.linalg.norm(np.ptp(q[:, :2], axis=0)))
        # Discard huge, poorly localized surfaces: e.g. wall or sloped terrain.
        if lateral_spread > 1.8:
            continue
        score = min(1.0, len(component)/max(3*min_pixels, 1))
        choices.append(DepthDetection(tuple(map(float, center)),
                                      center_dist, float(radius_m),
                                      len(component), float(score), int(step)))
    return min(choices, key=lambda v: v.range_m) if choices else None


class ObstacleNavigator:
    """Latch stable camera detections; don't use ground-truth obstacle pose.

    If a previously detected near-field obstacle disappears or observations
    become stale, request stop until reacquired. Any invalid geometric detour
    also triggers stop rather than commanding unverified forward motion.
    """

    def __init__(self, goal, *, min_confirmations=2, max_age_steps=10,
                 detection_gate_m=0.5, stop_range_m=1.6):
        self.goal = tuple(goal)
        self.min_confirmations = int(min_confirmations)
        self.max_age_steps = int(max_age_steps)
        self.detection_gate_m = float(detection_gate_m)
        self.stop_range_m = float(stop_range_m)
        self.pending = None
        self.confirmations = 0
        self.obstacle = None
        self.follower = None
        self.last_step = -1
        self.last_frame_step = -1
        if self.min_confirmations < 1 or self.max_age_steps < 1:
            raise ValueError("invalid detector timing")

    def ingest(self, detection, observed_step=None):
        """Register a depth frame even when it contains no detected obstacle."""
        if observed_step is None and detection is not None:
            observed_step = detection.observed_step
        if observed_step is not None:
            if observed_step <= self.last_frame_step:
                raise ValueError("camera frame steps must increase")
            self.last_frame_step = int(observed_step)
        if detection is None:
            return
        if observed_step is not None and detection.observed_step != observed_step:
            raise ValueError("detection and depth frame timestamps mismatch")
        if detection.observed_step <= self.last_step:
            raise ValueError("observation steps must increase")
        self.last_step = detection.observed_step
        if (self.pending is not None and
            hypot(detection.center_xy[0]-self.pending.center_xy[0],
                  detection.center_xy[1]-self.pending.center_xy[1]) <= self.detection_gate_m):
            self.confirmations += 1
        else:
            self.confirmations = 1
        self.pending = detection
        if self.confirmations >= self.min_confirmations:
            self.obstacle = detection

    def _memory_route_clear(self, x, y, margin_m=0.25):
        """Recheck full remaining route against last depth-estimated cylinder.

        Only apply after first detour waypoint is reached, only for this
        single STATIC obstacle experiment and only while camera frames remain
        fresh. Recheck all upcoming route segments with extra 0.25m margin;
        never use ground-truth geom or let a stale CAMERA frame drive.
        """
        from foresight.adaptive.obstacle_route import _segment_distance
        follower = self.follower
        if follower is None or follower.index < 1 or self.obstacle is None:
            return False
        route = follower.route
        if route.side == 0:
            return False
        remaining = tuple(route.waypoints[follower.index:])
        if not remaining:
            return False
        points = ((x, y),) + remaining
        radius = route.inflated_radius + margin_m
        if hypot(x-self.obstacle.center_xy[0],
                 y-self.obstacle.center_xy[1]) <= radius:
            return False
        return all(_segment_distance(self.obstacle.center_xy, a, b) >= radius
                   for a, b in zip(points, points[1:]))

    def command(self, x, y, yaw, step):
        if self.last_frame_step < 0 or step - self.last_frame_step > self.max_age_steps:
            # Never drive without a fresh image, even when no obstacle is seen.
            return 0.0, 0.0, "stop_stale_depth"
        if self.obstacle is None:
            if self.pending is not None and self.pending.range_m <= self.stop_range_m:
                return 0.0, 0.0, "stop_unconfirmed_obstacle"
            # Before any confirmed obstacle, conservatively approach the goal.
            from foresight.adaptive.obstacle_route import target_command
            vx, wz, reached = target_command(x, y, yaw, self.goal, speed=0.25)
            return vx, wz, "goal_reached" if reached else "approach_slow"
        dist = hypot(x-self.obstacle.center_xy[0], y-self.obstacle.center_xy[1])
        lost_near = (step-self.obstacle.observed_step > self.max_age_steps
                     and dist < self.stop_range_m)
        remembered_detour = False
        if lost_near:
            remembered_detour = self._memory_route_clear(x, y)
            if not remembered_detour:
                return 0.0, 0.0, "stop_lost_obstacle"
        if self.follower is None:
            try:
                route = plan_route((x, y), self.goal, self.obstacle.center_xy,
                                   self.obstacle.radius_m)
            except ValueError:
                return 0.0, 0.0, "stop_unsafe_geometry"
            self.follower = WaypointFollower(route)
        vx, wz, done = self.follower.command(x, y, yaw)
        if done:
            return vx, wz, "goal_reached"
        return vx, wz, "detour_memory" if remembered_detour else "detour"


def detect_from_depth(depth, camera_position, camera_rotation, fovy_degrees,
                      robot_xy, robot_heading_rad, step,
                      *, depth_is_ray=False, flip_rows=False, radius_m=0.30):
    """Adapter to the existing depth-rendering backprojection implementation.

    The depth convention and row order MUST be calibrated on ground pixels
    (see scripts/depth_detector.py), not picked by inspecting success labels.
    """
    from scripts.depth_detector import pixel_dirs, backproject
    raw = np.asarray(depth, dtype=np.float64)
    if raw.ndim != 2:
        raise ValueError("expected one (H,W) head-camera depth image")
    if raw.shape[0] != raw.shape[1]:
        raise ValueError("existing pixel_dirs requires square camera images")
    valid = np.isfinite(raw) & (raw > .05) & (raw < 50.)
    rays = pixel_dirs(raw.shape[0], fovy_degrees, flip_rows)
    xyz = backproject(np.where(valid, raw, 1.0),
                      np.asarray(camera_position), np.asarray(camera_rotation),
                      rays, depth_is_ray)
    xyz[~valid] = np.nan
    return detect_obstacle_points(xyz, robot_xy, robot_heading_rad, step,
                                  radius_m=radius_m)
