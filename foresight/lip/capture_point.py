# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Linear inverted pendulum (LIP) capture-point tools.

These are the whiteboard equations of the project:
    omega   = sqrt(g / z_com)
    x_cp    = x + xdot / omega                 (instantaneous capture point)
    J_max   = m * omega * (x_edge - x_cp)      (largest impulse absorbable without stepping)
An impulse J along a direction shifts xdot by J/m and therefore x_cp by J/(m*omega).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81


def omega(z_com: float) -> float:
  return float(np.sqrt(G / z_com))


def capture_point(x: np.ndarray, xdot: np.ndarray, z_com: float) -> np.ndarray:
  """Instantaneous capture point in the horizontal plane. x, xdot are shape (2,)."""
  return np.asarray(x, float) + np.asarray(xdot, float) / omega(z_com)


@dataclass
class SupportPolygon:
  """Convex support polygon in the horizontal plane, vertices (N,2) counter-clockwise."""
  vertices: np.ndarray

  @classmethod
  def from_feet(cls, left_foot_xy, right_foot_xy, foot_len=0.17, foot_wid=0.06, yaw=0.0):
    """Rectangle hull around two foot boxes (Playground G1 foot geom is 0.085 x 0.03 half-sizes)."""
    pts = []
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    for f in (np.asarray(left_foot_xy, float), np.asarray(right_foot_xy, float)):
      for dx in (-foot_len / 2, foot_len / 2):
        for dy in (-foot_wid / 2, foot_wid / 2):
          pts.append(f + R @ np.array([dx, dy]))
    pts = np.array(pts)
    return cls(_convex_hull(pts))

  def margin_along(self, point: np.ndarray, direction: np.ndarray) -> float:
    """Distance from point to the polygon edge along a unit direction (negative if outside).

    Found by ray-casting from point along direction against each edge.
    """
    p = np.asarray(point, float)
    d = np.asarray(direction, float)
    d = d / (np.linalg.norm(d) + 1e-12)
    if not self.contains(p):
      return -self.distance_outside(p)
    best = np.inf
    V = self.vertices
    for i in range(len(V)):
      a, b = V[i], V[(i + 1) % len(V)]
      e = b - a
      denom = d[0] * (-e[1]) + d[1] * e[0]
      if abs(denom) < 1e-12:
        continue
      t = ((a[0] - p[0]) * (-e[1]) + (a[1] - p[1]) * e[0]) / denom
      u = ((a[0] - p[0]) * (-d[1]) + (a[1] - p[1]) * d[0]) / denom
      if t >= 0 and 0 <= u <= 1:
        best = min(best, t)
    return float(best)

  def contains(self, p: np.ndarray) -> bool:
    V = self.vertices
    sign = None
    for i in range(len(V)):
      a, b = V[i], V[(i + 1) % len(V)]
      cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
      if abs(cross) < 1e-12:
        continue
      s = cross > 0
      if sign is None:
        sign = s
      elif s != sign:
        return False
    return True

  def distance_outside(self, p: np.ndarray) -> float:
    V = self.vertices
    best = np.inf
    for i in range(len(V)):
      a, b = V[i], V[(i + 1) % len(V)]
      e = b - a
      t = np.clip(np.dot(p - a, e) / (np.dot(e, e) + 1e-12), 0, 1)
      best = min(best, np.linalg.norm(p - (a + t * e)))
    return float(best)


def j_max(mass: float, z_com: float, polygon: SupportPolygon, x_cp: np.ndarray,
          push_dir: np.ndarray) -> float:
  """Largest impulse (N*s) along push_dir that keeps the capture point inside the polygon."""
  margin = polygon.margin_along(x_cp, push_dir)
  return float(mass * omega(z_com) * margin)


def predicted_fall(mass: float, z_com: float, polygon: SupportPolygon, x: np.ndarray,
                   xdot: np.ndarray, impulse: float, push_dir: np.ndarray) -> bool:
  """LIP prediction: does an impulse of this size push the capture point outside?"""
  return impulse > j_max(mass, z_com, polygon, capture_point(x, xdot, z_com), push_dir)


def _convex_hull(points: np.ndarray) -> np.ndarray:
  pts = sorted(map(tuple, points))
  if len(pts) <= 2:
    return np.array(pts)

  def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

  lower, upper = [], []
  for p in pts:
    while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
      lower.pop()
    lower.append(p)
  for p in reversed(pts):
    while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
      upper.pop()
    upper.append(p)
  return np.array(lower[:-1] + upper[:-1])
