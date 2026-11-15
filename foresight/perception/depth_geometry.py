"""Shared depth geometry and CPU diagnostics for the reviewed online experiment.

The centre estimator accepts an array backend (NumPy or jax.numpy), so the CPU
geometry checks exercise the actual function used inside the JAX renderer loop.
Simulator object position is used only by ErrorAccumulator after detection.
"""
from __future__ import annotations

import numpy as np


def pixel_directions(res: int, fovy_deg: float):
  """Unnormalised camera rays for axial z depth, with image y pointing up."""
  if res < 2 or not 0 < fovy_deg < 180:
    raise ValueError("Need res>=2 and 0<fovy<180")
  tangent = np.tan(np.deg2rad(fovy_deg) / 2)
  u = (np.arange(res) + 0.5) / res * 2 - 1
  v = 1 - (np.arange(res) + 0.5) / res * 2
  x, y = np.meshgrid(u * tangent, v * tangent)
  return np.stack([x, y, -np.ones_like(x)], axis=-1).astype(np.float32)


def detect_centre(depth, cam_pos, cam_mat, pelvis, directions, radius: float,
                  max_range: float = 6.0, xp=np):
  """Per-axis visible-surface median plus known sphere radius along the view ray.

  Returns image validity, centre, and candidate-pixel count. It accepts no launch
  flag, object identity/position/velocity or mass. The estimator matches the
  latest upstream/E8 median; errors are still measured on the online stream.
  """
  valid = xp.isfinite(depth) & (depth > 0.05) & (depth < 50.0)
  points = cam_pos + (directions * xp.where(valid, depth, 1.0)[..., None]) @ cam_mat.T
  distance = xp.linalg.norm(points[..., :2] - pelvis[:2], axis=-1)
  candidate = valid & (points[..., 2] > 0.3) & (points[..., 2] < 2.0)
  candidate &= (distance > 0.6) & (distance < max_range)
  count = xp.sum(candidate)
  selected = xp.where(candidate[..., None], points, xp.nan).reshape(-1, 3)
  # All-empty frames remain invalid; the zero fallback avoids an all-NaN
  # reduction while retaining a finite, ignored placeholder position.
  selected = xp.where(count > 0, selected, xp.zeros_like(selected))
  surface = xp.nanmedian(selected, axis=0)
  view = surface - cam_pos
  centre = surface + radius * view / (xp.linalg.norm(view) + 1e-9)
  return count >= 3, centre, count


def delivery_due(step: int, period: int, latency: int) -> tuple[bool, int]:
  """Captures occur at multiples of period; a capture arrives latency steps later."""
  if period <= 0 or latency < 0:
    raise ValueError("period>0 and latency>=0 are required")
  capture = step - latency
  return capture >= 0 and capture % period == 0, capture


class DetectionQueue:
  """Single camera's timestamp queue; validity is frozen at capture."""

  def __init__(self, latency: int):
    if latency < 0:
      raise ValueError("latency must be nonnegative")
    self.latency = latency
    self.pending = {}

  def capture(self, step: int, valid, position, sigma):
    delivery = step + self.latency
    if delivery in self.pending:
      raise ValueError("Duplicate camera capture/delivery step")
    # JAX arrays are immutable; copying also makes the NumPy/CPU contract safe
    # when a caller reuses a mutable capture buffer.
    snapshot = lambda x: x.copy() if hasattr(x, "copy") else x
    self.pending[delivery] = (step, snapshot(valid), snapshot(position), snapshot(sigma))

  def deliver(self, step: int):
    if any(t < step for t in self.pending):
      raise RuntimeError("A queued observation was not delivered on its scheduled step")
    return self.pending.pop(step, None)


class ErrorAccumulator:
  """Post-hoc measurement diagnostics, never passed back into the controller."""

  def __init__(self, edges=(0.0, 0.6, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, float("inf"))):
    self.edges = np.asarray(edges, float)
    self.frame_count = 0
    self.valid_count = 0
    self.prelaunch_detections = 0
    self.prelaunch_frames = 0
    self.n = np.zeros(len(edges) - 1, int)
    self.s = np.zeros((len(edges) - 1, 3))
    self.ss = np.zeros((len(edges) - 1, 3))
    self.max_norm = np.zeros(len(edges) - 1)

  def add(self, valid, detected, truth, pelvis, launched):
    valid, launched = np.asarray(valid, bool), np.asarray(launched, bool)
    detected, truth, pelvis = (np.asarray(x, float) for x in (detected, truth, pelvis))
    self.frame_count += len(valid)
    self.valid_count += int(valid.sum())
    self.prelaunch_detections += int((valid & ~launched).sum())
    self.prelaunch_frames += int((~launched).sum())
    error = detected - truth
    use = valid & launched & np.isfinite(error).all(axis=-1)
    ranges = np.linalg.norm(truth[:, :2] - pelvis[:, :2], axis=-1)
    for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
      e = error[use & (ranges >= lo) & (ranges < hi)]
      if not len(e):
        continue
      self.n[i] += len(e)
      self.s[i] += e.sum(axis=0)
      self.ss[i] += (e * e).sum(axis=0)
      self.max_norm[i] = max(self.max_norm[i], float(np.linalg.norm(e, axis=1).max()))

  def summary(self):
    bins = []
    for i, (lo, hi) in enumerate(zip(self.edges[:-1], self.edges[1:])):
      n = int(self.n[i])
      bins.append({"range_low_m": float(lo), "range_high_m": None if not np.isfinite(hi) else float(hi),
                   "n_successful_launched_detections": n,
                   "bias_xyz_m": (self.s[i] / n).tolist() if n else None,
                   "rmse_xyz_m": np.sqrt(self.ss[i] / n).tolist() if n else None,
                   "pooled_axis_rmse_m": float(np.sqrt(self.ss[i].sum() / (3 * n))) if n else None,
                   "max_position_error_m": float(self.max_norm[i]) if n else None})
    return {"frame_count": self.frame_count, "valid_detection_count": self.valid_count,
            "prelaunch_detection_count": self.prelaunch_detections,
            "prelaunch_frame_count": self.prelaunch_frames,
            "prelaunch_detection_rate": (self.prelaunch_detections / self.prelaunch_frames
                                          if self.prelaunch_frames else None),
            "conditioning": "successful detections of launched objects; errors include bias",
            "use": "diagnostic only; not fitted or fed back into evaluated likelihood",
            "bins": bins}
