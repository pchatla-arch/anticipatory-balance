# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Small numerical contracts shared by the controller and its CPU checks.

The optional array namespace makes these same routines work with NumPy or JAX;
the tests exercise analytical invariants, rather than substituting a fake JAX.
"""
from __future__ import annotations

import numpy as np


def next_delivery(step, period: int, latency: int, xp=np):
  """Next strictly future delivery; capture grid is 0, period, 2*period, ... ."""
  if period < 1 or latency < 0:
    raise ValueError("period must be positive and latency nonnegative")
  delivered = xp.maximum(step + 1 + (latency - step - 1) % period, latency)
  return delivered, delivered - latency, delivered - step


def transition(dt, xp=np):
  eye, zero = xp.eye(3), xp.zeros((3, 3))
  return xp.concatenate((xp.concatenate((eye, dt * eye), axis=1),
                         xp.concatenate((zero, eye), axis=1)), axis=0)


def transported_covariance(sigma, velocity_sigma, age_s, xp=np):
  """Covariance of [z + age*v, v], with independent capture error and v prior."""
  eye = xp.eye(3)
  vv = velocity_sigma ** 2 * eye
  pv = age_s * vv
  pp = sigma ** 2 * eye + age_s ** 2 * vv
  return xp.concatenate((xp.concatenate((pp, pv), axis=1),
                         xp.concatenate((pv, vv), axis=1)), axis=0)


def measurement_matrix(age_s, xp=np):
  return xp.concatenate((xp.eye(3), -age_s * xp.eye(3)), axis=1)


def gaussian_update(mu, cov, z, age_s, sigma, xp=np):
  """Timestamped position update under deterministic constant velocity.

  A tiny variance floor handles a nominally perfect detector numerically.
  This is not the correct delayed update for a stochastic acceleration model.
  """
  H = measurement_matrix(age_s, xp)
  R = xp.maximum(sigma ** 2, 1e-12) * xp.eye(3)
  S = H @ cov @ H.T + R
  K = xp.linalg.solve(S, H @ cov).T
  new_mu = mu + K @ (z - H @ mu)
  A = xp.eye(6) - K @ H
  new_cov = A @ cov @ A.T + K @ R @ K.T
  return new_mu, (new_cov + new_cov.T) * 0.5


def locked_duration(t_hit, hold, commit_steps: int, dt, max_hold_s, xp=np):
  """Duration executed after a commitment, with a fixed deadline (no sliding)."""
  minimum = commit_steps * dt
  if commit_steps < 1 or dt <= 0 or hold < 0 or max_hold_s < minimum:
    raise ValueError("invalid minimum commitment, hold, or maximum duration")
  # Round upward to the control grid: the executed duration is this many steps.
  duration = xp.clip(xp.maximum(t_hit, 0.0) + hold, minimum, max_hold_s)
  max_steps = xp.floor(max_hold_s / dt + 1e-6)
  steps = xp.clip(xp.ceil(duration / dt - 1e-6), commit_steps, max_steps)
  return steps * dt


def normalize_weights(weights, xp=np):
  total = xp.sum(weights)
  fallback = xp.ones_like(weights) / weights.shape[-1]
  return xp.where(total > 1e-20, weights / xp.maximum(total, 1e-20), fallback)


def observation_weights(prior, detection_probability, predicted_positions, z, sigma, xp=np):
  """Particle posterior for a delivered point, including state-dependent detection."""
  variance = xp.maximum(sigma ** 2, 1e-12)
  squared = xp.sum((predicted_positions - z) ** 2, axis=-1)
  logw = xp.log(xp.maximum(prior, 1e-30)) + xp.log(xp.maximum(detection_probability, 1e-30))
  logw = logw - 0.5 * squared / variance - 1.5 * xp.log(variance)
  supported = (prior > 0) & (detection_probability > 0)
  logw = xp.where(supported, logw, -1e30)
  raw = xp.exp(logw - xp.max(logw)) * supported
  return normalize_weights(raw, xp)
