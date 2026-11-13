# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Parametric object detections from the simulator's privileged object state.

A detection is a world-frame 3-D position of the object captured `latency` control steps
ago, delivered every `period` steps, with range-dependent Gaussian noise, a field-of-view
gate around the pelvis heading, and a random miss probability. The robot's own pose is
assumed known (it is on the robot). This is the observation model the particle filter
consumes; the rendered-depth pipeline (E7/E8) replaces the noise/dropout numbers, not
the interface.

All functions are JAX and operate on one environment; callers vmap over environments.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class DetectorConfig(NamedTuple):
  period: int = 5            # control steps between detections (5 -> 10 Hz at 50 Hz)
  latency: int = 5           # control steps between capture and delivery
  sigma0: float = 0.03       # m, position noise at zero range
  sigma1: float = 0.02       # m per m of range
  fov_deg: float = 120.0     # horizontal field of view around the pelvis heading
  miss_prob: float = 0.05    # random missed detection
  max_range: float = 6.0     # m
  buffer_len: int = 16       # ring buffer of past true states (>= latency + 1)


class DetectorState(NamedTuple):
  buf_pos: jax.Array         # (buffer_len, 3) true object positions, newest at index step % L
  buf_pelvis: jax.Array      # (buffer_len, 3)
  buf_fwd: jax.Array         # (buffer_len, 3)
  step: jax.Array            # int32 control step


class Detection(NamedTuple):
  valid: jax.Array           # bool: a detection was delivered this step
  pos: jax.Array             # (3,) noisy captured position (world frame)
  capture_step: jax.Array    # int32 step at which it was captured
  sigma: jax.Array           # position noise std used (for the filter's likelihood)


def init_state(cfg: DetectorConfig) -> DetectorState:
  z = jp.zeros((cfg.buffer_len, 3))
  return DetectorState(z, z, z, jp.int32(0))


def step(cfg: DetectorConfig, st: DetectorState, obj_pos, pelvis_pos, pelvis_fwd, launched,
         rng) -> tuple[DetectorState, Detection]:
  """Record the true state for this step and emit a (possibly empty) delayed detection."""
  L = cfg.buffer_len
  i = st.step % L
  st = st._replace(
      buf_pos=st.buf_pos.at[i].set(obj_pos),
      buf_pelvis=st.buf_pelvis.at[i].set(pelvis_pos),
      buf_fwd=st.buf_fwd.at[i].set(pelvis_fwd),
  )
  cap = st.step - cfg.latency
  j = cap % L
  p, pel, fwd = st.buf_pos[j], st.buf_pelvis[j], st.buf_fwd[j]
  rel = p - pel
  rng_dist = jp.linalg.norm(rel[:2]) + 1e-6
  heading = fwd[:2] / (jp.linalg.norm(fwd[:2]) + 1e-6)
  cos_ang = jp.dot(rel[:2] / rng_dist, heading)
  in_fov = cos_ang > jp.cos(jp.deg2rad(cfg.fov_deg / 2))
  k_miss, k_noise = jax.random.split(rng)
  due = (st.step % cfg.period == 0) & (cap >= 0)
  valid = due & launched & in_fov & (rng_dist < cfg.max_range) & \
      (jax.random.uniform(k_miss) >= cfg.miss_prob)
  sigma = cfg.sigma0 + cfg.sigma1 * rng_dist
  noisy = p + sigma * jax.random.normal(k_noise, (3,))
  det = Detection(valid=valid, pos=noisy, capture_step=cap.astype(jp.int32), sigma=sigma)
  return st._replace(step=st.step + 1), det
