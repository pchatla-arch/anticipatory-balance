# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Rao-Blackwellized belief over the approaching object.

Position and velocity: a Kalman filter (the flight is straight and unaccelerated, so the
constant-velocity model is exact). A detection captured at step k and delivered at step t
is a *linear* measurement of the current state, z = p_t - v (t - k) dt, so a delayed
observation is absorbed exactly without storing or replaying past beliefs.
Mass: samples from the prior (it is unobservable from position tracks).

Belief statistics for the decision layer are Monte-Carlo over N samples of (p, v) from the
Gaussian times the mass samples, in the robot's frame of motion:
  hit      : closest approach of the relative straight path to the pelvis axis is within
             r_contact, in the future, at a plausible height
  t_hit    : time (s from now) at which the path enters the contact radius
  momentum : m |v| (world frame)
All functions are JAX for one environment; callers vmap over environments.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class PFConfig(NamedTuple):
  n_particles: int = 128                # Monte-Carlo samples for the statistics
  mass_range: tuple = (4.0, 16.0)       # prior (uniform)
  init_speed: float = 5.0               # m/s, velocity prior mean toward the robot
  init_vel_sigma: float = 3.0           # m/s, velocity prior std per axis
  process_pos_sigma: float = 0.0        # exact model: no process noise on position
  process_vel_sigma: float = 0.02       # m/s per step, keeps the filter responsive
  r_contact: float = 0.45               # m, decision radius about the pelvis axis (contact ~0.3)
  dt: float = 0.02
  z_hit: tuple = (0.3, 1.4)             # plausible impact heights, m
  min_future: float = 0.0


class PFState(NamedTuple):
  mu: jax.Array        # (6,) [p, v] at the current step
  cov: jax.Array       # (6, 6)
  m: jax.Array         # (N,) mass samples from the prior
  n_obs: jax.Array     # int32 detections absorbed
  initialized: jax.Array  # bool


class BeliefStats(NamedTuple):
  p_hit: jax.Array
  t_hit: jax.Array           # E[t_hit | hit], s from now (inf if p_hit == 0)
  t_hit_std: jax.Array
  momentum_mean: jax.Array   # E[m|v| | hit]
  ess: jax.Array             # here: trace of the velocity covariance (uncertainty measure)
  n_obs: jax.Array
  pe_hit: jax.Array          # point estimate (KF mean) hit flag
  pe_t_hit: jax.Array
  pe_momentum: jax.Array
  pe_pos: jax.Array
  pe_vel: jax.Array
  mass_samples: jax.Array    # (N,)
  momentum_samples: jax.Array
  hit_samples: jax.Array
  weights: jax.Array         # (N,) uniform


def init_state(cfg: PFConfig) -> PFState:
  N = cfg.n_particles
  return PFState(jp.zeros(6), jp.eye(6), jp.full((N,), 8.0), jp.int32(0), jp.array(False))


def _init_from_detection(cfg: PFConfig, det_pos, det_sigma, delay_s, pelvis_pos, m_rng) -> PFState:
  to_robot = pelvis_pos - det_pos
  d = jp.linalg.norm(to_robot[:2]) + 1e-6
  v0 = jp.array([to_robot[0] / d, to_robot[1] / d, 0.0]) * cfg.init_speed
  # The detection is delay_s old: the object has moved on by v0 * delay_s (uncertain).
  mu = jp.concatenate([det_pos + v0 * delay_s, v0])
  sp = det_sigma ** 2 + (cfg.init_vel_sigma * delay_s) ** 2
  cov = jp.diag(jp.concatenate([jp.full(3, sp), jp.full(3, cfg.init_vel_sigma ** 2)]))
  m = jax.random.uniform(m_rng, (cfg.n_particles,), minval=cfg.mass_range[0], maxval=cfg.mass_range[1])
  return PFState(mu, cov, m, jp.int32(1), jp.array(True))


def predict(cfg: PFConfig, st: PFState) -> PFState:
  """One control step of constant-velocity motion."""
  F = jp.eye(6).at[0:3, 3:6].set(cfg.dt * jp.eye(3))
  Q = jp.diag(jp.concatenate([jp.full(3, cfg.process_pos_sigma ** 2),
                              jp.full(3, cfg.process_vel_sigma ** 2)]))
  mu = F @ st.mu
  cov = F @ st.cov @ F.T + Q
  return st._replace(mu=jp.where(st.initialized, mu, st.mu), cov=jp.where(st.initialized, cov, st.cov))


def update(cfg: PFConfig, st: PFState, det_valid, det_pos, det_sigma, capture_step, step_idx,
           pelvis_pos, rng) -> PFState:
  """Absorb one (possibly invalid) detection captured `step_idx - capture_step` steps ago."""
  delay_s = (step_idx - capture_step).astype(jp.float32) * cfg.dt

  def do_init(_):
    return _init_from_detection(cfg, det_pos, det_sigma, delay_s, pelvis_pos, rng)

  def do_update(_):
    H = jp.concatenate([jp.eye(3), -delay_s * jp.eye(3)], axis=1)      # z = p_t - v * delay
    R = det_sigma ** 2 * jp.eye(3)
    S = H @ st.cov @ H.T + R
    K = st.cov @ H.T @ jp.linalg.inv(S)
    mu = st.mu + K @ (det_pos - H @ st.mu)
    cov = (jp.eye(6) - K @ H) @ st.cov
    cov = 0.5 * (cov + cov.T)
    return st._replace(mu=mu, cov=cov, n_obs=st.n_obs + 1)

  def on_valid(_):
    return jax.lax.cond(st.initialized, do_update, do_init, None)

  return jax.lax.cond(det_valid, on_valid, lambda _: st, None)


def stats(cfg: PFConfig, st: PFState, pelvis_pos, pelvis_vel, rng) -> BeliefStats:
  """Belief statistics now, Monte-Carlo over the Gaussian and the mass samples."""
  N = cfg.n_particles
  L = jp.linalg.cholesky(st.cov + 1e-9 * jp.eye(6))
  eps = jax.random.normal(rng, (N, 6))
  x = st.mu[None] + eps @ L.T
  p, v_world = x[:, :3], x[:, 3:]
  w = jp.full((N,), 1.0 / N)

  def geom(p, v):
    rel = pelvis_pos[:2] - p[:2]
    vxy = v[:2]
    sp2 = jp.sum(vxy ** 2) + 1e-6
    t_ca = jp.dot(rel, vxy) / sp2
    d_min2 = jp.maximum(jp.sum(rel ** 2) - jp.dot(rel, vxy) ** 2 / sp2, 0.0)
    inside = d_min2 < cfg.r_contact ** 2
    t_enter = t_ca - jp.sqrt(jp.maximum(cfg.r_contact ** 2 - d_min2, 0.0) / sp2)
    z_at = p[2] + v[2] * t_enter
    hit = inside & (t_enter > cfg.min_future) & (z_at > cfg.z_hit[0]) & (z_at < cfg.z_hit[1])
    return hit, t_enter

  v_rel = v_world - pelvis_vel[None]
  hit, t_enter = jax.vmap(geom)(p, v_rel)
  hit = hit & st.initialized
  mom = st.m * jp.linalg.norm(v_world, axis=-1)
  wh = w * hit
  p_hit = jp.sum(wh)
  wh_n = wh / (p_hit + 1e-9)
  t_hit = jp.where(p_hit > 1e-6, jp.sum(wh_n * t_enter), jp.inf)
  t_std = jp.where(p_hit > 1e-6, jp.sqrt(jp.maximum(jp.sum(wh_n * t_enter ** 2) - t_hit ** 2, 0.0)), jp.inf)
  mom_mean = jp.where(p_hit > 1e-6, jp.sum(wh_n * mom), 0.0)
  # Point estimate: the KF mean.
  pe_hit, pe_t = geom(st.mu[:3], st.mu[3:] - pelvis_vel)
  pe_hit = pe_hit & st.initialized
  mm = jp.mean(st.m)
  return BeliefStats(p_hit, t_hit, t_std, mom_mean, jp.trace(st.cov[3:, 3:]), st.n_obs,
                     pe_hit, jp.where(pe_hit, pe_t, jp.inf), mm * jp.linalg.norm(st.mu[3:]),
                     st.mu[:3], st.mu[3:], st.m, mom, hit, w)
