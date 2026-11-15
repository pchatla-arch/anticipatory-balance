# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Gaussian kinematic belief with an independent, unobservable mass prior.

The corrected default models pre-impact flight as deterministic constant velocity.
Timestamped point updates are linear for that model. Empty scheduled observations
use a bounded particle moment match. This is an approximation to the full task:
detection selection, robot prediction and post-impact motion are not Gaussian.
The explicit ``inference_mode='legacy'`` retains the historical covariance/update for
controlled implementation comparisons, and must be labeled as such in results.

Lookahead uses finite particle quadrature, including a missing-observation branch. It
forecasts the next scheduled delivery and the position at that observation's capture
stamp. It does not use true projectile mass, launch state, position or velocity.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from foresight.decision import contracts as C


class PFConfig(NamedTuple):
  n_particles: int = 128
  mass_range: tuple = (4.0, 16.0)
  init_speed: float = 5.0
  init_vel_sigma: float = 3.0
  process_pos_sigma: float = 0.0
  process_vel_sigma: float = 0.0
  r_contact: float = 0.45
  dt: float = 0.02
  z_hit: tuple = (0.3, 1.4)
  min_future: float = 0.0
  inference_mode: str = "deterministic"  # deterministic | legacy


class PFState(NamedTuple):
  mu: jax.Array
  cov: jax.Array
  m: jax.Array
  n_obs: jax.Array
  initialized: jax.Array


class BeliefStats(NamedTuple):
  p_hit: jax.Array
  t_hit: jax.Array
  t_hit_std: jax.Array
  momentum_mean: jax.Array
  ess: jax.Array
  n_obs: jax.Array
  pe_hit: jax.Array
  pe_t_hit: jax.Array
  pe_momentum: jax.Array
  pe_pos: jax.Array
  pe_vel: jax.Array
  mass_samples: jax.Array
  momentum_samples: jax.Array
  hit_samples: jax.Array
  weights: jax.Array
  # Defaults keep old named/positional construction usable for compatibility tests.
  t_pass: jax.Array | None = None  # upstream closest-approach diagnostic; not deadline release
  t_hit_samples: jax.Array | None = None
  state_samples: jax.Array | None = None


class Lookahead(NamedTuple):
  future: BeliefStats           # every field has a leading observation-branch axis
  posterior_weights: jax.Array  # (observation branches, posterior particles)
  branch_probability: jax.Array # includes probability that contact has not resolved
  prefix_hit_weights: jax.Array # current particle weight hitting during the wait
  wait_steps: jax.Array
  delivery_step: jax.Array
  capture_step: jax.Array
  posterior_mean: jax.Array     # (branches, 6), Gaussian before entry-event conditioning
  posterior_covariance: jax.Array  # (branches, 6, 6), analytic for delivered-point branches
  posterior_fallback: jax.Array # branch had no active posterior samples; use blind prior


def validate_config(cfg: PFConfig):
  if cfg.inference_mode not in ("deterministic", "legacy"):
    raise ValueError("unknown inference_mode")
  if cfg.inference_mode == "deterministic" and (cfg.process_pos_sigma or cfg.process_vel_sigma):
    raise ValueError("timestamped deterministic filter requires Q=0; stochastic delayed dynamics need an augmented-state filter")
  if cfg.n_particles < 2 or cfg.dt <= 0 or cfg.init_vel_sigma <= 0:
    raise ValueError("invalid particle count, dt, or velocity prior")
  if not (0 < cfg.mass_range[0] <= cfg.mass_range[1]):
    raise ValueError("mass prior must be positive and ordered")


def init_state(cfg: PFConfig) -> PFState:
  validate_config(cfg)
  N = cfg.n_particles
  return PFState(jp.zeros(6), jp.eye(6), jp.full((N,), 8.0), jp.int32(0), jp.array(False))


def _init_from_detection(cfg: PFConfig, det_pos, det_sigma, delay_s, pelvis_pos, m_rng) -> PFState:
  to_robot = pelvis_pos - det_pos
  d = jp.linalg.norm(to_robot[:2]) + 1e-6
  v0 = jp.array([to_robot[0] / d, to_robot[1] / d, 0.0]) * cfg.init_speed
  mu = jp.concatenate([det_pos + v0 * delay_s, v0])
  if cfg.inference_mode == "legacy":
    sp = det_sigma ** 2 + (cfg.init_vel_sigma * delay_s) ** 2
    cov = jp.diag(jp.concatenate([jp.full(3, sp), jp.full(3, cfg.init_vel_sigma ** 2)]))
  else:
    cov = C.transported_covariance(det_sigma, cfg.init_vel_sigma, delay_s, jp)
  m = jax.random.uniform(m_rng, (cfg.n_particles,), minval=cfg.mass_range[0], maxval=cfg.mass_range[1])
  return PFState(mu, cov, m, jp.int32(1), jp.array(True))


def predict(cfg: PFConfig, st: PFState) -> PFState:
  F = C.transition(cfg.dt, jp)
  Q = jp.diag(jp.concatenate([jp.full(3, cfg.process_pos_sigma ** 2),
                              jp.full(3, cfg.process_vel_sigma ** 2)]))
  mu, cov = F @ st.mu, F @ st.cov @ F.T + Q
  return st._replace(mu=jp.where(st.initialized, mu, st.mu), cov=jp.where(st.initialized, cov, st.cov))


def update(cfg: PFConfig, st: PFState, det_valid, det_pos, det_sigma, capture_step, step_idx,
           pelvis_pos, rng) -> PFState:
  delay_s = (jp.asarray(step_idx) - capture_step).astype(jp.float32) * cfg.dt

  def do_init(_):
    return _init_from_detection(cfg, det_pos, det_sigma, delay_s, pelvis_pos, rng)

  def do_update(_):
    if cfg.inference_mode == "legacy":
      H = C.measurement_matrix(delay_s, jp)
      R = det_sigma ** 2 * jp.eye(3)
      S = H @ st.cov @ H.T + R
      K = st.cov @ H.T @ jp.linalg.inv(S)
      mu = st.mu + K @ (det_pos - H @ st.mu)
      cov = (jp.eye(6) - K @ H) @ st.cov
      cov = 0.5 * (cov + cov.T)
    else:
      mu, cov = C.gaussian_update(st.mu, st.cov, det_pos, delay_s, det_sigma, jp)
    return st._replace(mu=mu, cov=cov, n_obs=st.n_obs + 1)

  usable = det_valid & (capture_step >= 0) & (capture_step <= step_idx)
  return jax.lax.cond(usable, lambda _: jax.lax.cond(st.initialized, do_update, do_init, None), lambda _: st, None)


def _sample_states(st: PFState, rng, n):
  L = jp.linalg.cholesky(st.cov + 1e-8 * jp.eye(6))
  return st.mu[None] + jax.random.normal(rng, (n, 6)) @ L.T


def _capture_likelihood(detector_cfg, x, capture_dt, pelvis_pos, pelvis_vel, pelvis_fwd):
  """Shared approximate capture-time visibility/noise for filtering and planning."""
  captured_pos = x[:, :3] + capture_dt * x[:, 3:]
  capture_pelvis = pelvis_pos + capture_dt * pelvis_vel
  rel = captured_pos - capture_pelvis
  ranges = jp.linalg.norm(rel[:, :2], axis=-1)
  heading = pelvis_fwd[:2] / (jp.linalg.norm(pelvis_fwd[:2]) + 1e-9)
  cosine = jp.sum(rel[:, :2] * heading[None], axis=-1) / jp.maximum(ranges, 1e-9)
  visible = ((ranges < detector_cfg.max_range) & (ranges > getattr(detector_cfg, "min_range", 0.0)) &
             (cosine > jp.cos(jp.deg2rad(detector_cfg.fov_deg / 2))))
  detect_probability = visible.astype(jp.float32) * (1.0 - detector_cfg.miss_prob)
  sigma = jp.maximum(detector_cfg.sigma0 + detector_cfg.sigma1 * ranges, 1e-6)
  return captured_pos, detect_probability, sigma


def update_missing(cfg: PFConfig, st: PFState, detector_cfg, empty_scheduled,
                   pelvis_pos, pelvis_vel, pelvis_fwd, rng):
  """Moment-match a scheduled empty observation using the planner's same likelihood.

  Equal likelihood over all samples is explicitly a no-op, avoiding Monte Carlo
  drift from uninformative dropout. An impossible empty event under the model also
  leaves the belief unchanged; its zero likelihood is returned for diagnostics.
  Mass samples and n_obs (number of delivered points) are never modified.
  """
  if cfg.inference_mode == "legacy":
    return st, jp.array(False), jp.asarray(1.0)

  def condition(_):
    x = _sample_states(st, rng, cfg.n_particles)
    _, pd, _ = _capture_likelihood(detector_cfg, x, -detector_cfg.latency * cfg.dt,
                                    pelvis_pos, pelvis_vel, pelvis_fwd)
    likelihood = 1.0 - pd
    evidence = jp.mean(likelihood)
    informative = (jp.max(likelihood) - jp.min(likelihood) > 1e-6) & (evidence > 1e-12)
    weights = C.normalize_weights(likelihood, jp)
    mu = jp.sum(weights[:, None] * x, axis=0)
    centered = x - mu[None]
    cov = (centered.T * weights[None, :]) @ centered
    # Moment selection can leave only a few effective samples. A scale-aware
    # floor keeps float32 Cholesky stable without pretending Q became nonzero.
    cov += 1e-6 * jp.maximum(1.0, jp.trace(cov) / 6.0) * jp.eye(6)
    cov = (cov + cov.T) * 0.5
    new = st._replace(mu=jp.where(informative, mu, st.mu), cov=jp.where(informative, cov, st.cov))
    return new, informative, evidence

  return jax.lax.cond(empty_scheduled & st.initialized, condition,
                      lambda _: (st, jp.array(False), jp.asarray(1.0)), None)


def _geometry(cfg: PFConfig, p, v_relative, pelvis_pos, world_vz):
  rel = pelvis_pos[:2] - p[:2]
  vxy = v_relative[:2]
  sp2 = jp.sum(vxy ** 2) + 1e-6
  t_ca = jp.dot(rel, vxy) / sp2
  d_min2 = jp.maximum(jp.sum(rel ** 2) - jp.dot(rel, vxy) ** 2 / sp2, 0.0)
  inside = d_min2 < cfg.r_contact ** 2
  t_enter = t_ca - jp.sqrt(jp.maximum(cfg.r_contact ** 2 - d_min2, 0.0) / sp2)
  # The cylinder moves with the pelvis in XY; z_hit is a WORLD-height band.
  z_at = p[2] + world_vz * t_enter
  hit = inside & (t_enter > cfg.min_future) & (z_at > cfg.z_hit[0]) & (z_at < cfg.z_hit[1])
  return hit, t_enter


def _closest_time(p, v_relative, pelvis_pos):
  rel = pelvis_pos[:2] - p[:2]
  return jp.dot(rel, v_relative[:2]) / (jp.sum(v_relative[:2] ** 2) + 1e-6)


def stats_from_samples(cfg, st, x, weights, pelvis_pos, pelvis_vel):
  p, velocity = x[:, :3], x[:, 3:]
  hit, times = jax.vmap(lambda pi, vi: _geometry(cfg, pi, vi - pelvis_vel, pelvis_pos, vi[2]))(p, velocity)
  hit = hit & st.initialized
  mom = st.m * jp.linalg.norm(velocity, axis=-1)
  wh = weights * hit
  p_hit = jp.sum(wh)
  wh_n = wh / jp.maximum(p_hit, 1e-9)
  mean = jp.sum(wh_n * times)
  t_hit = jp.where(p_hit > 1e-6, mean, jp.inf)
  variance = jp.maximum(jp.sum(wh_n * (times - mean) ** 2), 0.0)
  t_std = jp.where(p_hit > 1e-6, jp.sqrt(variance), jp.inf)
  mu = jp.sum(weights[:, None] * x, axis=0)
  pe_hit, pe_t = _geometry(cfg, mu[:3], mu[3:] - pelvis_vel, pelvis_pos, mu[5])
  pe_hit = pe_hit & st.initialized
  t_pass = jp.where(st.initialized, _closest_time(mu[:3], mu[3:] - pelvis_vel, pelvis_pos), jp.inf)
  return BeliefStats(p_hit, t_hit, t_std, jp.sum(wh_n * mom), jp.trace(st.cov[3:, 3:]), st.n_obs,
                     pe_hit, jp.where(pe_hit, pe_t, jp.inf), jp.mean(st.m) * jp.linalg.norm(mu[3:]),
                     mu[:3], mu[3:], st.m, mom, hit, weights, t_pass, times, x)


def stats(cfg: PFConfig, st: PFState, pelvis_pos, pelvis_vel, rng) -> BeliefStats:
  N = cfg.n_particles
  x = _sample_states(st, rng, N)
  b = stats_from_samples(cfg, st, x, jp.full((N,), 1.0 / N), pelvis_pos, pelvis_vel)
  # TTC threshold uses the analytic Gaussian mean, not Monte Carlo mean noise.
  pe_hit, pe_t = _geometry(cfg, st.mu[:3], st.mu[3:] - pelvis_vel, pelvis_pos, st.mu[5])
  pe_hit = pe_hit & st.initialized
  return b._replace(pe_hit=pe_hit, pe_t_hit=jp.where(pe_hit, pe_t, jp.inf),
                    pe_pos=st.mu[:3], pe_vel=st.mu[3:],
                    t_pass=jp.where(st.initialized, _closest_time(st.mu[:3], st.mu[3:] - pelvis_vel, pelvis_pos), jp.inf),
                    pe_momentum=jp.mean(st.m) * jp.linalg.norm(st.mu[3:]))


def lookahead(cfg: PFConfig, st: PFState, b: BeliefStats, detector_cfg, step_idx,
              pelvis_pos, pelvis_vel, pelvis_fwd, rng, k: int = 16,
              observe: bool = True) -> Lookahead:
  """Next-delivery quadrature with analytic delayed-point posterior updates.

  Monte Carlo estimates whether the next capture is visible and samples its noisy
  point. Each valid point is then absorbed by the SAME Gaussian/Joseph update as
  the deployed filter; fresh samples from that posterior preserve unresolved
  velocity uncertainty. It is not finite-support importance conditioning.

  Empty observations use the deployed filter's bounded moment-match approximation.
  Both sensor likelihoods assume the declared pelvis-cone geometry. Reference-entry
  events inside the wait retain their CONTINUE response cost; subsequent action
  values condition on that single-impact event not having resolved yet.
  """
  delivery, capture, wait = C.next_delivery(step_idx, detector_cfg.period, detector_cfg.latency, jp)
  wait_s = wait * cfg.dt
  early = b.hit_samples & (b.t_hit_samples <= wait_s)
  prefix = b.weights * early
  active = b.weights * ~early
  active_mass = jp.sum(active)
  prior = C.normalize_weights(active, jp)
  x = b.state_samples
  future_x = jp.concatenate((x[:, :3] + wait_s * x[:, 3:], x[:, 3:]), axis=1)
  future_pelvis = pelvis_pos + wait_s * pelvis_vel
  F = C.transition(wait_s, jp)
  future_mu, future_cov = F @ st.mu, F @ st.cov @ F.T
  future_st = st._replace(mu=future_mu, cov=future_cov)
  blind = stats_from_samples(cfg, future_st, future_x, prior, future_pelvis, pelvis_vel)

  def one_branch(value):
    return jax.tree_util.tree_map(lambda a: a[None], value)

  if not observe or detector_cfg.miss_prob == 1.0:
    return Lookahead(one_branch(blind), prior[None], jp.asarray([active_mass]), prefix, wait,
                     delivery, capture, future_mu[None], future_cov[None], jp.asarray([False]))

  capture_dt = (capture - step_idx) * cfg.dt
  captured_pos, detect_probability, sigma = _capture_likelihood(
      detector_cfg, x, capture_dt, pelvis_pos, pelvis_vel, pelvis_fwd)
  detected_mass = jp.sum(prior * detect_probability)
  valid_prior = C.normalize_weights(prior * detect_probability, jp)
  key_i, key_noise, key_samples, key_empty = jax.random.split(rng, 4)
  indices = jax.random.categorical(key_i, jp.log(jp.maximum(valid_prior, 1e-30)), shape=(k,))
  observation_sigma = sigma[indices]
  z = captured_pos[indices] + observation_sigma[:, None] * jax.random.normal(key_noise, (k, 3))

  def condition_active(posterior, samples):
    # Reconstruct present geometry to identify which hypotheses cross the
    # reference-entry boundary during the wait. Fall/prefix cost is already paid.
    current_p = samples[:, :3] - wait_s * samples[:, 3:]
    hit_now, entry_now = jax.vmap(lambda pi, vi: _geometry(
        cfg, pi, vi - pelvis_vel, pelvis_pos, vi[2]))(current_p, samples[:, 3:])
    keep = ~(hit_now & (entry_now <= wait_s))
    any_active = jp.any(keep)
    weights = C.normalize_weights(keep.astype(jp.float32), jp)
    result = stats_from_samples(cfg, posterior, samples, weights, future_pelvis, pelvis_vel)
    # A finite draw can miss a rare active component. Falling back to its blind
    # prior keeps that component's costs instead of silently dropping them.
    result = jax.tree_util.tree_map(lambda new, old: jp.where(any_active, new, old), result, blind)
    return result, ~any_active

  sample_keys = jax.random.split(key_samples, k + 1)

  def point_branch(zi, sigi, key):
    mu, cov = C.gaussian_update(future_mu, future_cov, zi,
                               detector_cfg.latency * cfg.dt, sigi, jp)
    posterior = future_st._replace(mu=mu, cov=cov, n_obs=st.n_obs + 1)
    result, fallback = condition_active(posterior, _sample_states(posterior, key, cfg.n_particles))
    return result, mu, cov, fallback

  observed, mus, covs, fallbacks = jax.vmap(point_branch)(z, observation_sigma, sample_keys[1:])
  empty_st, empty_applied, _ = update_missing(cfg, future_st, detector_cfg, jp.array(True),
                                              future_pelvis, pelvis_vel, pelvis_fwd, key_empty)
  empty_samples = _sample_states(empty_st, sample_keys[0], cfg.n_particles)
  empty_stats, empty_fallback = condition_active(empty_st, empty_samples)
  # Exact no-op for uninformative dropout, with common samples for blind waiting.
  empty_stats = jax.tree_util.tree_map(lambda new, old: jp.where(empty_applied, new, old), empty_stats, blind)
  empty_fallback = empty_applied & empty_fallback
  branches = jax.tree_util.tree_map(lambda a, bs: jp.concatenate((a[None], bs), axis=0), empty_stats, observed)
  probabilities = active_mass * jp.concatenate((jp.asarray([1.0 - detected_mass]), jp.full((k,), detected_mass / k)))
  return Lookahead(branches, branches.weights, probabilities, prefix, wait, delivery, capture,
                   jp.concatenate((empty_st.mu[None], mus), axis=0),
                   jp.concatenate((empty_st.cov[None], covs), axis=0),
                   jp.concatenate((jp.asarray([empty_fallback]), fallbacks)))
