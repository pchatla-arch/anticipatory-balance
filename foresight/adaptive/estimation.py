# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Point estimation of actuator gain and the baseline correction methods (WP 2.1).

Model per actuator (pure proportional servo, unit gear, after control-range clipping):

    tau_i = g * Kp * (z_i - q_i)        =>   y_i = tau_i / Kp = g * phi_i,   phi_i = z_i - q_i

Every method below sees the same released telemetry packets, the same frozen actor and
the same common correction limits (applied by the environment). They differ only in how
they turn evidence into a requested offset x_k:

  none              x_k = 0
  oracle            privileged true gain (evaluation-only; implemented inside the env)
  estimated_inverse x_k = (1/g_hat - 1)(r_k - q_hat_k), always
  heuristic_gate    the same inverse, but only when validation-tuned thresholds on
                    sample count, residual, excitation, dead band and packet age pass
  candidate         ChatGPT's identification-aware projection (WP 1.2/2.2), consuming
                    the same RLS point estimate and the same packets

The RLS forgetting factor and the heuristic thresholds are tuned on the development split
with a recorded budget; the test split is run once. Nothing here receives the injected
gain or the fault onset.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from foresight.adaptive.sensing import MeasuredPacket


class RLSState(NamedTuple):
    """Scalar recursive least squares per actuator; arrays have shape (nu,)."""

    g_hat: jax.Array          # current point estimate
    cov: jax.Array            # scalar covariance P
    residual_sq: jax.Array    # exponentially weighted squared residual (y - g phi)^2
    excitation: jax.Array     # exponentially weighted |phi|
    n_valid: jax.Array        # int32, valid samples assimilated
    last_source_step: jax.Array   # int32, capture step of the newest assimilated sample (-1 none)
    last_position: jax.Array  # newest measured position (for the causal position estimate)
    last_velocity: jax.Array  # newest measured velocity


class RLSConfig(NamedTuple):
    forgetting: float = 0.98      # lambda in (0, 1]
    initial_cov: float = 1e3      # uninformative prior; a sample with |phi| = 0.03 already dominates it
    prior_gain: float = 1.0
    gain_min: float = 0.2         # projection of the point estimate onto the prior interval
    gain_max: float = 1.5
    ewma: float = 0.9             # for residual and excitation statistics
    min_excitation_rad: float = 1e-3   # ignore samples with |phi| below this (division safety)


def rls_init(nu, cfg: RLSConfig):
    return RLSState(
        g_hat=jp.full((nu,), cfg.prior_gain, jp.float32),
        cov=jp.full((nu,), cfg.initial_cov, jp.float32),
        residual_sq=jp.zeros((nu,), jp.float32),
        excitation=jp.zeros((nu,), jp.float32),
        n_valid=jp.zeros((nu,), jp.int32),
        last_source_step=jp.full((nu,), -1, jp.int32),
        last_position=jp.zeros((nu,), jp.float32),
        last_velocity=jp.zeros((nu,), jp.float32),
    )


def rls_update(state: RLSState, packet: MeasuredPacket, released, kp, cfg: RLSConfig):
    """Assimilate one released packet (all actuators at once). `released` is a bool scalar."""
    phi = packet.target_rad - packet.position_rad
    y = packet.torque_nm / kp
    use = released & packet.valid & (jp.abs(phi) >= cfg.min_excitation_rad)
    lam = cfg.forgetting
    denom = lam + phi * phi * state.cov
    gain_k = state.cov * phi / denom
    innovation = y - state.g_hat * phi
    g_new = state.g_hat + gain_k * innovation
    g_new = jp.clip(g_new, cfg.gain_min, cfg.gain_max)
    cov_new = (state.cov - gain_k * phi * state.cov) / lam
    residual_new = cfg.ewma * state.residual_sq + (1 - cfg.ewma) * innovation * innovation
    excitation_new = cfg.ewma * state.excitation + (1 - cfg.ewma) * jp.abs(phi)
    pos_update = released & packet.valid
    return RLSState(
        g_hat=jp.where(use, g_new, state.g_hat),
        cov=jp.where(use, cov_new, state.cov),
        residual_sq=jp.where(use, residual_new, state.residual_sq),
        excitation=jp.where(use, excitation_new, state.excitation),
        n_valid=state.n_valid + use.astype(jp.int32),
        last_source_step=jp.where(pos_update, packet.source_step, state.last_source_step),
        last_position=jp.where(pos_update, packet.position_rad, state.last_position),
        last_velocity=jp.where(pos_update, packet.velocity_rad_s, state.last_velocity),
    )


def causal_position(state: RLSState, now_step, dt):
    """Newest measured position propagated by its measured velocity over its age.

    This is the controller's best current estimate without the simulator's state; its
    error grows with packet age and joint speed, which is exactly the uncertainty the
    hold envelope (candidate method) accounts for.
    """
    age = jp.maximum(now_step - state.last_source_step, 0).astype(jp.float32) * dt
    return state.last_position + state.last_velocity * age, age


def inverse_offset(g_hat, reference, q_hat):
    """Point-estimate inverse: (1/g - 1)(r - q)."""
    return (1.0 / g_hat - 1.0) * (reference - q_hat)


class HeuristicConfig(NamedTuple):
    """Validation-tuned gate thresholds; a correction is requested only when all pass."""

    min_samples: int = 5
    max_residual_rms: float = 0.05     # in rad (units of y = tau/Kp)
    min_excitation_rad: float = 0.01   # EWMA |phi| must exceed this
    dead_band: float = 0.05            # |g_hat - 1| must exceed this
    max_age_steps: int = 5             # newest assimilated sample must be this fresh


REASON_APPLY, REASON_FEW_SAMPLES, REASON_RESIDUAL, REASON_EXCITATION, REASON_DEAD_BAND, REASON_STALE = range(6)


def heuristic_gate(state: RLSState, reference, now_step, dt, cfg: HeuristicConfig):
    """Return (requested offset, reason code per actuator)."""
    q_hat, _ = causal_position(state, now_step, dt)
    desired = inverse_offset(state.g_hat, reference, q_hat)
    age = now_step - state.last_source_step
    few = state.n_valid < cfg.min_samples
    noisy = jp.sqrt(state.residual_sq) > cfg.max_residual_rms
    weak = state.excitation < cfg.min_excitation_rad
    dead = jp.abs(state.g_hat - 1.0) <= cfg.dead_band
    stale = (age > cfg.max_age_steps) | (state.last_source_step < 0)
    reason = jp.where(few, REASON_FEW_SAMPLES,
             jp.where(stale, REASON_STALE,
             jp.where(noisy, REASON_RESIDUAL,
             jp.where(weak, REASON_EXCITATION,
             jp.where(dead, REASON_DEAD_BAND, REASON_APPLY)))))
    apply = reason == REASON_APPLY
    return jp.where(apply, desired, 0.0).astype(jp.float32), reason.astype(jp.int32)


def estimated_inverse(state: RLSState, reference, now_step, dt):
    """Ungated point inverse; the baseline that shows harmful adaptation under confounds."""
    q_hat, _ = causal_position(state, now_step, dt)
    return inverse_offset(state.g_hat, reference, q_hat).astype(jp.float32)


__all__ = ["RLSState", "RLSConfig", "rls_init", "rls_update", "causal_position", "inverse_offset",
           "HeuristicConfig", "heuristic_gate", "estimated_inverse",
           "REASON_APPLY", "REASON_FEW_SAMPLES", "REASON_RESIDUAL", "REASON_EXCITATION",
           "REASON_DEAD_BAND", "REASON_STALE"]
