# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Declared telemetry sensor models and in-order packet delivery (WP 1.1).

The simulator produces one exact, source-aligned actuator packet per control step
(`foresight.adaptive.telemetry`). Practical methods never see that packet. They see
what a declared sensor channel would deliver: encoder and effort readings with
noise, bias and an unknown effort scale, each packet arriving `latency` control
steps after its capture, sometimes never (dropout), and therefore sometimes out of
order. Capture time and arrival time are distinct fields.

Everything here is batched over environments and runs inside `jax.lax.scan`.

Delivery is in source order with a timeout. The consumer keeps a cursor at the next
unprocessed source step. A packet is released when it has arrived and every earlier
packet has been released or declared lost; a packet is declared lost once the
current step exceeds its capture step by more than the maximum latency. This keeps
the set-membership drift expansion exact (it needs the true time between processed
captures) and never pairs an old effort reading with a newer target.

Declared error bounds (`bound_position_rad`, `bound_torque_nm`) are what the
set-membership estimator is told, not what the noise actually is. With Gaussian noise
the bound is a coverage statement; the harness reports the empirical coverage.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp


class TelemetryConfig(NamedTuple):
    """Sensor channel and delivery model, all scalars or Python numbers.

    position_noise_rad / torque_noise_nm: Gaussian noise standard deviations.
    position_bias_rad / torque_bias_nm: constant offsets (per episode, same sign all joints).
    torque_scale: multiplicative effort-sensor scale c (1.0 = calibrated).
    latency_min/max: capture-to-arrival delay in control steps, uniform integer.
    dropout: probability a packet never arrives.
    bound_position_rad / bound_torque_nm: the error bounds declared to the estimator.
    """

    position_noise_rad: float = 0.0
    torque_noise_nm: float = 0.0
    position_bias_rad: float = 0.0
    torque_bias_nm: float = 0.0
    torque_scale: float = 1.0
    latency_min: int = 0
    latency_max: int = 0
    dropout: float = 0.0
    bound_position_rad: float = 0.0
    bound_torque_nm: float = 0.0

    def validate(self):
        if self.latency_min < 0 or self.latency_max < self.latency_min:
            raise ValueError("latency range must satisfy 0 <= min <= max")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.torque_scale <= 0:
            raise ValueError("torque_scale must be positive")
        for name in ("position_noise_rad", "torque_noise_nm", "bound_position_rad", "bound_torque_nm"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be nonnegative")
        return self


EXACT = TelemetryConfig()
DECLARED = TelemetryConfig(position_noise_rad=2e-3, torque_noise_nm=0.5,
                           latency_min=1, latency_max=3, dropout=0.02,
                           bound_position_rad=6e-3, bound_torque_nm=1.5)


class MeasuredPacket(NamedTuple):
    """What the sensor channel delivers for one capture; arrays have shape (nu,)."""

    source_step: jax.Array        # int32 scalar, control step of capture
    arrival_step: jax.Array       # int32 scalar, >= source_step; huge when dropped
    target_rad: jax.Array         # held command, known exactly to the controller
    position_rad: jax.Array       # encoder reading
    velocity_rad_s: jax.Array     # encoder-derived velocity reading
    torque_nm: jax.Array          # effort reading after scale, bias, noise
    valid: jax.Array              # bool (nu,): finite and unsaturated at the source


DROPPED = jp.int32(2**30)


def measure(exact, source_step, cfg: TelemetryConfig, key):
    """Corrupt one exact packet into a MeasuredPacket.

    `exact` is a mapping with the info["telemetry_*"] fields of one environment.
    The random key is split for noise, latency and dropout; the same key with the
    same config yields the same measurement, so paired comparisons share sensor noise.
    """
    k_q, k_tau, k_lat, k_drop = jax.random.split(key, 4)
    q = exact["telemetry_joint_position_rad"]
    nu = q.shape[0]
    q_meas = q + cfg.position_bias_rad + cfg.position_noise_rad * jax.random.normal(k_q, (nu,))
    tau = exact["telemetry_applied_joint_torque_nm"]
    tau_meas = (cfg.torque_scale * tau + cfg.torque_bias_nm
                + cfg.torque_noise_nm * jax.random.normal(k_tau, (nu,)))
    latency = jax.random.randint(k_lat, (), cfg.latency_min, cfg.latency_max + 1)
    dropped = jax.random.uniform(k_drop) < cfg.dropout
    arrival = jp.where(dropped, DROPPED, source_step + latency).astype(jp.int32)
    return MeasuredPacket(
        source_step=jp.asarray(source_step, jp.int32),
        arrival_step=arrival,
        target_rad=exact["telemetry_held_target_rad"],
        position_rad=q_meas.astype(jp.float32),
        velocity_rad_s=exact["telemetry_joint_velocity_rad_s"],
        torque_nm=tau_meas.astype(jp.float32),
        valid=exact["telemetry_regression_valid"],
    )


class DeliveryQueue(NamedTuple):
    """Fixed ring buffer of measured packets indexed by source step modulo `slots`."""

    source_step: jax.Array     # (slots,) int32, -1 when empty
    arrival_step: jax.Array    # (slots,) int32
    target_rad: jax.Array      # (slots, nu)
    position_rad: jax.Array    # (slots, nu)
    velocity_rad_s: jax.Array  # (slots, nu)
    torque_nm: jax.Array       # (slots, nu)
    valid: jax.Array           # (slots, nu) bool
    cursor: jax.Array          # int32 scalar: next source step to release
    lost: jax.Array            # int32 scalar: packets declared lost so far
    released: jax.Array        # int32 scalar: packets released so far


def queue_init(nu, slots):
    return DeliveryQueue(
        source_step=jp.full((slots,), -1, jp.int32),
        arrival_step=jp.full((slots,), DROPPED, jp.int32),
        target_rad=jp.zeros((slots, nu), jp.float32),
        position_rad=jp.zeros((slots, nu), jp.float32),
        velocity_rad_s=jp.zeros((slots, nu), jp.float32),
        torque_nm=jp.zeros((slots, nu), jp.float32),
        valid=jp.zeros((slots, nu), bool),
        cursor=jp.int32(0), lost=jp.int32(0), released=jp.int32(0),
    )


def queue_slots(cfg: TelemetryConfig, release_per_step):
    """Enough slots that a packet is never overwritten before its release or loss.

    The cursor lags the newest capture by at most latency_max + 1 steps (loss timeout)
    plus the backlog that `release_per_step` drains; a doubled margin covers both. If a
    slot were ever overwritten early, the stale source step would no longer match the
    cursor and the packet would be counted lost, never mis-paired.
    """
    return 2 * (int(cfg.latency_max) + 1) + int(release_per_step) + 4


def queue_push(queue: DeliveryQueue, packet: MeasuredPacket):
    slots = queue.source_step.shape[0]
    i = packet.source_step % slots
    return queue._replace(
        source_step=queue.source_step.at[i].set(packet.source_step),
        arrival_step=queue.arrival_step.at[i].set(packet.arrival_step),
        target_rad=queue.target_rad.at[i].set(packet.target_rad),
        position_rad=queue.position_rad.at[i].set(packet.position_rad),
        velocity_rad_s=queue.velocity_rad_s.at[i].set(packet.velocity_rad_s),
        torque_nm=queue.torque_nm.at[i].set(packet.torque_nm),
        valid=queue.valid.at[i].set(packet.valid),
    )


def queue_release(queue: DeliveryQueue, now_step, max_latency, count):
    """Release up to `count` packets in source order that are due at `now_step`.

    Returns (queue, packets, mask) where `packets` is a MeasuredPacket with a leading
    axis of length `count` and `mask` marks the real releases. A packet is due when it
    has arrived (arrival <= now) and its source step equals the cursor; a cursor
    position whose packet has not arrived by now - max_latency is declared lost and
    skipped. A cursor at a step not yet captured stops the release.
    """
    slots = queue.source_step.shape[0]

    def body(q, _):
        i = q.cursor % slots
        captured = q.source_step[i] == q.cursor
        arrived = captured & (q.arrival_step[i] <= now_step)
        lost = (q.cursor <= now_step - max_latency - 1) & ~arrived & (q.cursor < now_step + 1)
        # `lost` also covers packets that were never pushed for a past step (dropout
        # removes them before the push) once they are older than the maximum latency.
        release = arrived
        advance = release | lost
        packet = MeasuredPacket(
            source_step=q.cursor, arrival_step=q.arrival_step[i],
            target_rad=q.target_rad[i], position_rad=q.position_rad[i],
            velocity_rad_s=q.velocity_rad_s[i], torque_nm=q.torque_nm[i],
            valid=q.valid[i] & release)
        q = q._replace(cursor=q.cursor + advance.astype(jp.int32),
                       lost=q.lost + lost.astype(jp.int32),
                       released=q.released + release.astype(jp.int32))
        return q, (packet, release)

    queue, (packets, mask) = jax.lax.scan(body, queue, None, length=count)
    return queue, packets, mask


def declared_bounds(cfg: TelemetryConfig, kp, g_max):
    """Conservative per-sample bound epsilon_i for |y_i - g phi_i| in units of rad.

    epsilon = b_tau / Kp + g_max * b_q (proposal 0.3), with b_q, b_tau the declared bounds.
    """
    kp = jp.asarray(kp)
    return cfg.bound_torque_nm / kp + g_max * cfg.bound_position_rad


def summarize(cfg: TelemetryConfig):
    return dict(cfg._asdict())


def config_from_dict(values):
    return TelemetryConfig(**{k: (int(v) if k in ("latency_min", "latency_max") else float(v))
                              for k, v in values.items()}).validate()


__all__ = ["TelemetryConfig", "EXACT", "DECLARED", "MeasuredPacket", "DROPPED", "measure",
           "DeliveryQueue", "queue_init", "queue_slots", "queue_push", "queue_release",
           "declared_bounds", "summarize", "config_from_dict"]
