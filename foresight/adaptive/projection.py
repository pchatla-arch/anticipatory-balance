"""WP 1.2 numerical primitives for the bounded proportional-actuator model.

Inputs broadcast over environments and joints. Each gain update consumes a
source-ordered measurement. Sensor delivery, point estimation, reacquisition,
and the action on an unsupported projection belong to the caller. Comparisons
use the active JAX precision without directed rounding or an implicit tolerance.
At a boundary, floating-point rounding can change inclusion or a reason code;
the real-arithmetic force criterion is not a floating-point certificate.
"""
from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jnp


class Interval(NamedTuple):
    low: jnp.ndarray
    high: jnp.ndarray


class GainStatus(IntEnum):
    UPDATED = 0
    SKIPPED = 1
    INCONSISTENT = 2
    INVALID_INPUT = 3


class GainUpdate(NamedTuple):
    interval: Interval
    status: jnp.ndarray


class ProjectionStatus(IntEnum):
    FEASIBLE = 0
    INVALID_INPUT = 1
    UNSUPPORTED_SATURATION = 2
    NO_EFFECTIVE_OFFSET = 3
    CONSTRAINT_CONFLICT = 4


class Projection(NamedTuple):
    status: jnp.ndarray
    candidate: jnp.ndarray
    effective_interval: Interval
    requested_interval: Interval
    requested: jnp.ndarray
    effective: jnp.ndarray


def _ordered(interval):
    return (jnp.isfinite(interval.low) & jnp.isfinite(interval.high)
            & (interval.low <= interval.high))


def _mask_interval(mask, low, high):
    return Interval(jnp.where(mask, low, jnp.nan), jnp.where(mask, high, jnp.nan))


def propagate_gain(current, prior, drift_rate, elapsed):
    """Expand by the declared drift bound over elapsed seconds; clip to prior."""
    valid = (_ordered(current) & _ordered(prior) & (prior.low > 0)
             & (current.low >= prior.low) & (current.high <= prior.high)
             & jnp.isfinite(drift_rate) & (drift_rate >= 0)
             & jnp.isfinite(elapsed) & (elapsed >= 0))
    radius = drift_rate * elapsed
    return _mask_interval(valid, jnp.maximum(prior.low, current.low - radius),
                          jnp.minimum(prior.high, current.high + radius))


def update_gain_set(current, prior, *, drift_rate, elapsed, held_target, position,
                    torque, kp, position_error_bound, torque_error_bound, sample_valid):
    """Intersect an expanded set with one aligned packet using additive error bounds.

    elapsed is the difference between consecutive processed source timestamps.
    A delivered but rejected packet propagates to its real source time. With
    no delivered packet, keep the set and source clock unchanged (a padded
    scan entry may use sample_valid=False and elapsed=0). An
    inconsistent accepted packet yields an empty (NaN) result for this step;
    reacquisition must start explicitly from the full prior. Unknown scale or
    timestamp error requires a justified contribution to the supplied bounds.
    """
    predicted = propagate_gain(current, prior, drift_rate, elapsed)
    model_valid = (_ordered(predicted) & jnp.isfinite(kp) & (kp > 0)
                   & jnp.isfinite(position_error_bound) & (position_error_bound >= 0)
                   & jnp.isfinite(torque_error_bound) & (torque_error_bound >= 0))
    phi = held_target - position
    y = jnp.divide(torque, kp)
    epsilon = jnp.divide(torque_error_bound, kp) + prior.high * position_error_bound
    usable = jnp.isfinite(phi) & jnp.isfinite(y) & jnp.isfinite(epsilon)
    sample_valid = jnp.asarray(sample_valid, dtype=bool)
    valid = model_valid & (~sample_valid | usable)

    # Only divide when a measurement bound cuts through a prior endpoint. This
    # leaves weak but consistent excitation at the prior without a tuning cutoff.
    weight = jnp.abs(phi)
    signed_y = jnp.where(phi < 0, -y, y)
    min_y, max_y = signed_y - epsilon, signed_y + epsilon
    possible = (min_y <= weight * predicted.high) & (max_y >= weight * predicted.low)
    trim_low = (weight > 0) & (min_y > weight * predicted.low)
    trim_high = (weight > 0) & (max_y < weight * predicted.high)
    low = jnp.where(trim_low, min_y / jnp.where(trim_low, weight, 1), predicted.low)
    high = jnp.where(trim_high, max_y / jnp.where(trim_high, weight, 1), predicted.high)
    consistent = possible & (low <= high)
    status = jnp.select([~valid, ~sample_valid, ~consistent],
                        [GainStatus.INVALID_INPUT, GainStatus.SKIPPED, GainStatus.INCONSISTENT],
                        default=GainStatus.UPDATED)
    accepted = status == GainStatus.UPDATED
    supported = accepted | (status == GainStatus.SKIPPED)
    return GainUpdate(_mask_interval(supported, jnp.where(accepted, low, predicted.low),
                                     jnp.where(accepted, high, predicted.high)), status)


def hold_error(reference, measured_position, *, position_error_bound, max_speed, age, hold):
    """Tracking-error interval until hold end using a declared absolute speed bound."""
    valid = (jnp.isfinite(reference) & jnp.isfinite(measured_position)
             & jnp.isfinite(position_error_bound) & (position_error_bound >= 0)
             & jnp.isfinite(max_speed) & (max_speed >= 0)
             & jnp.isfinite(age) & (age >= 0) & jnp.isfinite(hold) & (hold > 0))
    center = reference - measured_position
    radius = position_error_bound + max_speed * (age + hold)
    return _mask_interval(valid, center - radius, center + radius)


def project_offset(gains, errors, *, current_gains, estimate, current_error, reference, target_range,
                   kp, available_force, max_offset, slew_rate, hold, previous, regularizer):
    """Project a point inverse under the four gain/error corners and requested slew.

    gains must cover the whole hold (propagate_gain with elapsed = age + hold).
    current_gains covers the decision time only and constrains the point estimate.
    current_error is a separate causal point estimate. available_force is the
    declared supported ceiling after subtracting any margin. previous is the last issued
    request before target clipping. Check status before using any command: a
    failure returns NaN, so no unsupported value is mistaken for a fallback.
    """
    valid = (_ordered(gains) & (gains.low > 0) & _ordered(current_gains)
             & (gains.low <= current_gains.low) & (current_gains.high <= gains.high)
             & _ordered(errors) & _ordered(target_range)
             & jnp.isfinite(reference) & (target_range.low <= reference)
             & (reference <= target_range.high) & jnp.isfinite(current_error)
             & jnp.isfinite(estimate) & jnp.isfinite(previous)
             & jnp.isfinite(kp) & (kp > 0) & jnp.isfinite(available_force)
             & (available_force > 0) & jnp.isfinite(hold) & (hold > 0)
             & jnp.isfinite(slew_rate) & (slew_rate >= 0)
             & jnp.isfinite(max_offset) & (max_offset >= 0)
             & jnp.isfinite(regularizer) & (regularizer >= 0))

    physical_low = target_range.low - reference
    physical_high = target_range.high - reference
    low, high = physical_low, physical_high
    unsaturated = jnp.asarray(True)
    for gain in gains:
        force_radius = jnp.divide(available_force, gain * kp)
        for error in errors:
            nominal = kp * error
            faulty = (gain * kp) * error
            unsaturated &= jnp.maximum(jnp.abs(nominal), jnp.abs(faulty)) <= available_force
            root = jnp.divide(-2 * (gain - 1) * error, gain)
            low = jnp.maximum(low, jnp.maximum(jnp.minimum(0, root), -force_radius - error))
            high = jnp.minimum(high, jnp.minimum(jnp.maximum(0, root), force_radius - error))

    # The inverse image includes requests beyond the limits when target
    # clipping lands exactly on an admissible end of the effective interval.
    request_low = jnp.where(low <= physical_low, -jnp.inf, low)
    request_high = jnp.where(high >= physical_high, jnp.inf, high)
    request_low = jnp.maximum(request_low, jnp.maximum(-max_offset, previous - slew_rate * hold))
    request_high = jnp.minimum(request_high, jnp.minimum(max_offset, previous + slew_rate * hold))
    status = jnp.select([~valid, ~unsaturated, low > high, request_low > request_high],
                        [ProjectionStatus.INVALID_INPUT, ProjectionStatus.UNSUPPORTED_SATURATION,
                         ProjectionStatus.NO_EFFECTIVE_OFFSET, ProjectionStatus.CONSTRAINT_CONFLICT],
                        default=ProjectionStatus.FEASIBLE)
    feasible = status == ProjectionStatus.FEASIBLE
    candidate = (1 / jnp.clip(estimate, current_gains.low, current_gains.high) - 1) * current_error
    center = (candidate + regularizer * previous) / (1 + regularizer)
    requested = jnp.clip(center, request_low, request_high)
    effective = jnp.clip(reference + requested, target_range.low, target_range.high) - reference
    return Projection(status, jnp.where(feasible, candidate, jnp.nan),
                      _mask_interval(feasible | (status == ProjectionStatus.CONSTRAINT_CONFLICT),
                                     low, high),
                      _mask_interval(feasible, request_low, request_high),
                      jnp.where(feasible, requested, jnp.nan),
                      jnp.where(feasible, effective, jnp.nan))
