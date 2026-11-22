"""WP 2.2 candidate. All controller inputs are causal and available at execution."""
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import NamedTuple

import jax.numpy as jp
import numpy as np

from foresight.adaptive import projection as P
from foresight.adaptive.estimation import causal_position


class Reason(IntEnum):
    APPLY = 0                    # shared bench apply counter
    NO_EFFECTIVE_CHANGE = 10
    NO_TELEMETRY = 11
    REACQUIRING = 12
    MODEL_MISMATCH = 13
    UNSUPPORTED_FORCE = 14
    CONSTRAINT_CONFLICT = 15
    INVALID_INPUT = 16
    NO_EFFECTIVE_OFFSET = 17


@dataclass(frozen=True)
class Config:
    gain_drift_per_s: float
    hold_speed_rad_s: float
    force_margin_nm: float
    roundoff_effort_bound_nm: float
    reacquire_packets: int
    regularizer: float
    hold_accel_rad_s2: float | None = None

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "hold_accel_rad_s2" and value is None:
                continue
            if name == "reacquire_packets":
                if type(value) is not int or value < 1:
                    raise ValueError("reacquire_packets must be a positive integer")
            elif isinstance(value, bool) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.hold_speed_rad_s == 0:
            raise ValueError("hold_speed_rad_s must be positive for the walking task")


class CandidateState(NamedTuple):
    gains: P.Interval
    last_source: object
    informative_count: object
    mismatch: object


@dataclass(frozen=True)
class Candidate:
    config: Config
    prior: P.Interval
    kp: object
    ctrl_range: object
    ctrl_limited: object
    force_ceiling: object
    max_offset: float
    max_rate: float
    dt: float
    encoder_error_rad: float
    effort_error_nm: float
    min_excitation_rad: float
    velocity_error_rad_s: float = 0.0

    @classmethod
    def bind(cls, env, telemetry, rls, config):
        """Use nominal public limits; hidden gain schedules and confound labels stay outside."""
        layout = env._telemetry_layout
        if layout is None:
            raise ValueError("source-aligned telemetry is required")
        # A symmetric working ceiling conservatively respects both sides of each limit.
        ceiling = np.full(env.action_size, np.inf)
        for limited, limits in ((layout.actuator_force_limited, layout.actuator_force_range),
                                (layout.joint_force_limited, layout.joint_force_range)):
            sides = np.minimum(-limits[:, 0], limits[:, 1])
            ceiling = np.minimum(ceiling, np.where(limited, sides, np.inf))
        return cls(config, P.Interval(rls.gain_min, rls.gain_max),
                   jp.asarray(env.mj_model.actuator_gainprm[:, 0], jp.float32),
                   jp.asarray(layout.ctrl_range, jp.float32),
                   jp.asarray(layout.ctrl_limited & layout.control_clipping_enabled),
                   jp.asarray(ceiling, jp.float32), env.max_correction_rad,
                   env.max_correction_rate_rad_s, env.dt, telemetry.bound_position_rad,
                   telemetry.bound_torque_nm, rls.min_excitation_rad, telemetry.velocity_error_rad_s)

    def init(self, n, nu):
        shape = n, nu
        return CandidateState(P.Interval(jp.full(shape, self.prior.low, jp.float32),
                                         jp.full(shape, self.prior.high, jp.float32)),
                              jp.full(shape, -1, jp.int32), jp.zeros(shape, jp.int32),
                              jp.zeros(shape, bool))

    def update_with_status(self, state, packet, released):
        """Consume one ordered release; quarantine inconsistent data and restart at prior."""
        source = packet.source_step[:, None]
        fresh = released[:, None] & (source > state.last_source)
        elapsed = jp.where(fresh & (state.last_source >= 0), source - state.last_source, 0) * self.dt
        updated = P.update_gain_set(
            state.gains, self.prior, drift_rate=self.config.gain_drift_per_s, elapsed=elapsed,
            held_target=packet.target_rad, position=packet.position_rad, torque=packet.torque_nm,
            kp=self.kp, position_error_bound=self.encoder_error_rad,
            torque_error_bound=self.effort_error_nm + self.config.roundoff_effort_bound_nm,
            sample_valid=fresh & packet.valid)
        bad = (updated.status == P.GainStatus.INCONSISTENT) | (updated.status == P.GainStatus.INVALID_INPUT)
        # The packet which caused an inconsistency is not also counted toward reacquisition.
        low = jp.where(bad, self.prior.low, updated.interval.low)
        high = jp.where(bad, self.prior.high, updated.interval.high)
        informative = ((updated.status == P.GainStatus.UPDATED)
                       & (jp.abs(packet.target_rad - packet.position_rad) >= self.min_excitation_rad))
        count = jp.where(bad, 0, jp.minimum(state.informative_count + informative,
                                            self.config.reacquire_packets))
        mismatch = jp.where(bad, True, jp.where(informative, False, state.mismatch))
        next_state = CandidateState(P.Interval(low, high), jp.where(fresh, source, state.last_source),
                                    count, mismatch)
        return next_state, updated.status

    def update(self, state, packet, released):
        return self.update_with_status(state, packet, released)[0]

    def evaluate(self, state, rls, reference, step, previous):
        """Project or withdraw from the last *issued* request at the common slew rate."""
        # The bench labels telemetry by the transition that just finished.
        capture_index = step - 1
        # Match the heuristic's point prediction exactly; the hold envelope
        # separately uses the physical capture time of the post-step encoder.
        q_hat, _ = causal_position(rls, step, self.dt)
        position_age = jp.maximum(capture_index - rls.last_source_step, 0) * self.dt
        gain_age = jp.maximum(capture_index - state.last_source, 0) * self.dt
        now = P.propagate_gain(state.gains, self.prior, self.config.gain_drift_per_s, gain_age)
        held = P.propagate_gain(state.gains, self.prior, self.config.gain_drift_per_s, gain_age + self.dt)
        speed = self.config.hold_speed_rad_s
        if self.config.hold_accel_rad_s2 is not None:
            # Source velocity and position come from the same released measurement.
            # For every t <= age + hold, displacement is bounded by (|v|+b_v)t + A*t²/2.
            speed = (jp.abs(rls.last_velocity) + self.velocity_error_rad_s
                     + 0.5 * self.config.hold_accel_rad_s2 * (position_age + self.dt))
        errors = P.hold_error(reference, rls.last_position, position_error_bound=self.encoder_error_rad,
                              max_speed=speed, age=position_age, hold=self.dt)
        # For a physically unlimited target the possible request spans ±max_offset.
        targets = P.Interval(jp.where(self.ctrl_limited, self.ctrl_range[:, 0], reference-self.max_offset),
                             jp.where(self.ctrl_limited, self.ctrl_range[:, 1], reference+self.max_offset))
        force = self.force_ceiling - self.config.force_margin_nm
        out = P.project_offset(
            held, errors, current_gains=now, estimate=rls.g_hat, current_error=reference-q_hat,
            reference=reference, target_range=targets, kp=self.kp, available_force=force,
            max_offset=self.max_offset, slew_rate=self.max_rate, hold=self.dt, previous=previous,
            regularizer=self.config.regularizer)
        reason = jp.select(
            [state.mismatch, rls.last_source_step < 0,
             state.informative_count < self.config.reacquire_packets,
             ~jp.isfinite(force) | (force <= 0) | (out.status == P.ProjectionStatus.UNSUPPORTED_SATURATION),
             out.status == P.ProjectionStatus.CONSTRAINT_CONFLICT,
             out.status == P.ProjectionStatus.NO_EFFECTIVE_OFFSET,
             out.status != P.ProjectionStatus.FEASIBLE, out.effective == 0],
            [Reason.MODEL_MISMATCH, Reason.NO_TELEMETRY, Reason.REACQUIRING,
             Reason.UNSUPPORTED_FORCE, Reason.CONSTRAINT_CONFLICT, Reason.NO_EFFECTIVE_OFFSET,
             Reason.INVALID_INPUT, Reason.NO_EFFECTIVE_CHANGE], default=Reason.APPLY)
        withdrawal = jp.clip(jp.zeros_like(previous),
                             previous-self.max_rate*self.dt, previous+self.max_rate*self.dt)
        issued = jp.where((reason == Reason.APPLY) | (reason == Reason.NO_EFFECTIVE_CHANGE),
                          out.requested, withdrawal)
        details = {"current_gain_low": now.low, "current_gain_high": now.high,
                   "hold_gain_low": held.low, "hold_gain_high": held.high,
                   "hold_error_low": errors.low, "hold_error_high": errors.high,
                   "effective_low": out.effective_interval.low, "effective_high": out.effective_interval.high,
                   "requested_low": out.requested_interval.low, "requested_high": out.requested_interval.high,
                   "projection_status": out.status,
                   "estimate_supported": ((state.informative_count >= self.config.reacquire_packets)
                                          & ~state.mismatch & (rls.last_source_step >= 0))}
        return issued, reason, details

    def correction(self, state, rls, reference, step, previous):
        requested, reason, _ = self.evaluate(state, rls, reference, step, previous)
        return requested, reason


def model_record(env, telemetry, rls_cfg, config):
    bound = Candidate.bind(env, telemetry, rls_cfg, config)
    return {
        "joint_names": list(env.actuator_joint_names), "kp": np.asarray(bound.kp).tolist(),
        "ctrl_range": np.asarray(bound.ctrl_range).tolist(),
        "ctrl_limited": np.asarray(bound.ctrl_limited).tolist(),
        "nominal_force_ceiling_nm": [float(v) if np.isfinite(v) else None for v in bound.force_ceiling],
        "dt_s": bound.dt, "max_offset_rad": bound.max_offset, "max_rate_rad_s": bound.max_rate,
        "prior_gain": list(bound.prior), "reason_codes": {r.name:int(r) for r in Reason},
        "source_step_convention":"post-transition; age at decision k = (k - 1 - source_step) * dt",
    }
