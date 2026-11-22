# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Known-gain feasibility environment; this module does not estimate or learn gains.

The fault changes each selected position actuator's proportional stiffness by
scaling BOTH its control gain and its position-dependent bias. Joint damping,
force limits, contact dynamics and the frozen actor are unchanged. The optional
oracle uses the true gain and current joint positions once per 20 ms control
step. Its bounded target correction is held through the ten physics substeps;
it is not an exact dynamics cancellation and cannot restore lost force authority.

Lag faults are intentionally outside this first gate.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
import numpy as np

from foresight.envs.g1_impact.impact_joystick import ImpactJoystick, default_config


def bounded_gain_correction(
    nominal_target,
    joint_position,
    gain,
    previous_delta,
    dt,
    max_correction_rad=0.20,
    max_correction_rate_rad_s=2.0,
    ctrl_range=None,
    ctrl_limited=None,
):
    """Return target, commanded correction, constrained fraction and clip fraction.

    The nominal reference is first subjected to the SAME control range clipping
    already present in MuJoCo. Correction limits are relative to that physically
    available nominal reference. This avoids describing the simulator's ordinary
    clipping of an out-of-range policy target as adapter corrective authority.

    In the unconstrained case, delta=(1/g-1)*(reference-q) gives
    g*kp*(reference+delta-q) == kp*(reference-q) at the sampled q.
    The identity is only instantaneous; q moves during the control hold.
    """
    nominal_target = jp.asarray(nominal_target)
    joint_position = jp.asarray(joint_position)
    gain = jp.asarray(gain)
    previous_delta = jp.asarray(previous_delta)
    if ctrl_range is None:
        reference = nominal_target
    else:
        limited = jp.asarray(ctrl_limited, dtype=bool)
        reference = jp.where(
            limited,
            jp.clip(nominal_target, ctrl_range[:, 0], ctrl_range[:, 1]),
            nominal_target,
        )
    desired = (1.0 / gain - 1.0) * (reference - joint_position)
    return bounded_offset(desired, reference, previous_delta, dt, max_correction_rad,
                          max_correction_rate_rad_s, ctrl_range, ctrl_limited)


def bounded_offset(
    desired,
    reference,
    previous_delta,
    dt,
    max_correction_rad=0.20,
    max_correction_rate_rad_s=2.0,
    ctrl_range=None,
    ctrl_limited=None,
):
    """Apply the common correction limits to a requested offset.

    These limits are shared by every method (oracle, estimated inverse, heuristic gate,
    identification-aware candidate): amplitude, requested slew against the previous
    requested offset, and the simulator's control-range clipping of the final target.
    `reference` is the nominal target after that same clipping. Returns the clipped
    target, the requested (rate-limited) offset, the fraction of actuators whose
    applied offset differs from the request, and the fraction clipped by control range.
    """
    desired = jp.asarray(desired)
    reference = jp.asarray(reference)
    previous_delta = jp.asarray(previous_delta)
    if ctrl_range is not None:
        limited = jp.asarray(ctrl_limited, dtype=bool)
    amplitude_limited = jp.clip(desired, -max_correction_rad, max_correction_rad)
    max_change = max_correction_rate_rad_s * dt
    delta = jp.clip(
        amplitude_limited, previous_delta - max_change, previous_delta + max_change
    )
    # Keep both constraints valid even if a caller supplies a bad previous delta.
    delta = jp.clip(delta, -max_correction_rad, max_correction_rad)
    before_control_clip = reference + delta
    if ctrl_range is None:
        target = before_control_clip
    else:
        target = jp.where(
            limited,
            jp.clip(before_control_clip, ctrl_range[:, 0], ctrl_range[:, 1]),
            before_control_clip,
        )
    applied_delta = target - reference
    bound_hit = jp.mean((jp.abs(applied_delta - desired) > 1e-7).astype(jp.float32))
    control_clip = jp.mean(
        (jp.abs(target - before_control_clip) > 1e-7).astype(jp.float32)
    )
    # Rate limiting applies to the commanded correction. Physical control clipping
    # may reduce its effective value more abruptly; report that independently.
    return target, delta, bound_hit, control_clip


class AdaptiveJoystick(ImpactJoystick):
    """Frozen-actor evaluation with a gain fault and optional privileged correction.

    After reset, evaluators may replace these info fields (per lane under vmap):
      adaptive_gain: positive shape(nu,) multiplicative stiffness factors;
      adaptive_onset_step: first control step with the fault;
      adaptive_oracle: whether to use privileged bounded compensation.

    `leg_actuator_mask` identifies the twelve hip, knee and ankle actuators from
    model joint names, rather than assuming a particular action-array ordering.
    """

    def __init__(
        self,
        task="flat_terrain",
        config=None,
        config_overrides=None,
        scene_dir=None,
        max_correction_rad=0.20,
        max_correction_rate_rad_s=2.0,
        telemetry=False,
    ):
        if not np.isfinite(max_correction_rad) or max_correction_rad <= 0:
            raise ValueError("max_correction_rad must be positive and finite")
        if not np.isfinite(max_correction_rate_rad_s) or max_correction_rate_rad_s <= 0:
            raise ValueError("max_correction_rate_rad_s must be positive and finite")
        cfg = default_config() if config is None else config
        if cfg.skill != "nominal":
            raise ValueError("The actuator gate requires the nominal walking skill")
        super().__init__(task, cfg, config_overrides, scene_dir)
        model = self.mj_model
        # The oracle formula is specific to these unit-gear position servos.
        if not (
            np.all(model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
            and np.allclose(model.actuator_gear[:, 0], 1.0)
            and np.allclose(model.actuator_gear[:, 1:], 0.0)
            and np.all(model.actuator_gainprm[:, 0] > 0)
            and np.allclose(model.actuator_biasprm[:, 0], 0.0)
            and np.allclose(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0])
            and np.allclose(model.actuator_biasprm[:, 2], 0.0)
        ):
            raise ValueError("Expected unit-gear proportional position actuators")
        joint_ids = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
        self.actuator_joint_names = tuple(model.joint(int(j)).name for j in joint_ids)
        self.leg_actuator_mask = np.asarray(
            [name.startswith(("left_hip_", "right_hip_", "left_knee_", "right_knee_",
                              "left_ankle_", "right_ankle_"))
             for name in self.actuator_joint_names], dtype=bool
        )
        if int(self.leg_actuator_mask.sum()) != 12:
            raise ValueError("Expected twelve leg position actuators")
        self._adaptive_joint_qpos = jp.asarray(model.jnt_qposadr[joint_ids])
        self._adaptive_joint_dof = jp.asarray(model.jnt_dofadr[joint_ids])
        self._adaptive_ctrl_range = jp.asarray(model.actuator_ctrlrange)
        self._adaptive_ctrl_limited = jp.asarray(model.actuator_ctrllimited, dtype=bool)
        self.max_correction_rad = float(max_correction_rad)
        self.max_correction_rate_rad_s = float(max_correction_rate_rad_s)
        # Opt-in exact source-aligned telemetry (AE-001/AE-002 sampler). The packet is
        # produced from the same plant model that generated the step and is written to
        # info["telemetry_*"]; it carries no gain or onset label.
        self.telemetry = bool(telemetry)
        self._telemetry_layout = None
        if self.telemetry:
            from foresight.adaptive.telemetry import ActuatorTelemetryLayout
            self._telemetry_layout = ActuatorTelemetryLayout.from_model(model)

    def measured_joint_positions(self, data):
        return data.qpos[self._adaptive_joint_qpos]

    def reset(self, rng):
        state = super().reset(rng)
        info = dict(state.info)
        nu = self.action_size
        info.update(
            adaptive_gain=jp.ones(nu, jp.float32),
            adaptive_onset_step=jp.int32(100),
            adaptive_oracle=jp.array(False),
            adaptive_prev_delta=jp.zeros(nu, jp.float32),
            adaptive_step_delta=jp.zeros(nu, jp.float32),
            adaptive_effective_delta=jp.zeros(nu, jp.float32),
            adaptive_bound_hit=jp.float32(0),
            adaptive_control_clip_fraction=jp.float32(0),
            adaptive_active=jp.array(False),
            # Gain schedule: the fault holds on [onset, return); a large return step means
            # "never recovers". An unannounced return to healthy gain is a test scenario.
            adaptive_return_step=jp.int32(2**30),
            # External methods write their requested offset here before calling step;
            # it passes through the same common limits as the oracle's request.
            adaptive_external=jp.array(False),
            adaptive_requested_offset=jp.zeros(nu, jp.float32),
            # Passive-loss confound (AE-003): scales joint damping and dry friction of the
            # actuated joints on the same schedule as the gain. It changes motion, not the
            # aligned actuator-effort relation, so a correct estimator must not react to it.
            adaptive_damping_scale=jp.float32(1.0),
        )
        if getattr(self, "telemetry", False):
            info.update(self._empty_telemetry(nu))
        return state.replace(info=info)

    @staticmethod
    def _empty_telemetry(nu):
        return dict(
            telemetry_source_time_s=jp.float32(-1.0),
            telemetry_held_target_rad=jp.zeros(nu, jp.float32),
            telemetry_joint_position_rad=jp.zeros(nu, jp.float32),
            telemetry_joint_velocity_rad_s=jp.zeros(nu, jp.float32),
            telemetry_applied_joint_torque_nm=jp.zeros(nu, jp.float32),
            telemetry_raw_actuator_force_nm=jp.zeros(nu, jp.float32),
            telemetry_actuator_saturated=jp.zeros(nu, bool),
            telemetry_joint_saturated=jp.zeros(nu, bool),
            telemetry_regression_valid=jp.zeros(nu, bool),
        )

    def step(self, state, action):
        info = dict(state.info)
        nu = self.action_size
        # Older evaluators and tests build info without the schedule/external fields.
        return_step = info.get("adaptive_return_step", jp.int32(2**30))
        external = info.get("adaptive_external", jp.array(False))
        requested = info.get("adaptive_requested_offset", jp.zeros(nu, jp.float32))
        active = ((info["ep_step"] >= info["adaptive_onset_step"])
                  & (info["ep_step"] < return_step))
        gain = jp.where(active, info["adaptive_gain"], 1.0)
        # The oracle is enabled whenever requested: with a healthy plant its desired offset is
        # exactly zero, so before onset nothing changes, and after an unannounced return to
        # healthy gain its withdrawal obeys the same slew limit as every other method instead
        # of vanishing in one step.
        enabled = info["adaptive_oracle"] | external
        nominal_target = self._default_pose + action * self._config.action_scale
        reference = jp.where(
            self._adaptive_ctrl_limited,
            jp.clip(nominal_target, self._adaptive_ctrl_range[:, 0], self._adaptive_ctrl_range[:, 1]),
            nominal_target,
        )
        # The oracle's request is the true-gain inverse at the measured position; an
        # external method's request is whatever it computed from its own telemetry.
        # Both pass through the same common limits.
        oracle_desired = (1.0 / gain - 1.0) * (reference - self.measured_joint_positions(state.data))
        desired = jp.where(external, requested, oracle_desired)
        target, delta, bound_hit, control_clip = bounded_offset(
            desired,
            reference,
            info["adaptive_prev_delta"],
            self.dt,
            self.max_correction_rad,
            self.max_correction_rate_rad_s,
            self._adaptive_ctrl_range,
            self._adaptive_ctrl_limited,
        )
        # Preserve the original parent call exactly whenever there is no
        # correction. In particular g==1 does not pre-clip policy commands.
        has_correction = enabled & jp.any(delta != 0)
        executed_action = jp.where(
            has_correction,
            (target - self._default_pose) / self._config.action_scale,
            action,
        )
        effective_delta = jp.where(enabled, target-reference, jp.zeros_like(delta))
        delta = jp.where(enabled, delta, jp.zeros_like(delta))
        info.update(
            adaptive_prev_delta=delta,
            adaptive_step_delta=delta,
            adaptive_effective_delta=effective_delta,
            adaptive_bound_hit=jp.where(enabled, bound_hit, 0.0),
            adaptive_control_clip_fraction=jp.where(enabled, control_clip, 0.0),
            adaptive_active=active,
        )
        state = state.replace(info=info)
        original_model = self._mjx_model
        damping_scale = jp.where(active, info.get("adaptive_damping_scale", jp.float32(1.0)), 1.0)
        dof = self._adaptive_joint_dof
        faulty_model = original_model.tree_replace({
            "actuator_gainprm": original_model.actuator_gainprm.at[:, 0].multiply(gain),
            "actuator_biasprm": original_model.actuator_biasprm.at[:, 1].multiply(gain),
            "dof_damping": original_model.dof_damping.at[dof].multiply(damping_scale),
            "dof_frictionloss": original_model.dof_frictionloss.at[dof].multiply(damping_scale),
        })
        # This mirrors Playground's model-binding wrapper. Restore on trace or
        # runtime errors, so another condition cannot inherit the fault model.
        try:
            self._mjx_model = faulty_model
            result = super().step(state, executed_action)
        finally:
            self._mjx_model = original_model
        result_info = dict(result.info)
        # Parent makes observations before this assignment, using the preceding
        # nominal history. Do not re-observe or consume another noise RNG key.
        result_info["last_act"] = action
        if getattr(self, "telemetry", False):
            from foresight.adaptive.telemetry import sample_mjx_actuators
            packet = sample_mjx_actuators(faulty_model, result.data, self._telemetry_layout)
            result_info.update(
                telemetry_source_time_s=jp.asarray(packet.source_time_s, jp.float32),
                telemetry_held_target_rad=jp.asarray(packet.held_target_rad, jp.float32),
                telemetry_joint_position_rad=jp.asarray(packet.joint_position_rad, jp.float32),
                telemetry_joint_velocity_rad_s=jp.asarray(packet.joint_velocity_rad_s, jp.float32),
                telemetry_applied_joint_torque_nm=jp.asarray(packet.applied_joint_torque_nm, jp.float32),
                telemetry_raw_actuator_force_nm=jp.asarray(packet.raw_actuator_force_nm, jp.float32),
                telemetry_actuator_saturated=jp.asarray(packet.actuator_saturated, bool),
                telemetry_joint_saturated=jp.asarray(packet.joint_saturated, bool),
                telemetry_regression_valid=jp.asarray(packet.regression_valid, bool),
            )
        return result.replace(info=result_info)
