# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Opt-in, source-aligned simulated effort for proportional G1 actuators.

MuJoCo's returned post-step qpos and cached actuator force need not describe the
same instant: force evaluation precedes the final integration. These samplers
perform a fresh forward evaluation without advancing time. Native sampling uses
a copied MjData; MJX sampling uses the functionally returned forward data. Neither
sampler replaces the caller's environment data, observations or action history.

The declared effort channel is qfrc_actuator at each actuator's hinge DoF, AFTER
joint actuator-force clipping. It excludes passive damping/friction, contact
force and total generalized joint force. Raw actuator_force is retained only as
a separately named diagnostic; it precedes joint-level clipping. Saturation at
either level invalidates the unsaturated proportional-gain regression.

This is an exact simulated sensor source, not a hardware sensor claim. A separate
sensor adapter must add declared noise/bias and queue packets by capture/arrival
time before practical identification. Injected gain and fault-onset labels are
not packet fields. The same plant model that generated the current step must be
supplied to forward; the model is simulator-only input, not estimator telemetry.

There is an extra forward evaluation per sample. Its GPU cost is unmeasured.
The existing privileged feasibility gate does not call this module. Native and
MJX/JAX CPU behavior is tested; Warp parity remains an integration check.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import mujoco
from mujoco import mjx
import numpy as np


class ActuatorSample(NamedTuple):
    """Same-source-time values, ordered by actuator; masks have shape (nu,).

    ``held_target_rad`` includes configured control-range clipping. The packet
    reports the command held at capture time, not the next controller command.
    ``regression_valid`` indicates finite, unsaturated samples under the declared
    model; it does not assert sufficient excitation, calibrated real sensors or
    a valid error bound after a downstream sensor corruption.
    """

    source_time_s: Any
    held_target_rad: Any
    joint_position_rad: Any
    joint_velocity_rad_s: Any
    applied_joint_torque_nm: Any
    raw_actuator_force_nm: Any
    actuator_saturated: Any
    joint_saturated: Any
    finite: Any
    regression_valid: Any


@dataclass(frozen=True)
class ActuatorTelemetryLayout:
    """Fixed, nominal interface metadata; construct once before JAX tracing.

    Sampling requires the same actuator mapping, configured limits and disable
    flags as this layout. Proportional gain and matching position bias may vary.
    Keep the layout in a JIT closure; it is not a dynamic simulator state.
    """

    qpos_adrs: np.ndarray
    dof_adrs: np.ndarray
    ctrl_limited: np.ndarray
    ctrl_range: np.ndarray
    actuator_force_limited: np.ndarray
    actuator_force_range: np.ndarray
    joint_force_limited: np.ndarray
    joint_force_range: np.ndarray
    control_clipping_enabled: bool
    saturation_margin_nm: float

    @classmethod
    def from_model(cls, model, saturation_margin_nm=1e-5):
        if not np.isfinite(saturation_margin_nm) or saturation_margin_nm < 0:
            raise ValueError("saturation_margin_nm must be finite and nonnegative")
        if model.nu < 1:
            raise ValueError("At least one actuator is required")
        if not np.all(model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT)):
            raise ValueError("Telemetry requires direct joint transmission")
        joints = np.asarray(model.actuator_trnid[:, 0], dtype=np.int32)
        if (len(np.unique(joints)) != model.nu
                or not np.all(model.jnt_type[joints] == int(mujoco.mjtJoint.mjJNT_HINGE))
                or not np.allclose(model.actuator_gear[:, 0], 1.0, rtol=0, atol=1e-12)
                or not np.allclose(model.actuator_gear[:, 1:], 0.0, rtol=0, atol=1e-12)):
            raise ValueError("Expected one unit-gear hinge actuator per joint")
        if not (
            np.all(model.actuator_dyntype == int(mujoco.mjtDyn.mjDYN_NONE))
            and np.all(model.actuator_gaintype == int(mujoco.mjtGain.mjGAIN_FIXED))
            and np.all(model.actuator_biastype == int(mujoco.mjtBias.mjBIAS_AFFINE))
            and np.all(np.isfinite(model.actuator_gainprm[:, 0]))
            and np.all(np.isfinite(model.actuator_biasprm[:, :3]))
            and np.all(model.actuator_gainprm[:, 0] > 0)
            and np.allclose(model.actuator_biasprm[:, 0], 0.0, rtol=0, atol=1e-12)
            and np.allclose(model.actuator_biasprm[:, 1], -model.actuator_gainprm[:, 0],
                            rtol=1e-12, atol=1e-12)
            and np.allclose(model.actuator_biasprm[:, 2], 0.0, rtol=0, atol=1e-12)
        ):
            raise ValueError("Expected pure proportional position actuators")
        if np.any(model.jnt_actgravcomp[joints]):
            raise ValueError("Actuator-level gravity compensation is unsupported")
        if int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_ACTUATION):
            raise ValueError("Disabled actuation is outside the gain-only model")
        if int(model.opt.disableactuator):
            raise ValueError("Disabled actuator groups are outside the gain-only model")

        def copied(value, dtype=None):
            a = np.array(value, dtype=dtype, copy=True)
            a.flags.writeable = False
            return a

        return cls(
            qpos_adrs=copied(model.jnt_qposadr[joints], np.int32),
            dof_adrs=copied(model.jnt_dofadr[joints], np.int32),
            ctrl_limited=copied(model.actuator_ctrllimited, bool),
            ctrl_range=copied(model.actuator_ctrlrange),
            actuator_force_limited=copied(model.actuator_forcelimited, bool),
            actuator_force_range=copied(model.actuator_forcerange),
            joint_force_limited=copied(model.jnt_actfrclimited[joints], bool),
            joint_force_range=copied(model.jnt_actfrcrange[joints]),
            control_clipping_enabled=not bool(
                int(model.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_CLAMPCTRL)),
            saturation_margin_nm=float(saturation_margin_nm),
        )


def _packet(data, layout, xp):
    # xp.array copies native views so later mutation of the caller's state cannot
    # silently change a packet that has already entered a sensor queue.
    time = xp.array(data.time)
    ctrl = xp.array(data.ctrl)
    cr = xp.asarray(layout.ctrl_range)
    target = xp.where(
        xp.asarray(layout.ctrl_limited) & layout.control_clipping_enabled,
        xp.clip(ctrl, cr[:, 0], cr[:, 1]), ctrl)
    q = xp.array(data.qpos)[layout.qpos_adrs]
    qvel = xp.array(data.qvel)[layout.dof_adrs]
    raw = xp.array(data.actuator_force)
    applied = xp.array(data.qfrc_actuator)[layout.dof_adrs]

    def at_limit(value, enabled, bounds):
        bounds = xp.asarray(bounds)
        return xp.asarray(enabled) & (
            (value <= bounds[:, 0] + layout.saturation_margin_nm)
            | (value >= bounds[:, 1] - layout.saturation_margin_nm))

    actuator_sat = at_limit(raw, layout.actuator_force_limited,
                            layout.actuator_force_range)
    joint_sat = at_limit(applied, layout.joint_force_limited,
                         layout.joint_force_range)
    finite = (xp.isfinite(time) & xp.isfinite(ctrl) & xp.isfinite(target)
              & xp.isfinite(q) & xp.isfinite(qvel)
              & xp.isfinite(raw) & xp.isfinite(applied))
    return ActuatorSample(time, target, q, qvel, applied, raw,
                          actuator_sat, joint_sat, finite,
                          finite & ~actuator_sat & ~joint_sat)


def sample_native_actuators(model, data, layout):
    """Forward a copied native state and return aligned exact simulated effort.

    ``model`` must be the plant model at capture time, including its actual gain.
    It is used only by the simulator; no gain or model object enters the packet.
    The input data and its cached observations remain unchanged.
    """
    aligned = mujoco.MjData(model)
    mujoco.mj_copyData(aligned, model, data)
    mujoco.mj_forward(model, aligned)
    return _packet(aligned, layout, np)


def sample_mjx_actuators(model, data, layout):
    """Functional MJX equivalent; supports JIT/vmap with layout in a closure.

    Use the returned packet only. Substituting forward-refreshed data into the
    environment would change observation/cache timing and is not this API's role.
    """
    import jax.numpy as jp

    aligned = mjx.forward(model, data)
    return _packet(aligned, layout, jp)
