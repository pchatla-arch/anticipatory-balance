"""Replay identical native G1 knee states under motor gain loss and sensor scale error."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import mujoco
import numpy as np

from foresight.adaptive.telemetry import ActuatorTelemetryLayout, sample_native_actuators
from foresight.envs.g1_impact.impact_joystick import ImpactJoystick, default_config


def run():
    env = ImpactJoystick(config=default_config(), config_overrides={"impl": "jax"})
    model = env.mj_model
    layout = ActuatorTelemetryLayout.from_model(model)
    joint = model.joint("left_knee_joint").id
    actuator, = np.flatnonzero(model.actuator_trnid[:, 0] == joint)
    kp = float(model.actuator_gainprm[actuator, 0])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    data.qvel[:] = 0
    data.ctrl[:] = data.qpos[layout.qpos_adrs]
    q = float(data.qpos[layout.qpos_adrs[actuator]])
    gain_loss, step_s = 0.70, 0.020
    cases = []
    for error_rad in (-0.10, -0.05, 0.05, 0.10):
        # Illustrative point-estimate correction from rest with the gate's limits.
        delta = float(np.clip((1 / gain_loss - 1) * error_rad, -0.2, 0.2))
        delta = float(np.clip(delta, -2.0 * step_s, 2.0 * step_s))
        # Minimize the worst instantaneous torque error across these two worlds.
        # With symmetric absolute error, the two extreme signed errors balance.
        balanced_delta = float(np.clip((1 - gain_loss) * error_rad / (1 + gain_loss),
                                       -min(0.2, 2.0 * step_s), min(0.2, 2.0 * step_s)))
        rows = {}
        for label, gain, scale in (("weak_motor", gain_loss, 1.0),
                                   ("scaled_sensor", 1.0, gain_loss)):
            model.actuator_gainprm[actuator, 0] = kp * gain
            model.actuator_biasprm[actuator, 1] = -kp * gain
            rows[label] = {}
            for action, correction in (("leave_target", 0.0), ("apply_correction", delta),
                                       ("balanced_correction", balanced_delta)):
                data.ctrl[actuator] = q + error_rad + correction
                sample = sample_native_actuators(model, data, layout)
                if (not sample.regression_valid[actuator] or
                        abs(sample.held_target_rad[actuator] - data.ctrl[actuator]) > 1e-12):
                    raise ValueError("Clipping or invalid torque sample in the proposed comparison")
                torque = float(sample.applied_joint_torque_nm[actuator])
                rows[label][action] = {"physical_effort_nm": torque, "observed_effort_nm": scale * torque}
        for action in ("leave_target", "apply_correction", "balanced_correction"):
            np.testing.assert_allclose(rows["weak_motor"][action]["observed_effort_nm"],
                                       rows["scaled_sensor"][action]["observed_effort_nm"], atol=1e-12, rtol=0)
        target_torque = kp * error_rad
        for label, should_improve in (("weak_motor", True), ("scaled_sensor", False)):
            old, new = [abs(rows[label][a]["physical_effort_nm"] - target_torque)
                        for a in ("leave_target", "apply_correction")]
            if (new < old) != should_improve or np.isclose(old, new):
                raise AssertionError("Unexpected effect of the point-estimate correction")
        balanced_errors = [rows[label]["balanced_correction"]["physical_effort_nm"] - target_torque
                           for label in ("weak_motor", "scaled_sensor")]
        np.testing.assert_allclose(balanced_errors[0], -balanced_errors[1], rtol=0, atol=1e-12)
        cases.append({"error_rad": error_rad, "delta_rad": delta, "balanced_delta_rad": balanced_delta,
                      "healthy_reference_nm": target_torque,
                      "worlds": rows})
    return {"experiment": "matched-state effort-scale ambiguity, v1.1", "backend": "native MuJoCo",
            "mujoco_version": version("mujoco"), "joint": "left_knee_joint", "kp_nm_rad": kp,
            "control_period_s": step_s, "position_bound_rad": 0.2, "rate_bound_rad_s": 2.0,
            "gain_loss": gain_loss, "sensor_scale_on_healthy_plant": gain_loss,
            "cases": cases, "closed_loop_episodes": 0}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    result = run()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
