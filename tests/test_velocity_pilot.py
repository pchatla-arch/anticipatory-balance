from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import os

import jax
import jax.numpy as jp
import numpy as np
import pytest

from foresight.adaptive import candidate as C, projection as P, sensing as S, estimation as E


def test_velocity_error_is_source_paired_and_preserves_other_sensor_draws():
    exact = {"telemetry_joint_position_rad": jp.arange(8.) / 10,
             "telemetry_joint_velocity_rad_s": jp.arange(8.),
             "telemetry_held_target_rad": jp.zeros(8),
             "telemetry_applied_joint_torque_nm": jp.ones(8),
             "telemetry_regression_valid": jp.ones(8, bool)}
    sensor = S.DECLARED._replace(velocity_error_rad_s=0.5)
    earliest = sensor._replace(latency_min=0, latency_max=0)
    for t in range(16):
        key = jax.random.fold_in(jax.random.PRNGKey(72), t)
        previous = S.measure(exact, t, S.DECLARED, key)
        delayed = S.measure(exact, t, sensor, key)
        early = S.measure(exact, t, earliest, key)
        for field in previous._fields:
            if field != "velocity_rad_s":
                np.testing.assert_array_equal(getattr(previous, field), getattr(delayed, field))
            if field != "arrival_step":
                np.testing.assert_array_equal(getattr(delayed, field), getattr(early, field))
        np.testing.assert_array_equal(previous.velocity_rad_s, exact["telemetry_joint_velocity_rad_s"])
        diff = delayed.velocity_rad_s - previous.velocity_rad_s
        assert np.all(np.abs(diff) <= .5) and np.any(diff != 0)
        assert bool(early.arrival_step == S.DROPPED) == bool(delayed.arrival_step == S.DROPPED)
    for invalid in (float("nan"), -1, float("inf")):
        with pytest.raises(ValueError):
            sensor._replace(velocity_error_rad_s=invalid).validate()


def test_velocity_envelope_uses_latest_valid_source_and_future_hold():
    cfg = C.Config(0., 10., .01, 1e-4, 2, 0.)
    speed = C.Candidate(cfg, P.Interval(.2, 1.5), jp.array([20.]), jp.array([[-2., 2.]]),
                        jp.ones(1, bool), jp.array([20.]), .2, 2., .02, .006, 1.5, .001, .5)
    velo = replace(speed, config=replace(cfg, hold_accel_rad_s2=400.))
    rls = jax.vmap(lambda _: E.rls_init(1, E.RLSConfig()))(jp.arange(1))
    packet = S.MeasuredPacket(jp.array([3]), jp.array([4]), jp.array([[.1]]), jp.array([[.05]]),
                              jp.array([[-2.]]), jp.array([[1.]]), jp.array([[True]]))
    update = jax.vmap(lambda r, p, m: E.rls_update(r, p, m, speed.kp, E.RLSConfig()))
    rls = update(rls, packet, jp.array([True]))
    invalid = packet._replace(source_step=jp.array([4]), velocity_rad_s=jp.array([[500.]]),
                              valid=jp.array([[False]]))
    rls = update(rls, invalid, jp.array([True]))
    rls = update(rls, invalid._replace(valid=jp.array([[True]])), jp.array([False]))
    for model, radius in ((speed, .006 + 10. * .06), (velo, .006 + 2.5 * .06 + 200. * .06**2)):
        # Decision 6 sees a post-transition capture at 3. Two elapsed ticks plus one future hold.
        _, _, obs = jax.jit(model.evaluate)(model.init(1,1), rls, jp.array([[.2]]), 6, jp.zeros((1,1)))
        np.testing.assert_allclose(obs["hold_error_low"], .15 - radius, atol=2e-7)
        np.testing.assert_allclose(obs["hold_error_high"], .15 + radius, atol=2e-7)
    for invalid_value in (-1, float("nan")):
        with pytest.raises(ValueError):
            replace(cfg, hold_accel_rad_s2=invalid_value)


def test_audit_masks_unobserved_transitions_and_missing_sources():
    path = Path(__file__).resolve().parents[1] / "scripts/run_velocity_pilot.py"
    spec = spec_from_file_location("run_velocity_pilot", path)
    pilot = module_from_spec(spec)
    spec.loader.exec_module(pilot)
    q = np.zeros((3, 1, 2, 2))
    q[1, 0, 0, 1] = .15  # Outside first substep radius .106, inside end-of-hold radius .206.
    q[2] = 9.  # padded post-termination
    trace = dict(pilot_substep_q=q, alive=np.array([[True],[True],[False]]),
                 authority_source_q_truth=np.zeros((3,1,2)), authority_decision_q_truth=np.zeros((3,1,2)),
                 authority_rls_last_position=np.zeros((3,1,2)),
                 authority_rls_last_source_step=np.array([[[-1,-1]],[[0,0]],[[1,1]]]))
    result = pilot.inspect_hold(trace, ["left_hip_pitch_joint","left_knee_joint"], .02,
                                C.Config(0.,10.,.01,1e-4,2,0.), S.DECLARED)
    assert result["hip"]["live_joint_ticks"] == 2
    assert result["hip"]["source_age_ticks_histogram"] == {"0": 1}
    assert result["hip"]["whole_hold_coverage"] == 1.
    assert result["knee"]["whole_hold_coverage"] == 0.

@pytest.mark.skipif(os.environ.get("FORESIGHT_RUN_CPU_PHYSICS") != "1",
                    reason="set FORESIGHT_RUN_CPU_PHYSICS=1 for the MJX observer parity check")
def test_observer_preserves_healthy_and_faulted_transitions():
    from foresight.adaptive.actuator_env import AdaptiveJoystick
    from foresight.adaptive.pilot_observer import ObservedJoystick
    from foresight.envs.g1_impact import default_config

    cfg = default_config()
    cfg.impact.resample_on_autoreset = False
    cfg.impact.reset_perturb = 1.0
    cfg.push_config.enable = False
    plain_env = AdaptiveJoystick(config=cfg, config_overrides={"impl": "jax"}, telemetry=True)
    observed_env = ObservedJoystick(config=cfg, config_overrides={"impl": "jax"}, telemetry=True)
    state = jax.jit(plain_env.reset)(jax.random.PRNGKey(98213))
    advance, observe = jax.jit(plain_env.step), jax.jit(observed_env.step)
    action = jp.linspace(-0.4, 0.4, plain_env.action_size)
    for gain in (1.0, 0.7):
        info = dict(state.info, adaptive_gain=jp.full_like(state.info["adaptive_gain"], gain),
                    adaptive_onset_step=jp.int32(0))
        plain = advance(state.replace(info=info), action)
        observed_info = dict(info, pilot_substep_q=jp.zeros((observed_env.n_substeps, observed_env.action_size)))
        seen = observe(state.replace(info=observed_info), action)
        records = dict(seen.info)
        q = records.pop("pilot_substep_q")
        np.testing.assert_array_equal(q[-1], plain.info["telemetry_joint_position_rad"])
        baseline_leaves = jax.tree_util.tree_leaves(plain)
        seen_leaves = jax.tree_util.tree_leaves(seen.replace(info=records))
        for expected, actual in zip(baseline_leaves, seen_leaves, strict=True):
            np.testing.assert_array_equal(expected, actual)
