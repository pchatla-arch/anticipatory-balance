"""Gain feasibility contracts: physical stiffness, limits, model isolation, actor history."""
import os
import unittest
from types import SimpleNamespace
from unittest import mock

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from foresight.adaptive.actuator_env import AdaptiveJoystick, bounded_gain_correction
from foresight.envs.g1_impact.impact_joystick import ImpactJoystick, default_config


class CorrectionTests(unittest.TestCase):
    def test_identity_and_jit(self):
        reference = jp.array([.2, -.4])
        q = jp.array([.1, .2])
        target, delta, hit, clip = jax.jit(bounded_gain_correction)(
            reference, q, jp.ones(2), jp.zeros(2), .02)
        np.testing.assert_array_equal(target, reference)
        np.testing.assert_array_equal(delta, [0., 0.])
        self.assertEqual(float(hit), 0.)
        self.assertEqual(float(clip), 0.)

    def test_unconstrained_instantaneous_position_force_recovery(self):
        reference, q = jp.array([.4, -.2]), jp.array([.1, .05])
        gain, kp = jp.array([.7, .55]), jp.array([75., 20.])
        target, delta, _, _ = bounded_gain_correction(
            reference, q, gain, jp.zeros(2), .02, 10., 1000.)
        np.testing.assert_allclose(gain * kp * (target-q), kp * (reference-q), rtol=2e-7)
        self.assertTrue(np.all(np.isfinite(delta)))

    def test_amplitude_and_slew_bound(self):
        delta = jp.zeros(2)
        for _ in range(20):
            previous = delta
            _, delta, hit, _ = bounded_gain_correction(
                jp.array([1., -1.]), jp.zeros(2), jp.array([.55, .55]),
                delta, .02, .2, 2.)
            self.assertLessEqual(float(jp.max(jp.abs(delta))), .200001)
            self.assertLessEqual(float(jp.max(jp.abs(delta-previous))), .040001)
            self.assertEqual(float(hit), 1.)
        np.testing.assert_allclose(delta, [.2, -.2], rtol=1e-6)

    def test_existing_control_limits_respected_only_when_enabled(self):
        target, delta, hit, clipped = bounded_gain_correction(
            jp.array([.19, .19]), jp.zeros(2), jp.array([.5, .5]),
            jp.zeros(2), .02, .2, 2.,
            jp.array([[-.2, .2], [-.2, .2]]), jp.array([True, False]))
        np.testing.assert_allclose(target, [.2, .23], atol=1e-7)
        np.testing.assert_allclose(delta, [.04, .04], atol=1e-7)
        self.assertEqual(float(clipped), .5)
        self.assertEqual(float(hit), 1.)

    def test_commanded_slew_remains_bounded_when_hardware_clipping_changes(self):
        previous = jp.zeros(1)
        effective = []
        for reference in [0., .02, .04, .06, .08, .1, .19, -.1, -.3, .3]:
            target, delta, _, _ = bounded_gain_correction(
                jp.array([reference]), jp.array([-1.]), jp.array([.5]),
                previous, .02, .2, 2., jp.array([[-.2, .2]]), jp.array([True]))
            self.assertLessEqual(float(jp.max(jp.abs(delta-previous))), .040001)
            self.assertLessEqual(float(jp.max(jp.abs(delta))), .200001)
            self.assertLessEqual(float(jp.max(jp.abs(target))), .200001)
            effective.append(float(target[0])-float(np.clip(reference, -.2, .2)))
            previous = delta
        # Control-range clipping can remove a previously feasible correction
        # faster than the controller's slew cap; it must remain a separate metric.
        self.assertGreater(float(np.max(np.abs(np.diff(effective)))), .04)

    def test_gain_fault_scales_stiffness_not_absolute_target(self):
        # Native MuJoCo verifies the physical actuation interpretation, independent
        # of the JAX helper algebra. Damping and force caps stay in the model.
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="joint" type="hinge" damping="2"/>
          <geom type="capsule" size=".05 .2" mass="1"/></body></worldbody>
          <actuator><position joint="joint" kp="75" forcerange="-100 100"/></actuator>
          </mujoco>''')
        data = mujoco.MjData(model)
        data.qpos[0] = .1
        data.ctrl[0] = .4
        mujoco.mj_forward(model, data)
        healthy = data.actuator_force.copy()
        damping, limits = model.dof_damping.copy(), model.actuator_forcerange.copy()
        model.actuator_gainprm[:, 0] *= .7
        model.actuator_biasprm[:, 1] *= .7
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, healthy * .7, rtol=1e-12)
        target, _, _, _ = bounded_gain_correction(
            jp.array([.4]), jp.array([.1]), jp.array([.7]), jp.zeros(1), .02, 10., 1000.)
        data.ctrl[:] = np.asarray(target)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(data.actuator_force, healthy, rtol=2e-7)
        np.testing.assert_array_equal(model.dof_damping, damping)
        np.testing.assert_array_equal(model.actuator_forcerange, limits)


class StepContractTests(unittest.TestCase):
    """Exercise real MJX parameter arrays through a lightweight parent-step spy."""

    def setUp(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="joint" type="hinge"/>
          <geom type="capsule" size=".05 .2" mass="1"/></body></worldbody>
          <actuator><position joint="joint" kp="75"/></actuator></mujoco>''')
        self.original_model = mjx.put_model(model, impl="jax")
        self.env = AdaptiveJoystick.__new__(AdaptiveJoystick)
        self.env._mjx_model = self.original_model
        self.env._mj_model = model
        self.env._config = SimpleNamespace(action_scale=.5, ctrl_dt=.02, sim_dt=.002)
        self.env._ctrl_dt = .02
        self.env._sim_dt = .002
        self.env._default_pose = jp.zeros(1)
        self.env._adaptive_joint_qpos = jp.array([0])
        self.env._adaptive_joint_dof = jp.array([0])
        self.env._adaptive_ctrl_range = jp.array([[-1., 1.]])
        self.env._adaptive_ctrl_limited = jp.array([False])
        self.env.max_correction_rad = .2
        self.env.max_correction_rate_rad_s = 2.
        self.state = mjx_env.State(
            data=SimpleNamespace(qpos=jp.array([.1])),
            obs={"state": jp.array([-.2])}, reward=jp.float32(0), done=jp.float32(0),
            metrics={}, info={
                "ep_step": jp.int32(99), "last_act": jp.array([-.2]),
                "last_last_act": jp.array([-.3]),
                "adaptive_gain": jp.array([.7]), "adaptive_onset_step": jp.int32(100),
                "adaptive_oracle": jp.array(True), "adaptive_prev_delta": jp.zeros(1),
                "adaptive_step_delta": jp.zeros(1), "adaptive_bound_hit": jp.float32(0),
                "adaptive_control_clip_fraction": jp.float32(0), "adaptive_active": jp.array(False),
            })

    def parent_spy(self, captures):
        def step(env, state, action):
            captures.append((env.mjx_model, np.asarray(action)))
            info = dict(state.info)
            obs = {"state": info["last_act"]}
            info["last_last_act"] = info["last_act"]
            info["last_act"] = action
            return state.replace(info=info, obs=obs)
        return step

    def test_onset_scales_gain_and_bias_and_preserves_nominal_action_history(self):
        captures = []
        with mock.patch.object(ImpactJoystick, "step", self.parent_spy(captures)):
            result = self.env.step(self.state, jp.array([.8]))
            np.testing.assert_allclose(captures[0][0].actuator_gainprm[:, 0], [75.])
            np.testing.assert_array_equal(captures[0][1], np.asarray(jp.array([.8])))
            state = self.state.replace(info={**self.state.info, "ep_step": jp.int32(100)})
            result = self.env.step(state, jp.array([.8]))
        faulty_model, executed = captures[1]
        np.testing.assert_allclose(faulty_model.actuator_gainprm[:, 0], [52.5])
        np.testing.assert_allclose(faulty_model.actuator_biasprm[:, 1], [-52.5])
        np.testing.assert_allclose(executed, [.88], atol=1e-7)
        np.testing.assert_allclose(result.info["last_act"], [.8])
        np.testing.assert_allclose(result.info["last_last_act"], [-.2])
        np.testing.assert_allclose(result.obs["state"], [-.2])
        self.assertIs(self.env._mjx_model, self.original_model)

    def test_model_restored_even_if_parent_raises(self):
        with mock.patch.object(ImpactJoystick, "step", side_effect=RuntimeError("fixture")):
            with self.assertRaisesRegex(RuntimeError, "fixture"):
                self.env.step(self.state, jp.array([.8]))
        self.assertIs(self.env._mjx_model, self.original_model)

    def test_no_oracle_changes_only_plant(self):
        captures = []
        state = self.state.replace(info={**self.state.info, "ep_step": jp.int32(100),
                                       "adaptive_oracle": jp.array(False)})
        with mock.patch.object(ImpactJoystick, "step", self.parent_spy(captures)):
            result = self.env.step(state, jp.array([.8]))
        np.testing.assert_allclose(captures[0][0].actuator_gainprm[:, 0], [52.5])
        np.testing.assert_array_equal(captures[0][1], np.asarray(jp.array([.8])))
        np.testing.assert_array_equal(result.info["adaptive_step_delta"], [0.])


@unittest.skipUnless(os.environ.get("FORESIGHT_RUN_CPU_PHYSICS") == "1",
                     "Set FORESIGHT_RUN_CPU_PHYSICS=1 and JAX_PLATFORMS=cpu for full G1 identity")
class FullG1IdentityTests(unittest.TestCase):
    def test_healthy_control_step_matches_original_environment(self):
        self.assertTrue(all(d.platform == "cpu" for d in jax.devices()))
        def build(cls):
            cfg = default_config()
            cfg.impact.reset_perturb = 0.
            cfg.impact.resample_on_autoreset = False
            cfg.noise_config.level = 0.
            return cls(config=cfg, config_overrides={"impl": "jax"})
        adapted, original = build(AdaptiveJoystick), build(ImpactJoystick)
        self.assertEqual(int(adapted.leg_actuator_mask.sum()), 12)
        key = jax.random.PRNGKey(7)
        adapted_state = jax.jit(adapted.reset)(key)
        original_state = jax.jit(original.reset)(key)
        action = jp.linspace(-.1, .1, 29)
        adapted_state = jax.jit(adapted.step)(adapted_state, action)
        original_state = jax.jit(original.step)(original_state, action)
        for x, y in (
            (adapted_state.data.qpos, original_state.data.qpos),
            (adapted_state.data.qvel, original_state.data.qvel),
            (adapted_state.obs["state"], original_state.obs["state"]),
            (adapted_state.info["last_act"], original_state.info["last_act"]),
            (adapted_state.info["rng"], original_state.info["rng"]),
        ):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


if __name__ == "__main__":
    unittest.main()
