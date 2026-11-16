"""AE-001/002 contracts using native MuJoCo and the MJX CPU implementation."""
import unittest

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import numpy as np

from foresight.adaptive.telemetry import (
    ActuatorTelemetryLayout, sample_mjx_actuators, sample_native_actuators,
)


def model_fixture(joint_limit=None, actuator_limit=None, ctrl_limit=None,
                  damping=0., friction=0.):
    joint = '' if joint_limit is None else f'actuatorfrcrange="-{joint_limit} {joint_limit}"'
    act = '' if actuator_limit is None else f'forcerange="-{actuator_limit} {actuator_limit}"'
    ctrl = '' if ctrl_limit is None else f'ctrlrange="-{ctrl_limit} {ctrl_limit}"'
    return mujoco.MjModel.from_xml_string(f'''<mujoco>
      <option timestep=".002" gravity="0 0 0" integrator="Euler"/>
      <worldbody><body><joint name="joint" type="hinge" damping="{damping}"
        frictionloss="{friction}" {joint}/>
        <geom type="capsule" size=".05 .2" mass="1"/></body></worldbody>
      <actuator><position joint="joint" kp="75" {act} {ctrl}/></actuator>
    </mujoco>''')


def data_fixture(model):
    data = mujoco.MjData(model)
    data.qpos[0] = .1
    data.qvel[0] = .3
    data.ctrl[0] = .4
    mujoco.mj_forward(model, data)
    return data


class NativeTelemetryTests(unittest.TestCase):
    def test_post_step_force_is_stale_and_sampler_aligns_without_mutation(self):
        model = model_fixture()
        layout = ActuatorTelemetryLayout.from_model(model)
        data = data_fixture(model)
        pre_force = data.actuator_force.copy()
        mujoco.mj_step(model, data)
        np.testing.assert_array_equal(data.actuator_force, pre_force)
        self.assertGreater(abs(float(data.actuator_force[0]
                                     - 75*(data.ctrl[0]-data.qpos[0]))), .01)
        state_before = {k: np.array(getattr(data, k), copy=True)
                        for k in ('time', 'qpos', 'qvel', 'ctrl', 'actuator_force',
                                  'qfrc_actuator', 'qacc_warmstart')}
        packet = sample_native_actuators(model, data, layout)
        self.assertEqual(float(packet.source_time_s), float(data.time))
        np.testing.assert_allclose(packet.applied_joint_torque_nm,
                                   75*(packet.held_target_rad-packet.joint_position_rad),
                                   rtol=1e-12, atol=1e-12)
        for key, before in state_before.items():
            np.testing.assert_array_equal(getattr(data, key), before)
        self.assertTrue(packet.regression_valid[0])
        saved_position = packet.joint_position_rad.copy()
        data.qpos[0] += .2
        np.testing.assert_array_equal(packet.joint_position_rad, saved_position)

    def test_exact_gain_recovered_without_gain_label_in_packet(self):
        for gain in (1., .7):
            with self.subTest(gain=gain):
                model = model_fixture()
                layout = ActuatorTelemetryLayout.from_model(model)
                model.actuator_gainprm[:, 0] *= gain
                model.actuator_biasprm[:, 1] *= gain
                data = data_fixture(model)
                packet = sample_native_actuators(model, data, layout)
                estimate = packet.applied_joint_torque_nm / (
                    75*(packet.held_target_rad-packet.joint_position_rad))
                np.testing.assert_allclose(estimate, [gain], rtol=1e-12)
                self.assertTrue(packet.regression_valid[0])
                self.assertFalse(any('gain' in name or 'onset' in name
                                     for name in packet._fields))

    def test_joint_only_clipping_uses_applied_force_and_rejects_regression(self):
        model = model_fixture(joint_limit=5.)
        packet = sample_native_actuators(model, data_fixture(model),
                                        ActuatorTelemetryLayout.from_model(model))
        np.testing.assert_allclose(packet.raw_actuator_force_nm, [22.5])
        np.testing.assert_allclose(packet.applied_joint_torque_nm, [5.])
        self.assertFalse(packet.actuator_saturated[0])
        self.assertTrue(packet.joint_saturated[0])
        self.assertFalse(packet.regression_valid[0])

    def test_actuator_only_clipping_rejects_regression(self):
        model = model_fixture(actuator_limit=4.)
        packet = sample_native_actuators(model, data_fixture(model),
                                        ActuatorTelemetryLayout.from_model(model))
        np.testing.assert_allclose(packet.raw_actuator_force_nm, [4.])
        np.testing.assert_allclose(packet.applied_joint_torque_nm, [4.])
        self.assertTrue(packet.actuator_saturated[0])
        self.assertFalse(packet.joint_saturated[0])
        self.assertFalse(packet.regression_valid[0])

    def test_effective_target_matches_control_clipping(self):
        model = model_fixture(ctrl_limit=.2)
        packet = sample_native_actuators(model, data_fixture(model),
                                        ActuatorTelemetryLayout.from_model(model))
        np.testing.assert_allclose(packet.held_target_rad, [.2])
        np.testing.assert_allclose(packet.applied_joint_torque_nm, [7.5])
        self.assertTrue(packet.regression_valid[0])

    def test_passive_friction_and_damping_are_not_actuator_effort(self):
        packets = []
        passive = []
        for damping, friction in ((0., 0.), (4., 2.)):
            model = model_fixture(damping=damping, friction=friction)
            data = data_fixture(model)
            packets.append(sample_native_actuators(
                model, data, ActuatorTelemetryLayout.from_model(model)))
            passive.append(data.qfrc_passive.copy())
        np.testing.assert_array_equal(packets[0].applied_joint_torque_nm,
                                      packets[1].applied_joint_torque_nm)
        self.assertGreater(float(np.max(np.abs(passive[1]-passive[0]))), .1)

    def test_unsupported_mapping_and_nonproportional_model_rejected(self):
        for change in ('gear', 'slide', 'velocity_bias', 'gravity_comp'):
            with self.subTest(change=change):
                model = model_fixture()
                if change == 'gear':
                    model.actuator_gear[0, 0] = 2.
                elif change == 'slide':
                    model.jnt_type[0] = int(mujoco.mjtJoint.mjJNT_SLIDE)
                elif change == 'velocity_bias':
                    model.actuator_biasprm[0, 2] = -2.
                else:
                    model.jnt_actgravcomp[0] = True
                with self.assertRaises(ValueError):
                    ActuatorTelemetryLayout.from_model(model)
        duplicate = mujoco.MjModel.from_xml_string('''<mujoco>
          <worldbody><body><joint name="j"/><geom type="sphere" size=".1"/></body></worldbody>
          <actuator><position joint="j" kp="75"/><position joint="j" kp="75"/></actuator>
        </mujoco>''')
        with self.assertRaises(ValueError):
            ActuatorTelemetryLayout.from_model(duplicate)


class MjxTelemetryTests(unittest.TestCase):
    def test_cpu_backend(self):
        self.assertTrue(all(device.platform == 'cpu' for device in jax.devices()),
                        'Run these checks with JAX_PLATFORMS=cpu')

    def test_jit_vmap_native_parity_and_functional_input(self):
        model = model_fixture()
        layout = ActuatorTelemetryLayout.from_model(model)
        model.actuator_gainprm[:, 0] *= .7
        model.actuator_biasprm[:, 1] *= .7
        data = data_fixture(model)
        mujoco.mj_step(model, data)
        expected = sample_native_actuators(model, data, layout)
        mjx_model = mjx.put_model(model, impl='jax')
        mjx_data = mjx.put_data(model, data, impl='jax')
        original = {k: np.array(getattr(mjx_data, k), copy=True)
                    for k in ('time', 'qpos', 'qvel', 'ctrl', 'actuator_force', 'qfrc_actuator')}
        sample = jax.jit(lambda state: sample_mjx_actuators(mjx_model, state, layout))
        actual = sample(mjx_data)
        for want, got in zip(expected, actual):
            np.testing.assert_allclose(np.asarray(got), want, rtol=3e-6, atol=1e-6)
        for key, before in original.items():
            np.testing.assert_array_equal(np.asarray(getattr(mjx_data, key)), before)
        batch = jax.tree_util.tree_map(lambda a: jp.stack([a, a]), mjx_data)
        batched = jax.jit(jax.vmap(sample))(batch)
        for want, got in zip(actual, batched):
            np.testing.assert_allclose(np.asarray(got), np.stack([want, want]),
                                       rtol=3e-6, atol=1e-6)

    def test_mjx_saturation_and_nonfinite_sample_rejected(self):
        for joint_limit, actuator_limit in ((5., None), (None, 4.)):
            with self.subTest(joint_limit=joint_limit, actuator_limit=actuator_limit):
                model = model_fixture(joint_limit=joint_limit, actuator_limit=actuator_limit)
                layout = ActuatorTelemetryLayout.from_model(model)
                mm = mjx.put_model(model, impl='jax')
                data = mjx.put_data(model, data_fixture(model), impl='jax')
                sample = jax.jit(lambda state: sample_mjx_actuators(mm, state, layout))
                packet = sample(data)
                self.assertFalse(bool(packet.regression_valid[0]))
                self.assertEqual(bool(packet.joint_saturated[0]), joint_limit is not None)
                self.assertEqual(bool(packet.actuator_saturated[0]), actuator_limit is not None)
                invalid = sample(data.replace(ctrl=jp.full_like(data.ctrl, jp.nan)))
                self.assertFalse(bool(invalid.finite[0]))
                self.assertFalse(bool(invalid.regression_valid[0]))


if __name__ == '__main__':
    unittest.main()
