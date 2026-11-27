"""CPU JAX causal single-G1 adapter contract, no MuJoCo / GPU required."""
import unittest
from types import SimpleNamespace
import numpy as np
import jax.numpy as jp
from foresight.adaptive.single_causal_controller import SingleCausalAdapter, frozen_heuristic_configs
from foresight.adaptive import sensing as S


class StubG1:
    def __init__(self):
        self.action_size = 3
        self.mj_model = SimpleNamespace(actuator_gainprm=np.array(
            [[1.0, 0, 0], [1.0, 0, 0], [1.0, 0, 0]], dtype=np.float32))
        self._default_pose = jp.zeros((3,), jp.float32)
        self._config = SimpleNamespace(action_scale=1.0)
        self._adaptive_ctrl_limited = jp.zeros((3,), dtype=bool)
        self._adaptive_ctrl_range = jp.array([[-1., 1.]]*3, jp.float32)
        self.leg_actuator_mask = jp.array([True, True, True])
        self.dt=0.02


def exact_packet(gain):
    return {
        "telemetry_source_time_s": jp.float32(0),
        "telemetry_held_target_rad": jp.ones((3,), jp.float32),
        "telemetry_joint_position_rad": jp.zeros((3,), jp.float32),
        "telemetry_joint_velocity_rad_s": jp.zeros((3,), jp.float32),
        "telemetry_applied_joint_torque_nm": jp.full((3,), gain, jp.float32),
        "telemetry_regression_valid": jp.ones((3,), dtype=bool),
    }


class CausalAdapterTests(unittest.TestCase):
    def config(self):
        return {"rls": {"forgetting": 0.9, "initial_cov": 1000.0,
                        "min_excitation_rad": 0.001, "ewma": 0.9},
                "heuristic": {"min_samples": 2, "max_residual_rms": 0.5,
                              "min_excitation_rad": 0.001, "dead_band": 0.1,
                              "max_age_steps": 10}}

    def telemetry(self):
        return S.TelemetryConfig(position_noise_rad=0, torque_noise_nm=0,
                                 latency_min=1, latency_max=1, dropout=0,
                                 velocity_error_rad_s=0)

    def test_starts_without_correction_before_any_observation(self):
        adapter=SingleCausalAdapter(StubG1(), self.config(), seed=1,
                                    telemetry=self.telemetry())
        desired=adapter.requested_offset(jp.ones((3,),jp.float32),0)
        self.assertAlmostEqual(float(jp.max(jp.abs(desired))),0.0)

    def test_gain_estimate_changes_from_causal_effort_packets_only(self):
        adapter=SingleCausalAdapter(StubG1(),self.config(),seed=1,
                                    telemetry=self.telemetry())
        for t in range(7):
            adapter.assimilate_captured(exact_packet(.7),t)
        obs=adapter.evidence()
        self.assertGreater(obs["valid_samples_total"],0)
        self.assertLess(obs["gain_estimate_leg_mean"],0.9)
        desired=adapter.requested_offset(jp.ones((3,),jp.float32),8)
        self.assertGreater(float(jp.max(jp.abs(desired))),0.01)
        self.assertEqual(obs["released_packets"],6)

    def test_healthy_effort_cannot_trigger_correction(self):
        adapter=SingleCausalAdapter(StubG1(),self.config(),seed=2,
                                    telemetry=self.telemetry())
        for t in range(7):
            adapter.assimilate_captured(exact_packet(1.0),t)
        desired=adapter.requested_offset(jp.ones((3,),jp.float32),8)
        self.assertAlmostEqual(float(jp.max(jp.abs(desired))),0.0,places=4)

    def test_no_true_gain_or_onset_required(self):
        adapter=SingleCausalAdapter(StubG1(),self.config(),seed=3,
                                    telemetry=self.telemetry())
        # The packet contains source aligned telemetry, but no injected
        # plant/simulator labels; the API must not demand those labels.
        for t in range(5):
            adapter.assimilate_captured(exact_packet(.72),t)
        self.assertGreater(adapter.evidence()["released_packets"],0)

    def test_real_baseline_tuning_provenance_accepted_but_not_controlled(self):
        original = self.config()
        tagged = dict(original, tuned_on={
            "split": "dev", "families": ["gain_070"], "telemetry": "declared",
            "episodes": 8, "evaluations": 17, "objective": "development"})
        r1, h1 = frozen_heuristic_configs(original)
        r2, h2 = frozen_heuristic_configs(tagged)
        self.assertEqual(r1, r2)
        self.assertEqual(h1, h2)

    def test_unknown_frozen_fields_rejected(self):
        with self.assertRaisesRegex(ValueError,"unexpected"):
            frozen_heuristic_configs({"rls":{}, "adaptive_gain_truth":.7})


if __name__ == "__main__":
    unittest.main()
