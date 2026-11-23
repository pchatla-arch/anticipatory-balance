"""Bench scenarios, command schedule and the closed loop with methods (WP 1.4, 2.3, 4.1)."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

import jax.numpy as jp
import numpy as np

from foresight.adaptive import bench as B
from foresight.adaptive import sensing as S

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_adaptive_bench as H  # noqa: E402


class FakeEnv:
    actuator_joint_names = ("left_hip_pitch", "left_knee", "left_ankle_pitch", "right_hip_pitch",
                            "right_knee", "right_ankle_pitch", "waist_yaw")
    leg_actuator_mask = np.array([True] * 6 + [False])


class ScenarioTests(unittest.TestCase):
    def test_scenarios_are_deterministic_paired_and_grouped(self):
        env = FakeEnv()
        a = B.draw_scenarios(B.FAMILIES["gain_070_return"], 11000, 8, env)
        b = B.draw_scenarios(B.FAMILIES["gain_070_return"], 11000, 8, env)
        for key in ("onset", "return", "gain", "speed", "sensor_key"):
            np.testing.assert_array_equal(a[key], b[key])
        self.assertTrue(np.all((a["onset"] >= 75) & (a["onset"] < 125)))
        self.assertTrue(np.all((a["return"] - a["onset"] >= 100) & (a["return"] - a["onset"] < 150)))
        np.testing.assert_array_equal(a["gain"][:, 6], 1.0, "non-leg actuators keep gain 1")
        np.testing.assert_allclose(a["gain"][:, :6], 0.7)
        left = B.draw_scenarios(B.FAMILIES["gain_070_left"], 11000, 2, env)["gain"][0]
        np.testing.assert_allclose(left, [0.7, 0.7, 0.7, 1.0, 1.0, 1.0, 1.0])
        healthy = B.draw_scenarios(B.FAMILIES["healthy"], 11000, 2, env)
        self.assertTrue(np.all(healthy["return"] > 2**29))
        np.testing.assert_array_equal(healthy["gain"], 1.0)
        other = B.draw_scenarios(B.FAMILIES["gain_070"], 12000, 8, env)
        self.assertFalse(np.array_equal(a["onset"], other["onset"]), "splits draw different scenarios")

    def test_command_schedule_walk_turn_stop_resume(self):
        sched = B.WALK_TURN_STOP_RESUME
        speed = jp.float32(0.6)
        np.testing.assert_allclose(B.command_at(sched, jp.int32(0), speed), [0.6, 0.0, 0.0])
        np.testing.assert_allclose(B.command_at(sched, jp.int32(150), speed), [0.36, 0.0, 0.5])
        np.testing.assert_allclose(B.command_at(sched, jp.int32(250), speed), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(B.command_at(sched, jp.int32(399), speed), [0.6, 0.0, 0.0])
        np.testing.assert_allclose(B.command_at(B.WALK_ONLY, jp.int32(399), speed), [0.6, 0.0, 0.0])


class HarnessTests(unittest.TestCase):
    def rows(self, method, loss, fell=False, harmful=0.0, family="gain_070", n=6):
        return [dict(split="dev", phase="eval", family=family, method=method, telemetry="exact",
                     scenario_id=f"s{i}", gain=0.7, onset_step=80, return_step=-1, speed_cmd_mps=0.5,
                     fell=fell, nonfinite=False, termination_step=-1, pre_fault_failed=False,
                     postfault_loss=loss + 0.001 * i, progress_m=2.0, correction_bound_fraction=0.0,
                     control_clip_fraction=0.0, max_correction_rad=0.1, max_slew_rad_s=0.5,
                     max_abs_actuator_force_nm=40.0, harmful_offset_rad=harmful, mean_applied_offset_rad=0.02,
                     withdrawal_steps=-1, gain_abs_error=0.05, apply_steps=100, lost_packets=0, released_packets=300)
                for i in range(n)]

    def test_summary_pairs_against_the_reference_and_applies_margins(self):
        rows = self.rows("heuristic_gate", 0.05) + self.rows("candidate", 0.03) + self.rows("none", 0.09, fell=True)
        rows += self.rows("heuristic_gate", 0.012, family="healthy") + self.rows("candidate", 0.014, harmful=0.01, family="healthy")
        summary = H.summarize(rows, margins={"healthy": 0.005})
        self.assertEqual(summary["cells"]["gain_070|exact|none"]["fall_rate"], 1.0)
        paired = summary["paired_vs_reference"]["gain_070|exact|candidate"]["postfault_loss"]
        self.assertAlmostEqual(paired["mean_difference"], -0.02)
        self.assertLess(paired["ci95"][1], 0.0, "candidate beats the reference with a clear interval")
        healthy = summary["paired_vs_reference"]["healthy|exact|candidate"]
        self.assertTrue(healthy["noninferior_loss"], "+0.002 is inside the 0.005 margin")
        self.assertNotIn("gain_070|exact|heuristic_gate", summary["paired_vs_reference"])
        json.dumps(summary, allow_nan=False)
        report = H.render_report(summary, dict(split="dev", source_commit="abc", impl="jax", steps=10, schedule="walk", episodes=6))
        self.assertIn("candidate", report)

    def test_test_split_requires_a_matching_prior_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = Path(tmp) / "registration.json"
            with self.assertRaises(SystemExit):
                H.main(["--out", tmp, "--split", "test", "--allow-cpu", "--impl", "jax", "--episodes", "2"])
            reg.write_text(json.dumps({"families": ["gain_070"], "methods": ["none"], "telemetry": ["exact"],
                                       "episodes": 2, "steps": 10, "schedule": "walk", "registered_at": "earlier"}))
            with self.assertRaises(SystemExit):
                # episodes differ from the registration
                H.main(["--out", tmp, "--split", "test", "--registration", str(reg), "--allow-cpu", "--impl", "jax",
                        "--families", "gain_070", "--methods", "none", "--episodes", "4", "--steps", "10", "--schedule", "walk"])

    def test_registration_binds_the_whole_protocol(self):
        """AE-012: a test run must match the prior record on split, seeds, margins, configs and
        checkpoint, not only on the scenario dimensions; --register-only writes that record."""
        common = ["--allow-cpu", "--impl", "jax", "--families", "healthy", "--methods", "heuristic_gate",
                  "--telemetry", "declared", "--episodes", "2", "--steps", "10", "--schedule", "walk"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "reg"
            self.assertEqual(H.main(["--out", str(out), "--split", "test", "--register-only"] + common), 0)
            record = json.loads((out / "registration.json").read_text())
            for key in ("split", "family_seeds", "noninferiority_margins_loss", "telemetry_configs", "rls",
                        "heuristic", "checkpoint_sha256", "impl"):
                self.assertIn(key, record)
            self.assertEqual(record["split"], "test")
            self.assertEqual(record["family_seeds"], {"healthy": H.family_seed("test", "healthy")})
            # the same invocation agrees with its own record on every protocol key
            self.assertEqual(H.check_registration(record, record), [])
            mutations = {
                "split": "dev", "noninferiority_margins_loss": {"healthy": 0.123},
                "family_seeds": {"healthy": 11000}, "checkpoint_sha256": "0" * 64,
                "heuristic": dict(record["heuristic"], dead_band=0.5),
                "telemetry_configs": {"declared": dict(record["telemetry_configs"]["declared"], latency_max=0)},
            }
            for key, value in mutations.items():
                mutated = dict(record, **{key: value})
                self.assertEqual(H.check_registration(mutated, record), [key], key)
            # the development registration from the first test is rejected on split and more, before physics
            dev = dict(record, split="dev", noninferiority_margins_loss={"healthy": 0.005})
            reg = Path(tmp) / "dev_registration.json"
            reg.write_text(json.dumps(dev))
            with self.assertRaises(SystemExit):
                H.main(["--out", tmp, "--split", "test", "--registration", str(reg), "--margin", "0.123"] + common)


@unittest.skipUnless(os.environ.get("FORESIGHT_RUN_CPU_PHYSICS") == "1",
                     "set FORESIGHT_RUN_CPU_PHYSICS=1 to run the closed loop on CPU")
class ClosedLoopTests(unittest.TestCase):
    """The external correction path reproduces the in-env oracle bitwise, and estimators run."""

    @classmethod
    def setUpClass(cls):
        import jax
        from foresight.adaptive.actuator_env import AdaptiveJoystick
        from foresight.envs.g1_impact import default_config
        from foresight.eval import rollout as R
        cfg = default_config()
        cfg.impact.resample_on_autoreset = False
        cfg.impact.miss_fraction = 0.0
        cfg.impact.reset_perturb = 1.0
        cfg.push_config.enable = False
        cfg.episode_length = 60
        cls.env = AdaptiveJoystick(config=cfg, config_overrides={"impl": "jax"}, telemetry=True)
        # staticmethod: a plain function stored on the class would be bound to the instance.
        cls.policy = staticmethod(R.load_policy(R.latest_checkpoint(ROOT / "checkpoints/nominal_s1"), deterministic=True))
        cls.family = B.ScenarioFamily("test_gain", gain=0.7, onset_range=(20, 21), return_after=(25, 26))
        cls.scenario = B.draw_scenarios(cls.family, 500, 2, cls.env)
        keys = jax.random.split(jax.random.PRNGKey(0), 2)
        initial = jax.jit(jax.vmap(cls.env.reset))(keys)
        cls.initial = R.set_scenario(cls.env, initial, launch=False, command=jp.zeros((2, 3), jp.float32))
        cls.sub = {k: jp.asarray(v) for k, v in cls.scenario.items() if k != "ids"}

    def run_method(self, name, telemetry=S.EXACT, trace=False):
        import jax
        spec = B.METHODS[name]
        execute = B.build_bench_rollout(self.env, self.policy, spec, telemetry, B.WALK_TURN_STOP_RESUME, 60, trace=trace)
        return jax.device_get(execute(self.initial, self.sub))

    def test_external_oracle_matches_in_env_oracle_bitwise_and_estimators_run(self):
        oracle = self.run_method("oracle")
        external = self.run_method("oracle_external")
        for key in ("loss_sum", "progress", "max_correction", "max_slew", "max_force", "harmful_sum", "term_step"):
            np.testing.assert_array_equal(oracle[key], external[key], err_msg=key)
        none = self.run_method("none")
        self.assertTrue(np.all(none["max_correction"] == 0.0))
        est = self.run_method("estimated_inverse", trace=True)
        g = est["trace"]["g_hat"]            # (steps, n, nu)
        leg = np.asarray(self.env.leg_actuator_mask)
        np.testing.assert_allclose(g[19][:, leg], 1.0, atol=1e-6, err_msg="healthy before onset with exact telemetry")
        self.assertLess(g[45][:, leg].mean(), 0.95, "the estimate moves toward the 0.7 fault after onset")
        self.assertTrue(np.all(est["released_packets"] >= 55), "nearly every capture is released with zero latency")
        self.assertTrue(np.all(est["lost_packets"] == 0))
        gate = self.run_method("heuristic_gate", telemetry=S.DECLARED)
        # AE-011: with no actuator change the plant is healthy at every step, so the harmful
        # offset must be averaged over all observed steps, not just outside the schedule window.
        healthy_family = B.ScenarioFamily("test_healthy", gain=1.0, onset_range=(20, 21))
        hs = B.draw_scenarios(healthy_family, 501, 2, self.env)
        hsub = {k: jp.asarray(v) for k, v in hs.items() if k != "ids"}
        execute = B.build_bench_rollout(self.env, self.policy, B.METHODS["estimated_inverse"], S.DECLARED,
                                        B.WALK_TURN_STOP_RESUME, 60)
        import jax
        res = jax.device_get(execute(self.initial, hsub))
        alive_steps = np.where(res["term_step"] >= 0, res["term_step"] + 1, 60)
        np.testing.assert_array_equal(res["healthy_steps"], alive_steps)
        self.assertTrue(np.all(np.isfinite(gate["postfault_loss"])))
        self.assertTrue(np.all(gate["max_correction"] <= 0.2 + 1e-5))
        self.assertTrue(np.all(gate["max_slew"] <= 2.0 + 1e-3))


if __name__ == "__main__":
    unittest.main()
