"""CPU algebra and scalar-reference parity for WP 1.2."""
import json
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from foresight.adaptive.projection import (
    GainStatus, Interval, ProjectionStatus, hold_error, project_offset,
    propagate_gain, update_gain_set,
)


REFERENCE = Path(__file__).resolve().parents[1] / "docs/adaptive_execution/research_candidate/analytical_check.json"


def projection_inputs(cases, shape):
    def values(key):
        return jnp.asarray([case[key] for case in cases]).reshape(shape)

    def intervals(key):
        return Interval(*(jnp.asarray([case[key][idx] for case in cases]).reshape(shape)
                          for idx in range(2)))

    fields = ("estimate", "reference", "kp", "available_force", "max_offset",
              "slew_rate", "hold", "previous", "regularizer")
    kwargs = {key: values(key) for key in fields}
    kwargs["current_error"] = jnp.asarray([
        sum(case["errors"]) / 2 if case["current_error"] is None else case["current_error"]
        for case in cases]).reshape(shape)
    return dict(gains=intervals("gains"), current_gains=intervals("gains"), errors=intervals("errors"),
                target_range=intervals("target_range"), **kwargs)


class ProjectionParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads(REFERENCE.read_text())
        cls.kernel = staticmethod(jax.jit(project_offset))

    def test_all_published_cases_as_environment_joint_batch(self):
        records = self.reference["cases"]
        result = self.kernel(**projection_inputs([r["case"] for r in records], (2, 5)))
        for index, record in enumerate(records):
            expected = record["result"]
            loc = np.unravel_index(index, (2, 5))
            with self.subTest(case=record["case"]["name"]):
                self.assertEqual(ProjectionStatus(int(result.status[loc])).name, expected["status"])
                for key in ("candidate", "requested", "effective"):
                    actual = getattr(result, key)[loc]
                    if key in expected:
                        np.testing.assert_allclose(actual, expected[key], rtol=2e-6, atol=1e-7)
                    else:
                        self.assertTrue(np.isnan(actual))
                for key in ("requested_interval", "effective_interval"):
                    actual = [a[loc] for a in getattr(result, key)]
                    if key in expected:
                        np.testing.assert_allclose(actual, expected[key], rtol=2e-6, atol=1e-7)
                    else:
                        self.assertTrue(np.all(np.isnan(actual)))

    def test_published_interval_sweep(self):
        sweep = self.reference["sweep"]
        base = self.reference["cases"][0]["case"]
        cases = [dict(base, gains=(0.6, row["upper_gain"])) for row in sweep]
        result = self.kernel(**projection_inputs(cases, (len(cases),)))
        np.testing.assert_equal(np.asarray(result.status), ProjectionStatus.FEASIBLE)
        for key in ("requested", "effective", "candidate"):
            np.testing.assert_allclose(getattr(result, key), [row[key] for row in sweep],
                                       rtol=2e-6, atol=1e-7)

    def test_point_gain_uses_decision_time_interval(self):
        base = dict(self.reference["cases"][0]["case"], estimate=.4, current_error=.01)
        kwargs = projection_inputs([base], (1,))
        kwargs["current_gains"] = Interval(.7, .71)
        result = self.kernel(**kwargs)
        self.assertEqual(int(result.status[0]), ProjectionStatus.FEASIBLE)
        np.testing.assert_allclose(result.candidate[0], (1 / .7 - 1) * .01, rtol=2e-6)

    def test_invalid_inputs_and_separate_causal_point(self):
        base = self.reference["cases"][0]["case"]
        cases = [dict(base, estimate=np.nan), dict(base, current_error=np.inf),
                 dict(base, kp=0), dict(base, current_error=0.01, regularizer=1.0)]
        result = self.kernel(**projection_inputs(cases, (4,)))
        np.testing.assert_array_equal(result.status, [ProjectionStatus.INVALID_INPUT] * 3
                                      + [ProjectionStatus.FEASIBLE])
        np.testing.assert_allclose(result.requested[3], (1 / .7 - 1) * .01 / 2, rtol=2e-6)


class GainAndHoldTests(unittest.TestCase):
    def test_signed_measurement_weak_excitation_saturation_and_contradiction(self):
        prior = Interval(.5, 1.5)
        phi = jnp.array([.25, -.25, 0., 0., .25, .25, 1e-35])
        result = jax.jit(update_gain_set)(prior, prior, drift_rate=.1, elapsed=.5,
                                 held_target=phi, position=0., kp=4.,
                                 torque=jnp.array([.75, -.75, 0., 1., jnp.nan, jnp.nan, 0.]),
                                 torque_error_bound=.02, position_error_bound=.005,
                                 sample_valid=jnp.array([1, 1, 1, 1, 0, 1, 1], bool))
        np.testing.assert_array_equal(result.status, [GainStatus.UPDATED, GainStatus.UPDATED,
                                      GainStatus.UPDATED, GainStatus.INCONSISTENT,
                                      GainStatus.SKIPPED, GainStatus.INVALID_INPUT, GainStatus.UPDATED])
        np.testing.assert_allclose(result.interval.low, [.7, .7, .5, np.nan, .5, np.nan, .5],
                                   atol=1e-7, equal_nan=True)
        np.testing.assert_allclose(result.interval.high, [.8, .8, 1.5, np.nan, 1.5, np.nan, 1.5],
                                   atol=1e-7, equal_nan=True)

    def test_jump_rejected_and_explicit_restart_from_prior(self):
        prior, previous = Interval(.5, 1.5), Interval(.7, .8)
        params = dict(drift_rate=.01, elapsed=.5, held_target=1., position=0., kp=1.,
                      torque=1., torque_error_bound=.01, position_error_bound=0., sample_valid=True)
        mismatch = update_gain_set(previous, prior, **params)
        self.assertEqual(int(mismatch.status), GainStatus.INCONSISTENT)
        self.assertTrue(np.isnan(mismatch.interval.low))
        restarted = update_gain_set(prior, prior, **params)
        np.testing.assert_allclose(restarted.interval, [.99, 1.01], rtol=1e-6)

    def test_hold_expands_with_age_and_rejects_negative_elapsed(self):
        current = Interval(.7, .8)
        prior = Interval(.5, 1.5)
        gains = jax.jit(propagate_gain)(current, prior, 1., jnp.array([.07, .4, -.005]))
        np.testing.assert_allclose(gains.low, [.63, .5, np.nan], atol=1e-7, equal_nan=True)
        np.testing.assert_allclose(gains.high, [.87, 1.2, np.nan], atol=1e-7, equal_nan=True)
        errors = jax.jit(hold_error)(.1, 0., position_error_bound=.01, max_speed=2.,
                                     age=jnp.array([0., .05, -.01]), hold=.02)
        np.testing.assert_allclose(errors.low, [.05, -.05, np.nan], atol=1e-7, equal_nan=True)
        np.testing.assert_allclose(errors.high, [.15, .25, np.nan], atol=1e-7, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
