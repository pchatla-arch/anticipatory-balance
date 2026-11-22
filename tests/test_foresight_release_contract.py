"""Check the paper's discrete-time release contract against the actual JAX controller."""
from types import SimpleNamespace
import unittest

import jax
import jax.numpy as jp

from foresight.belief import particle_filter as pf
from foresight.decision import rules


class ReleaseContract(unittest.TestCase):
    cfg = rules.RuleConfig(kind="ttc_threshold", theta=0.3)
    rm = rules.ResponseModel(0, 0, 0, 0)

    def tick(self, mem, k, *, want=False, t_pass=float("inf"), delivery=False):
        # Only these fields are consumed by the TTC start predicate and shared wrapper.
        b = SimpleNamespace(pe_hit=jp.array(want), pe_t_hit=jp.float32(0.1),
                            t_pass=jp.float32(t_pass))
        return rules.step(self.cfg, self.rm, mem, b, jp.array(delivery), jp.int32(k))

    def test_minimum_commit_and_strict_pass_threshold(self):
        mem = rules.init_memory()
        mem, active = self.tick(mem, 8, want=True, delivery=True)
        self.assertTrue(active)
        # Even a reliable "already passed" estimate does not interrupt these 15 intervals.
        for k in range(9, 23):
            mem, active = self.tick(mem, k, t_pass=-1.)
            self.assertTrue(active)
        self.assertEqual(int(mem.brace_steps), 15)
        # From interval 23, release is allowed. At exactly −0.30 the condition is false.
        still, active = self.tick(mem, 23, t_pass=-0.3)
        self.assertTrue(active)
        for starting_mem, k in ((mem, 23), (still, 24)):
            released, active = self.tick(starting_mem, k, want=True, t_pass=-0.301)
            self.assertFalse(active)
            self.assertEqual(int(released.release_reason), rules.RELEASE_PASSED)
            # An asserted start predicate is eligible again on the next update.
            _, active = self.tick(released, k + 1, want=True, t_pass=-0.301)
            self.assertTrue(active)

    def test_strict_staleness_and_current_delivery(self):
        mem = rules.init_memory()
        mem, _ = self.tick(mem, 0, want=True, delivery=True)
        for k in range(1, 51):
            mem, active = self.tick(mem, k)
            self.assertTrue(active)
        released, active = self.tick(mem, 51)
        self.assertFalse(active)
        self.assertEqual(int(released.release_reason), rules.RELEASE_STALE)
        refreshed, active = self.tick(mem, 51, delivery=True)
        self.assertTrue(active)
        self.assertEqual(int(refreshed.last_det_step), 51)

    def test_closest_approach_sign_and_uninitialized_belief(self):
        cfg = pf.PFConfig()
        state = pf.init_state(cfg)
        # At x=2 the object recedes at 1.5 m/s while the pelvis moves at 0.5 m/s.
        s = state._replace(mu=jp.array([2., 0., 1., 1.5, 0., 0.]))
        pelvis = jp.zeros(3)
        velocity = jp.array([0.5, 0., 0.])
        key = jax.random.PRNGKey(0)
        self.assertTrue(jp.isposinf(pf.stats(cfg, s, pelvis, velocity, key).t_pass))
        s = s._replace(initialized=jp.array(True))
        self.assertAlmostEqual(float(pf.stats(cfg, s, pelvis, velocity, key).t_pass),
                               -2. / (1. + 1e-6), places=5)
        s = s._replace(mu=s.mu.at[0].set(-2.))
        self.assertAlmostEqual(float(pf.stats(cfg, s, pelvis, velocity, key).t_pass),
                               2. / (1. + 1e-6), places=5)


if __name__ == "__main__":
    unittest.main()
