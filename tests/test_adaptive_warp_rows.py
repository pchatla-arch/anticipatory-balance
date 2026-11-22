"""Row selection must tolerate the unbatched leaves of MuJoCo-Warp state.

Regression for the first two GPU gate runs (2026-11-16), which crashed with
"Incompatible shapes for broadcasting: shapes=[(32,), (0,), (0,)]" because the
CPU check used --impl jax, whose state has no unbatched leaves.
"""
import os
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))


class SelectRowsTests(unittest.TestCase):
    def test_batched_rows_selected_and_unbatched_leaves_passed_through(self):
        import jax.numpy as jp
        from eval_adaptive_gate import select_batched_rows
        n = 4
        mask = jp.array([True, False, True, False])
        new = {'q': jp.ones((n, 3)), 'flag': jp.ones(n), 'empty': jp.zeros((0,)),
               'pool': jp.full((7, 2), 5.), 'count': jp.array([9]), 'scalar': jp.array(2.)}
        old = {'q': jp.zeros((n, 3)), 'flag': jp.zeros(n), 'empty': jp.zeros((0,)),
               'pool': jp.full((7, 2), 1.), 'count': jp.array([3]), 'scalar': jp.array(1.)}
        out = select_batched_rows(mask, new, old)
        np.testing.assert_array_equal(out['q'][:, 0], [1, 0, 1, 0])
        np.testing.assert_array_equal(out['flag'], [1, 0, 1, 0])
        self.assertEqual(out['empty'].shape, (0,))
        np.testing.assert_array_equal(out['pool'], new['pool'])
        np.testing.assert_array_equal(out['count'], [9])
        self.assertEqual(float(out['scalar']), 2.)

    def test_unbatched_leaf_shape_change_is_rejected(self):
        import jax.numpy as jp
        from eval_adaptive_gate import select_batched_rows
        with self.assertRaises(ValueError):
            select_batched_rows(jp.ones(2, bool), {'a': jp.zeros((0,))}, {'a': jp.zeros((1,))})


@unittest.skipUnless(os.environ.get('FORESIGHT_RUN_CPU_PHYSICS') == '1',
                     'set FORESIGHT_RUN_CPU_PHYSICS=1 to trace the Warp rollout on CPU')
class WarpTraceTests(unittest.TestCase):
    """Traces (does not execute) the gate rollout with the pod's Warp backend."""

    def test_warp_rollout_traces_at_pod_batch_sizes(self):
        import jax
        import eval_adaptive_gate as G
        from foresight.adaptive.actuator_env import AdaptiveJoystick
        from foresight.envs.g1_impact import default_config
        from foresight.eval import rollout as R
        cfg = default_config()
        cfg.impact.resample_on_autoreset = False
        cfg.impact.miss_fraction = 0.
        cfg.impact.reset_perturb = 1.
        cfg.push_config.enable = False
        cfg.episode_length = G.HORIZON
        env = AdaptiveJoystick(config=cfg, config_overrides={'impl': 'warp'})
        policy = R.load_policy(R.latest_checkpoint(ROOT / 'checkpoints/nominal_s1'), deterministic=True)
        execute = G.build_rollout(env, policy, steps=3)
        for n in (8, 32):
            initial = jax.eval_shape(jax.vmap(env.reset), jax.random.split(jax.random.PRNGKey(0), n))
            out = jax.eval_shape(execute, initial)
            self.assertEqual(out['alive'].shape, (n,))



class ReproducibilityCheckTests(unittest.TestCase):
    """The CPU/pod reproducibility probe reports first divergence with step, channel and units."""

    def test_first_divergence_locates_step_channel_and_onset_relation(self):
        from check_warp_reproducibility import first_divergence, compare, KEYS
        steps, n, nu = 10, 3, 4
        a = {'ctrl': np.zeros((steps, n, nu)), 'qpos': np.zeros((steps, n, 7)), 'actuator_force': np.zeros((steps, n, nu))}
        b = {k: v.copy() for k, v in a.items()}
        b['ctrl'][6, 1, 2] = 1e-7      # first divergence
        b['ctrl'][8, 0, 3] = 5e-3      # largest divergence
        onsets = np.array([5, 7, 9])
        report = first_divergence(a, b, onsets)
        self.assertTrue(report['qpos']['bitwise_equal'])
        ctrl = report['ctrl']
        self.assertFalse(ctrl['bitwise_equal'])
        self.assertEqual((ctrl['first_divergence']['step'], ctrl['first_divergence']['env'], ctrl['first_divergence']['channel']), (6, 1, 2))
        self.assertTrue(ctrl['first_divergence']['before_onset'], 'env 1 onset is 7, divergence at step 6')
        self.assertEqual(ctrl['first_divergence']['onset_step_of_that_env'], 7)
        self.assertEqual((ctrl['largest']['step'], ctrl['largest']['channel']), (8, 3))
        self.assertAlmostEqual(ctrl['largest']['abs_difference'], 5e-3)
        self.assertIn('rad', ctrl['units'])
        same = {k: np.ones(n) for k in KEYS}
        other = dict(same, max_force=np.ones(n) + 1e-3)
        self.assertTrue(compare(same, same)['bitwise_equal'])
        self.assertFalse(compare(same, other)['bitwise_equal'])
        self.assertAlmostEqual(compare(same, other)['max_abs_difference']['max_force'], 1e-3)


if __name__ == '__main__':
    unittest.main()
