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


if __name__ == '__main__':
    unittest.main()
