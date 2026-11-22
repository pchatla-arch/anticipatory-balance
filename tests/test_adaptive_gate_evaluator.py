"""Rollout bookkeeping with a tiny JAX plant, without MuJoCo or GPU work."""
from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import NamedTuple
import unittest

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
import jax
import jax.numpy as jp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import eval_adaptive_gate as evaluator


class FakeData(NamedTuple):
    qpos: object
    qvel: object
    xpos: object
    sensordata: object
    actuator_force: object
    xmat: object


class FakeState(NamedTuple):
    data: FakeData
    obs: dict
    info: dict
    done: object


class FakeEnv:
    dt = .02
    _pelvis_body = 0

    def _pelvis_pos(self, data):
        return data.qpos[:3]

    def get_local_linvel(self, data, name):
        return data.qvel[:3]

    def step(self, state, action):
        info = dict(state.info)
        def assert_live_input(terminated, command):
            if bool(terminated):
                raise AssertionError('The evaluator simulated a terminal state instead of a safe reset.')
            if not np.isfinite(command).all():
                raise AssertionError('The evaluator sent a nonfinite policy action to physics.')
        jax.debug.callback(assert_live_input, info['terminated'], action, ordered=True)
        step = info['ep_step'] + 1
        terminal = (info['fake_terminal_step'] >= 0) & (step >= info['fake_terminal_step'])
        poison = (info['fake_nan_step'] >= 0) & (step >= info['fake_nan_step'])
        qpos = state.data.qpos.at[0].add(.5 * self.dt)
        qpos = jp.where(poison, jp.full_like(qpos, jp.nan), qpos)
        data = state.data._replace(qpos=qpos, xpos=qpos[None, :])
        info.update(ep_step=step, terminated=terminal,
                    adaptive_step_delta=jp.zeros(2), adaptive_prev_delta=jp.zeros(2))
        return state._replace(data=data, obs={'state': qpos[:2]}, info=info,
                              done=terminal.astype(jp.float32))


def policy(obs, key):
    return jp.zeros(2), {}


def initial_state(terminal_steps, nan_steps, onsets):
    n = len(onsets)
    data = FakeData(qpos=jp.zeros((n, 3)),
                    qvel=jp.tile(jp.array([.5, 0., 0.]), (n, 1)),
                    xpos=jp.zeros((n, 1, 3)),
                    sensordata=jp.zeros((n, 2)),
                    actuator_force=jp.full((n, 2), 10.),
                    xmat=jp.tile(jp.eye(3).reshape(1, 1, 9), (n, 1, 1)))
    info = {
        'ep_step': jp.zeros(n, jp.int32),
        'adaptive_onset_step': jp.array(onsets, jp.int32),
        'command_nominal': jp.tile(jp.array([.5, 0., 0.]), (n, 1)),
        'rng': jax.random.split(jax.random.PRNGKey(12), n),
        'terminated': jp.zeros(n, bool),
        'adaptive_step_delta': jp.zeros((n, 2)),
        'adaptive_prev_delta': jp.zeros((n, 2)),
        'adaptive_bound_hit': jp.zeros(n),
        'adaptive_control_clip_fraction': jp.zeros(n),
        'fake_terminal_step': jp.array(terminal_steps, jp.int32),
        'fake_nan_step': jp.array(nan_steps, jp.int32),
    }
    return FakeState(data, {'state': jp.zeros((n, 2))}, info, jp.zeros(n))


class AdaptiveEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.execute = staticmethod(evaluator.build_rollout(FakeEnv(), policy, steps=6))

    def test_nan_after_finite_terminal_does_not_change_outcome_or_progress(self):
        # Lane 1 becomes non-finite only after its finite terminal transition;
        # lane 2 has the same termination with entirely finite subsequent states.
        result = jax.device_get(self.execute(initial_state([-1, 3, 3], [-1, 4, -1], [1, 1, 1])))
        for key, values in result.items():
            self.assertTrue(np.isfinite(values).all(), key)
            np.testing.assert_array_equal(values[1], values[2], err_msg=key)
        np.testing.assert_array_equal(result['term_step'], [-1, 2, 2])
        np.testing.assert_array_equal(result['first_nonfinite_step'], [-1, -1, -1])
        np.testing.assert_allclose(result['progress'], [.05, .02, .02], atol=1e-7)
        np.testing.assert_allclose(result['postfault_loss'], [0., 3.2, 3.2], atol=1e-7)

    def test_nan_while_alive_is_retained_as_failure_with_remaining_horizon_loss(self):
        result = jax.device_get(self.execute(initial_state([-1, -1, -1], [-1, 3, -1], [1, 1, 1])))
        for key, values in result.items():
            self.assertTrue(np.isfinite(values).all(), key)
            np.testing.assert_array_equal(values[0], values[2], err_msg=key)
        np.testing.assert_array_equal(result['term_step'], [-1, 2, -1])
        np.testing.assert_array_equal(result['first_nonfinite_step'], [-1, 2, -1])
        np.testing.assert_allclose(result['progress'], [.05, .01, .05], atol=1e-7)
        # One healthy post-fault step, the invalid step, and three remaining
        # failed steps: (0 + 4 + 4 + 4 + 4) / 5 = 3.2, not zero or NaN.
        np.testing.assert_allclose(result['postfault_loss'], [0., 3.2, 0.], atol=1e-7)
        self.assertFalse(result['alive'][1])

    def test_prefault_failure_receives_full_fixed_postfault_penalty(self):
        result = jax.device_get(self.execute(initial_state([3, -1, -1], [-1, -1, -1], [4, 4, 1])))
        np.testing.assert_array_equal(result['pre_fault_failed'], [True, False, False])
        np.testing.assert_array_equal(result['first_nonfinite_step'], [-1, -1, -1])
        np.testing.assert_allclose(result['postfault_loss'], [4., 0., 0.], atol=1e-7)
        np.testing.assert_allclose(result['progress'], [0., .02, .05], atol=1e-7)
        np.testing.assert_array_equal(result['exposed'], [0, 2, 5])

    def test_nonfinite_policy_action_is_recorded_without_reaching_physics(self):
        def invalid_policy(obs, key):
            return jp.full(2, jp.nan), {}
        execute = evaluator.build_rollout(FakeEnv(), invalid_policy, steps=6)
        result = jax.device_get(execute(initial_state([-1, -1], [-1, -1], [1, 1])))
        np.testing.assert_array_equal(result['term_step'], [0, 0])
        np.testing.assert_array_equal(result['first_nonfinite_step'], [0, 0])
        np.testing.assert_array_equal(result['postfault_loss'], [4., 4.])
        np.testing.assert_array_equal(result['progress'], [0., 0.])

    def test_scenarios_are_repeatable_with_disjoint_confirmation_ids(self):
        screen = evaluator.scenarios(evaluator.SCREEN_SEED, 32)
        repeat = evaluator.scenarios(evaluator.SCREEN_SEED, 32)
        confirm = evaluator.scenarios(evaluator.CONFIRM_SEED, 128)
        for key in screen:
            np.testing.assert_array_equal(screen[key], repeat[key])
        self.assertFalse(set(screen['ids']) & set(confirm['ids']))
        self.assertTrue(np.all((screen['onset'] >= 75) & (screen['onset'] <= 125)))
        arms = evaluator.conditions('gain_070')
        self.assertEqual([arm[0] for arm in arms],
                         ['healthy', 'identity', 'healthy_repeat', 'gain_070_plain', 'gain_070_oracle'])
        self.assertEqual(arms[0][1], arms[1][1])
        # The probe is the healthy arm again: same gain, correction off.
        self.assertEqual(arms[2][1:], arms[0][1:])
        self.assertEqual([arm[0] for arm in evaluator.conditions('gain_070', repeat_probe=False)],
                         ['healthy', 'identity', 'gain_070_plain', 'gain_070_oracle'])

    def test_deterministic_execution_flags_and_model_option(self):
        import os
        saved = os.environ.get('XLA_FLAGS')
        try:
            os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
            self.assertFalse(evaluator.configure_determinism(False))
            self.assertEqual(os.environ['XLA_FLAGS'], '--xla_gpu_autotune_level=0')
            self.assertTrue(evaluator.configure_determinism(True))
            self.assertEqual(os.environ['XLA_FLAGS'].split(),
                             ['--xla_gpu_autotune_level=0', '--xla_gpu_deterministic_ops=true'])
            evaluator.configure_determinism(True)
            self.assertEqual(os.environ['XLA_FLAGS'].count('deterministic_ops'), 1)
        finally:
            if saved is None:
                os.environ.pop('XLA_FLAGS', None)
            else:
                os.environ['XLA_FLAGS'] = saved
        sentinel = object()
        self.assertEqual(evaluator.deterministic_model(sentinel, 'jax'), (sentinel, None))

        class FakeModel:
            def tree_replace(self, updates):
                self.updates = updates
                return self
        model, mask = evaluator.deterministic_model(FakeModel(), 'warp')
        self.assertEqual(mask, 7, 'CONTACTS | CONSTRAINT | ATOMICS')
        self.assertEqual(model.updates, {'opt._impl.deterministic': 7})


if __name__ == '__main__':
    unittest.main()
