"""Exercise the production transition with NumPy; JAX integration is optional."""
from importlib.util import find_spec
from types import SimpleNamespace
import unittest

import numpy as np

from foresight.decision import commitment as C


def config(**overrides):
  values = dict(release_mode="deadline", dt=0.02, hold=0.5, commit_steps=15,
                stale_steps=50, max_brace_seconds=3.0)
  values.update(overrides)
  return SimpleNamespace(**values)


def advance(cfg, memory, step, *, want=True, ttc=0.6, valid=True):
  return C.transition(cfg, memory, want, ttc, valid, step, xp=np)


class CommitmentTests(unittest.TestCase):
  def test_want_dies_before_impact_but_deadline_holds(self):
    cfg = config()
    memory = advance(cfg, C.init_memory(np), 0)
    deadline = int(memory.release_deadline_step)
    self.assertEqual(deadline, 55)  # (0.6 + 0.5) / 0.02
    for step in range(1, deadline):
      ttc = 0.6 - step * cfg.dt
      memory = advance(cfg, memory, step, want=ttc > 0.22,
                       ttc=ttc if ttc > 0 else np.inf)
      self.assertTrue(memory.braced, step)
      self.assertEqual(memory.release_deadline_step, deadline)
      self.assertEqual(memory.release_reason, C.RELEASE_NONE)
    memory = advance(cfg, memory, deadline, want=False, ttc=np.inf)
    self.assertFalse(memory.braced)
    self.assertEqual(memory.release_reason, C.RELEASE_DEADLINE)
    self.assertEqual(memory.n_switches, 2)

  def test_positive_to_infinity_does_not_release_or_extend(self):
    cfg = config()
    memory = advance(cfg, C.init_memory(np), 0, ttc=0.3)
    deadline = int(memory.release_deadline_step)
    for step in range(1, deadline):
      memory = advance(cfg, memory, step, want=False, ttc=np.inf, valid=False)
      self.assertTrue(memory.braced)
      self.assertEqual(memory.release_deadline_step, deadline)
    self.assertFalse(advance(cfg, memory, deadline, ttc=np.inf).braced)

  def test_fresh_detections_forever_still_release_and_never_retrigger(self):
    cfg = config()
    memory = C.init_memory(np)
    for step in range(200):
      memory = advance(cfg, memory, step, want=True, ttc=0.6, valid=True)
      self.assertEqual(bool(memory.braced), step < 55)
      self.assertEqual(memory.release_deadline_step, 55)
      self.assertTrue(memory.has_committed)
      self.assertEqual(memory.release_reason,
                       C.RELEASE_DEADLINE if step == 55 else C.RELEASE_NONE)
    self.assertEqual(memory.n_switches, 2)

  def test_never_observed_or_stale_belief_cannot_start(self):
    cfg = config()
    memory = advance(cfg, C.init_memory(np), 0, valid=False)
    self.assertFalse(memory.braced)
    memory = advance(cfg, memory, 1, want=False, valid=True)
    memory = advance(cfg, memory, 52, want=True, valid=False)
    self.assertFalse(memory.braced)
    self.assertFalse(memory.has_committed)
    self.assertEqual(memory.release_deadline_step, -1)

  def test_can_start_between_detections_while_observation_fresh(self):
    cfg = config()
    memory = advance(cfg, C.init_memory(np), 0, want=False)
    memory = advance(cfg, memory, 3, want=True, valid=False)
    self.assertTrue(memory.braced)
    self.assertEqual(memory.release_deadline_step, 58)

  def test_invalid_ttc_cannot_start(self):
    for ttc in (np.inf, np.nan, -0.1, 0.0):
      with self.subTest(ttc=ttc):
        memory = advance(config(), C.init_memory(np), 0, ttc=ttc)
        self.assertFalse(memory.braced)
        self.assertFalse(memory.has_committed)

  def test_staleness_after_entry_does_not_shorten_deadline(self):
    cfg = config(stale_steps=1)
    memory = advance(cfg, C.init_memory(np), 0)
    for step in range(1, 55):
      memory = advance(cfg, memory, step, want=False, valid=False)
      self.assertTrue(memory.braced)
    self.assertFalse(advance(cfg, memory, 55, valid=False).braced)

  def test_fractional_step_rounds_up_and_deadline_is_absolute(self):
    cfg = config(dt=0.03, hold=0.02, commit_steps=1)
    memory = advance(cfg, C.init_memory(np), 10, ttc=0.05)
    self.assertEqual(memory.release_deadline_step, 13)
    for step in (11, 12):
      memory = advance(cfg, memory, step, want=False)
      self.assertTrue(memory.braced)
    self.assertFalse(advance(cfg, memory, 13, want=False).braced)

  def test_minimum_commitment_counts_active_ticks(self):
    cfg = config(hold=0.0)
    memory = C.init_memory(np)
    active = []
    for step in range(16):
      memory = advance(cfg, memory, step, want=(step == 0), ttc=0.001)
      active.append(bool(memory.braced))
    self.assertEqual(active, [True] * 15 + [False])
    self.assertEqual(memory.release_deadline_step, 15)

  def test_duration_cap_and_minimum_commitment_precedence(self):
    cfg = config()
    memory = advance(cfg, C.init_memory(np), 7, ttc=1e30)
    self.assertEqual(memory.release_deadline_step, 157)
    cfg = config(commit_steps=200)
    memory = advance(cfg, C.init_memory(np), 7, ttc=1e30)
    self.assertEqual(memory.release_deadline_step, 207)

  def test_cap_rounds_down_to_remain_a_bound(self):
    cfg = config(dt=0.03, max_brace_seconds=0.10, commit_steps=1)
    memory = advance(cfg, C.init_memory(np), 0, ttc=5.0)
    self.assertEqual(memory.release_deadline_step, 3)

  def test_forecast_matches_entry_rule(self):
    belief = SimpleNamespace(t_hit=0.9, pe_t_hit=0.2)
    self.assertEqual(C.forecast_ttc("ttc_threshold", belief), 0.2)
    self.assertEqual(C.forecast_ttc("belief_commit", belief), 0.9)
    self.assertEqual(C.forecast_ttc("always", belief), 0.9)

  def test_legacy_reproduces_early_release_and_retrigger(self):
    cfg = config(release_mode="legacy")
    memory = advance(cfg, C.init_memory(np), 0)
    for step in range(1, 15):
      memory = advance(cfg, memory, step, want=False)
      self.assertTrue(memory.braced)
    memory = advance(cfg, memory, 15, want=False)
    self.assertFalse(memory.braced)
    self.assertEqual(memory.release_reason, C.RELEASE_REQUEST_FALSE)
    memory = advance(cfg, memory, 16, want=True)
    self.assertTrue(memory.braced)
    self.assertEqual(memory.release_deadline_step, -1)

  def test_stale_only_holds_on_infinite_ttc_and_releases_on_timeout(self):
    cfg = config(release_mode="stale_only")
    memory = advance(cfg, C.init_memory(np), 0)
    for step in range(1, 51):
      memory = advance(cfg, memory, step, want=False, ttc=np.inf, valid=False)
      self.assertTrue(memory.braced)
    memory = advance(cfg, memory, 51, want=False, ttc=np.inf, valid=False)
    self.assertFalse(memory.braced)
    self.assertEqual(memory.release_reason, C.RELEASE_STALE)

  def test_historical_modes_intentionally_retain_stale_start(self):
    for mode in ("legacy", "stale_only"):
      memory = advance(config(release_mode=mode), C.init_memory(np), 0, valid=False)
      self.assertTrue(memory.braced)

  def test_invalid_static_config_rejected(self):
    for changes in ({"release_mode": "unknown"}, {"dt": 0}, {"dt": np.inf},
                    {"hold": -1}, {"max_brace_seconds": np.inf},
                    {"commit_steps": 0}, {"commit_steps": 1.5}, {"stale_steps": -1}):
      with self.subTest(changes=changes), self.assertRaises(ValueError):
        advance(config(**changes), C.init_memory(np), 0)

  @unittest.skipUnless(find_spec("jax") is not None, "JAX is not installed locally")
  def test_jitted_transition_matches_numpy(self):
    import jax
    import jax.numpy as jnp

    for mode in C.RELEASE_MODES:
      cfg = config(release_mode=mode)
      fn = jax.jit(lambda memory, want, ttc, valid, step:
                   C.transition(cfg, memory, want, ttc, valid, step, xp=jnp))
      nmem, jmem = C.init_memory(np), C.init_memory(jnp)
      for step in range(100):
        want, valid = step < 20, step % 5 == 0
        # Avoid a ceil boundary in backend equivalence: float32 division and
        # XLA reciprocal multiplication may conservatively differ by one tick.
        ttc = np.float32(max(0.613 - step * cfg.dt, 0.001))
        nmem = C.transition(cfg, nmem, want, ttc, valid, step, xp=np)
        jmem = fn(jmem, want, ttc, valid, step)
        for expected, actual in zip(nmem, jmem):
          np.testing.assert_equal(np.asarray(actual), expected)


if __name__ == "__main__":
  unittest.main()
