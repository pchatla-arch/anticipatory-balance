"""Persistence/counting tests, runnable with stdlib unittest and NumPy."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from foresight.eval import frontier_output as O


class FrontierOutputTests(unittest.TestCase):
  def test_counts_keep_distinct_denominators(self):
    table = dict(is_miss=np.array([0, 0, 0, 1, 1], bool),
                 contact=np.array([1, 1, 0, 1, 0], bool),
                 fell=np.array([1, 0, 1, 1, 0], bool),
                 false_prep=np.array([0, 0, 0, 1, 0], bool),
                 brace_steps_total=np.array([1, 0, 4, 2, 0]))
    counts = O.outcome_counts(table)
    self.assertEqual(counts['n_hits'], 3)
    self.assertEqual(counts['n_contact_hits'], 2)
    self.assertEqual(counts['n_falls_hits'], 2)
    self.assertEqual(counts['n_falls_contact_hits'], 1)
    self.assertEqual(counts['n_misses'], 2)
    rates = O.pooled_rates(counts)
    self.assertEqual(rates['fall_rate_hits_pooled'], 2/3)
    self.assertEqual(rates['fall_rate_given_contact_pooled'], 1/2)
    self.assertEqual(rates['false_prep_rate_pooled'], 1/2)

  def test_zero_denominator_is_explicit(self):
    counts = O.outcome_counts(dict(is_miss=np.array([False]), contact=np.array([False]),
                                  fell=np.array([False]), false_prep=np.array([False]),
                                  brace_steps_total=np.array([0])))
    self.assertIsNone(O.pooled_rates(counts)['false_prep_rate_pooled'])
    self.assertIsNone(O.pooled_rates(counts)['fall_rate_given_contact_pooled'])

  def test_csv_has_stable_paired_ids_and_full_precision(self):
    with tempfile.TemporaryDirectory() as tmp:
      a, b = Path(tmp)/'a.csv', Path(tmp)/'b.csv'
      table = dict(progress_m=np.array([0.123456789012345, 0.987654321098765]))
      for path, kind, period in [(a, 'reactive', 5), (b, 'belief_commit', 2.5)]:
        O.write_episodes(path, table, batch=1, seed=201, method=kind, param=.2,
                         latency_steps=5, period=period)
      with a.open() as fa, b.open() as fb:
        aa=list(csv.DictReader(fa)); bb=list(csv.DictReader(fb))
      self.assertEqual([r['scenario_id'] for r in aa], [r['scenario_id'] for r in bb])
      self.assertEqual(float(aa[0]['progress_m']), table['progress_m'][0])

  def test_existing_results_refused_and_invalid_overwrite_nondestructive(self):
    with tempfile.TemporaryDirectory() as tmp:
      path=O.prepare_output(Path(tmp)/'run')
      O.atomic_json(path/'config.json', {'important': 1})
      with self.assertRaises(ValueError):
        O.prepare_output(path)
      (path/'episodes'/'unrelated.txt').write_text('keep')
      with self.assertRaises(ValueError):
        O.prepare_output(path, overwrite=True)
      self.assertEqual(json.loads((path/'config.json').read_text()), {'important': 1})
      self.assertEqual((path/'episodes'/'unrelated.txt').read_text(), 'keep')

  def test_failed_atomic_write_preserves_prior_file(self):
    with tempfile.TemporaryDirectory() as tmp:
      path=Path(tmp)/'batches.json'
      O.atomic_json(path, {'complete': [1, 2]})
      with self.assertRaises(TypeError):
        O.atomic_json(path, {'bad': object()})
      self.assertEqual(json.loads(path.read_text()), {'complete': [1, 2]})
      self.assertEqual([p.name for p in Path(tmp).iterdir()], ['batches.json'])

  def test_preparation_excludes_actions_after_first_termination(self):
    # First world fails on transition 0, second never fails, third fails on 1.
    active=np.array([[0, 1, 0], [1, 1, 1], [1, 1, 1]], bool)
    got=O.preparation_until_termination(active, np.array([0, -1, 1]),
                                        np.ones(3, bool), np.ones_like(active))
    np.testing.assert_array_equal(got['brace_steps_until_termination'], [0, 3, 1])
    np.testing.assert_array_equal(got['false_prep_until_termination'], [False, True, True])
    np.testing.assert_array_equal(got['brace_onset_step_until_termination'], [-1, 0, 1])
    np.testing.assert_array_equal(got['n_detections_until_termination'], [1, 3, 2])


if __name__ == '__main__':
  unittest.main()
