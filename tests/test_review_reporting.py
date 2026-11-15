"""Synthetic, simulator-independent integration checks for the evidence gate."""
from __future__ import annotations
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from foresight.eval import statistics as ST

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


V = module('validate_review_stage')
S = module('summarize_review_run')
SHA = 'a'*40


def fixture(path, kind='frontier', all_failed=False):
    path.mkdir(parents=True, exist_ok=True)
    cfg = {'schema_version': ST.SCHEMA_VERSION, 'kind': kind, 'dt': .02,
           'steps': 30, 'release_mode': 'deadline', 'single_commit': True,
           'commit_steps': 2, 'max_hold_s': .1, 'period': 5}
    if kind != 'depth':
        cfg['timing_reference' if kind == 'oracle' else 'oracle_timing'] = 'reference_entry'
    prov = {'source_commit': SHA, 'source_dirty': False, 'argv': ['synthetic'],
            'artifacts': {'nominal': {'path': 'synthetic/checkpoint',
                         'files': [{'path': 'params', 'bytes': 12, 'sha256': 'b'*64}]}},
            'config_sha256': hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()}
    tables, rows = [], []
    for i in range(2):
        brace = np.array([3, 3, 3, 0]) if i else np.zeros(4, int)
        miss = np.array([False, False, False, kind != 'oracle'])
        contact = ~miss
        fell = np.array([False, True, False, False]) if not all_failed else ~miss
        table = {'scenario_id': np.array([f'seed1_env{x}' for x in range(4)]),
                 'is_miss': miss, 'contact': contact, 'fell': fell,
                 'censored': np.zeros(4, bool), 'brace_steps_total': brace,
                 'brace_onset_step': np.where(brace > 0, 1, -1),
                 'brace_release_step': np.where(brace > 0, 4, -1),
                 'contact_step': np.where(contact, 3, -1), 'term_step': np.where(fell, 10, -1),
                 'progress_m': np.array([.4, .5, .6, .7]) - .1*i,
                 'brace_at_contact': contact & (brace > 0),
                 'mass': np.full(4, 10.), 'speed': np.full(4, 4.), 't_hit_nominal': np.full(4, .4)}
        if kind == 'oracle':
            condition = {'mode': 'brace', 'launch': 1, 'lead': .1 if i else -1.}
            summary_key = {'mode': 'brace', 'launch': 1, 'lead_s': condition['lead']}
        elif kind == 'frontier':
            condition = {'method': 'ttc_threshold' if i else 'reactive', 'param': .3 if i else 0., 'latency_steps': 0}
            summary_key = condition
        else:
            condition = {'source': 'depth' if i else 'none', 'method': 'ttc_threshold' if i else 'reactive', 'param': .3 if i else 0.}
            summary_key = condition
        table.update({k: np.full(4, v) for k, v in condition.items()})
        rows.append({**summary_key, **ST.summarize_episodes(table, .02)})
        tables.append(table)
    summary = {'schema_version': ST.SCHEMA_VERSION, 'status': 'complete',
               'provenance': prov, 'episodes_per_condition': 4,
               'curve' if kind == 'oracle' else 'rows': rows}
    if kind == 'oracle':
        summary['reference_quality'] = [{'batch': 0, 'seed': 1, 'n': 4, 'n_reference_events': 4}]
        pair = {'mode': 'brace', 'launch': 1, 'condition_a': 'continue', 'condition_b': 'lead_0.1',
                **ST.paired_comparison(tables[0], tables[1], .02)}
    else:
        if kind == 'frontier':
            summary['reference_quality'] = [{'batch': 0, 'seed': 1, 'n_hits': 3, 'n_reference_events_hits': 3}]
            a, b = 'ttc_threshold:p=0.3:lat=0', 'reactive:p=0:lat=0'
        else:
            a, b = ['depth', 'ttc_threshold', .3], ['none', 'reactive', 0.]
        pair = {'condition_a': a, 'condition_b': b, **ST.paired_comparison(tables[1], tables[0], .02)}
    config_file = {**cfg, 'provenance': prov} if kind == 'depth' else {'config': cfg, 'provenance': prov}
    ST.write_json(path/'config.json', config_file)
    ST.write_json(path/'summary.json', summary)
    ST.write_json(path/'paired.json', [pair])
    if kind == 'depth':
        (path/'episodes').mkdir()
        for i, table in enumerate(tables):
            ST.write_csv(path/'episodes'/f'condition{i}.csv', table)
    else:
        ST.write_csv(path/'episodes.csv', ST.concatenate_tables(tables))
    return summary


def change_json(path, fn):
    value = json.loads(path.read_text())
    fn(value)
    ST.write_json(path, value)


def change_csv(path, fn):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    fn(rows)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


class StageValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)/'stage'
        fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def check(self, kind='frontier'):
        return V.validate_stage(self.path, kind, 4, 2, SHA)

    def test_valid_frontier_recomputes_counts_and_pairs(self):
        result = self.check()
        self.assertTrue(result['passed'])
        self.assertEqual(result['episode_rows'], 8)
        self.assertEqual(result['reference_entry_coverage'], 1.)

    def test_valid_depth_flat_configuration_and_split_csv(self):
        path = Path(self.temp.name)/'depth'
        fixture(path, 'depth')
        self.assertTrue(V.validate_stage(path, 'depth', 4, 2, SHA)['passed'])

    def test_no_misses_null_rates_are_valid(self):
        path = Path(self.temp.name)/'oracle'
        fixture(path, 'oracle')
        self.assertTrue(V.validate_stage(path, 'oracle', 4, 2, SHA)['passed'])

    def test_all_failed_is_valid_evidence_not_a_performance_gate(self):
        path = Path(self.temp.name)/'all_failed'
        fixture(path, 'frontier', all_failed=True)
        self.assertTrue(V.validate_stage(path, 'frontier', 4, 2, SHA)['passed'])

    def test_incomplete_and_wrong_source_rejected(self):
        change_json(self.path/'summary.json', lambda s:s.update(status='running'))
        with self.assertRaisesRegex(ValueError, 'complete'):
            self.check()
        change_json(self.path/'summary.json', lambda s:s.update(status='complete'))
        with self.assertRaisesRegex(ValueError, 'provenance'):
            V.validate_stage(self.path, 'frontier', 4, 2, 'c'*40)

    def test_configuration_content_hash_rejected(self):
        change_json(self.path/'config.json', lambda s:s['config'].update(steps=31))
        with self.assertRaisesRegex(ValueError, 'hash'):
            self.check()

    def test_duplicate_ids_and_missing_episode_rejected(self):
        change_csv(self.path/'episodes.csv', lambda rows:rows[1].update(scenario_id=rows[0]['scenario_id']))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.check()
        change_csv(self.path/'episodes.csv', lambda rows:rows.pop())
        with self.assertRaises(ValueError):
            self.check()

    def test_wrong_pooled_rate_rejected(self):
        change_json(self.path/'summary.json', lambda s:s['rows'][0].update(fall_rate_hits=.9))
        with self.assertRaisesRegex(ValueError, 'pooled'):
            self.check()

    def test_nonfinite_progress_rejected(self):
        change_csv(self.path/'episodes.csv', lambda rows:rows[0].update(progress_m='inf'))
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.check()

    def test_impossible_commitment_rejected(self):
        change_csv(self.path/'episodes.csv', lambda rows:rows[4].update(brace_steps_total=6, brace_release_step=7))
        with self.assertRaisesRegex(ValueError, 'maximum'):
            self.check()

    def test_reference_coverage_and_missing_evidence_rejected(self):
        change_json(self.path/'summary.json', lambda s:s['reference_quality'][0].update(n_reference_events_hits=2))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.check()
        change_json(self.path/'summary.json', lambda s:s.pop('reference_quality'))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.check()

    def test_no_reactive_contacts_rejected(self):
        change_csv(self.path/'episodes.csv', lambda rows:[r.update(contact=False, contact_step=-1) for r in rows[:4]])
        summary = json.loads((self.path/'summary.json').read_text())
        with (self.path/'episodes.csv').open() as stream:
            rows = list(csv.DictReader(stream))
        summary['rows'][0].update(ST.summarize_episodes(V.episode_table(rows[:4]), .02))
        ST.write_json(self.path/'summary.json', summary)
        with self.assertRaisesRegex(ValueError, 'reactive'):
            self.check()

    def test_paired_interval_must_match_raw_episodes(self):
        change_json(self.path/'paired.json', lambda p:p[0]['failure_hits'].update(normal95=[-.7,.7]))
        with self.assertRaisesRegex(ValueError, 'paired episode'):
            self.check()

    def test_cli_failure_writes_audit(self):
        code = V.main(['--stage-dir', str(self.path), '--kind', 'frontier',
                       '--episodes-per-condition', '5', '--conditions', '2', '--source-commit', SHA])
        self.assertEqual(code, 1)
        self.assertFalse(json.loads((self.path/'audit.json').read_text())['passed'])


class RunSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        stage = self.path/'frontier_test_p5'
        fixture(stage)
        ST.write_json(stage/'audit.json', V.validate_stage(stage, 'frontier', 4, 2, SHA))
        self.stage = {'name': 'frontier_test_p5', 'kind': 'frontier', 'split': 'test',
                      'status': 'passed', 'episodes_per_condition': 4, 'conditions': 2}
        self.run = {'run_id': 'synthetic', 'source_commit': SHA, 'mode': 'full',
                    'status': 'completed', 'elapsed_seconds': 1.,
                    'planned_stages': [self.stage], 'stages': [self.stage]}
        ST.write_json(self.path/'run.json', self.run)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_run_generates_report_and_machine_readable_pairs(self):
        analysis = S.summarize_run(self.path, plots=False)
        self.assertEqual(analysis['status'], 'complete')
        self.assertIn('paired', analysis['stages'][0])
        self.assertIn('Fixed controller comparisons', (self.path/'report.md').read_text())
        self.assertTrue((self.path/'analysis.json').is_file())

    def test_unstarted_planned_stage_is_partial(self):
        self.run['status'] = 'failed'
        self.run['planned_stages'].append({'name':'depth_test_l0', 'kind':'depth', 'split':'test'})
        ST.write_json(self.path/'run.json', self.run)
        analysis = S.summarize_run(self.path, plots=False)
        self.assertEqual(analysis['status'], 'partial')
        self.assertEqual(analysis['stages'][1]['status'], 'not_started')
        self.assertIn('not zero-valued results', (self.path/'report.md').read_text())

    def test_missing_audit_cannot_be_labeled_complete(self):
        (self.path/'frontier_test_p5'/'audit.json').unlink()
        analysis = S.summarize_run(self.path, plots=False)
        self.assertEqual(analysis['status'], 'partial')
        self.assertEqual(S.preferred_stages(analysis, 'frontier'), [])

    def test_summary_must_match_requested_source(self):
        self.run['source_commit'] = 'c'*40
        ST.write_json(self.path/'run.json', self.run)
        self.assertEqual(S.collect_run(self.path)['status'], 'partial')

    def test_no_stages_does_not_invent_success(self):
        self.run.update(stages=[], planned_stages=[])
        ST.write_json(self.path/'run.json', self.run)
        analysis = S.summarize_run(self.path, plots=False)
        self.assertEqual(analysis['status'], 'partial')
        self.assertIn('No experiment stages recorded', (self.path/'report.md').read_text())

    def test_plots_only_for_available_validated_data(self):
        analysis = S.summarize_run(self.path, plots=True)
        if analysis['plots'].get('availability', {}).get('status') == 'unavailable':
            self.assertTrue((self.path/'report.md').is_file())
            return
        self.assertTrue((self.path/'fixed_controller_comparison.png').is_file())
        self.assertFalse((self.path/'depth_comparison.png').exists())
        self.assertFalse((self.path/'readiness.png').exists())


if __name__ == '__main__':
    unittest.main()
