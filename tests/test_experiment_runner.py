"""Regression checks for false-success cases in the pod execution gate."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment_runner', ROOT / 'scripts/run_experiments_v2.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class StageValidationTests(unittest.TestCase):
    def fixture(self, path):
        (path / 'episodes').mkdir()
        rows = []
        for i, method in enumerate(('reactive', 'ttc_threshold')):
            rows.append(dict(method=method, n_total=2, fall_rate_hits=.5,
                             contact_rate_hits=1., false_prep_rate=0., progress_m=1., braced_s=.2))
            with (path / 'episodes' / f'{i}.csv').open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['method', 'n_detections_until_termination'])
                writer.writeheader()
                writer.writerows([dict(method=method, n_detections_until_termination=3)] * 2)
        (path / 'summary.json').write_text(json.dumps(dict(status='complete', episodes_per_condition=2, rows=rows)))
        (path / 'config.json').write_text(json.dumps(dict(status='complete', period=2.5)))
        (path / 'batches.json').write_text(json.dumps(dict(completed_condition_batches=2, rows=rows)))

    def test_complete_stage_passes(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.fixture(path)
            self.assertTrue(runner.validate_stage(path, 2, 2)['passed'])

    def test_missing_condition_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.fixture(path)
            (path / 'episodes/1.csv').unlink()
            with self.assertRaisesRegex(RuntimeError, 'missing per-condition'):
                runner.validate_stage(path, 2, 2)

    def test_detections_only_after_termination_do_not_pass(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.fixture(path)
            (path / 'episodes/1.csv').write_text('method,n_detections,n_detections_until_termination\nttc_threshold,15,0\nttc_threshold,15,0\n')
            with self.assertRaisesRegex(RuntimeError, 'no valid detections'):
                runner.validate_stage(path, 2, 2)

    def test_nonfinite_result_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            self.fixture(path)
            data = json.loads((path / 'summary.json').read_text())
            data['rows'][0]['progress_m'] = float('nan')
            (path / 'summary.json').write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
                runner.validate_stage(path, 2, 2)


if __name__ == '__main__':
    unittest.main()
