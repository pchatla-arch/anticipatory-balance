"""Data publication stays within the dedicated results tree."""
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('experiment_publisher', ROOT / 'scripts/publish_experiment_results.py')
publisher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publisher)


class PublicationTests(unittest.TestCase):
    def test_nested_tables_are_allowed_and_code_is_rejected(self):
        self.assertTrue(publisher.allowed_tree_path('results/run1/pilot_20hz/episodes/batch000_condition000_reactive.csv'))
        self.assertTrue(publisher.allowed_tree_path('results/run1/run.json'))
        for path in ('scripts/start.py', 'results/run1/policy.py', 'results/run1/../secrets.json',
                     'results/run1/checkpoint.npz', 'results/run1/.credentials.json'):
            self.assertFalse(publisher.allowed_tree_path(path), path)

    def test_credential_bearing_logs_are_redacted(self):
        text = publisher.sanitize('https://example-user:password@github.com/pchatla-arch/anticipatory-balance.git ghp_TESTTOKEN123')
        self.assertNotIn('password', text)
        self.assertNotIn('TESTTOKEN123', text)


if __name__ == '__main__':
    unittest.main()
