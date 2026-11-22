from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import publish_experiment_results as publisher
import s1_dev_bench as s1


class DevelopmentBenchTest(unittest.TestCase):
    def setUp(self):
        self.plan = s1.read(s1.ROOT / s1.MANIFEST)
        self.files = {'heuristic': Path('heuristic.json'), 'candidate': Path('candidate.json')}

    def test_only_development_warp_matrix(self):
        args = s1.command(self.plan, self.files, Path('results'), False)
        self.assertEqual(args[args.index('--split') + 1], 'dev')
        self.assertEqual(args[args.index('--impl') + 1], 'warp')
        self.assertNotIn('--registration', args)
        self.assertNotIn('--allow-cpu', args)
        for smoke in (False, True):
            self.assertEqual(s1.dimensions(self.plan, smoke)[1], ['exact', 'declared', 'declared_fast'])
        families, telemetry, methods, episodes, _, steps = s1.dimensions(self.plan, False)
        self.assertEqual((len(families), len(telemetry), len(methods), episodes, steps), (11, 3, 5, 32, 400))

    def test_incomplete_or_changed_protocol_fails(self):
        plan = {**self.plan, 'split': 'test'}
        with self.assertRaisesRegex(ValueError, 'only development'):
            s1.frozen_inputs(plan)
        plan = {**self.plan, 'frozen_inputs': {'heuristic': None, 'candidate': None}}
        with self.assertRaisesRegex(ValueError, 'not been frozen'):
            s1.frozen_inputs(plan)

    def test_each_bench_artifact_survives_publication(self):
        for prefix in ('', 'cpu_smoke/'):
            for filename in ('registration.json', 'actuators.json', 'episodes.csv', 'summary.json', 'report.md', 'config.json'):
                self.assertTrue(publisher.allowed_tree_path('results/run/' + prefix + filename))
        self.assertFalse(publisher.allowed_tree_path('results/run/other/registration.json'))
        self.assertFalse(publisher.allowed_tree_path('results/run/cpu_smoke/unknown.npz'))
