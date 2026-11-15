"""Experiment splits, fixed test parameters, and evidence publication contract."""
import importlib.util
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import review_plan as plan
import publish_experiment_results as publisher
import run_review_experiments as runner


class ReviewRunnerTests(unittest.TestCase):
    def test_splits_disjoint_with_intentional_within_test_pairing(self):
        stages = plan.make_plan(Path('OUT'))
        self.assertTrue(plan.assert_disjoint_splits(stages))
        test = [s for s in stages if s['split']=='test']
        self.assertEqual(len(test), 4)
        self.assertEqual([s['conditions'] for s in test], [14, 14, 13, 13])
        self.assertEqual([s['episodes_per_condition'] for s in test], [512]*4)
        stages[-1]['seed'] = 7001
        with self.assertRaises(ValueError):
            plan.assert_disjoint_splits(stages)

    def test_fitter_precedes_fixed_heldout_policy_tests(self):
        stages = plan.make_plan(Path('OUT'))
        fit = next(i for i,s in enumerate(stages) if s['kind']=='fit')
        for i,s in enumerate(stages):
            if s['kind'] in ('frontier', 'depth'):
                self.assertGreater(i, fit)
                self.assertIn('OUT/response_model.json', s['args'])
            if s['kind']=='frontier':
                self.assertEqual(s['args'][s['args'].index('--thetas')+1], '0.3')
                self.assertEqual(s['args'][s['args'].index('--c_preps')+1], '0.1')

    def test_plan_only_needs_system_python_and_does_not_start_experiments(self):
        r = subprocess.run([sys.executable, str(ROOT/'scripts/run_review_experiments.py'),
                            '--plan', '--mode', 'smoke'], capture_output=True, text=True, timeout=20)
        self.assertEqual(r.returncode, 0, r.stderr)
        stages = json.loads(r.stdout)
        self.assertEqual(len(stages), 6)
        self.assertFalse(any(s['split']=='test' for s in stages))

    def test_new_evidence_paths_and_binary_roundtrip(self):
        names = ['tests.json','plan.json','response_model.json','report.md','analysis.json',
                 'fixed_controller_comparison.png','depth_comparison.png','readiness.png',
                 'oracle_gate/audit.json','oracle_gate/paired.json','oracle_gate/episodes_b000_brace_launch1.csv',
                 'oracle_gate/traces/oracle_b000_brace_launch1.npz',
                 'frontier_gate/episodes_b000_c001.csv','frontier_gate/episodes.npz',
                 'frontier_gate/traces/frontier_b000_c001.npz',
                 'depth_gate_l0/calibration.json','depth_gate_l0/table.csv','depth_gate_l0/episodes.npz',
                 'depth_gate_l0/episodes/parametric_matched_belief_voi_p0p1_batch000.csv',
                 'depth_gate_l0/traces/depth_belief_wait_p0p1_batch000.csv']
        for name in names:
            self.assertTrue(publisher.allowed_tree_path('results/run1/'+name), name)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)/'episodes.npz'
            np.savez_compressed(target, fell=np.array([False, True]), time=np.array([1., 2.]))
            self.assertEqual(publisher.read_result(target), target.read_bytes())
            np.savez_compressed(target, invalid=np.array(['secret']))
            with self.assertRaises(publisher.PublicationError):
                publisher.read_result(target)
            np.savez_compressed(target, invalid=np.array([{'a':1}], dtype=object))
            with self.assertRaises(publisher.PublicationError):
                publisher.read_result(target)


class ReviewLifecycleTests(unittest.TestCase):
    def run_case(self, failure=None, stale_receipt=False):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        calls = []
        source = 'a'*40
        def child(command, env, log, timeout_seconds, **kwargs):
            calls.append(command)
            if failure and failure in command:
                raise TimeoutError('fixture stage budget')
            if 'scripts/fit_review_response.py' in command:
                p = Path(command[command.index('--out')+1])
                p.write_text(json.dumps(dict(schema_version='foresight.review_response.v1',
                        contact_conditioned=False,protocol=dict(timing_reference='reference_entry'),
                        n_cont=64,n_brace_lead=320)))
            if 'scripts/validate_review_stage.py' in command:
                p = Path(command[command.index('--stage-dir')+1]); p.mkdir(parents=True,exist_ok=True)
                (p/'audit.json').write_text('{"passed":true}')
            if any(str(c).endswith('publish_experiment_results.py') for c in command):
                p = Path(command[command.index('--run-dir')+1])
                rec = json.loads((p/'run.json').read_text())
                (p/'publication_receipt.json').write_text(json.dumps(dict(schema_version=1,status='verified',
                    verified=True,run_id='old' if stale_receipt else rec['run_id'],run_source=source,
                    commit='b'*40,run_exit_code=rec['exit_code'])))
            return 0
        with contextlib.ExitStack() as s:
            s.enter_context(mock.patch.object(runner,'ROOT',root))
            s.enter_context(mock.patch.object(runner,'run_child',side_effect=child))
            s.enter_context(mock.patch.object(runner,'choose_python',return_value=sys.executable))
            s.enter_context(mock.patch.object(runner.signal,'signal'))
            s.enter_context(mock.patch.object(runner.shutil,'which',return_value=None))
            s.enter_context(mock.patch.object(runner.subprocess,'run',return_value=subprocess.CompletedProcess([],0,source+'\n','')))
            s.enter_context(contextlib.redirect_stdout(io.StringIO()))
            s.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code=runner.main(['--mode','smoke'])
        last=json.loads((root/'logs/last_run.json').read_text())
        rec=json.loads((Path(last['run_dir'])/'run.json').read_text())
        return code,last,rec,calls

    def test_failed_gate_stops_later_gpu_stages_and_publishes_partial_evidence(self):
        code,last,rec,calls=self.run_case('scripts/eval_oracle_brace.py')
        self.assertEqual(code,124)
        self.assertEqual(rec['status'],'timed_out')
        self.assertEqual(len(rec['stages']),1)
        self.assertEqual(len(rec['planned_stages']),6)
        self.assertFalse(any('scripts/eval_decision_frontier.py' in c for c in calls))
        self.assertEqual(last['publication_status'],'verified')

    def test_success_executes_all_six_smoke_stages(self):
        code,last,rec,calls=self.run_case()
        self.assertEqual(code,0)
        self.assertEqual(rec['status'],'completed')
        self.assertEqual(len(rec['stages']),6)
        self.assertEqual(last['publication_status'],'verified')

    def test_stale_publication_receipt_cannot_authorize_stop(self):
        code,last,rec,calls=self.run_case(stale_receipt=True)
        self.assertEqual(code,2)
        self.assertEqual(rec['status'],'completed')
        self.assertEqual(last['publication_status'],'failed')


if __name__ == '__main__':
    unittest.main()
