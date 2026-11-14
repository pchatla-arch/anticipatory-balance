"""Mock only external work; exercise the real runner's lifecycle and finalization."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/run_experiments_v2.py'
spec = importlib.util.spec_from_file_location('runner_lifecycle_subject', SOURCE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
SOURCE_SHA = 'a' * 40
PUBLICATION_SHA = 'b' * 40


class LifecycleTests(unittest.TestCase):
    def run_case(self, *, extra=(), git_error=None, python_error=None,
                 experiment_error=None, archive_error=None, publication_code=0,
                 stale_receipt=False, wrong_exit_code=False):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        calls = []
        frozen = []

        def child(command, env, log, timeout_seconds, *, check=True):
            calls.append((command, timeout_seconds, check))
            if any(str(v).endswith('publish_experiment_results.py') for v in command):
                run_dir = Path(command[command.index('--run-dir') + 1])
                frozen.append((run_dir / 'run.json').read_bytes())
                record = json.loads(frozen[-1])
                if publication_code == 0:
                    receipt = dict(schema_version=1, status='verified', verified=True,
                                   run_id='old_run' if stale_receipt else record['run_id'],
                                   run_source=record['source_commit'], commit=PUBLICATION_SHA,
                                   run_exit_code=99 if wrong_exit_code else record['exit_code'])
                    (run_dir / 'publication_receipt.json').write_text(json.dumps(receipt))
                return publication_code
            if 'scripts/eval_decision_frontier.py' in command:
                if experiment_error:
                    raise experiment_error
                Path(command[command.index('--out') + 1]).mkdir()
            if 'foresight.eval.release_audit' in command:
                out = Path(command[command.index('--run-dir') + 1])
                (out / 'release_audit.json').write_text('{"status":"passed"}')
            return 0

        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(patch.object(runner, 'ROOT', root))
            stack.enter_context(patch.dict(runner.os.environ, {'FORESIGHT_INVOCATION_ID': 'fixture-invocation'}))
            stack.enter_context(patch.object(runner.signal, 'signal'))
            stack.enter_context(patch.object(runner.shutil, 'which', return_value=None))
            stack.enter_context(patch.object(runner, 'choose_python', return_value=sys.executable,
                                            side_effect=python_error))
            stack.enter_context(patch.object(runner.subprocess, 'run',
                                            return_value=subprocess.CompletedProcess([], 0, SOURCE_SHA + '\n', ''),
                                            side_effect=git_error))
            stack.enter_context(patch.object(runner, 'run_child', side_effect=child))
            stack.enter_context(patch.object(runner, 'validate_stage', return_value={'passed': True}))
            if archive_error:
                stack.enter_context(patch.object(runner.shutil, 'make_archive', side_effect=archive_error))
            stack.enter_context(contextlib.redirect_stdout(stdout))
            stack.enter_context(contextlib.redirect_stderr(stderr))
            code = runner.main(['--mode', 'release', *extra])
        latest = json.loads((root / 'logs/last_run.json').read_text())
        run_dir = Path(latest['run_dir'])
        return code, latest, json.loads((run_dir / 'run.json').read_text()), run_dir, calls, frozen, stdout.getvalue()

    def test_larger_budget_is_not_silently_capped_at_1000_seconds(self):
        code, latest, record, run_dir, calls, frozen, output = self.run_case(extra=['--max-minutes', '120'])
        timeout = next(t for cmd, t, _ in calls if 'scripts/eval_decision_frontier.py' in cmd)
        self.assertGreater(timeout, 1000)
        self.assertLessEqual(timeout, 7200)
        self.assertEqual(code, 0)
        self.assertEqual(latest['experiment_status'], 'completed')
        self.assertEqual(latest['publication_status'], 'verified')
        self.assertEqual(latest['overall_exit_code'], 0)
        self.assertEqual(latest['invocation_id'], 'fixture-invocation')
        self.assertEqual((run_dir / 'run.json').read_bytes(), frozen[0])
        self.assertIn('Experiment status: completed (exit 0)', output)
        self.assertIn('Publication status: verified (exit 0)', output)

    def test_git_failure_is_saved_with_traceback_before_preflight(self):
        code, latest, record, run_dir, calls, frozen, _ = self.run_case(
            extra=['--no-publish'], git_error=subprocess.CalledProcessError(128, ['git', 'rev-parse', 'HEAD']))
        self.assertEqual(code, 1)
        self.assertEqual(record['status'], 'failed')
        self.assertIn('Traceback (most recent call last)', (run_dir / 'console.log').read_text())
        self.assertEqual(calls, [])
        self.assertEqual(latest['publication_status'], 'not_requested')

    def test_missing_python_is_logged_and_failed_run_still_publishes(self):
        code, latest, record, run_dir, calls, frozen, _ = self.run_case(python_error=RuntimeError('fixture venv absent'))
        self.assertEqual(code, 1)
        self.assertEqual(latest['publication_status'], 'verified')
        self.assertIn('fixture venv absent', (run_dir / 'console.log').read_text())
        self.assertEqual(record['stages'], [])

    def test_archive_failure_does_not_skip_publication(self):
        code, latest, record, run_dir, calls, frozen, _ = self.run_case(archive_error=OSError('fixture disk error'))
        self.assertEqual(code, 1)
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(latest['archive_status'], 'failed')
        self.assertEqual(latest['publication_status'], 'verified')
        self.assertEqual(len(frozen), 1)
        self.assertIn('fixture disk error', (run_dir / 'console.log').read_text())

    def test_publisher_failure_does_not_relabel_successful_experiment(self):
        code, latest, record, run_dir, calls, frozen, output = self.run_case(publication_code=7)
        self.assertEqual(code, 2)
        self.assertEqual(record['exit_code'], 0)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(latest['publication_status'], 'failed')
        self.assertEqual(latest['publication_exit_code'], 7)
        self.assertEqual(latest['overall_exit_code'], 2)
        self.assertEqual((run_dir / 'run.json').read_bytes(), frozen[0])
        self.assertIn('Experiment status: completed (exit 0)', output)
        self.assertIn('Publication status: failed (exit 7)', output)

    def test_zero_publisher_exit_requires_receipt_for_this_run(self):
        code, latest, record, run_dir, calls, frozen, _ = self.run_case(stale_receipt=True)
        self.assertEqual(code, 2)
        self.assertEqual(latest['publication_status'], 'failed')
        self.assertEqual(latest['publication_exit_code'], 0)
        self.assertIsNone(latest['publication_receipt'])

    def test_timeout_remains_timeout_after_successful_publication(self):
        code, latest, record, run_dir, calls, frozen, _ = self.run_case(experiment_error=TimeoutError('fixture budget reached'))
        self.assertEqual(code, 124)
        self.assertEqual(record['status'], 'timed_out')
        self.assertEqual(record['stages'][0]['status'], 'timed_out')
        self.assertEqual(latest['experiment_exit_code'], 124)
        self.assertEqual(latest['publication_status'], 'verified')
        self.assertEqual(latest['overall_exit_code'], 124)

    def test_receipt_must_match_actual_experiment_exit_code(self):
        code, latest, record, run_dir, calls, frozen, _ = self.run_case(wrong_exit_code=True)
        self.assertEqual(code, 2)
        self.assertEqual(latest['publication_status'], 'failed')
        self.assertEqual(record['exit_code'], 0)
        self.assertIsNone(latest['publication_receipt'])


if __name__ == '__main__':
    unittest.main()
