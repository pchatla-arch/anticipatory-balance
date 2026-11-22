"""Offline coverage of adaptive launch, budget accounting, and publication state."""
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
import adaptive_gate_pod as worker
import publish_experiment_results as publisher
import test_session_h_launchers as launch_fixtures


class AdaptiveLauncherTests(unittest.TestCase):
    # Reuse only the H tests' isolated fake executables; no inherited test cases.
    make_auth_fixture = launch_fixtures.LauncherTests.make_auth_fixture
    reset_auth_calls = launch_fixtures.LauncherTests.reset_auth_calls

    def setUp(self):
        launch_fixtures.LauncherTests.setUp(self)
        self.env = {key: value for key, value in self.env.items()
                    if not key.startswith('ADAPTIVE_')}

    def invoke(self, args=(), env=None, script='start_adaptive_gate.sh'):
        return subprocess.run(['bash', str(ROOT / 'scripts' / script), *args],
                              env={**self.env, **(env or {})}, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=15)

    def clock(self):
        started = int(time.time())
        return {'ADAPTIVE_STARTED_EPOCH': str(started),
                'ADAPTIVE_DEADLINE_EPOCH': str(started + 7200)}

    def test_help_requires_neither_gpu_nor_download(self):
        for name in ('start_adaptive_gate.sh', 'start_adaptive_gate_pod.sh'):
            result = self.invoke(['--help'], script=name)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('120', result.stdout)
            self.assertIn('RunPod', result.stdout)
            self.assertFalse(self.calls.exists())
            result = subprocess.run(['bash', '-n', str(ROOT / 'scripts' / name)],
                                    capture_output=True, text=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_wrapper_sets_original_deadline_and_isolated_source(self):
        self.download.write_text('''#!/usr/bin/env bash
set -euo pipefail
python3 - <<'PY'
import json, os
print(json.dumps({k:v for k,v in os.environ.items()
                  if k.startswith(('FORESIGHT_', 'ADAPTIVE_')) or k == 'NO_STOP'}))
PY
''')
        for args, profile, minutes in (([], 'gate', 120), (['smoke'], 'smoke', 30)):
            result = self.invoke(args, {'NO_STOP': '1'})
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout.splitlines()[-1])
            self.assertEqual(data['ADAPTIVE_PROFILE'], profile)
            self.assertEqual(int(data['ADAPTIVE_DEADLINE_EPOCH']) - int(data['ADAPTIVE_STARTED_EPOCH']), minutes * 60)
            self.assertEqual(data['FORESIGHT_REPO'], '/workspace/adaptive-execution-gate')
            self.assertEqual(data['FORESIGHT_CODE_REF'], 'portfolio/adaptive-execution-gate')
            self.assertEqual(data['NO_STOP'], '1')

    def test_invalid_budget_and_revision_fail_before_download(self):
        for values in ({'ADAPTIVE_MAX_MINUTES': '121'}, {'ADAPTIVE_MAX_MINUTES': '14'},
                       {'ADAPTIVE_MAX_MINUTES': 'nan'}, {'ADAPTIVE_MAX_MINUTES': '1;false'},
                       {'FORESIGHT_CODE_REF': '../main'}):
            result = self.invoke(env=values)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.calls.exists())

    def test_pod_entry_requires_outer_deadline_before_gpu_work(self):
        result = self.invoke(script='start_adaptive_gate_pod.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('original attempt deadline', result.stdout)
        self.assertFalse(self.calls.exists())

    def test_permission_and_exact_deadline_reach_only_new_tmux_worker(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        for can_push, expected in (('ok', True), ('no', False)):
            self.reset_auth_calls()
            clock = self.clock()
            result = self.invoke(env={**clock, 'MOCK_GIT_PUSH': can_push}, script='start_adaptive_gate_pod.sh')
            self.assertEqual(self.started.exists(), expected, result.stdout + result.stderr)
            self.assertEqual(result.returncode == 0, expected, result.stdout + result.stderr)
            if expected:
                command = self.started.read_text()
                self.assertIn('-s adaptive-gate', command)
                self.assertIn('scripts/start_adaptive_gate_pod.sh --worker', command)
                self.assertIn('ADAPTIVE_DEADLINE_EPOCH=' + clock['ADAPTIVE_DEADLINE_EPOCH'], command)
                self.assertNotIn('scripts/start_session_h_pod.sh --worker', command)

    def test_worker_rechecks_account_before_python(self):
        repo = self.make_auth_fixture()
        launched = self.root / 'python_started.txt'
        (repo / 'scripts/adaptive_gate_pod.py').write_text(
            'import os\nfrom pathlib import Path\nPath(os.environ["MOCK_PYTHON_STARTED"]).touch()\n')
        for push, expected in (('ok', True), ('no', False)):
            launched.unlink(missing_ok=True)
            result = self.invoke(['--worker'], env={**self.clock(), 'MOCK_GIT_PUSH': push,
                                 'MOCK_PYTHON_STARTED': str(launched)}, script='start_adaptive_gate_pod.sh')
            self.assertEqual(launched.exists(), expected, result.stdout + result.stderr)
            self.assertEqual(result.returncode == 0, expected)


    GOOD = 'good_token_0123456789abcdef'

    def test_saved_git_credentials_need_no_token_and_no_gh(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        git_calls = self.root / 'git_calls.txt'
        result = self.invoke(env={**self.clock(), 'MOCK_GIT_PUSH': 'ok', 'MOCK_GIT_CALLS': str(git_calls)},
                             script='start_adaptive_gate_pod.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.started.exists())
        self.assertIn('saved Git credentials can push', result.stdout)
        self.assertFalse(self.gh_calls.exists(), 'the GitHub CLI must not be used')
        self.assertIn('origin HEAD:refs/heads/foresight/credential-probe', git_calls.read_text())

    def test_access_token_from_environment_is_saved_privately_and_never_printed(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        result = self.invoke(env={**self.clock(), 'MOCK_GIT_PUSH': 'token',
                                  'FORESIGHT_GITHUB_TOKEN': self.GOOD},
                             script='start_adaptive_gate_pod.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.started.exists())
        self.assertIn('access token can push', result.stdout)
        self.assertNotIn(self.GOOD, result.stdout + result.stderr)
        self.assertNotIn(self.GOOD, self.started.read_text())
        saved = self.root / '.foresight-git-credentials'
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertEqual(saved.read_text(), f'https://pchatla-arch:{self.GOOD}@github.com\n')
        self.assertNotIn('login', self.gh_calls.read_text() if self.gh_calls.exists() else '')

    def test_existing_gh_login_is_reused_without_token_prompt(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        result = self.invoke(env={**self.clock(), 'MOCK_GIT_PUSH': 'gh', 'MOCK_GH_TOKEN': 'gho_secretvalue0123456789'},
                             script='start_adaptive_gate_pod.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertTrue(self.started.exists())
        self.assertIn('existing GitHub CLI login can push', result.stdout)
        self.assertNotIn('gho_secretvalue0123456789', result.stdout + result.stderr + self.started.read_text())
        self.assertFalse((self.root / '.foresight-git-credentials').exists())
        self.assertNotIn('login', self.gh_calls.read_text())

    def test_rejected_token_is_deleted_and_nothing_starts(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        result = self.invoke(env={**self.clock(), 'MOCK_GIT_PUSH': 'token',
                                  'FORESIGHT_GITHUB_TOKEN': 'wrong_token_0123456789abcdef'},
                             script='start_adaptive_gate_pod.sh')
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.started.exists())
        self.assertIn('refused that token', result.stderr)
        self.assertFalse((self.root / '.foresight-git-credentials').exists())
        self.assertNotIn('wrong_token_0123456789abcdef', result.stdout + result.stderr)

    def test_malformed_token_or_no_terminal_fails_before_saving(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        for token, message in (('not a token!', 'does not look like a GitHub token'),
                               ('', 'no terminal is available')):
            with self.subTest(token=token):
                result = self.invoke(env={**self.clock(), 'MOCK_GIT_PUSH': 'token',
                                          'FORESIGHT_GITHUB_TOKEN': token},
                                     script='start_adaptive_gate_pod.sh')
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(message, result.stderr)
                self.assertFalse(self.started.exists())
                self.assertFalse((self.root / '.foresight-git-credentials').exists())

    def test_saved_token_file_is_reused_on_the_next_run(self):
        repo = self.make_auth_fixture()
        (repo / 'scripts/adaptive_gate_pod.py').touch()
        saved = self.root / '.foresight-git-credentials'
        saved.write_text(f'https://pchatla-arch:{self.GOOD}@github.com\n')
        result = self.invoke(env={**self.clock(), 'MOCK_GIT_PUSH': 'token'}, script='start_adaptive_gate_pod.sh')
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('saved Git credentials can push', result.stdout)


class AdaptiveWorkerTests(unittest.TestCase):
    def test_deadline_does_not_restart_after_setup(self):
        started, deadline, remaining = worker.parse_deadline(
            {'ADAPTIVE_STARTED_EPOCH': '1000', 'ADAPTIVE_DEADLINE_EPOCH': '8200'}, now=2800)
        self.assertEqual((started, deadline, remaining), (1000, 8200, 5400))
        with mock.patch.object(worker.time, 'monotonic', return_value=2800):
            self.assertEqual(worker.remaining_timeout(8200, 7200, reserve=600), 4800)
            with self.assertRaises(TimeoutError):
                worker.remaining_timeout(3200, 7200, reserve=600)

    def test_invalid_or_future_deadlines_rejected(self):
        for env in ({}, {'ADAPTIVE_STARTED_EPOCH': '1000', 'ADAPTIVE_DEADLINE_EPOCH': '9000'},
                    {'ADAPTIVE_STARTED_EPOCH': '2000', 'ADAPTIVE_DEADLINE_EPOCH': '9200'}):
            with self.assertRaises(ValueError):
                worker.parse_deadline(env, now=1000)

    def test_new_evidence_names_are_published(self):
        for name in ('run.json', 'summary.json', 'episodes.csv', 'report.md',
                     'plan.json', 'config.json', 'preflight.json', 'setup/preflight.json'):
            self.assertTrue(publisher.allowed_tree_path('results/adaptive_gate_1/' + name), name)

    def test_nonfinite_or_missing_scientific_summary_is_not_success(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for text in ('{"scientific_gate_status":"pass","metric":NaN}', '{}'):
                (output / 'summary.json').write_text(text)
                with self.assertRaises(ValueError):
                    worker.validate_outputs(output)

    def run_fixture(self, *, timeout=False, scientific='fail', upload_ok=True, determinism=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        calls = []

        def child(command, env, log, budget, **kwargs):
            calls.append((command, budget))
            if 'scripts/preflight_session_h.py' in command:
                Path(command[-1]).write_text('{}\n')
            if 'scripts/eval_adaptive_gate.py' in command:
                output = Path(command[command.index('--out') + 1])
                if timeout:
                    raise TimeoutError('fixture experiment deadline')
                (output / 'summary.json').write_text(json.dumps({'scientific_gate_status': scientific}))
                (output / 'episodes.csv').write_text('scenario_id,fell\n1,0\n')
                (output / 'report.md').write_text('Fixture result.\n')
            if 'scripts/publish_experiment_results.py' in command:
                if not upload_ok:
                    return 1
                output = Path(command[command.index('--run-dir') + 1])
                record = json.loads((output / 'run.json').read_text())
                (output / 'publication_receipt.json').write_text(json.dumps({
                    'schema_version': 1, 'status': 'verified', 'verified': True,
                    'run_id': record['run_id'], 'run_source': record['source_commit'],
                    'run_exit_code': record['exit_code'], 'commit': 'b' * 40}))
            return 0

        old_cwd = Path.cwd()
        self.addCleanup(os.chdir, old_cwd)
        now = int(time.time())
        env = {'ADAPTIVE_STARTED_EPOCH': str(now - 1800),
               'ADAPTIVE_DEADLINE_EPOCH': str(now + 5400),
               'FORESIGHT_SETUP_ROOT': str(root)}
        if determinism is not None:
            env['ADAPTIVE_DETERMINISTIC'] = determinism
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.dict(os.environ, env))
            stack.enter_context(mock.patch.object(worker, 'ROOT', root))
            stack.enter_context(mock.patch.object(worker, 'run_child', child))
            stack.enter_context(mock.patch.object(worker, 'require_idle_gpu', return_value=None))
            stack.enter_context(mock.patch.object(worker, 'setup_command', return_value=(['mock_setup'], 1500, 'fixture setup')))
            stack.enter_context(mock.patch.object(worker.signal, 'signal', return_value=None))
            stack.enter_context(mock.patch.object(worker.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'a' * 40 + '\n', '')))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            code = worker.main(['--profile', 'gate'])
        os.chdir(old_cwd)
        latest = json.loads((root / 'logs/last_run.json').read_text())
        record = json.loads((Path(latest['run_dir']) / 'run.json').read_text())
        return code, latest, record, calls

    def test_scientific_failure_is_completed_pipeline_with_verified_evidence(self):
        code, latest, record, calls = self.run_fixture(scientific='fail')
        self.assertEqual(code, 0)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(record['scientific_gate_status'], 'fail')
        self.assertEqual(latest['publication_status'], 'verified')
        evaluation = [entry for entry in calls if 'scripts/eval_adaptive_gate.py' in entry[0]]
        self.assertEqual(len(evaluation), 1)
        self.assertLessEqual(evaluation[0][1], 4800)
        self.assertEqual(evaluation[0][0][-2:], ['--deterministic', 'auto'], 'pod runs request deterministic execution by default')

    def test_determinism_escape_hatch_reaches_the_evaluator_and_rejects_typos(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ADAPTIVE_DETERMINISTIC', None)
            _, _, _, calls = self.run_fixture(determinism='off')
        evaluation = [entry[0] for entry in calls if 'scripts/eval_adaptive_gate.py' in entry[0]]
        self.assertEqual(evaluation[0][-2:], ['--deterministic', 'off'])
        code, _, record, calls = self.run_fixture(determinism='maybe')
        self.assertEqual(code, 1)
        self.assertEqual(record['status'], 'failed')
        self.assertFalse([entry for entry in calls if 'scripts/eval_adaptive_gate.py' in entry[0]], 'no GPU work on a bad setting')

    def test_timeout_still_publishes_partial_evidence(self):
        code, latest, record, _ = self.run_fixture(timeout=True)
        self.assertEqual(code, 124)
        self.assertEqual(record['status'], 'timed_out')
        self.assertNotIn('scientific_gate_status', record)
        self.assertEqual(latest['publication_status'], 'verified')

    def test_failed_upload_cannot_produce_verified_stop_receipt(self):
        code, latest, record, _ = self.run_fixture(upload_ok=False)
        self.assertEqual(code, 2)
        self.assertEqual(record['status'], 'completed')
        self.assertEqual(latest['publication_status'], 'failed')
        self.assertIsNone(latest['publication_receipt'])
        self.assertEqual(latest['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
