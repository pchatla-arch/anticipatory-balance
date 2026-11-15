"""Stopping uses only current verified results; all pod commands are mocked."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('finish_pod_session', ROOT / 'scripts/finish_pod_session.py')
finish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(finish)


class FinishPodTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'logs').mkdir()
        self.run_dir = self.root / 'results/run_one'
        self.run_dir.mkdir(parents=True)
        self.env = {'FORESIGHT_AUTO_STOP': '1', 'FORESIGHT_INVOCATION_ID': 'invocation-one',
                    'RUNPOD_POD_ID': 'this-pod-123'}
        self.receipt = {'schema_version': 1, 'status': 'verified', 'verified': True,
                        'run_id': 'run_one', 'run_source': 'a' * 40,
                        'run_exit_code': 0, 'commit': 'b' * 40}
        self.run = {'run_id': 'run_one', 'source_commit': 'a' * 40,
                    'status': 'completed', 'exit_code': 0}
        self.latest = {'schema_version': 1, 'invocation_id': 'invocation-one',
                       'run_id': 'run_one', 'run_dir': str(self.run_dir),
                       'source_commit': 'a' * 40, 'status': 'completed',
                       'experiment_status': 'completed', 'experiment_exit_code': 0,
                       'publication_status': 'verified', 'publication_exit_code': 0,
                       'publication_receipt': dict(self.receipt)}
        self.calls = []
        self.help_style = 'modern'
        self.gpu_stdout = ''
        self.stop_code = 0
        self.save()

    def save(self):
        for path, value in ((self.root / 'logs/last_run.json', self.latest),
                            (self.run_dir / 'run.json', self.run),
                            (self.run_dir / 'publication_receipt.json', self.receipt)):
            path.write_text(json.dumps(value))

    def command(self, command, timeout):
        self.calls.append(command)
        self.assertLessEqual(timeout, 30)
        self.assertNotIn('terminate', command)
        self.assertNotIn('remove', command)
        if command[0].endswith('nvidia-smi'):
            return subprocess.CompletedProcess(command, 0, self.gpu_stdout, '')
        if command[-1] == '--help':
            modern = command[1:3] == ['pod', 'stop']
            if self.help_style == 'modern' and modern:
                return subprocess.CompletedProcess(command, 0, 'Usage: runpodctl pod stop <id>', '')
            if self.help_style == 'legacy' and not modern:
                return subprocess.CompletedProcess(command, 0, 'Usage: runpodctl stop pod [podId]', '')
            return subprocess.CompletedProcess(command, 1, '', 'unknown command')
        self.assertEqual(command[-1], self.env['RUNPOD_POD_ID'])
        return subprocess.CompletedProcess(command, self.stop_code, '', '')

    def execute(self, env=None, command=None, which=None):
        with mock.patch.object(finish, 'run_command', side_effect=command or self.command), \
                mock.patch.object(finish.shutil, 'which', side_effect=which or (lambda tool: '/mock/' + tool)), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = finish.finalize(self.root, self.env if env is None else env)
        return result, output.getvalue()

    def stop_calls(self):
        return [c for c in self.calls if c[0].endswith('runpodctl') and c[-1] != '--help']

    def test_current_verified_run_requests_modern_stop(self):
        code, output = self.execute()
        self.assertEqual(code, 0)
        self.assertEqual(self.stop_calls(), [['/mock/runpodctl', 'pod', 'stop', 'this-pod-123']])
        self.assertIn('Upload verified', output)
        self.assertIn('Stop request accepted', output)
        self.assertIn('not been independently observed', output)
        receipt = json.loads((self.root / 'logs/pod_stop_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'stop_requested')

    def test_legacy_cli_is_detected_using_help(self):
        self.help_style = 'legacy'
        self.assertEqual(self.execute()[0], 0)
        self.assertEqual(self.stop_calls(), [['/mock/runpodctl', 'stop', 'pod', 'this-pod-123']])

    def test_no_stop_override_and_no_authorization_do_not_read_records_or_run_commands(self):
        (self.root / 'logs/last_run.json').unlink()
        for env in ({**self.env, 'NO_STOP': '1'}, {**self.env, 'FORESIGHT_AUTO_STOP': '0'}, {}):
            with self.subTest(env=env):
                self.assertEqual(self.execute(env=env)[0], 0)
                self.assertEqual(self.calls, [])

    def test_stale_or_missing_invocation_is_rejected(self):
        for token in ('old-invocation', ''):
            with self.subTest(token=token):
                self.assertEqual(self.execute(env={**self.env, 'FORESIGHT_INVOCATION_ID': token})[0], 2)
                self.assertEqual(self.calls, [])

    def test_publisher_failure_is_rejected_even_with_valid_old_receipt(self):
        self.latest['publication_exit_code'] = 1
        self.save()
        self.assertEqual(self.execute()[0], 2)
        self.assertEqual(self.calls, [])

    def test_receipt_identity_source_exit_and_verified_flag_must_match(self):
        for key, value in (('run_id', 'other'), ('run_source', 'c' * 40),
                           ('run_exit_code', 1), ('verified', False), ('commit', 'not-a-hash')):
            with self.subTest(key=key):
                prior = self.receipt[key]
                self.receipt[key] = value
                self.latest['publication_receipt'] = dict(self.receipt)
                self.save()
                self.assertEqual(self.execute()[0], 2)
                self.assertEqual(self.calls, [])
                self.receipt[key] = prior

    def test_receipt_file_must_equal_receipt_from_current_publisher(self):
        self.receipt['commit'] = 'c' * 40
        self.save()
        self.assertEqual(self.execute()[0], 2)
        self.assertEqual(self.calls, [])

    def test_frozen_run_must_match_last_run_record(self):
        self.run['source_commit'] = 'c' * 40
        self.save()
        self.assertEqual(self.execute()[0], 2)
        self.assertEqual(self.calls, [])

    def test_failed_and_timed_out_runs_can_stop_after_verified_log_upload(self):
        for state, exit_code in (('failed', 1), ('timed_out', 124)):
            with self.subTest(state=state):
                self.calls.clear()
                self.run.update(status=state, exit_code=exit_code)
                self.receipt['run_exit_code'] = exit_code
                self.latest.update(status='failed', experiment_status=state,
                                   experiment_exit_code=exit_code, publication_receipt=dict(self.receipt))
                self.save()
                self.assertEqual(self.execute()[0], 0)
                self.assertEqual(len(self.stop_calls()), 1)

    def test_running_and_interrupted_states_are_not_stopped(self):
        for state in ('running', 'interrupted'):
            with self.subTest(state=state):
                self.latest['experiment_status'] = state
                self.save()
                self.assertEqual(self.execute()[0], 2)
                self.assertEqual(self.calls, [])

    def test_active_or_unrecognizable_gpu_process_list_prevents_stop(self):
        for output in ('12345\n', '[Not Supported]\n'):
            with self.subTest(output=output):
                self.calls.clear()
                self.gpu_stdout = output
                self.assertEqual(self.execute()[0], 2)
                self.assertEqual(self.stop_calls(), [])

    def test_missing_cli_gpu_checker_or_pod_id_prevents_stop(self):
        for missing in ('runpodctl', 'nvidia-smi'):
            with self.subTest(missing=missing):
                self.calls.clear()
                code, output = self.execute(which=lambda tool: None if tool == missing else '/mock/' + tool)
                self.assertEqual(code, 2)
                self.assertIn('Upload verified', output)
                self.assertEqual(self.stop_calls(), [])
        self.assertEqual(self.execute(env={**self.env, 'RUNPOD_POD_ID': ''})[0], 2)
        self.assertEqual(self.stop_calls(), [])

    def test_unrecognized_cli_never_tries_a_stop_or_terminate(self):
        self.help_style = 'unsupported'
        self.assertEqual(self.execute()[0], 2)
        self.assertEqual(self.stop_calls(), [])

    def test_failed_stop_request_is_not_reported_as_success(self):
        self.stop_code = 1
        code, output = self.execute()
        self.assertEqual(code, 2)
        self.assertNotIn('Stop request accepted', output)
        receipt = json.loads((self.root / 'logs/pod_stop_receipt.json').read_text())
        self.assertEqual(receipt['status'], 'request_failed')

    def test_subprocess_timeout_is_bounded_and_sanitized(self):
        with mock.patch.object(finish.subprocess, 'run', side_effect=subprocess.TimeoutExpired('secret', 1)) as run:
            with self.assertRaises(finish.StopBlocked) as raised:
                finish.run_command(['fake'], timeout=10)
        self.assertNotIn('secret', str(raised.exception))
        self.assertEqual(run.call_args.kwargs['timeout'], 10)


if __name__ == '__main__':
    unittest.main()
