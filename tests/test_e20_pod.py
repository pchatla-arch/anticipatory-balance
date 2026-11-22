"""Offline launch and publication regressions; these tests do not use a GPU."""
from contextlib import ExitStack
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import publish_experiment_results as publisher
import run_e20_pod as worker


def git(path, *args):
    return subprocess.check_output(['git', '-C', str(path), *args], text=True).strip()


class PodTests(unittest.TestCase):
    def test_empty_workspace_plan(self):
        with tempfile.TemporaryDirectory() as empty:
            env = dict(os.environ, FORESIGHT_SETUP_ROOT=empty)
            result = subprocess.run(['bash', str(ROOT / 'scripts/start_foresight_e20.sh'), '--plan'],
                                    cwd=empty, env=env, text=True, capture_output=True, check=True)
            self.assertIn('4 x 512', result.stdout)
            self.assertEqual(list(Path(empty).iterdir()), [])

    def test_empty_pod_clones_then_blocks_on_busy_gpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            volume, origin, bin_dir = [base / child for child in ('workspace', 'remote', 'bin')]
            for directory in (volume, origin, bin_dir):
                directory.mkdir()
            git(origin, 'init', '-q')
            (origin / 'scripts').mkdir()
            (origin / 'scripts/run_e20_pod.py').write_text("raise AssertionError('GPU must stay idle')")
            git(origin, 'add', 'scripts')
            git(origin, '-c', 'user.email=test@example.invalid', '-c', 'user.name=Test',
                'commit', '-qm', 'offline source')
            source = git(origin, 'rev-parse', 'HEAD')
            nvidia = bin_dir / 'nvidia-smi'
            nvidia.write_text('#!/bin/sh\necho 1111\n')
            nvidia.chmod(0o755)
            canonical = 'https://github.com/pchatla-arch/anticipatory-balance.git'
            # Route only this public clone to the local fixture; do not contact GitHub.
            env = dict(os.environ, FORESIGHT_SETUP_ROOT=str(volume), FORESIGHT_CODE_REF=source,
                       PATH=str(bin_dir) + os.pathsep + os.environ['PATH'], GIT_CONFIG_COUNT='1',
                       GIT_CONFIG_KEY_0=f'url.{origin.as_uri()}.insteadOf', GIT_CONFIG_VALUE_0=canonical)
            result = subprocess.run(['bash', str(ROOT / 'scripts/start_foresight_e20.sh'), '--run'],
                                    env=env, text=True, capture_output=True, timeout=15)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('GPU already has a compute process', result.stderr)
            self.assertEqual(git(volume / 'foresight-e20', 'rev-parse', 'HEAD'), source)
            self.assertFalse((volume / 'foresight-e20/logs/e20_attempt.json').exists())

    def test_publish_and_retry_on_local_git_remote(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary) / 'isolated'
            root.mkdir()
            (root / 'logs').mkdir()
            (root / 'results').mkdir()
            (root / 'scripts').mkdir()
            git(root, 'init', '-q')
            git(root, '-c', 'user.email=test@example.invalid', '-c', 'user.name=Test',
                'commit', '-q', '--allow-empty', '-m', 'offline fixture')
            source = git(root, 'rev-parse', 'HEAD')
            bare = Path(temporary) / 'mirror.git'
            git(root, 'init', '--bare', '-q', str(bare))
            git(root, 'remote', 'add', 'origin', str(bare))
            stack.enter_context(patch.multiple(worker, ROOT=root, ATTEMPT=root / 'logs/e20_attempt.json',
                                               LOG=root / 'logs/e20_stage.log'))
            stack.enter_context(patch.object(publisher, 'SOURCE_DIRECTORY', root))
            # Only the permitted remote identity is substituted; the real publisher does all
            # Git object creation, push, remote readback, receipt creation and verification.
            stack.enter_context(patch.object(publisher.Git, 'validate_origin', lambda self: None))
            stack.enter_context(patch.dict(os.environ, FORESIGHT_SETUP_ROOT=temporary, FORESIGHT_AUTO_STOP='0'))
            calls = []

            def fake_runtime(command, env, log, cap, **kwargs):
                calls.append(command[1])
                if command[1] == 'scripts/run_foresight_matched_duration.py':
                    out = Path(command[-1])
                    out.mkdir()
                    for name in ['protocol.json', 'frozen.json', 'verdict.json']:
                        (out / name).write_text('{"fixture": true}')
                    (out / 'confirmation/episodes').mkdir(parents=True)
                    (out / 'confirmation/episodes/batch003_condition002.csv').write_text('id\n1\n')
                if command[1] == 'scripts/publish_experiment_results.py':
                    publisher.publish(Path(command[3]), Path(command[5]), int(command[7]))
                return 0

            stack.enter_context(patch.object(worker, 'run_child', fake_runtime))
            attempt = worker.execute(source)
            self.assertEqual(calls.count('scripts/run_foresight_matched_duration.py'), 1)
            self.assertEqual(worker.publish(attempt, source), 0)
            tree = git(bare, 'ls-tree', '-r', '--name-only', 'foresight/experiment-results')
            for suffix in ['protocol.json', 'frozen.json', 'verdict.json',
                           'confirmation/episodes/batch003_condition002.csv']:
                self.assertIn(f"results/{attempt['run_id']}/{suffix}", tree)
            self.assertEqual(worker.publish(json.loads(worker.ATTEMPT.read_text()), source), 0)
            self.assertEqual(calls.count('scripts/run_foresight_matched_duration.py'), 1)
            latest = json.loads((root / 'logs/last_run.json').read_text())
            self.assertEqual(latest['publication_receipt']['run_id'], attempt['run_id'])


if __name__ == '__main__':
    unittest.main()
