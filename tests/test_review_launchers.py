"""Exercise public/worker dispatch without network access, GPUs, or installs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location(
    'review_pod_session_for_tests', ROOT / 'scripts/review_pod_session.py')
SESSION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SESSION)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.download = self.root / 'downloaded_entry.sh'
        self.calls = self.root / 'curl_args.txt'
        self.download.write_text('''#!/usr/bin/env bash
set -euo pipefail
[[ -f "${BASH_SOURCE[0]}" && ! -p "${BASH_SOURCE[0]}" ]]
python3 - <<'PY'
import json, os
print(json.dumps({k:v for k,v in os.environ.items()
                  if k.startswith('FORESIGHT_') or k == 'NO_STOP'}))
PY
''')
        curl = self.bin / 'curl'
        curl.write_text('''#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "$@" > "$MOCK_CURL_ARGS"
while (( $# )); do
  if [[ "$1" == --output ]]; then output="$2"; shift 2; else shift; fi
done
cp -- "$MOCK_ENTRY" "$output"
''')
        curl.chmod(0o755)
        self.env = {k:v for k,v in os.environ.items()
                    if not k.startswith('FORESIGHT_') and k != 'NO_STOP'}
        self.env.update(PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        MOCK_ENTRY=str(self.download), MOCK_CURL_ARGS=str(self.calls))

    def invoke(self, args=(), env=None, script='start_review.sh'):
        return subprocess.run(['bash', str(ROOT / 'scripts' / script), *args],
                              env={**self.env, **(env or {})},
                              capture_output=True, text=True, timeout=10)

    def dispatch(self, args=(), env=None):
        result = self.invoke(args, env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_help_and_shell_syntax_require_no_network_or_gpu(self):
        for script in ('start_review.sh', 'start_review_pod.sh'):
            with self.subTest(script=script):
                result = self.invoke(['--help'], script=script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('90', result.stdout)
                self.assertIn('30', result.stdout)
                self.assertIn('PPO training', result.stdout)
                self.assertIn('RunPod', result.stdout)
                self.assertFalse(self.calls.exists())
                check = subprocess.run(['bash', '-n', str(ROOT/'scripts'/script)],
                                       capture_output=True, text=True, timeout=5)
                self.assertEqual(check.returncode, 0, check.stderr)

    def test_public_defaults_aliases_and_small_budget(self):
        for args, mode, minutes in (([], 'full', '90'), (['review'], 'full', '90'),
                                    (['full'], 'full', '90'), (['smoke'], 'smoke', '30')):
            with self.subTest(args=args):
                data = self.dispatch(args)
                self.assertEqual(data['FORESIGHT_MODE'], mode)
                self.assertEqual(data['FORESIGHT_MAX_MINUTES'], minutes)
                self.assertEqual(data['FORESIGHT_REPO'], '/workspace/foresight-review-runner')
                self.assertEqual(data['FORESIGHT_AUTO_STOP'], '1')
                self.assertIn('/foresight/review-rerun-v1/scripts/start_review_pod.sh',
                              self.calls.read_text())

    def test_explicit_paths_budget_pinned_revision_and_no_stop_survive_dispatch(self):
        overrides = {'FORESIGHT_SETUP_ROOT': str(self.root / 'with space'),
                     'FORESIGHT_VENV': str(self.root / 'python env'),
                     'FORESIGHT_CODE_REF': 'a' * 40,
                     'FORESIGHT_MAX_MINUTES': '45.5',
                     'FORESIGHT_AUTO_STOP': '0', 'NO_STOP': '1'}
        data = self.dispatch(['review'], overrides)
        for name, value in overrides.items():
            self.assertEqual(data[name], value)
        self.assertEqual(data['FORESIGHT_REPO'], str(self.root / 'with space/foresight-review-runner'))
        self.assertIn('/' + 'a' * 40 + '/scripts/start_review_pod.sh', self.calls.read_text())

    def test_invalid_dispatch_is_rejected_before_download(self):
        cases = [(['unknown'], {}), (['review', 'extra'], {}),
                 (['review'], {'FORESIGHT_CODE_REF': '../main'}),
                 (['review'], {'FORESIGHT_REPO': 'relative'}),
                 (['review'], {'FORESIGHT_AUTO_STOP': 'yes'})]
        cases += [(['review'], {'FORESIGHT_MAX_MINUTES': value})
                  for value in ('0', '0.5', '-1', 'nan', 'inf', '1; echo bad')]
        for args, env in cases:
            with self.subTest(args=args, env=env):
                result = self.invoke(args, env)
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(self.calls.exists())

    def test_invalid_download_is_never_executed(self):
        for content in ('<html>missing</html>\n', '#!/usr/bin/env bash\nif then\n'):
            with self.subTest(content=content):
                self.download.write_text(content)
                result = self.invoke(['review'])
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue('launcher' in result.stderr)

    def make_publish_fixture(self, status='completed'):
        repo = self.root / 'publish repo'
        (repo / 'scripts').mkdir(parents=True)
        run = repo / 'results/run_one'
        run.mkdir(parents=True)
        (run / 'run.json').write_text(json.dumps({'status': status, 'exit_code': 0}))
        (repo / 'scripts/publish_experiment_results.py').write_text(
            'import json,sys\nprint(json.dumps(sys.argv[1:]))\n')
        return repo, run

    def test_retry_publish_only_dispatches_completed_local_run(self):
        repo, run = self.make_publish_fixture()
        result = self.invoke(['retry-publish', 'run_one'], {'FORESIGHT_REPO': str(repo)})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout),
                         ['--run-dir', str(run), '--log', str(run / 'console.log'), '--exit-code', '0'])
        self.assertFalse(self.calls.exists())

    def test_retry_publish_rejects_running_and_traversal(self):
        repo, _ = self.make_publish_fixture(status='running')
        for run in ('run_one', '../run_one'):
            result = self.invoke(['retry-publish', run], {'FORESIGHT_REPO': str(repo)})
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(self.calls.exists())

    def test_worker_modes_reach_review_session_with_matching_budgets(self):
        repo = self.root / 'worker repo'
        (repo / 'scripts').mkdir(parents=True)
        (repo / 'scripts/review_pod_session.py').write_text('''import json, os, sys
print(json.dumps({'args':sys.argv[1:], 'no_stop':os.environ.get('NO_STOP'),
                  'auto_stop':os.environ.get('FORESIGHT_AUTO_STOP')}))
''')
        git = self.bin / 'git'
        git.write_text('#!/usr/bin/env bash\nprintf "%s\\n" ' + 'a' * 40 + '\n')
        git.chmod(0o755)
        for mode, expected, minutes in (('full', 'full', '90'), ('review', 'full', '90'),
                                       ('smoke', 'smoke', '30')):
            with self.subTest(mode=mode):
                result = self.invoke(['--worker'],
                                     {'FORESIGHT_SETUP_ROOT': str(self.root),
                                      'FORESIGHT_REPO': str(repo),
                                      'FORESIGHT_MODE': mode, 'NO_STOP': '1',
                                      'FORESIGHT_AUTO_STOP': '1'}, script='start_review_pod.sh')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(data['args'], ['--mode', expected, '--max-minutes', minutes])
                self.assertEqual(data['no_stop'], '1')
                self.assertEqual(data['auto_stop'], '1')
                self.assertFalse(self.calls.exists())

    def test_pod_entry_rejects_invalid_mode_or_budget_before_gpu_probe(self):
        for env in ({'FORESIGHT_MODE': 'release'}, {'FORESIGHT_MAX_MINUTES': '0.5'}):
            result = self.invoke(env=env, script='start_review_pod.sh')
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertNotIn('NVIDIA', result.stderr)
            self.assertFalse(self.calls.exists())

    def test_push_permission_is_required_before_worker_start_unless_explicitly_skipped(self):
        repo = self.root / 'permission repo'
        (repo / '.git').mkdir(parents=True)
        (repo / 'scripts').mkdir()
        for name in ('bootstrap_fresh_pod.sh', 'review_pod_session.py'):
            (repo / 'scripts' / name).touch()
        # All external programs that could contact a service or start GPU work
        # are mocked; local flock/tee and shell control flow remain real.
        mock_tools = {
            'git': '#!/usr/bin/env bash\nexit 0\n',
            'nvidia-smi': '''#!/usr/bin/env bash
case "$*" in
  *query-compute-apps*) ;;
  *name,driver_version*) printf 'Mock GPU, 595.0\\n' ;;
  *driver_version*) printf '595.0\\n' ;;
  *) exit 2 ;;
esac
''',
            'gh': '''#!/usr/bin/env bash
printf '%s\\n' "$*" >> "$MOCK_GH_CALLS"
if [[ "$1" == api ]]; then
  printf 'mock API stderr must stay private\\n' >&2
  printf '%s\\n' "$MOCK_CAN_PUSH"
  exit "$MOCK_API_EXIT"
fi
exit 0
''',
            'tmux': '''#!/usr/bin/env bash
if [[ "$1" == has-session ]]; then exit 1; fi
if [[ "$1" == new-session ]]; then printf '%s\\n' "$*" > "$MOCK_WORKER_STARTED"; exit 0; fi
exit 2
''',
        }
        for name, content in mock_tools.items():
            target = self.bin / name
            target.write_text(content)
            target.chmod(0o755)
        started = self.root / 'worker_started.txt'
        gh_calls = self.root / 'gh_calls.txt'
        cases = (('true', '0', '0', True), ('false', '0', '0', False),
                 ('true', '7', '0', False), ('false', '7', '1', True))
        for permission, api_exit, skip, expected_start in cases:
            with self.subTest(permission=permission, api_exit=api_exit, skip=skip):
                started.unlink(missing_ok=True)
                gh_calls.unlink(missing_ok=True)
                result = self.invoke(env={
                    'FORESIGHT_SETUP_ROOT': str(self.root), 'FORESIGHT_REPO': str(repo),
                    'FORESIGHT_SKIP_GITHUB_LOGIN': skip, 'NO_STOP': '1',
                    'MOCK_CAN_PUSH': permission, 'MOCK_API_EXIT': api_exit,
                    'MOCK_GH_CALLS': str(gh_calls), 'MOCK_WORKER_STARTED': str(started),
                }, script='start_review_pod.sh')
                self.assertEqual(started.exists(), expected_start, result.stdout + result.stderr)
                self.assertEqual(result.returncode == 0, expected_start)
                self.assertNotIn('mock API stderr', result.stdout + result.stderr)
                if skip == '0':
                    self.assertIn('api repos/pchatla-arch/anticipatory-balance --jq .permissions.push',
                                  gh_calls.read_text())
                else:
                    self.assertFalse(gh_calls.exists())
                if not expected_start:
                    self.assertIn('No experiment has started', result.stderr)
                    self.assertIn('RunPod dashboard', result.stderr)


class SessionTests(unittest.TestCase):
    def test_all_referenced_repository_scripts_exist(self):
        # Check the launcher chain, including the reused-environment preflight
        # and fresh bootstrap. External Playground/uv installation paths are
        # not repository scripts and intentionally are not matched here.
        sources = ('scripts/start_review.sh', 'scripts/start_review_pod.sh',
                   'scripts/review_pod_session.py', 'scripts/bootstrap_fresh_pod.sh',
                   'scripts/preflight_review.py')
        referenced = set()
        for source in sources:
            self.assertTrue((ROOT / source).is_file(), source)
            for target in re.findall(r'\b(?:scripts|foresight)/[A-Za-z0-9_./-]+\.(?:py|sh)\b',
                                     (ROOT / source).read_text()):
                with self.subTest(source=source, target=target):
                    self.assertTrue((ROOT / target).is_file(),
                                    f'{source} references missing repository script {target}')
                    referenced.add(target)
        self.assertTrue({'scripts/bootstrap_fresh_pod.sh', 'scripts/preflight_review.py',
                         'scripts/finish_pod_session.py'} <= referenced)

    def test_mode_dependent_budget_and_invalid_limits(self):
        self.assertEqual(SESSION.parse_args([]).max_minutes, 90)
        self.assertEqual(SESSION.parse_args(['--mode', 'smoke']).max_minutes, 30)
        self.assertEqual(SESSION.parse_args(['--max-minutes', '45.5']).max_minutes, 45.5)
        for value in ('0', '.5', '-1', 'nan', 'inf'):
            with self.subTest(value=value), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    SESSION.parse_args(['--max-minutes', value])

    def test_existing_environment_gets_preflight_without_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for venv in (root / 'mujoco_playground/rss2025/.venv', root / 'custom'):
                (venv / 'bin').mkdir(parents=True)
                (venv / 'bin/python').touch()
                command, seconds, _ = SESSION.setup_command(root, venv, root / 'logs')
                self.assertEqual(command[:2], [str(venv / 'bin/python'), 'scripts/preflight_review.py'])
                self.assertEqual(seconds, 300)

    def test_fresh_default_only_gets_bounded_bootstrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command, seconds, _ = SESSION.setup_command(
                root, root / 'mujoco_playground/rss2025/.venv', root / 'logs')
            self.assertEqual(command, ['bash', 'scripts/bootstrap_fresh_pod.sh'])
            self.assertEqual(seconds, 1500)
            with self.assertRaises(RuntimeError):
                SESSION.setup_command(root, root / 'custom', root / 'logs')


if __name__ == '__main__':
    unittest.main()
