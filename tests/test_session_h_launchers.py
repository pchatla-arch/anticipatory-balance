"""Exercise public/worker dispatch without network access, GPUs, or installs."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location(
    'review_pod_session_for_tests', ROOT / 'scripts/session_h_pod.py')
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
                    if not k.startswith('FORESIGHT_')
                    and k not in ('NO_STOP', 'GH_TOKEN', 'GITHUB_TOKEN', 'GH_HOST')}
        self.env.update(PATH=str(self.bin) + os.pathsep + os.environ['PATH'],
                        MOCK_ENTRY=str(self.download), MOCK_CURL_ARGS=str(self.calls))

    def invoke(self, args=(), env=None, script='start_session_h.sh'):
        return subprocess.run(['bash', str(ROOT / 'scripts' / script), *args],
                              env={**self.env, **(env or {})},
                              capture_output=True, text=True, timeout=10)

    def dispatch(self, args=(), env=None):
        result = self.invoke(args, env)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return json.loads(result.stdout.splitlines()[-1])

    def test_help_and_shell_syntax_require_no_network_or_gpu(self):
        for script in ('start_session_h.sh', 'start_session_h_pod.sh'):
            with self.subTest(script=script):
                result = self.invoke(['--help'], script=script)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('75', result.stdout)
                self.assertIn('20', result.stdout)
                self.assertIn('PPO training', result.stdout)
                self.assertIn('RunPod', result.stdout)
                self.assertFalse(self.calls.exists())
                check = subprocess.run(['bash', '-n', str(ROOT/'scripts'/script)],
                                       capture_output=True, text=True, timeout=5)
                self.assertEqual(check.returncode, 0, check.stderr)

    def test_public_defaults_aliases_and_small_budget(self):
        for args, mode, minutes in (([], 'full', '75'), (['h'], 'full', '75'),
                                    (['full'], 'full', '75'), (['smoke'], 'smoke', '20')):
            with self.subTest(args=args):
                data = self.dispatch(args)
                self.assertEqual(data['FORESIGHT_MODE'], mode)
                self.assertEqual(data['FORESIGHT_MAX_MINUTES'], minutes)
                self.assertEqual(data['FORESIGHT_REPO'], '/workspace/foresight-session-h')
                self.assertEqual(data['FORESIGHT_AUTO_STOP'], '1')
                self.assertIn('/session-d/scripts/start_session_h_pod.sh',
                              self.calls.read_text())

    def test_explicit_paths_budget_pinned_revision_and_no_stop_survive_dispatch(self):
        overrides = {'FORESIGHT_SETUP_ROOT': str(self.root / 'with space'),
                     'FORESIGHT_VENV': str(self.root / 'python env'),
                     'FORESIGHT_CODE_REF': 'a' * 40,
                     'FORESIGHT_MAX_MINUTES': '45.5',
                     'FORESIGHT_AUTO_STOP': '0', 'NO_STOP': '1'}
        data = self.dispatch(['h'], overrides)
        for name, value in overrides.items():
            self.assertEqual(data[name], value)
        self.assertEqual(data['FORESIGHT_REPO'], str(self.root / 'with space/foresight-session-h'))
        self.assertIn('/' + 'a' * 40 + '/scripts/start_session_h_pod.sh', self.calls.read_text())

    def test_invalid_dispatch_is_rejected_before_download(self):
        cases = [(['unknown'], {}), (['h', 'extra'], {}),
                 (['h'], {'FORESIGHT_CODE_REF': '../main'}),
                 (['h'], {'FORESIGHT_REPO': 'relative'}),
                 (['h'], {'FORESIGHT_AUTO_STOP': 'yes'})]
        cases += [(['h'], {'FORESIGHT_MAX_MINUTES': value})
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
                result = self.invoke(['h'])
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
        (repo / 'scripts/session_h_pod.py').write_text('''import json, os, sys
print(json.dumps({'args':sys.argv[1:], 'no_stop':os.environ.get('NO_STOP'),
                  'auto_stop':os.environ.get('FORESIGHT_AUTO_STOP')}))
''')
        git = self.bin / 'git'
        git.write_text('#!/usr/bin/env bash\nprintf "%s\\n" ' + 'a' * 40 + '\n')
        git.chmod(0o755)
        for mode, expected, minutes in (('full', 'full', '75'), ('h', 'full', '75'),
                                       ('smoke', 'smoke', '20')):
            with self.subTest(mode=mode):
                result = self.invoke(['--worker'],
                                     {'FORESIGHT_SETUP_ROOT': str(self.root),
                                      'FORESIGHT_REPO': str(repo),
                                      'FORESIGHT_MODE': mode, 'NO_STOP': '1',
                                      'FORESIGHT_SKIP_GITHUB_LOGIN': '1',
                                      'FORESIGHT_AUTO_STOP': '1'}, script='start_session_h_pod.sh')
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                data = json.loads(result.stdout.splitlines()[-1])
                self.assertEqual(data['args'], ['--mode', expected, '--max-minutes', minutes])
                self.assertEqual(data['no_stop'], '1')
                self.assertEqual(data['auto_stop'], '1')
                self.assertFalse(self.calls.exists())

    def test_pod_entry_rejects_invalid_mode_or_budget_before_gpu_probe(self):
        for env in ({'FORESIGHT_MODE': 'release'}, {'FORESIGHT_MAX_MINUTES': '0.5'}):
            result = self.invoke(env=env, script='start_session_h_pod.sh')
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertNotIn('NVIDIA', result.stderr)
            self.assertFalse(self.calls.exists())

    def make_auth_fixture(self):
        repo = self.root / 'permission repo'
        (repo / '.git').mkdir(parents=True)
        (repo / 'scripts').mkdir()
        for name in ('bootstrap_fresh_pod.sh', 'session_h_pod.py'):
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
  [[ "$2" == --hostname && "$3" == github.com ]] || exit 90
  case "$4" in
    user)
      if [[ -f "$MOCK_SWITCHED" ]]; then
        printf '%s\\n' "$MOCK_LOGIN_AFTER_SWITCH"
      else
        printf '%s\\n' "$MOCK_LOGIN"
      fi
      exit "$MOCK_USER_API_EXIT"
      ;;
    repos/pchatla-arch/anticipatory-balance)
      printf '%s\\n' "$MOCK_CAN_PUSH"
      exit "$MOCK_API_EXIT"
      ;;
    *) exit 91 ;;
  esac
fi
if [[ "$1 $2" == 'auth switch' ]]; then
  if [[ "$MOCK_SWITCH_EXIT" == 0 ]]; then touch "$MOCK_SWITCHED"; fi
  exit "$MOCK_SWITCH_EXIT"
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
        self.started = self.root / 'worker_started.txt'
        self.gh_calls = self.root / 'gh_calls.txt'
        self.switched = self.root / 'account_switched.txt'
        self.env.update({
            'FORESIGHT_SETUP_ROOT': str(self.root), 'FORESIGHT_REPO': str(repo),
            'FORESIGHT_SKIP_GITHUB_LOGIN': '0', 'NO_STOP': '1',
            'MOCK_CAN_PUSH': 'true', 'MOCK_API_EXIT': '0', 'MOCK_USER_API_EXIT': '0',
            'MOCK_LOGIN': 'pchatla-arch', 'MOCK_LOGIN_AFTER_SWITCH': 'pchatla-arch',
            'MOCK_SWITCH_EXIT': '0', 'MOCK_SWITCHED': str(self.switched),
            'MOCK_GH_CALLS': str(self.gh_calls), 'MOCK_WORKER_STARTED': str(self.started),
        })
        return repo

    def reset_auth_calls(self):
        for path in (self.started, self.gh_calls, self.switched):
            path.unlink(missing_ok=True)
        # The worker's `exec > >(tee ...)` child inherits the setup lock descriptor, so the
        # lock can stay held for a moment after a worker exits. Wait for it before the next
        # invocation instead of letting flock -n fail spuriously.
        lock = self.root / '.foresight-setup.lock'
        if lock.exists():
            deadline = time.time() + 5
            while time.time() < deadline:
                with open(lock, 'w') as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        time.sleep(0.02)

    def test_push_permission_is_required_before_worker_start_unless_explicitly_skipped(self):
        self.make_auth_fixture()
        cases = (('true', '0', '0', True), ('false', '0', '0', False),
                 ('true', '7', '0', False), ('false', '7', '1', True))
        for permission, api_exit, skip, expected_start in cases:
            with self.subTest(permission=permission, api_exit=api_exit, skip=skip):
                self.reset_auth_calls()
                result = self.invoke(env={
                    'FORESIGHT_SKIP_GITHUB_LOGIN': skip,
                    'MOCK_CAN_PUSH': permission, 'MOCK_API_EXIT': api_exit,
                }, script='start_session_h_pod.sh')
                self.assertEqual(self.started.exists(), expected_start, result.stdout + result.stderr)
                self.assertEqual(result.returncode == 0, expected_start)
                self.assertNotIn('mock API stderr', result.stdout + result.stderr)
                if skip == '0':
                    calls = self.gh_calls.read_text()
                    self.assertIn('api --hostname github.com user --jq .login', calls)
                    self.assertIn('api --hostname github.com repos/pchatla-arch/anticipatory-balance --jq .permissions.push',
                                  calls)
                else:
                    self.assertFalse(self.gh_calls.exists())
                    self.assertIn('FORESIGHT_SKIP_GITHUB_LOGIN=1', self.started.read_text())
                if not expected_start:
                    self.assertIn('No experiment has started', result.stderr)
                    self.assertIn('RunPod dashboard', result.stderr)

    def test_wrong_stored_account_switch_must_produce_expected_identity(self):
        self.make_auth_fixture()
        cases = (('0', 'pchatla-arch', True), ('1', 'pchatla-arch', False),
                 ('0', 'other-account', False))
        for switch_exit, switched_login, expected_start in cases:
            with self.subTest(switch_exit=switch_exit, switched_login=switched_login):
                self.reset_auth_calls()
                result = self.invoke(env={
                    'MOCK_LOGIN': 'other-account', 'MOCK_SWITCH_EXIT': switch_exit,
                    'MOCK_LOGIN_AFTER_SWITCH': switched_login,
                }, script='start_session_h_pod.sh')
                self.assertEqual(self.started.exists(), expected_start, result.stdout + result.stderr)
                self.assertEqual(result.returncode == 0, expected_start)
                calls = self.gh_calls.read_text()
                self.assertIn('auth switch --hostname github.com --user pchatla-arch', calls)
                if expected_start:
                    self.assertGreaterEqual(calls.count('api --hostname github.com user --jq .login'), 2)
                else:
                    self.assertNotIn('auth setup-git', calls)
                    self.assertNotIn('repos/pchatla-arch/anticipatory-balance', calls)

    def test_wrong_environment_token_identity_cannot_be_fixed_by_switching_stored_account(self):
        self.make_auth_fixture()
        for name in ('GH_TOKEN', 'GITHUB_TOKEN'):
            with self.subTest(name=name):
                self.reset_auth_calls()
                result = self.invoke(env={
                    'MOCK_LOGIN': 'other-account',
                    name: 'dummy-test-value',
                }, script='start_session_h_pod.sh')
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertFalse(self.started.exists())
                self.assertNotIn('auth switch', self.gh_calls.read_text())
                self.assertNotIn('auth login', self.gh_calls.read_text())
                self.assertNotIn('dummy-test-value', result.stdout + result.stderr)

    def test_worker_rechecks_identity_and_permission_before_python_setup(self):
        repo = self.make_auth_fixture()
        python_started = self.root / 'python_started.txt'
        (repo / 'scripts/session_h_pod.py').write_text(
            'import os\nfrom pathlib import Path\n'
            'Path(os.environ["MOCK_PYTHON_STARTED"]).touch()\n'
        )
        cases = (('pchatla-arch', 'true', True), ('other-account', 'true', False),
                 ('pchatla-arch', 'false', False))
        for login, permission, expected_start in cases:
            with self.subTest(login=login, permission=permission):
                self.reset_auth_calls()
                python_started.unlink(missing_ok=True)
                result = self.invoke(['--worker'], env={
                    'MOCK_LOGIN': login, 'MOCK_CAN_PUSH': permission,
                    'MOCK_PYTHON_STARTED': str(python_started),
                }, script='start_session_h_pod.sh')
                self.assertEqual(python_started.exists(), expected_start, result.stdout + result.stderr)
                self.assertEqual(result.returncode == 0, expected_start)
                calls = self.gh_calls.read_text()
                self.assertIn('api --hostname github.com user --jq .login', calls)
                self.assertNotIn('auth switch', calls)
                self.assertNotIn('auth login', calls)
                self.assertNotIn('mock API stderr', result.stdout + result.stderr)


class SessionTests(unittest.TestCase):
    def test_all_referenced_repository_scripts_exist(self):
        # Check the launcher chain, including the reused-environment preflight
        # and fresh bootstrap. External Playground/uv installation paths are
        # not repository scripts and intentionally are not matched here.
        sources = ('scripts/start_session_h.sh', 'scripts/start_session_h_pod.sh',
                   'scripts/session_h_pod.py', 'scripts/bootstrap_fresh_pod.sh',
                   'scripts/preflight_session_h.py')
        referenced = set()
        for source in sources:
            self.assertTrue((ROOT / source).is_file(), source)
            for target in re.findall(r'\b(?:scripts|foresight)/[A-Za-z0-9_./-]+\.(?:py|sh)\b',
                                     (ROOT / source).read_text()):
                with self.subTest(source=source, target=target):
                    self.assertTrue((ROOT / target).is_file(),
                                    f'{source} references missing repository script {target}')
                    referenced.add(target)
        self.assertTrue({'scripts/bootstrap_fresh_pod.sh', 'scripts/preflight_session_h.py',
                         'scripts/finish_pod_session.py'} <= referenced)

    def test_mode_dependent_budget_and_invalid_limits(self):
        self.assertEqual(SESSION.parse_args([]).max_minutes, 75)
        self.assertEqual(SESSION.parse_args(['--mode', 'smoke']).max_minutes, 20)
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
                self.assertEqual(command[:2], [str(venv / 'bin/python'), 'scripts/preflight_session_h.py'])
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
