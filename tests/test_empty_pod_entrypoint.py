"""Exercise entrypoint routing without installing packages or touching a GPU."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/start_empty_pod.sh'
FAKE = '''#!/usr/bin/env python3
import json, os, pathlib, sys
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
with open(os.environ['FAKE_CALLS'],'a') as f: f.write(json.dumps([name,*args])+'\\n')
if name=='nvidia-smi':
    driver=os.environ.get('FAKE_DRIVER','595.10')
    if '--query-gpu=driver_version' in args: print(driver)
    elif '--query-gpu=name,driver_version' in args: print('Mock GPU, '+driver)
    elif '--query-gpu=index,name,driver_version,memory.total' in args: print('0, Mock GPU, '+driver+', 24576')
elif name=='tmux':
    if args[0]=='has-session': sys.exit(0 if os.environ.get('FAKE_ACTIVE')=='1' else 1)
elif name=='git' and args and args[0]=='clone':
    p=pathlib.Path(args[-1]); (p/'.git').mkdir(parents=True)
    (p/'scripts').mkdir()
    for n in ['bootstrap_fresh_pod.sh','start_empty_pod.sh','pod_session.py']: (p/'scripts'/n).write_text('# stub\\n')
elif name=='git' and args[:2]==['rev-parse','HEAD']:
    print('a'*40)
elif name=='gh':
    if args[:2]==['auth','status']: sys.exit(int(os.environ.get('FAKE_GH_AUTH_EXIT','0')))
    if args[:2]==['auth','login']: sys.exit(int(os.environ.get('FAKE_GH_LOGIN_EXIT','0')))
elif name=='findmnt':
    print(json.dumps({'filesystems':[{'target':os.environ['FORESIGHT_SETUP_ROOT'],'fstype':'ext4'}]}))
elif name=='apt-get':
    raise SystemExit('Unexpected real installation route in mocked launch')
'''


class EmptyPodEntrypointTests(unittest.TestCase):
    def launch(self, directory, *, script=SCRIPT, arguments=(), interactive=False, **overrides):
        directory=Path(directory)
        bin_dir=directory/'bin'; bin_dir.mkdir()
        for name in ('nvidia-smi','git','tmux','flock','findmnt','timeout','gh','apt-get'):
            p=bin_dir/name; p.write_text(FAKE); p.chmod(0o755)
        calls=directory/'calls.jsonl'
        env=os.environ.copy()
        env.update(PATH=str(bin_dir)+os.pathsep+env['PATH'], FAKE_CALLS=str(calls),
                   FORESIGHT_SETUP_ROOT=str(directory/'work'), FORESIGHT_SKIP_GITHUB_LOGIN='1',
                   FORESIGHT_CODE_REF='foresight/experiments-v2')
        # Do not inherit a developer's real checkout, venv, or stop preference.
        for name in ('FORESIGHT_REPO','FORESIGHT_VENV','FORESIGHT_AUTO_STOP','NO_STOP'):
            env.pop(name,None)
        env.update(overrides)
        Path(env['FORESIGHT_SETUP_ROOT']).mkdir(parents=True,exist_ok=True)
        if interactive:
            master,slave=os.openpty()
            try:
                result=subprocess.run(['bash',str(script),*arguments],env=env,stdin=slave,
                                      text=True,capture_output=True,timeout=10)
            finally:
                os.close(master); os.close(slave)
        else:
            result=subprocess.run(['bash',str(script),*arguments],env=env,stdin=subprocess.DEVNULL,
                                  text=True,capture_output=True,timeout=10)
        recorded=[json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, recorded

    def test_downloaded_script_routes_to_requested_root(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            repo=Path(directory)/'work/foresight-runner'
            self.assertTrue((repo/'scripts/bootstrap_fresh_pod.sh').exists())
            self.assertTrue((repo/'scripts/pod_session.py').exists())
            self.assertFalse((Path(directory)/'work/anticipatory-balance').exists())
            tmux=[c for c in calls if c[:2]==['tmux','new-session']]
            self.assertEqual(len(tmux),1)
            self.assertIn(str(repo),tmux[0][-1])
            self.assertIn('FORESIGHT_SETUP_ROOT=',tmux[0][-1])
            self.assertIn('bash scripts/start_empty_pod.sh --worker',tmux[0][-1])
            self.assertFalse(list((Path(directory)/'work').glob('.foresight-clone.*')))
            self.assertNotIn('apt-get',[c[0] for c in calls])

    def test_existing_warm_venv_and_lower_driver_reach_preflight_route(self):
        with tempfile.TemporaryDirectory() as directory:
            venv=Path(directory)/'existing environment'
            (venv/'bin').mkdir(parents=True)
            (venv/'bin/python').write_text('# existing fixture; never executed\n')
            (venv/'bin/python').chmod(0o755)
            result,calls=self.launch(directory,FAKE_DRIVER='570.12',FORESIGHT_VENV=str(venv))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            tmux=[c for c in calls if c[:2]==['tmux','new-session']]
            self.assertEqual(len(tmux),1)
            self.assertIn('FORESIGHT_VENV='+str(venv),shlex.split(tmux[0][-1]))
            self.assertFalse(any(c[0]=='apt-get' for c in calls))

    def test_fresh_bootstrap_still_rejects_lower_driver_before_install(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,script=ROOT/'scripts/bootstrap_fresh_pod.sh',FAKE_DRIVER='570.12')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('driver >= 580',result.stdout+result.stderr)
            self.assertFalse(any(c[0] in ('git','apt-get') for c in calls))

    def test_failed_interactive_auth_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,interactive=True,FORESIGHT_SKIP_GITHUB_LOGIN='0',
                                     FAKE_GH_AUTH_EXIT='1',FAKE_GH_LOGIN_EXIT='1')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('No experiment has started',result.stderr)
            self.assertTrue(any(c[:3]==['gh','auth','login'] for c in calls))
            self.assertFalse(any(c[:2]==['tmux','new-session'] for c in calls))

    def test_missing_noninteractive_auth_never_starts_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,FORESIGHT_SKIP_GITHUB_LOGIN='0',FAKE_GH_AUTH_EXIT='1')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('No terminal for GitHub login',result.stderr)
            self.assertFalse(any(c[:2]==['tmux','new-session'] for c in calls))

    def test_authenticated_launch_starts_worker_only_after_git_auth_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,FORESIGHT_SKIP_GITHUB_LOGIN='0')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            auth=next(i for i,c in enumerate(calls) if c[:3]==['gh','auth','setup-git'])
            start=next(i for i,c in enumerate(calls) if c[:2]==['tmux','new-session'])
            self.assertLess(auth,start)

    def test_worker_executes_pod_session_with_preserved_custom_venv(self):
        with tempfile.TemporaryDirectory() as directory:
            repo=Path(directory)/'work/foresight-runner'
            (repo/'scripts').mkdir(parents=True)
            observed=Path(directory)/'worker.json'
            (repo/'scripts/pod_session.py').write_text(
                'import json,os,pathlib,sys\n'
                'pathlib.Path(os.environ["FAKE_WORKER_JSON"]).write_text(json.dumps('
                '{"args":sys.argv[1:],"venv":os.environ.get("FORESIGHT_VENV"),'
                '"repo":os.environ.get("FORESIGHT_REPO"),"stop":os.environ.get("FORESIGHT_AUTO_STOP")}))\n')
            venv=Path(directory)/'existing environment'
            result,calls=self.launch(directory,arguments=('--worker',),FORESIGHT_VENV=str(venv),
                                     FORESIGHT_MODE='smoke',FORESIGHT_MAX_MINUTES='45',FORESIGHT_AUTO_STOP='1',
                                     FAKE_WORKER_JSON=str(observed))
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            value=json.loads(observed.read_text())
            self.assertEqual(value['args'],['--mode','smoke','--max-minutes','45'])
            self.assertEqual(value['venv'],str(venv))
            self.assertEqual(value['repo'],str(repo))
            self.assertEqual(value['stop'],'1')
            self.assertFalse(any(c[0] in ('nvidia-smi','apt-get','tmux') for c in calls))

    def test_active_session_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,FAKE_ACTIVE='1')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('already exists',result.stdout)
            self.assertFalse(any(c[0]=='git' for c in calls))
            self.assertFalse(any(c[:2]==['tmux','kill-session'] for c in calls))


def load_pod_worker():
    """Load the real standard-library worker without changing global import paths."""
    spec=importlib.util.spec_from_file_location('run_experiments_v2',ROOT/'scripts/run_experiments_v2.py')
    runner=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    spec=importlib.util.spec_from_file_location('entrypoint_test_pod_session',ROOT/'scripts/pod_session.py')
    worker=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'run_experiments_v2':runner}):
        spec.loader.exec_module(worker)
    return worker


class PodWorkerRoutingTests(unittest.TestCase):
    def test_custom_warm_environment_ignores_unrelated_default_setup_marker(self):
        worker=load_pod_worker()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'.foresight-bootstrap').mkdir()
            (root/'.foresight-bootstrap/state.json').write_text('{}')
            venv=root/'custom_existing_venv'
            (venv/'bin').mkdir(parents=True)
            (venv/'bin/python').touch()
            command,budget,message=worker.setup_command(root,venv,root/'logs')
            self.assertEqual(command[:2],[str(venv/'bin/python'),'scripts/preflight_experiments_v2.py'])
            self.assertEqual(budget,300)
            self.assertIn('no Python packages',message)

    def test_each_worker_invocation_replaces_stale_token_and_passes_it_to_finish(self):
        worker=load_pod_worker()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            venv=root/'existing_venv'
            (venv/'bin').mkdir(parents=True)
            (venv/'bin/python').touch()
            launched=[]
            finalized=[]

            def popen(command,**kwargs):
                launched.append(kwargs['env']['FORESIGHT_INVOCATION_ID'])
                process=MagicMock()
                process.wait.return_value=0
                return process

            def run(command,**kwargs):
                if command[:2]==['git','rev-parse']:
                    return subprocess.CompletedProcess(command,0,'a'*40+'\n','')
                self.assertTrue(any(str(value).endswith('finish_pod_session.py') for value in command))
                finalized.append(kwargs['env']['FORESIGHT_INVOCATION_ID'])
                return subprocess.CompletedProcess(command,0)

            with contextlib.ExitStack() as stack:
                stack.enter_context(patch.object(worker,'ROOT',root))
                stack.enter_context(patch.object(worker.os,'chdir'))
                stack.enter_context(patch.dict(os.environ,{
                    'FORESIGHT_SETUP_ROOT':str(root),'FORESIGHT_VENV':str(venv),
                    'FORESIGHT_INVOCATION_ID':'stale-parent-token'}))
                stack.enter_context(patch.object(worker.signal,'signal'))
                stack.enter_context(patch.object(worker,'run_child',return_value=0))
                stack.enter_context(patch.object(worker.subprocess,'Popen',side_effect=popen))
                stack.enter_context(patch.object(worker.subprocess,'run',side_effect=run))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                self.assertEqual(worker.main(['--mode','smoke']),0)
                self.assertEqual(worker.main(['--mode','smoke']),0)
            self.assertEqual(len(launched),2)
            self.assertEqual(launched,finalized)
            self.assertNotEqual(launched[0],launched[1])
            for token in launched:
                self.assertRegex(token,r'^[0-9a-f]{32}$')
                self.assertNotEqual(token,'stale-parent-token')


if __name__=='__main__': unittest.main()
