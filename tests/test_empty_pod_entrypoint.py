"""Exercise entrypoint routing without installing packages or touching a GPU."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/start_empty_pod.sh'
FAKE = '''#!/usr/bin/env python3
import json, os, pathlib, sys
name=pathlib.Path(sys.argv[0]).name
args=sys.argv[1:]
with open(os.environ['FAKE_CALLS'],'a') as f: f.write(json.dumps([name,*args])+'\\n')
if name=='nvidia-smi':
    if '--query-gpu=driver_version' in args: print(os.environ.get('FAKE_DRIVER','595.10'))
elif name=='tmux':
    if args[0]=='has-session': sys.exit(0 if os.environ.get('FAKE_ACTIVE')=='1' else 1)
elif name=='git' and args and args[0]=='clone':
    p=pathlib.Path(args[-1]); (p/'.git').mkdir(parents=True)
    (p/'scripts').mkdir()
    for n in ['bootstrap_fresh_pod.sh','start_empty_pod.sh']: (p/'scripts'/n).write_text('# stub\\n')
elif name=='apt-get':
    raise SystemExit('Unexpected real installation route in mocked launch')
'''


class EmptyPodEntrypointTests(unittest.TestCase):
    def launch(self, directory, **overrides):
        directory=Path(directory)
        bin_dir=directory/'bin'; bin_dir.mkdir()
        for name in ('nvidia-smi','git','tmux','flock','findmnt','timeout','gh','apt-get'):
            p=bin_dir/name; p.write_text(FAKE); p.chmod(0o755)
        calls=directory/'calls.jsonl'
        env=os.environ.copy()
        env.update(PATH=str(bin_dir)+os.pathsep+env['PATH'], FAKE_CALLS=str(calls),
                   FORESIGHT_SETUP_ROOT=str(directory/'work'), FORESIGHT_SKIP_GITHUB_LOGIN='1',
                   FORESIGHT_CODE_REF='foresight/experiments-v2')
        env.update(overrides)
        result=subprocess.run(['bash',str(SCRIPT)],env=env,text=True,capture_output=True,timeout=10)
        recorded=[json.loads(line) for line in calls.read_text().splitlines()] if calls.exists() else []
        return result, recorded

    def test_downloaded_script_routes_to_requested_root(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            repo=Path(directory)/'work/anticipatory-balance'
            self.assertTrue((repo/'scripts/bootstrap_fresh_pod.sh').exists())
            tmux=[c for c in calls if c[:2]==['tmux','new-session']]
            self.assertEqual(len(tmux),1)
            self.assertIn(str(repo),tmux[0][-1])
            self.assertIn('FORESIGHT_SETUP_ROOT=',tmux[0][-1])
            self.assertFalse(list((Path(directory)/'work').glob('.foresight-clone.*')))
            self.assertNotIn('apt-get',[c[0] for c in calls])

    def test_unsupported_driver_stops_before_install_or_clone(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,FAKE_DRIVER='570.12')
            self.assertNotEqual(result.returncode,0)
            self.assertIn('580 or newer',result.stderr)
            self.assertFalse(any(c[0] in ('git','apt-get','tmux') for c in calls))

    def test_active_session_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            result,calls=self.launch(directory,FAKE_ACTIVE='1')
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertIn('already exists',result.stdout)
            self.assertFalse(any(c[0]=='git' for c in calls))
            self.assertFalse(any(c[:2]==['tmux','kill-session'] for c in calls))


if __name__=='__main__': unittest.main()
