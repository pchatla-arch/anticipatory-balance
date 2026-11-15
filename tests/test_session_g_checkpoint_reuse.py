"""Fresh-pod Session G reuses verified published skills without starting PPO."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PublishedCheckpointReuseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='foresight-g-reuse-')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        (self.root/'scripts').mkdir()
        shutil.copyfile(ROOT/'scripts/run_session_g.sh', self.root/'scripts/run_session_g.sh')
        self.calls = self.root/'calls.txt'
        self.env = {**os.environ, 'PY':sys.executable, 'SESSION_G_CALLS':str(self.calls),
                    'PYTHONDONTWRITEBYTECODE':'1'}
        tools = self.root/'tools'; tools.mkdir()
        git = tools/'git'; git.write_text('#!/bin/sh\nexit 0\n'); git.chmod(0o755)
        self.env['PATH'] = str(tools)+os.pathsep+os.environ['PATH']
        # Only the fixture's small stand-ins are imported, never JAX or a simulator.
        (self.root/'jax.py').write_text('# import-only fixture\n')
        (self.root/'absl.py').write_text('# import-only fixture\n')
        (self.root/'foresight/eval').mkdir(parents=True)
        (self.root/'foresight/__init__.py').write_text('')
        (self.root/'foresight/eval/__init__.py').write_text('')
        (self.root/'foresight/eval/rollout.py').write_text('''import os
from pathlib import Path
def load_policy(path):
    assert os.environ.get('JAX_PLATFORMS') == 'cpu'
    with open(os.environ['SESSION_G_CALLS'],'a') as f: f.write('RESTORE '+str(path)+'\\n')
    if os.environ.get('RESTORE_FAIL_MATCH','__never__') in str(path):
        raise RuntimeError('fixture unresolved OCDBT reference')
    return object()
''')
        (self.root/'scripts/eval_decision_frontier.py').write_text('''import os,sys
from pathlib import Path
with open(os.environ['SESSION_G_CALLS'],'a') as f: f.write('EVAL '+' '.join(sys.argv[1:])+'\\n')
p=Path(sys.argv[sys.argv.index('--out')+1]);p.mkdir(parents=True,exist_ok=True)
(p/'summary.json').write_text('{"fixture":true}')
''')
        (self.root/'scripts/train_skill.py').write_text('''import os,sys
with open(os.environ['SESSION_G_CALLS'],'a') as f: f.write('TRAIN '+' '.join(sys.argv[1:])+'\\n')
raise SystemExit(73)  # Dispatch marker only: no training is implemented.
''')
        for tag in ('E6c_readiness','E18_rule_family_p5','E18h_heldout_p5',
                    'E17b_depth_lat0','E17b_depth_lat100'):
            p=self.root/'results'/tag;p.mkdir(parents=True);(p/'summary.json').write_text('{}')
        (self.root/'results/response_model.json').write_text('{"c_lead":-1.0}')
        for tag in ('nominal_s1','brace2_s1'):
            (self.root/'checkpoints'/tag).mkdir(parents=True)
        for tag in ('nominal_s2','brace2_s2'):
            self.make_bundle(tag)

    def make_bundle(self, tag, step='000202342400'):
        p=self.root/'checkpoints'/tag/step;p.mkdir(parents=True,exist_ok=True)
        for name,obj in [('ppo_network_config.json',{'action_size':29}),
                         ('_METADATA',{'tree_metadata':{'fixture':1}}),
                         ('_CHECKPOINT_METADATA',{'commit_timestamp_nsecs':123})]:
            (p/name).write_text(json.dumps(obj))
        (p/'commit_success.txt').write_text('fixture successful commit')
        for q in (p,p/'ocdbt.process_0'):
            q.mkdir(exist_ok=True);(q/'manifest.ocdbt').write_bytes(b'fixture manifest')
            (q/'d').mkdir(exist_ok=True);(q/'d/payload').write_bytes(b'fixture numeric array data')
        return p

    def run_session(self):
        return subprocess.run(['bash','scripts/run_session_g.sh'],cwd=self.root,env=self.env,
                              text=True,capture_output=True,timeout=20)

    def events(self):
        return self.calls.read_text().splitlines() if self.calls.exists() else []

    def test_fresh_pod_restores_both_published_bundles_without_training_logs(self):
        self.assertFalse((self.root/'logs').exists())
        r=self.run_session()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        events=self.events()
        self.assertEqual(sum(e.startswith('RESTORE ') for e in events),2)
        self.assertEqual(sum(e.startswith('EVAL ') for e in events),1)
        self.assertFalse(any(e.startswith('TRAIN ') for e in events))
        self.assertIn('no retraining or log recovery needed',r.stdout)

    def test_finished_e19_does_not_restore_or_train(self):
        p=self.root/'results/E19_seed2_p5';p.mkdir();(p/'summary.json').write_text('{}')
        r=self.run_session()
        self.assertEqual(r.returncode,0,r.stdout+r.stderr)
        self.assertEqual(self.events(),[])

    def test_incomplete_or_unfetched_bundle_fails_before_training(self):
        p=self.root/'checkpoints/nominal_s2/000202342400'
        for failure in ('manifest','payload','config','lfs','commit'):
            with self.subTest(failure=failure):
                self.make_bundle('nominal_s2')
                if failure=='manifest': (p/'manifest.ocdbt').unlink()
                elif failure=='payload': (p/'ocdbt.process_0/d/payload').unlink()
                elif failure=='config': (p/'ppo_network_config.json').write_text('not json')
                elif failure=='lfs': (p/'d/payload').write_bytes(b'version https://git-lfs.github.com/spec/v1\n')
                else: (p/'_CHECKPOINT_METADATA').write_text('{"commit_timestamp_nsecs":null}')
                r=self.run_session()
                self.assertNotEqual(r.returncode,0)
                self.assertIn('no replacement training was started',r.stderr)
                self.assertFalse(any(e.startswith(('TRAIN ','EVAL ')) for e in self.events()))

    def test_structural_success_does_not_bypass_actual_restore_failure(self):
        self.env['RESTORE_FAIL_MATCH']='nominal_s2'
        r=self.run_session()
        self.assertNotEqual(r.returncode,0)
        self.assertEqual(sum(e.startswith('RESTORE ') for e in self.events()),1)
        self.assertFalse(any(e.startswith(('TRAIN ','EVAL ')) for e in self.events()))

    def test_latest_partial_checkpoint_is_not_replaced_by_older_complete_one(self):
        (self.root/'checkpoints/nominal_s2/000202342401').mkdir()
        r=self.run_session()
        self.assertNotEqual(r.returncode,0)
        self.assertIn('000202342401',r.stderr)
        self.assertEqual(self.events(),[])

    def test_missing_bundle_retains_training_dispatch_without_executing_training(self):
        shutil.rmtree(self.root/'checkpoints/nominal_s2')
        r=self.run_session()
        self.assertEqual(r.returncode,73,r.stdout+r.stderr)
        self.assertEqual(len(self.events()),1)
        self.assertTrue(self.events()[0].startswith('TRAIN '))
        self.assertIn('--skill nominal',self.events()[0])


if __name__=='__main__':
    unittest.main()
