"""Bounded H execution, same-code paired design, and publication evidence paths."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import session_h_plan as plan
import run_session_h as runner
import publish_experiment_results as publisher
import summarize_session_h as report


class HRunnerTests(unittest.TestCase):
    def test_hold_baseline_and_longer_arms_share_everything_except_release_and_out(self):
        stages=plan.make_plan('OUT')
        self.assertEqual(len(stages),8)
        self.assertTrue(plan.assert_disjoint_splits(stages))
        holds=[s for s in stages if s['name'].startswith('E18b')]
        self.assertEqual([s['release_after'] for s in holds],[.3,.6,1.])
        self.assertEqual([s['episodes_per_condition'] for s in holds],[2048]*3)
        reference=None
        for s in holds:
            args=s['args'][:]
            for flag in ('--release_after','--out'):
                i=args.index(flag);del args[i:i+2]
            if reference is None: reference=args
            else:self.assertEqual(args,reference)
            self.assertIn('reactive,ttc_threshold,belief_lead',args)
        self.assertFalse(any('train_skill' in ' '.join(s['args']) for s in stages))

    def test_plan_only_requires_standard_python(self):
        r=subprocess.run([sys.executable,str(ROOT/'scripts/run_session_h.py'),'--plan','--mode','smoke'],capture_output=True,text=True,timeout=10)
        self.assertEqual(r.returncode,0,r.stderr)
        self.assertEqual(len(json.loads(r.stdout)),3)

    def test_all_h_evidence_paths_are_published(self):
        for p in ('report.md','analysis.json','hold_comparisons.csv',
                  'E17c_depth_lat0/episodes/depth_belief_voi_0.05_b000.csv',
                  'E17c_depth_lat0/episodes/none_reactive_0_b001.csv',
                  'E17c_depth_lat0/closed_loop.png','E17c_depth_lat0/episodes.npz',
                  'E18b_hold_0p3/episodes/batch000_condition000.csv',
                  'E18b_hold_0p3/frontier.png','E18b_hold_0p3/batch_summaries.json'):
            self.assertTrue(publisher.allowed_tree_path('results/run1/'+p),p)

    def test_stage_timeout_halts_later_work_and_publishes_partial_record(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);calls=[]
            def child(command,env,log,timeout_seconds,**kwargs):
                calls.append(command)
                if 'scripts/eval_depth_closed_loop.py' in command:raise TimeoutError('fixture deadline')
                if any(str(x).endswith('publish_experiment_results.py') for x in command):
                    p=Path(command[command.index('--run-dir')+1]);rec=json.loads((p/'run.json').read_text())
                    (p/'publication_receipt.json').write_text(json.dumps(dict(schema_version=1,status='verified',verified=True,
                        run_id=rec['run_id'],run_source='a'*40,commit='b'*40,run_exit_code=124)))
                return 0
            with contextlib.ExitStack() as s:
                for obj,name,value in ((runner,'ROOT',root),(runner,'choose_python',lambda:sys.executable),
                                       (runner,'run_child',child),(runner.signal,'signal',lambda *a:None),
                                       (runner.shutil,'which',lambda *a:None)):
                    s.enter_context(mock.patch.object(obj,name,value))
                s.enter_context(mock.patch.object(runner.subprocess,'run',return_value=subprocess.CompletedProcess([],0,'a'*40+'\n','')))
                s.enter_context(contextlib.redirect_stdout(io.StringIO()));s.enter_context(contextlib.redirect_stderr(io.StringIO()))
                code=runner.main(['--mode','full'])
            self.assertEqual(code,124)
            last=json.loads((root/'logs/last_run.json').read_text())
            rec=json.loads((Path(last['run_dir'])/'run.json').read_text())
            self.assertEqual(len(rec['stages']),1)
            self.assertEqual(len(rec['planned_stages']),8)
            self.assertEqual(last['publication_status'],'verified')
            self.assertFalse(any('scripts/eval_decision_frontier.py' in c for c in calls))

    def test_partial_report_names_unstarted_stages(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            (p/'run.json').write_text(json.dumps(dict(status='failed',run_id='fixture',source_commit='a'*40,
                planned_stages=plan.make_plan(p),stages=[])))
            with contextlib.redirect_stdout(io.StringIO()):a=report.summarize(p)
            self.assertEqual(a['status'],'partial')
            self.assertEqual(len(a['stages']),8)
            self.assertEqual(a['paired_hold_comparisons'],[])
            self.assertIn('not_started',(p/'report.md').read_text())


if __name__=='__main__':unittest.main()
