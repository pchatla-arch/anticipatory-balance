"""CPU fixtures for Session H stage validity gates."""
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from foresight.eval import session_h_evidence as ST

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('h_validator',ROOT/'scripts/validate_session_h.py')
V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
SHA='a'*40


class ValidationTests(unittest.TestCase):
    def fixture(self,kind='depth'):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup);p=Path(d.name)
        (p/'episodes').mkdir()
        config={'schema':ST.SCHEMA_VERSION,'dt':.02,'steps':10,'period':5,'release_after':.3}
        if kind=='frontier':config['rule']={'release_after':config.pop('release_after')}
        digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        prov={'source_commit':SHA,'source_dirty':False,'config_sha256':digest,
              'artifacts':{'policy':{'files':[{'sha256':'b'*64,'bytes':1}]}}}
        rows=[]
        for i,method in enumerate(('reactive','belief_lead')):
            tab=dict(scenario_id=np.array(['seed1_env00000','seed1_env00001','seed1_env00002']),
                     scenario_seed=np.ones(3,int),scenario_index=np.arange(3),mass=np.array([4.,5.,6.]),
                     speed=np.array([3.,4.,5.]),t_hit_nominal=np.ones(3),
                     is_miss=np.array([False,False,True]),fell=np.array([True,False,False]),
                     contact=np.array([True,True,False]),censored=np.zeros(3,bool),
                     contact_step=np.array([1,1,-1]),term_step=np.array([3,-1,-1]),
                     progress_m=np.array([1.,2.,3.]),brace_steps_total=np.full(3,0 if i==0 else 2))
            metrics=ST.summarize_episodes(tab,.02)
            meta={'method':method,'param':float(i)*.05}
            if kind=='depth':
                meta['source']='none' if i==0 else 'depth'
                tab['brace_steps']=tab.pop('brace_steps_total')
                for suffix in ('','_raw'):
                    tab['prelaunch_frames'+suffix]=np.full(3,2 if i else 0)
                    tab['prelaunch_detections'+suffix]=np.full(3,1 if i else 0)
                    metrics['prelaunch_frames'+suffix]=6 if i else 0
                    metrics['prelaunch_detections'+suffix]=3 if i else 0
                    metrics['false_alarm_rate'+suffix]=.5 if i else None
            else:meta['latency_steps']=0
            rows.append({**meta,**metrics})
            with (p/'episodes'/f'condition{i}.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=[*meta,*tab]);w.writeheader()
                for k in range(3):w.writerow({**meta,**{name:np.asarray(v[k]).item() for name,v in tab.items()}})
        ST.write_json(p/'config.json',{**config,'provenance':prov})
        ST.write_json(p/'summary.json',{'schema_version':ST.SCHEMA_VERSION,'status':'complete',
                          'config':config,'provenance':prov,'episodes_per_condition':3,'rows':rows})
        (p/'episodes.npz').write_bytes(b'fixture archive')
        ST.write_json(p/'paired.json',[])
        return p

    def check(self,p,kind='depth',**kw):
        args=dict(stage_dir=p,kind=kind,episodes_per_condition=3,conditions=2,expected_release=.3,source_commit=SHA)
        args.update(kw);return V.validate(**args)

    def rewrite_csv(self,p,change):
        with p.open(newline='') as f:r=list(csv.DictReader(f))
        change(r)
        with p.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(r[0]));w.writeheader();w.writerows(r)

    def test_valid_depth_allows_nonzero_false_alarms(self):
        self.assertTrue(self.check(self.fixture())['passed'])

    def test_valid_frontier(self):
        self.assertTrue(self.check(self.fixture('frontier'),'frontier')['passed'])

    def test_wrong_count(self):
        with self.assertRaisesRegex(ValueError,'episodes_per_condition'):self.check(self.fixture(),episodes_per_condition=4)

    def test_stale_hash(self):
        p=self.fixture();cfg=V.load(p/'config.json');cfg['dt']=.03
        ST.write_json(p/'config.json',cfg)
        with self.assertRaises(ValueError):self.check(p)

    def test_duplicate_id(self):
        p=self.fixture();self.rewrite_csv(p/'episodes/condition1.csv',lambda r:r[1].update(scenario_id=r[0]['scenario_id']))
        with self.assertRaisesRegex(ValueError,'duplicate'):self.check(p)

    def test_invalid_release(self):
        with self.assertRaisesRegex(ValueError,'release_after'):self.check(self.fixture(),expected_release=.6)

    def test_nonfinite_progress(self):
        p=self.fixture();self.rewrite_csv(p/'episodes/condition1.csv',lambda r:r[0].update(progress_m='nan'))
        with self.assertRaisesRegex(ValueError,'progress_m'):self.check(p)

    def test_prelaunch_numerator_cannot_exceed_frames(self):
        p=self.fixture();self.rewrite_csv(p/'episodes/condition1.csv',lambda r:r[0].update(prelaunch_detections='3'))
        with self.assertRaisesRegex(ValueError,'prelaunch'):self.check(p)

    def test_wrong_pooled_rate(self):
        p=self.fixture();s=V.load(p/'summary.json');s['rows'][1]['fall_rate_hits']=.9;ST.write_json(p/'summary.json',s)
        with self.assertRaisesRegex(ValueError,'fall_rate_hits'):self.check(p)

    def test_cli_writes_failure_audit(self):
        p=self.fixture()
        with contextlib.redirect_stdout(io.StringIO()):
            code=V.main(['--stage-dir',str(p),'--kind','depth','--episodes-per-condition','5','--conditions','2','--expected-release','.3','--source-commit',SHA])
        self.assertEqual(code,1);self.assertFalse(V.load(p/'audit.json')['passed'])


if __name__=='__main__':unittest.main()
