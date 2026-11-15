"""Session H depth evidence and compiled observation path, without a GPU."""
import ast
import csv
from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jp
import numpy as np

from foresight.eval import session_h_evidence as SE

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'scripts/eval_depth_closed_loop.py'
TREE = ast.parse(SOURCE.read_text())
NS = dict(np=np, jp=jp, jax=jax, csv=csv, Path=Path)
for node in TREE.body:
    if isinstance(node, ast.FunctionDef) and node.name in ('pixel_dirs','make_detector','_pool','_paired','_episode_csv'):
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(SOURCE),'exec'),NS)


def table(n, *, hits=None, falls=None, fa=0, frames=0):
    return dict(fell=np.asarray(falls if falls is not None else [False]*n),
                is_miss=~np.asarray(hits if hits is not None else [True]*n),
                contact=np.ones(n,bool),brace_steps=np.arange(n),brace_steps_raw=np.arange(n)+1,
                progress_m=np.ones(n),n_detections=np.zeros(n,int),braced_prelaunch=np.zeros(n,bool),
                prelaunch_detections=np.full(n,fa),prelaunch_frames=np.full(n,frames),
                prelaunch_detections_raw=np.full(n,fa),prelaunch_frames_raw=np.full(n,frames),
                censored=np.zeros(n,bool),scenario_seed=np.ones(n,int),scenario_index=np.arange(n),
                mass=np.full(n,4.),speed=np.full(n,3.))


class EvidenceTests(unittest.TestCase):
    def test_pool_counts_instead_of_average_conditional_rates(self):
        a=table(1,falls=[True],fa=1,frames=1)
        b=table(3,falls=[False]*3,fa=0,frames=10)
        _,m=NS['_pool']([a,b],.02)
        self.assertEqual(m['fall_rate_hits'],.25)
        self.assertEqual(m['false_alarm_rate'],1/31)
        self.assertEqual(m['false_alarm_rate_raw'],1/31)
        self.assertEqual(m['n_falls_hits'],1)
        self.assertEqual(m['n_hits'],4)
        self.assertIsNone(m['false_prep_rate'])
        self.assertGreater(m['braced_s_raw'],m['braced_s'])

    def test_unobserved_groups_are_null(self):
        _,m=NS['_pool']([table(2,hits=[False,False])],.02)
        self.assertIsNone(m['fall_rate_hits'])
        self.assertIsNone(m['false_alarm_rate'])

    def test_pairing_keeps_discordance_and_checks_ids(self):
        a=table(4,falls=[True,False,True,False]); b=table(4,falls=[False,True,True,False])
        p=NS['_paired'](a,b)
        self.assertEqual((p['a_only'],p['b_only']),(1,1))
        self.assertEqual(p['difference_a_minus_b'],0.)
        b['scenario_index']=np.array([1,0,2,3])
        with self.assertRaises(ValueError): NS['_paired'](a,b)

    def test_csv_retains_ids_and_full_float_precision(self):
        a=table(1);a['progress_m'][0]=.123456789012345
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episodes.csv'
            NS['_episode_csv'](p,a,'depth','belief_lead',.05)
            with p.open() as stream: row=list(csv.DictReader(stream))[0]
            self.assertEqual(row['scenario_id'],'seed1_env00000')
            self.assertEqual(float(row['progress_m']),a['progress_m'][0])

    def test_float32_progress_pool_agrees_with_saved_episode_evidence(self):
        # Reactive progress from the first uploaded GPU gate. Reducing these
        # float32 values before promotion differs from the saved CSV mean by
        # 6.89e-8 m, enough to fail the deliberately strict evidence validator.
        progress=np.asarray([
            1.5633423328399658, .0919535756111145, 1.6154876947402954,
            .8632070422172546, .7335505485534668, .8451490998268127,
            2.16880464553833, 1.9157700538635254, .6903193593025208,
            1.9477788209915161, .2832622528076172, 2.5334737300872803,
            2.9313483238220215, .39469870924949646, 1.5092474222183228,
            2.4433841705322266],dtype=np.float32)
        a=table(len(progress));a['progress_m']=progress
        _,pooled=NS['_pool']([a],.02)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'episodes.csv'
            NS['_episode_csv'](p,a,'none','reactive',0.)
            with p.open() as stream: rows=list(csv.DictReader(stream))
        restored={**a,'brace_steps_total':a['brace_steps'],
                  'progress_m':np.asarray([float(row['progress_m']) for row in rows])}
        expected=SE.summarize_episodes(restored,.02)['progress_m']
        self.assertNotEqual(float(progress.mean()),expected)
        self.assertEqual(pooled['progress_m'],expected)
        # Batch partitioning cannot change the summary of the same saved values.
        parts=[{k:v[:7] for k,v in a.items()},{k:v[7:] for k,v in a.items()}]
        self.assertEqual(NS['_pool'](parts,.02)[1]['progress_m'],expected)

    def test_script_refuses_overwrite_and_marks_completion_only_at_end(self):
        s=SOURCE.read_text()
        self.assertIn('if outdir.exists() and any(outdir.iterdir())',s)
        self.assertLess(s.index('np.savez_compressed'),s.index('"status": "complete"'))
        self.assertIn('controller_cache[spec] = cached, jax.jit(cached.step_with_detections)',s)
        self.assertIn('per_ep.setdefault(("none", "reactive", 0.)',s)
        self.assertIn('brace_steps += applied_before_failure',s)


class SensorTests(unittest.TestCase):
    def test_parked_projectile_is_out_of_range_in_built_scene(self):
        import xml.etree.ElementTree as ET
        from foresight.envs.g1_impact.scene import build_scene
        with tempfile.TemporaryDirectory() as d:
            path,_=build_scene(Path(d))
            xml=ET.parse(path)
            body=xml.find('.//body[@name="projectile"]')
            position=[float(x) for x in body.attrib["pos"].split()]
            self.assertLess(position[0],-70.)
            self.assertGreater(position[2],0.3)

    def test_median_detector_rejects_parked_range(self):
        dirs=NS['pixel_dirs'](8,90.)
        detect=NS['make_detector'](dirs,.15)
        cam=jp.array([[0.,0.,1.]])
        matrix=jp.array([[[0.,0.,-1.],[-1.,0.,0.],[0.,1.,0.]]])
        valid,centre=detect(jp.full((1,8,8),80.),cam,matrix,jp.zeros((1,3)))
        self.assertFalse(bool(valid[0]))
        self.assertTrue(np.isfinite(np.asarray(centre)).all())

    def test_actual_controller_jits_for_h_methods_and_two_worlds(self):
        from foresight.belief import particle_filter as PF
        from foresight.decision import rules as RU
        from foresight.decision.controller import AnticipationController
        from foresight.perception import detections as D
        for kind in ('ttc_threshold','belief_lead','belief_wait','belief_voi'):
            ctrl=AnticipationController(D.DetectorConfig(period=5,latency=5),PF.PFConfig(),RU.RuleConfig(kind=kind),RU.ResponseModel(-4.,1.,-5.,1.))
            carry=ctrl.init(2,jax.random.PRNGKey(0)); call=jax.jit(ctrl.step_with_detections)
            for t in (0,5,10):
                carry,brace,logs=call(carry,jp.full(2,t>0),jp.array([[2.,0.,.9],[3.,0.,.9]]),jp.full(2,.03),
                                      jp.full(2,t-5),jp.array([[0.,0.,.8],[0.,0.,.8]]),jp.array([[.5,0.,0.],[.5,0.,0.]]),
                                      jp.int32(t),jax.random.PRNGKey(t))
                jax.block_until_ready(brace)
                self.assertEqual(brace.shape,(2,))
                self.assertTrue(np.isfinite(np.asarray(logs['p_hit'])).all())


if __name__=='__main__': unittest.main()
