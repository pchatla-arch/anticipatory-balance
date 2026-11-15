"""Session H release semantics and termination-aware evidence, without GPU work."""
import ast
import csv
from pathlib import Path
import tempfile
import types
import unittest

import jax
import jax.numpy as jp
import numpy as np

from foresight.belief.particle_filter import BeliefStats
from foresight.decision import rules as RU
from foresight.eval import session_h_evidence as ST


SOURCE = Path(__file__).resolve().parents[1]/'scripts/eval_decision_frontier.py'
# Load pure evidence functions without creating global absl flags or importing the
# simulator. The actual Session G rule implementation is imported above for JIT tests.
tree = ast.parse(SOURCE.read_text())
defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and
        n.name in ('_episode_evidence', '_summarize', '_write_episodes')]
ns = dict(np=np, Path=Path, csv=csv, RU=RU, ST=ST)
exec(compile(ast.Module(body=defs, type_ignores=[]), str(SOURCE), 'exec'), ns)


def fixture():
    requested=np.array([[0,0],[1,0],[1,0],[0,1],[1,1],[0,0]],bool)
    table=dict(fell=np.array([False,True]),is_miss=np.array([False,True]),
               contact=np.array([True,False]),contact_step=np.array([4,-1]),
               term_step=np.array([-1,2]),progress_m=np.array([1.,.2]),
               censored=np.array([True,False]),brace_steps_total=requested.sum(0),
               brace_onset_step=np.array([1,3]),false_prep=np.array([False,True]),
               mass=np.array([8.,12.]),speed=np.array([3.,4.]),t_hit_nominal=np.array([2.,2.]))
    reasons=np.array([[0,0],[0,0],[0,0],[1,0],[1,0],[2,2]])
    trace=dict(brace_active=requested,ctrl_release_reason=reasons,
               ctrl_t_pass=np.tile(np.arange(6)[:,None]*-.2,(1,2)))
    return table,trace


class HoldEvidenceTests(unittest.TestCase):
    def test_after_failure_braces_do_not_count_as_preparation(self):
        tab,traj=fixture()
        out=ns['_episode_evidence'](tab,traj,200,.02)
        np.testing.assert_array_equal(out['brace_steps_total'],[3,0])
        np.testing.assert_array_equal(out['brace_steps_total_legacy'],[3,2])
        np.testing.assert_array_equal(out['brace_steps_after_termination'],[0,2])
        np.testing.assert_array_equal(out['brace_onset_step'],[1,-1])
        np.testing.assert_array_equal(out['false_prep'],[False,False])
        np.testing.assert_array_equal(out['false_prep_legacy'],[False,True])
        # Inputs are retained so evidence extraction cannot alter simulation state.
        np.testing.assert_array_equal(tab['brace_onset_step'],[1,3])

    def test_actual_transitions_distinguish_stale_and_passed_releases(self):
        tab,traj=fixture()
        out=ns['_episode_evidence'](tab,traj,200,.02)
        np.testing.assert_array_equal(out['brace_segments'],[2,0])
        np.testing.assert_array_equal(out['brace_releases'],[2,0])
        np.testing.assert_array_equal(out['passed_releases'],[1,0])
        np.testing.assert_array_equal(out['stale_releases'],[1,0])
        np.testing.assert_array_equal(out['brace_release_step'],[3,-1])
        self.assertAlmostEqual(out['t_pass_at_first_release'][0],-.6)
        self.assertTrue(np.isnan(out['t_pass_at_first_release'][1]))

    def test_terminating_interval_is_included(self):
        tab,traj=fixture()
        traj['brace_active'][2,1]=True
        out=ns['_episode_evidence'](tab,traj,200,.02)
        self.assertEqual(out['brace_steps_total'][1],1)
        self.assertEqual(out['brace_onset_step'][1],2)

    def test_horizon_truncation_is_not_a_controller_release(self):
        tab,traj=fixture()
        traj['brace_active'][:,0]=True
        out=ns['_episode_evidence'](tab,traj,200,.02)
        self.assertTrue(out['brace_horizon_truncated'][0])
        self.assertEqual(out['brace_releases'][0],0)
        self.assertEqual(out['brace_release_step'][0],-1)

    def test_contact_records_the_brace_in_that_physical_interval(self):
        tab,traj=fixture()
        out=ns['_episode_evidence'](tab,traj,200,.02)
        np.testing.assert_array_equal(out['brace_at_contact'],[True,False])
        np.testing.assert_array_equal(out['contact_before_termination'],[True,False])

    def test_pooled_metrics_use_corrected_duration_and_keep_legacy(self):
        tab,traj=fixture()
        out=ns['_episode_evidence'](tab,traj,200,.02)
        row=ns['_summarize'](out,.02)
        self.assertAlmostEqual(row['braced_s'],.03)
        self.assertAlmostEqual(row['braced_s_legacy'],.05)
        self.assertEqual(row['n_episodes_rebraced'],1)
        self.assertEqual(row['n_releases_stale'],1)
        self.assertEqual(row['false_prep_rate'],0.)

    def test_csv_preserves_scenario_ids_and_float_precision(self):
        tab,traj=fixture()
        out=ns['_episode_evidence'](tab,traj,200,.02)
        out['progress_m'][0]=.123456789123
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'episodes.csv'
            ns['_write_episodes'](p,out,'belief_lead',.1,5,0)
            with p.open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(rows[0]['scenario_id'],'seed200_env00000')
            self.assertEqual(float(rows[0]['progress_m']),.123456789123)
            self.assertEqual(rows[0]['latency_steps'],'5')

    def test_config_names_distinct_execution_cost_and_oracle_holds(self):
        text=SOURCE.read_text()
        self.assertIn('"oracle_hold", 0.5',text)
        self.assertIn('"hold": _HOLD.value, "release_after": _RELAFTER.value',text)
        self.assertIn('"release_after_does_not_apply": True',text)
        self.assertIn('"response_model_refit_for_hold": False',text)


class OriginalReleaseSemanticsTests(unittest.TestCase):
    def belief(self,t_pass=-.4,pe_hit=False):
        return BeliefStats(jp.float32(0),jp.inf,jp.float32(0),jp.float32(0),jp.float32(0),jp.int32(4),
                           jp.asarray(pe_hit),jp.float32(.1),jp.float32(10),jp.zeros(3),jp.zeros(3),
                           jp.ones(2),jp.ones(2),jp.zeros(2,bool),jp.ones(2)/2,jp.float32(t_pass))

    def rule_step(self,release_after=.3,steps=15,last_det=40,at=40,t_pass=-.4,det=True,pe_hit=False):
        cfg=RU.RuleConfig(kind='ttc_threshold',release_after=release_after)
        rm=RU.ResponseModel(0.,0.,0.,0.)
        mem=RU.CtrlMemory(jp.array(True),jp.int32(steps),jp.int32(last_det),jp.int32(1),jp.int32(0))
        f=jax.jit(lambda m,b,d,t:RU.step(cfg,rm,m,b,d,t))
        return f(mem,self.belief(t_pass,pe_hit),jp.asarray(det),jp.int32(at))

    def test_longer_release_waits_for_later_closest_approach_threshold(self):
        _,short=self.rule_step(.3)
        _,long=self.rule_step(.6)
        self.assertFalse(bool(short)); self.assertTrue(bool(long))

    def test_minimum_commitment_precedes_passed_release(self):
        mem,braced=self.rule_step(.3,steps=14,t_pass=-.8)
        self.assertTrue(bool(braced))
        self.assertEqual(int(mem.brace_steps),15)

    def test_staleness_can_preempt_requested_long_hold(self):
        mem,braced=self.rule_step(1.,last_det=0,at=51,t_pass=-.2,det=False)
        self.assertFalse(bool(braced))
        self.assertEqual(int(mem.release_reason),RU.RELEASE_STALE)

    def test_session_g_allows_a_new_brace_after_release(self):
        cfg=RU.RuleConfig(kind='ttc_threshold')
        rm=RU.ResponseModel(0.,0.,0.,0.)
        mem=RU.CtrlMemory(jp.array(False),jp.int32(0),jp.int32(40),jp.int32(2),jp.int32(1))
        mem,braced=RU.step(cfg,rm,mem,self.belief(-.4,True),True,41)
        self.assertTrue(bool(braced))
        self.assertEqual(int(mem.n_switches),3)


if __name__ == '__main__':
    unittest.main()
