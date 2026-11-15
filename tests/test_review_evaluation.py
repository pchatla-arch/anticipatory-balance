"""Numerical contract checks for paired evidence; no GPU/JAX dependency."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from foresight.eval import statistics as S


def table(ids, fell, miss=None, brace=None, contact=None, progress=None):
  n = len(ids)
  return {'scenario_id': np.asarray(ids), 'fell': np.asarray(fell, bool),
          'is_miss': np.zeros(n, bool) if miss is None else np.asarray(miss, bool),
          'contact': np.ones(n, bool) if contact is None else np.asarray(contact, bool),
          'brace_steps_total': np.zeros(n) if brace is None else np.asarray(brace),
          'progress_m': np.ones(n) if progress is None else np.asarray(progress)}


def readiness_traj(T=30, contact_at=10):
  tr = {'brace_active': np.ones((T,1), bool),
        'pre_pelvis_pos': np.zeros((T,1,3)), 'pre_torso_up': np.ones((T,1)),
        'pre_feet_contact': np.ones((T,1,2), bool),
        'pre_terminated': np.zeros((T,1), bool), 'impact_contact': np.zeros((T,1), bool)}
  tr['pre_pelvis_pos'][:,:,2] = .65
  if contact_at is not None: tr['impact_contact'][contact_at:] = True
  return tr


class ReadinessTests(unittest.TestCase):
  def test_window_requires_full_physical_duration(self):
    t=readiness_traj(contact_at=9)
    r=S.readiness_metrics(t,.02,.65)
    self.assertFalse(r['ready_at_contact'][0])  # 0.18s before interval9 is not0.2s
    t['impact_contact'][:]=False;t['impact_contact'][10:]=True
    r=S.readiness_metrics(t,.02,.65)
    self.assertTrue(r['ready_at_contact'][0])
    self.assertAlmostEqual(r['readiness_window_start_s'][0],0.)
    self.assertAlmostEqual(r['readiness_window_complete_s'][0],.2)

  def test_late_postcontact_posture_is_not_ready_at_contact(self):
    t=readiness_traj(contact_at=5);t['pre_torso_up'][:10]=0.
    r=S.readiness_metrics(t,.02,.65)
    self.assertTrue(r['readiness_ever_after_switch'][0])
    self.assertFalse(r['ready_at_contact'][0])
    self.assertTrue(np.isnan(r['readiness_precontact_complete_s'][0]))

  def test_window_requires_brace_active_every_sample(self):
    t=readiness_traj(contact_at=10);t['brace_active'][5]=False
    r=S.readiness_metrics(t,.02,.65)
    self.assertFalse(r['ready_at_contact'][0])
    self.assertAlmostEqual(r['readiness_window_complete_s'][0],.32)

  def test_terminated_pose_cannot_establish_readiness(self):
    t=readiness_traj(contact_at=10);t['pre_terminated'][9:]=True
    r=S.readiness_metrics(t,.02,.65)
    self.assertFalse(r['ready_at_contact'][0])
    self.assertFalse(r['readiness_ever_after_switch'][0])

  def test_no_contact_is_not_ready_at_contact(self):
    r=S.readiness_metrics(readiness_traj(contact_at=None),.02,.65)
    self.assertFalse(r['readiness_contact_observed'][0]);self.assertFalse(r['ready_at_contact'][0])
    self.assertTrue(r['readiness_ever_after_switch'][0])

  def test_foot_contact_loss_breaks_precontact_window(self):
    t=readiness_traj(contact_at=10);t['pre_feet_contact'][9,0,0]=False
    self.assertFalse(S.readiness_metrics(t,.02,.65)['ready_at_contact'][0])


class ReferenceEntryTests(unittest.TestCase):
  def fixture(self, x, vx):
    T=len(x)
    z=np.zeros((T,1,3));p=z.copy();p[:,0,0]=x;p[:,:,2]=.8
    v=z.copy();v[:,0,0]=vx
    return {'pre_proj_pos':p,'pre_proj_vel':v,'pre_pelvis_pos':z.copy(),
            'pre_pelvis_vel':z.copy(),'pre_proj_launched':np.ones((T,1),bool),
            'pre_terminated':np.zeros((T,1),bool),'pre_impact_contact':np.zeros((T,1),bool)}

  def test_high_speed_crossing_within_single_step_is_kept(self):
    t=self.fixture([.50],[-8.])
    self.assertAlmostEqual(S.reference_entry_times(t,.02)[0],.00625)

  def test_relative_robot_motion_is_included(self):
    t=self.fixture([.50],[-4.]);t['pre_pelvis_vel'][0,0,0]=1.
    self.assertAlmostEqual(S.reference_entry_times(t,.02)[0],.01)

  def test_postcontact_crossing_is_not_used(self):
    t=self.fixture([.5],[-8.]);t['pre_impact_contact'][:]=True
    self.assertTrue(np.isnan(S.reference_entry_times(t,.02)[0]))

  def test_no_future_crossing_stays_missing(self):
    self.assertTrue(np.isnan(S.reference_entry_times(self.fixture([.5],[8.]),.02)[0]))

  def test_helper_does_not_mutate_visibility(self):
    t=self.fixture([.5],[-8.]);t['pre_terminated'][:]=True
    S.reference_entry_times(t,.02)
    self.assertTrue(t['pre_proj_launched'].all())


class PoolingAndPairingTests(unittest.TestCase):
  def test_pooling_uses_counts_not_mean_batch_rates(self):
    a=table(['a'],[1]);b=table(list('bcdefghij'),[0]*9)
    out=S.summarize_episodes(S.concatenate_tables([a,b]))
    self.assertEqual(out['n_hits'],10);self.assertAlmostEqual(out['fall_rate_hits'],.1)
    self.assertIsNone(out['false_prep_rate'])

  def test_metrics_use_separate_denominators(self):
    out=S.summarize_episodes(table(['a','b','c','d'],[1,0,1,0],miss=[0,0,1,1],brace=[1,2,3,0],contact=[1,0,0,0]))
    self.assertEqual(out['n_hits'],2);self.assertEqual(out['n_contact_hits'],1)
    self.assertEqual(out['fall_rate_hits'],.5);self.assertEqual(out['fall_rate_given_contact'],1.)
    self.assertEqual(out['false_prep_rate'],.5)

  def test_pairing_aligns_ids_and_reports_discordance(self):
    a=table(['a','b','c'],[1,0,1]);b=table(['c','a','b'],[1,0,1])
    r=S.paired_comparison(a,b)['failure_hits']
    self.assertEqual(r['a_only'],1);self.assertEqual(r['b_only'],1);self.assertEqual(r['both'],1)
    self.assertEqual(r['difference'],0.)

  def test_pairing_rejects_duplicates_and_label_mismatch(self):
    a=table(['a','a'],[0,0]);b=table(['a','b'],[0,0])
    with self.assertRaises(ValueError):S.paired_comparison(a,b)
    a=table(['a','b'],[0,0],miss=[1,0])
    with self.assertRaises(ValueError):S.paired_comparison(a,b)

  def test_paired_interval_reproduces_saved_large_effect(self):
    a=np.r_[np.ones(159),np.zeros(11),np.ones(200),np.zeros(142)]
    b=np.r_[np.zeros(159),np.ones(11),np.ones(200),np.zeros(142)]
    r=S.paired_difference(a,b)
    self.assertAlmostEqual(r['difference'],.2890625)
    np.testing.assert_allclose(r['normal95'],[.245842356666,.332282643334])

  def test_csv_preserves_string_ids_and_float_precision(self):
    with tempfile.TemporaryDirectory() as d:
      p=Path(d)/'x.csv'
      S.write_csv(p,{'scenario_id':np.array(['seed1_env0']),'t':np.array([1.2345678901234567]),'flag':np.array([True])})
      with p.open() as f:
        r=list(csv.DictReader(f))[0]
      self.assertEqual(r['scenario_id'],'seed1_env0')
      self.assertEqual(float(r['t']),1.2345678901234567);self.assertEqual(r['flag'],'1')

  def test_strict_json_maps_unavailable_values_to_null(self):
    with tempfile.TemporaryDirectory() as d:
      p=Path(d)/'x.json';S.write_json(p,{'x':np.array([np.nan,np.inf,1.]),'b':np.bool_(True)})
      self.assertEqual(json.loads(p.read_text()),{'x':[None,None,1.],'b':True})


class ProtocolTests(unittest.TestCase):
  def test_new_model_requires_matching_contract(self):
    expect={'timing_reference':'reference_entry','hold_s':.5,'single_commit':True}
    m={'schema_version':'foresight.review_response.v1','protocol':expect}
    self.assertTrue(S.validate_response_protocol(m,expect)['matched'])
    with self.assertRaises(ValueError):S.validate_response_protocol(m,{**expect,'hold_s':.7})

  def test_old_model_requires_explicit_override(self):
    with self.assertRaises(ValueError):S.validate_response_protocol({}, {'hold_s':.5})
    self.assertFalse(S.validate_response_protocol({}, {'hold_s':.5}, True)['matched'])


class EpisodeAccountingTests(unittest.TestCase):
  def fixture(self):
    T,n=5,2
    pre=np.zeros((T,n,3));pre[:,:,0]=np.arange(T)[:,None]*.1
    post=pre.copy();post[:,:,0]+=.1
    fwd=np.zeros_like(pre);fwd[:,:,0]=1.
    terminated=np.zeros((T,n),bool);terminated[3:,0]=True
    active=np.array([[1,0],[1,0],[1,0],[0,1],[0,1]],bool)
    traj={'pelvis_pos':post,'pre_pelvis_pos':pre,'pre_pelvis_fwd':fwd,
          'pre_terminated':terminated,'pre_com':pre.copy(),
          'pre_feet_pos':np.zeros((T,n,2,3)),'pre_proj_vel':np.ones((T,n,3)),
          'brace_active':active}
    info={k:np.zeros(n) for k in ('proj_mass','proj_speed','proj_momentum','t_hit_nominal',
          'proj_launch_step','proj_lateral','term_reason')}
    info.update(fell=np.array([1,0],bool),proj_is_miss=np.array([0,1],bool),
                proj_launched=np.ones(n,bool),impact_contact=np.array([1,0],bool),
                contact_step=np.array([2,-1]),fell_before_launch=np.zeros(n,bool),
                term_step=np.array([2,-1]))
    return info,traj

  def test_progress_includes_first_interval_and_stops_after_failure(self):
    info,traj=self.fixture()
    out=S.episode_table(info,traj,.02)
    np.testing.assert_allclose(out['progress_m'],[.3,.5])
    self.assertAlmostEqual(out['actual_onset_time_s'][0],0.)
    self.assertAlmostEqual(out['actual_release_time_s'][0],.06)
    self.assertAlmostEqual(out['contact_time_lower_s'][0],.04)
    self.assertAlmostEqual(out['contact_time_upper_s'][0],.06)
    self.assertTrue(out['brace_at_contact'][0])

  def test_no_contact_has_no_fabricated_preimpact_state(self):
    info,traj=self.fixture()
    out=S.episode_table(info,traj,.02)
    self.assertTrue(np.isnan(out['com_x_pre'][1]))
    self.assertTrue(np.isnan(out['contact_time_lower_s'][1]))
    self.assertFalse(out['brace_at_contact'][1])

  def test_censoring_uses_end_of_contact_interval(self):
    info,traj=self.fixture();info['fell'][:]=False
    out=S.episode_table(info,traj,.02)
    self.assertAlmostEqual(out['recovery_observed_s'][0],.04)
    self.assertTrue(out['censored'][0]);self.assertFalse(out['censored'][1])


if __name__=='__main__':
  unittest.main()
