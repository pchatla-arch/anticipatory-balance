"""Analytical controller contracts; real-JAX integration tests skip if unavailable.

Run: python -m unittest discover -s tests -p test_review_belief_rules.py -v
No simulator, package installation, policy checkpoint or GPU is needed.
"""
from __future__ import annotations

import math
import unittest

import numpy as np

from foresight.decision import contracts as C

try:
  import jax
  import jax.numpy as jp
except ImportError:
  jax = None

if jax is not None:
  from foresight.belief import particle_filter as PF
  from foresight.decision import rules as RU
  from foresight.decision.controller import AnticipationController
  from foresight.perception import detections as D


class NumericalContracts(unittest.TestCase):
  def test_next_delivery_is_first_valid_capture_grid_event(self):
    for period in (1, 2, 5):
      for latency in (0, 1, 7, 15):
        for step in range(41):
          delivery, capture, wait = C.next_delivery(step, period, latency)
          expected = next(t for t in range(step + 1, step + latency + period + 2)
                          if t >= latency and (t - latency) % period == 0)
          self.assertEqual(delivery, expected)
          self.assertEqual(capture, delivery - latency)
          self.assertEqual(wait, delivery - step)

  def test_next_delivery_can_have_past_capture_stamp(self):
    self.assertEqual(tuple(C.next_delivery(4, 5, 5)), (5, 0, 1))
    self.assertEqual(tuple(C.next_delivery(8, 5, 7)), (12, 5, 4))

  def test_capture_uncertainty_is_not_counted_twice(self):
    for lag in (0.0, 0.1, 0.3):
      P = C.transported_covariance(.03, 3.0, lag)
      H = C.measurement_matrix(lag)
      np.testing.assert_allclose(H @ P @ H.T, np.eye(3) * .03**2, atol=2e-15)
      np.testing.assert_allclose(P[:3, 3:], np.eye(3) * lag * 9)
      self.assertGreater(np.linalg.eigvalsh(P).min(), 0)

  def test_delayed_update_matches_update_then_transport(self):
    initial_mu = np.array([2., .2, .9, -3., .1, 0.])
    initial_cov = np.diag([.1, .2, .3, 2., 3., 4.])
    z = np.array([1.9, .3, .85])
    age = .3
    F = C.transition(age)
    m_old, p_old = C.gaussian_update(initial_mu, initial_cov, z, 0., .03)
    m_now, p_now = C.gaussian_update(F @ initial_mu, F @ initial_cov @ F.T, z, age, .03)
    np.testing.assert_allclose(m_now, F @ m_old, atol=1e-12)
    np.testing.assert_allclose(p_now, F @ p_old @ F.T, atol=1e-12)

  def test_joseph_update_preserves_positive_covariance(self):
    m, p = np.zeros(6), C.transported_covariance(.03, 3., .3)
    for i in range(40):
      F = C.transition(.02)
      m, p = C.gaussian_update(F @ m, F @ p @ F.T, np.ones(3)*.01, .3, .03)
      np.testing.assert_allclose(p, p.T, atol=1e-14)
      self.assertGreaterEqual(np.linalg.eigvalsh(p).min(), -1e-13)

  def test_duration_enforces_grid_minimum_and_cap(self):
    values = C.locked_duration(np.array([0.0, .001, .25, 10.]), .1, 15, .02, 4.)
    np.testing.assert_allclose(values, [.3, .3, .36, 4.])
    self.assertTrue(np.all(values >= .3))
    self.assertTrue(np.all(values <= 4.))
    self.assertLessEqual(float(C.locked_duration(10.,.1,15,.02,.31)),.31)

  def test_invalid_duration_rejected(self):
    with self.assertRaises(ValueError):
      C.locked_duration(.5, .5, 15, .02, .1)

  def test_missing_observation_can_change_geometry_belief(self):
    prior = np.array([.5, .5])
    detection_probability = np.array([1., 0.])
    empty = C.normalize_weights(prior * (1-detection_probability))
    np.testing.assert_array_equal(empty, [0., 1.])

  def test_old_capture_indistinguishable_states_stay_indistinguishable(self):
    x = np.array([[1., 0., .8, -1., 0., 0.], [.6, 0., .8, -6., 0., 0.]])
    captured = x[:, :3] - .08*x[:, 3:]
    np.testing.assert_allclose(captured[0], captured[1])
    posterior = C.observation_weights(np.array([.5, .5]), np.ones(2), captured,
                                      np.array([1.07, .01, .8]), np.full(2, .03))
    np.testing.assert_allclose(posterior, [.5, .5], atol=1e-12)

  def test_zero_visibility_weights_are_finite(self):
    w = C.observation_weights(np.array([.5, .5]), np.zeros(2), np.zeros((2, 3)),
                              np.zeros(3), np.ones(2)*.03)
    self.assertTrue(np.isfinite(w).all())
    self.assertAlmostEqual(w.sum(), 1.)


@unittest.skipIf(jax is None, "real JAX is unavailable; no simulator/JIT validation claimed")
class JaxContracts(unittest.TestCase):
  def setUp(self):
    self.pfc = PF.PFConfig(n_particles=2)
    self.rm = RU.ResponseModel(math.log(4), 0., -math.log(4), 0., -math.log(4), 0., -1., 2., True)
    self.det = D.DetectorConfig(period=5, latency=5, sigma1=0., miss_prob=0., fov_deg=180.)

  def belief(self, positions=(1., 1.1), velocities=(-1., -1.)):
    x = jp.array([[positions[0], 0., .8, velocities[0], 0., 0.],
                  [positions[1], 0., .8, velocities[1], 0., 0.]])
    st = PF.PFState(x.mean(0), jp.eye(6)*1e-6, jp.array([4.,16.]), jp.int32(1), jp.array(True))
    b = PF.stats_from_samples(self.pfc, st, x, jp.array([.5,.5]), jp.zeros(3), jp.zeros(3))
    return st, b

  def forecast(self, st, b, step=4, observe=True, detector=None):
    return PF.lookahead(self.pfc, st, b, detector or self.det, step, jp.zeros(3),
                        jp.zeros(3), jp.array([1.,0.,0.]), jax.random.PRNGKey(3), observe=observe)

  def test_stochastic_delayed_model_is_rejected(self):
    with self.assertRaises(ValueError):
      PF.init_state(self.pfc._replace(process_vel_sigma=.02))
    PF.init_state(self.pfc._replace(process_vel_sigma=.02, inference_mode="legacy"))

  def test_world_height_gate_is_independent_of_pelvis_vertical_velocity(self):
    st,b = self.belief()
    moving = PF.stats_from_samples(self.pfc,st,b.state_samples,b.weights,jp.zeros(3),jp.array([0.,0.,5.]))
    np.testing.assert_array_equal(moving.hit_samples,b.hit_samples)
    np.testing.assert_allclose(moving.t_hit_samples,b.t_hit_samples)

  def broad_state(self):
    cfg = self.pfc._replace(n_particles=128)
    st = PF.PFState(jp.array([0.,2.,.8,0.,0.,0.]),
                    jp.diag(jp.array([1.,.001,.001,.01,.01,.01])),
                    jp.linspace(4.,16.,128),jp.int32(4),jp.array(True))
    return cfg,st

  def test_uninformative_empty_observation_is_exact_noop(self):
    cfg,st=self.broad_state()
    new,applied,pnone=PF.update_missing(cfg,st,self.det._replace(latency=0,miss_prob=1.),jp.array(True),
                                        jp.zeros(3),jp.zeros(3),jp.array([1.,0.,0.]),jax.random.PRNGKey(12))
    self.assertFalse(bool(applied))
    self.assertEqual(float(pnone),1.)
    np.testing.assert_array_equal(new.mu,st.mu)
    np.testing.assert_array_equal(new.cov,st.cov)
    np.testing.assert_array_equal(new.m,st.m)

  def test_informative_empty_update_preserves_mass_and_psd(self):
    cfg,st=self.broad_state()
    new,applied,pnone=PF.update_missing(cfg,st,self.det._replace(latency=0,miss_prob=0.),jp.array(True),
                                        jp.zeros(3),jp.zeros(3),jp.array([1.,0.,0.]),jax.random.PRNGKey(12))
    self.assertTrue(bool(applied))
    self.assertGreater(float(pnone),.2)
    self.assertLess(float(pnone),.8)
    self.assertLess(float(new.mu[0]),-.4)
    self.assertGreaterEqual(np.linalg.eigvalsh(np.asarray(new.cov)).min(),-1e-7)
    np.testing.assert_array_equal(new.m,st.m)
    self.assertEqual(int(new.n_obs),int(st.n_obs))

  def test_empty_update_only_on_real_scheduled_delivery(self):
    cfg,st=self.broad_state()
    ctrl=AnticipationController(self.det._replace(latency=0),cfg,RU.RuleConfig(kind="reactive"),self.rm)
    carry=ctrl.init(1,jax.random.PRNGKey(0))
    carry=carry._replace(pf=jax.tree_util.tree_map(lambda a:a[None],st))
    def run(t,capture):
      return ctrl.step_with_detections(carry,jp.array([False]),jp.zeros((1,3)),jp.array([.03]),
                                        jp.array([capture]),jp.zeros((1,3)),jp.zeros((1,3)),t,
                                        jax.random.PRNGKey(4),pelvis_fwd=jp.array([[1.,0.,0.]]))
    predicted=PF.predict(cfg,st)
    for t,capture in ((1,1),(5,-1)):
      updated,_,logs=run(t,capture)
      self.assertFalse(bool(logs['empty_observation'][0]))
      np.testing.assert_allclose(updated.pf.mu[0],predicted.mu,atol=1e-7)
      np.testing.assert_allclose(updated.pf.cov[0],predicted.cov,atol=1e-7)
    updated,_,logs=run(5,5)
    self.assertTrue(bool(logs['empty_observation'][0]))
    self.assertTrue(bool(logs['empty_update_applied'][0]))

  def test_empty_observation_cannot_initialize_track(self):
    st=PF.init_state(self.pfc)
    new,applied,_=PF.update_missing(self.pfc,st,self.det,jp.array(True),jp.zeros(3),jp.zeros(3),
                                    jp.array([1.,0.,0.]),jax.random.PRNGKey(3))
    self.assertFalse(bool(new.initialized))
    self.assertFalse(bool(applied))

  def test_impossible_capture_stamp_does_not_initialize(self):
    st = PF.init_state(self.pfc)
    updated = PF.update(self.pfc, st, jp.array(True), jp.array([2.,0.,.8]), jp.array(.03),
                        jp.int32(11), jp.int32(10), jp.zeros(3), jax.random.PRNGKey(0))
    self.assertFalse(bool(updated.initialized))

  def test_forecast_uses_past_capture_position(self):
    st,b = self.belief((1.,.6),(-1.,-6.))
    la = self.forecast(st,b)
    self.assertEqual(int(la.delivery_step),5)
    self.assertEqual(int(la.capture_step),0)
    self.assertEqual(int(la.wait_steps),1)
    np.testing.assert_allclose(la.posterior_weights[1:], .5, atol=2e-5)
    F=C.transition(.02)
    _,expected=C.gaussian_update(F@np.asarray(st.mu),F@np.asarray(st.cov)@F.T,
                                 np.zeros(3),.1,.03)
    for covariance in np.asarray(la.posterior_covariance[1:]):
      np.testing.assert_allclose(covariance,expected,rtol=2e-5,atol=1e-8)

  def test_single_point_posterior_does_not_reveal_velocity(self):
    cfg=PF.PFConfig(n_particles=128)
    st=PF.PFState(jp.array([2.,0.,.8,-1.,0.,0.]),
                   jp.diag(jp.array([.09,.09,.01,9.,9.,9.])),
                   jp.linspace(4.,16.,128),jp.int32(1),jp.array(True))
    b=PF.stats(cfg,st,jp.zeros(3),jp.zeros(3),jax.random.PRNGKey(7))
    det=self.det._replace(period=5,latency=1)
    la=PF.lookahead(cfg,st,b,det,0,jp.zeros(3),jp.zeros(3),jp.array([1.,0.,0.]),
                    jax.random.PRNGKey(8),k=4)
    # The next delivered point was captured NOW. Present p and v are independent,
    # so that point cannot reduce velocity variance even with a narrow .03m noise.
    for covariance in np.asarray(la.posterior_covariance[1:]):
      np.testing.assert_allclose(covariance[3:,3:],9.*np.eye(3),atol=2e-5)
    velocity=np.asarray(la.future.state_samples[1:,:,3:])
    self.assertTrue(np.all(velocity.std(axis=1)>1.5))

  def test_forecast_empty_branch_matches_deployed_moment_update(self):
    cfg,st=self.broad_state()
    det=self.det._replace(latency=0,miss_prob=0.)
    rng=jax.random.PRNGKey(9)
    b=PF.stats(cfg,st,jp.zeros(3),jp.zeros(3),jax.random.PRNGKey(10))
    la=PF.lookahead(cfg,st,b,det,0,jp.zeros(3),jp.zeros(3),jp.array([1.,0.,0.]),rng,k=2)
    F=C.transition(.1,jp)
    future=st._replace(mu=F@st.mu,cov=F@st.cov@F.T)
    expected,_,_=PF.update_missing(cfg,future,det,jp.array(True),jp.zeros(3),jp.zeros(3),
                                  jp.array([1.,0.,0.]),jax.random.split(rng,4)[3])
    np.testing.assert_allclose(la.posterior_mean[0],expected.mu,atol=1e-6)
    np.testing.assert_allclose(la.posterior_covariance[0],expected.cov,atol=1e-6)

  def test_empty_observation_branch_is_retained(self):
    st,b = self.belief()
    la = self.forecast(st,b,detector=self.det._replace(miss_prob=.2))
    self.assertAlmostEqual(float(la.branch_probability[0]), .2, places=6)
    self.assertAlmostEqual(float(la.branch_probability.sum()),1.,places=6)

  def test_blind_and_observation_wait_agree_when_detection_impossible(self):
    st,b = self.belief()
    det = self.det._replace(miss_prob=1.)
    blind = self.forecast(st,b,observe=False,detector=det)
    observed = self.forecast(st,b,observe=True,detector=det)
    cfg = RU.RuleConfig(kind="belief_voi")
    vb = RU.decision_values(cfg,self.rm,b,blind)[2]
    vo = RU.decision_values(cfg,self.rm,b,observed)[2]
    self.assertAlmostEqual(float(vb),float(vo),places=6)

  def test_contact_during_wait_is_not_erased(self):
    st,b = self.belief((.50,.52))
    la = self.forecast(st,b,step=0,detector=self.det._replace(latency=0))
    self.assertAlmostEqual(float(la.prefix_hit_weights.sum()),1.,places=6)
    self.assertAlmostEqual(float(la.branch_probability.sum()),0.,places=6)
    cw = RU.decision_values(RU.RuleConfig(kind="belief_voi"),self.rm,b,la)[2]
    self.assertAlmostEqual(float(cw),.8,places=6)

  def test_wait_after_horizon_equals_continue(self):
    st,b = self.belief()
    la = self.forecast(st,b,step=4)
    _,cc,cw = RU.decision_values(RU.RuleConfig(kind="belief_voi",horizon_steps=5),self.rm,b,la)
    self.assertAlmostEqual(float(cc),float(cw),places=6)

  def test_deadline_does_not_slide_and_single_commit_does_not_restart(self):
    _,b = self.belief()
    b=b._replace(p_hit=jp.array(1.),t_hit=jp.array(.5))
    cfg=RU.RuleConfig(kind="always",hold=.5)
    mem=RU.init_memory()
    for t in range(80):
      changing=b._replace(t_hit=jp.array(.5 if t==0 else 3.0))
      mem,braced=RU.step(cfg,self.rm,mem,changing,jp.array(True),t,capture_step=t)
      self.assertEqual(int(mem.release_step),50)
      self.assertEqual(bool(braced),t<50)
      if t==50:self.assertEqual(int(mem.release_reason),1)
    self.assertEqual(int(mem.n_switches),2)

  def test_old_capture_cannot_initiate_on_fresh_delivery(self):
    _,b=self.belief()
    cfg=RU.RuleConfig(kind="always",p_min=0.)
    mem,braced=RU.step(cfg,self.rm,RU.init_memory(),b,jp.array(True),100,capture_step=0)
    self.assertFalse(bool(braced))
    self.assertEqual(int(mem.last_det_step),100)
    self.assertEqual(int(mem.last_capture_step),0)

  def test_mass_prior_risk_is_independent_of_geometry_weight(self):
    _,b=self.belief()
    rm=self.rm._replace(b_cont=1.,b_brace_lead=1.)
    pc,_=RU._risks(rm,b)
    self.assertAlmostEqual(float(pc[0]),float(pc[1]),places=6)
    # Equal speeds have identical mass-integrated risk even with different m_i.
    old=rm.p_fall(b.momentum_samples,False)
    self.assertNotAlmostEqual(float(old[0]),float(old[1]),places=4)

  def test_real_jit_vmap_controller_accepts_all_corrected_methods(self):
    for method in ("belief_lead","belief_wait","belief_voi"):
      ctrl=AnticipationController(self.det,self.pfc,RU.RuleConfig(kind=method,horizon_steps=50),self.rm)
      carry=ctrl.init(2,jax.random.PRNGKey(0))
      advance=jax.jit(ctrl.step_with_detections)
      carry,brace,logs=advance(carry,jp.ones(2,bool),jp.array([[2.,0.,.8],[2.1,0.,.8]]),
                              jp.full(2,.03),jp.zeros(2,jp.int32),jp.zeros((2,3)),
                              jp.zeros((2,3)),jp.int32(5),jax.random.PRNGKey(1))
      jax.block_until_ready(brace)
      self.assertTrue(np.isfinite(np.asarray(carry.pf.cov)).all())
      self.assertEqual(brace.shape,(2,))
      self.assertEqual(int(logs['next_delivery_step'][0]),10)

  def test_default_particle_count_in_compiled_batched_scan(self):
    det=self.det._replace(latency=7,period=5,miss_prob=0.)
    ctrl=AnticipationController(det,PF.PFConfig(),RU.RuleConfig(kind='belief_voi',horizon_steps=40),self.rm)
    initial=ctrl.init(2,jax.random.PRNGKey(21))
    pelvis=jp.zeros((2,3)); fwd=jp.tile(jp.array([1.,0.,0.]),(2,1))
    def advance(carry,t):
      pos=jp.array([[3.,0.,.8],[3.,.2,.8]])-jp.array([.04,0.,0.])[None]*t
      info={'obj_pos':pos,'proj_launched':jp.ones(2,bool)}
      carry,braced,logs=ctrl.step(carry,info,pelvis,fwd,pelvis,t,jax.random.fold_in(jax.random.PRNGKey(22),t))
      return carry,(braced,logs['cost_brace'],logs['cost_continue'],logs['cost_wait'],logs['det_valid'])
    final,outputs=jax.jit(lambda c:jax.lax.scan(advance,c,jp.arange(18)))(initial)
    jax.block_until_ready(final.pf.cov)
    self.assertTrue(np.isfinite(np.asarray(final.pf.mu)).all())
    self.assertTrue(np.isfinite(np.asarray(final.pf.cov)).all())
    for cost in outputs[1:4]:self.assertTrue(np.isfinite(np.asarray(cost)).all())
    self.assertEqual(np.flatnonzero(np.asarray(outputs[4])[:,0]).tolist(),[7,12,17])


if __name__ == '__main__':
  unittest.main()
