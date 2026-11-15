# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Anticipation controller = detector + particle filter + decision rule, batched over envs.

Plugs into foresight.eval.rollout.rollout(..., controller=...): the rollout calls
  carry = controller.init(n, rng)
  carry, brace, logs = controller.step(carry, info, pelvis_pos, pelvis_fwd, step_idx, rng)
every control step, where everything is batched over the n environments.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from foresight.perception import detections as D
from foresight.belief import particle_filter as PF
from foresight.decision import rules as RU


class Carry(NamedTuple):
  det: D.DetectorState
  pf: PF.PFState
  mem: RU.CtrlMemory


class AnticipationController:

  def __init__(self, det_cfg: D.DetectorConfig, pf_cfg: PF.PFConfig, rule_cfg: RU.RuleConfig,
               response: RU.ResponseModel):
    self.det_cfg, self.pf_cfg, self.rule_cfg, self.rm = det_cfg, pf_cfg, rule_cfg, response
    self.lookahead_k = 16

  def init(self, n: int, rng) -> Carry:
    one = Carry(D.init_state(self.det_cfg), PF.init_state(self.pf_cfg), RU.init_memory())
    return jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (n,) + jp.shape(x)), one)

  def _lookahead(self, pf, pelvis_pos, pelvis_vel, step_idx, rng):
    """(observation-branch stats, no-detection stats, P(no detection)) for the lookahead
    rules, or None. The wait runs to the sensor's next delivery (period phase), the
    hypothetical detection carries the sensor's latency, and the no-detection probability is
    the sensor's miss probability (field-of-view loss is not forecast)."""
    if self.rule_cfg.kind not in ("belief_voi", "belief_wait"):
      return None
    k1, k2 = jax.random.split(rng)
    per = self.det_cfg.period
    wait = per - (step_idx % per)                       # steps to the next delivery
    wait = jp.where(wait <= 0, per, wait)
    sigma_next = self.det_cfg.sigma0 + self.det_cfg.sigma1 * jp.linalg.norm((pf.mu[:3] - pelvis_pos)[:2])
    la_none = PF.lookahead(self.pf_cfg, pf, sigma_next, wait, self.det_cfg.latency, pelvis_pos, pelvis_vel, k1,
                           self.lookahead_k, observe=False)
    la_obs = la_none
    if self.rule_cfg.kind == "belief_voi":
      la_obs = PF.lookahead(self.pf_cfg, pf, sigma_next, wait, self.det_cfg.latency, pelvis_pos, pelvis_vel, k2,
                            self.lookahead_k, observe=True)
    return la_obs, la_none, jp.float32(self.det_cfg.miss_prob)

  def _step_one(self, c: Carry, obj_pos, pelvis_pos, pelvis_fwd, pelvis_vel, launched, step_idx, rng):
    k_det, k_pf, k_mc = jax.random.split(rng, 3)
    det_st, det = D.step(self.det_cfg, c.det, obj_pos, pelvis_pos, pelvis_fwd, launched, k_det)
    pf = PF.predict(self.pf_cfg, c.pf)
    pf = PF.update(self.pf_cfg, pf, det.valid, det.pos, det.sigma, det.capture_step, step_idx,
                   pelvis_pos, k_pf)
    b = PF.stats(self.pf_cfg, pf, pelvis_pos, pelvis_vel, k_mc)
    la = self._lookahead(pf, pelvis_pos, pelvis_vel, step_idx, k_mc)
    mem, brace = RU.step(self.rule_cfg, self.rm, c.mem, b, det.valid, step_idx, la)
    logs = {"p_hit": b.p_hit, "t_hit": b.t_hit, "t_hit_std": b.t_hit_std,
            "momentum_mean": b.momentum_mean, "ess": b.ess, "n_obs": b.n_obs,
            "det_valid": det.valid, "pe_t_hit": b.pe_t_hit,
            "pe_pos": b.pe_pos, "pe_vel": b.pe_vel, "t_pass": b.t_pass, "release_reason": mem.release_reason}
    return Carry(det_st, pf, mem), brace, logs

  def _step_one_det(self, c: Carry, det_valid, det_pos, det_sigma, capture_step, pelvis_pos, pelvis_vel,
                    step_idx, rng):
    """Same as _step_one but with a detection supplied from outside (e.g. a depth detector)."""
    k_pf, k_mc = jax.random.split(rng)
    pf = PF.predict(self.pf_cfg, c.pf)
    pf = PF.update(self.pf_cfg, pf, det_valid, det_pos, det_sigma, capture_step, step_idx, pelvis_pos, k_pf)
    b = PF.stats(self.pf_cfg, pf, pelvis_pos, pelvis_vel, k_mc)
    la = self._lookahead(pf, pelvis_pos, pelvis_vel, step_idx, k_mc)
    mem, brace = RU.step(self.rule_cfg, self.rm, c.mem, b, det_valid, step_idx, la)
    logs = {"p_hit": b.p_hit, "t_hit": b.t_hit, "det_valid": det_valid, "n_obs": b.n_obs,
            "t_pass": b.t_pass, "release_reason": mem.release_reason}
    return Carry(c.det, pf, mem), brace, logs

  def step_with_detections(self, carry: Carry, det_valid, det_pos, det_sigma, capture_step, pelvis_pos,
                           pelvis_vel, step_idx, rng):
    n = pelvis_pos.shape[0]
    keys = jax.random.split(rng, n)
    f = jax.vmap(self._step_one_det, in_axes=(0, 0, 0, 0, 0, 0, 0, None, 0))
    return f(carry, det_valid, det_pos, det_sigma, capture_step, pelvis_pos, pelvis_vel, step_idx, keys)

  def step(self, carry: Carry, info, pelvis_pos, pelvis_fwd, pelvis_vel, step_idx, rng):
    n = pelvis_pos.shape[0]
    keys = jax.random.split(rng, n)
    f = jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, 0, 0, None, 0))
    return f(carry, info["obj_pos"], pelvis_pos, pelvis_fwd, pelvis_vel, info["proj_launched"],
             step_idx, keys)
