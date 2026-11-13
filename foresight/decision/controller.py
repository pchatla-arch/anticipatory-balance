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

  def init(self, n: int, rng) -> Carry:
    one = Carry(D.init_state(self.det_cfg), PF.init_state(self.pf_cfg), RU.init_memory())
    return jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (n,) + jp.shape(x)), one)

  def _step_one(self, c: Carry, obj_pos, pelvis_pos, pelvis_fwd, pelvis_vel, launched, step_idx, rng):
    k_det, k_pf, k_mc = jax.random.split(rng, 3)
    det_st, det = D.step(self.det_cfg, c.det, obj_pos, pelvis_pos, pelvis_fwd, launched, k_det)
    pf = PF.predict(self.pf_cfg, c.pf)
    pf = PF.update(self.pf_cfg, pf, det.valid, det.pos, det.sigma, det.capture_step, step_idx,
                   pelvis_pos, k_pf)
    b = PF.stats(self.pf_cfg, pf, pelvis_pos, pelvis_vel, k_mc)
    mem, brace = RU.step(self.rule_cfg, self.rm, c.mem, b, det.valid, step_idx)
    logs = {"p_hit": b.p_hit, "t_hit": b.t_hit, "t_hit_std": b.t_hit_std,
            "momentum_mean": b.momentum_mean, "ess": b.ess, "n_obs": b.n_obs,
            "det_valid": det.valid, "pe_t_hit": b.pe_t_hit,
            "pe_pos": b.pe_pos, "pe_vel": b.pe_vel}
    return Carry(det_st, pf, mem), brace, logs

  def step(self, carry: Carry, info, pelvis_pos, pelvis_fwd, pelvis_vel, step_idx, rng):
    n = pelvis_pos.shape[0]
    keys = jax.random.split(rng, n)
    f = jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, 0, 0, None, 0))
    return f(carry, info["obj_pos"], pelvis_pos, pelvis_fwd, pelvis_vel, info["proj_launched"],
             step_idx, keys)
