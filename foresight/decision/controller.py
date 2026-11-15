# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Batched detector, timestamped kinematic belief and frozen-skill selector.

A control decision occurs every physics-control tick. Detector period and latency
control captures/deliveries, not this decision cadence. The future observation model
uses an explicit pelvis-cone approximation even when current observations come from
rendered depth; it must not be described as an exact camera-frustum prediction.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from foresight.perception import detections as D
from foresight.belief import particle_filter as PF
from foresight.decision import rules as RU
from foresight.decision import contracts as C


class Carry(NamedTuple):
  det: D.DetectorState
  pf: PF.PFState
  mem: RU.CtrlMemory


class AnticipationController:

  def __init__(self, det_cfg: D.DetectorConfig, pf_cfg: PF.PFConfig, rule_cfg: RU.RuleConfig,
               response: RU.ResponseModel):
    PF.validate_config(pf_cfg)
    RU.validate_config(rule_cfg)
    if det_cfg.period < 1 or det_cfg.latency < 0 or det_cfg.buffer_len <= det_cfg.latency:
      raise ValueError("invalid detector cadence, latency or buffer capacity")
    if abs(pf_cfg.dt - rule_cfg.dt) > 1e-12:
      raise ValueError("belief and rule control intervals must match")
    if rule_cfg.kind in ("belief_wait", "belief_voi") and pf_cfg.inference_mode != "deterministic":
      raise ValueError("corrected lookahead requires the deterministic timestamped filter")
    self.det_cfg, self.pf_cfg, self.rule_cfg, self.rm = det_cfg, pf_cfg, rule_cfg, response
    self.lookahead_k = 16

  def init(self, n: int, rng) -> Carry:
    one = Carry(D.init_state(self.det_cfg), PF.init_state(self.pf_cfg), RU.init_memory())
    return jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (n,) + jp.shape(x)), one)

  def _analyse(self, c, det, pelvis_pos, pelvis_fwd, pelvis_vel, step_idx, rng):
    k_pf, k_missing, k_mc, k_la = jax.random.split(rng, 4)
    pf = PF.predict(self.pf_cfg, c.pf)
    pf = PF.update(self.pf_cfg, pf, det.valid, det.pos, det.sigma, det.capture_step, step_idx,
                   pelvis_pos, k_pf)
    expected_capture = step_idx - self.det_cfg.latency
    scheduled = (expected_capture >= 0) & (expected_capture % self.det_cfg.period == 0)
    empty = scheduled & ~det.valid & (det.capture_step == expected_capture)
    pf, empty_applied, empty_probability = PF.update_missing(
        self.pf_cfg, pf, self.det_cfg, empty, pelvis_pos, pelvis_vel, pelvis_fwd, k_missing)
    b = PF.stats(self.pf_cfg, pf, pelvis_pos, pelvis_vel, k_mc)
    la = None
    if self.rule_cfg.kind in ("belief_wait", "belief_voi"):
      la = PF.lookahead(self.pf_cfg, pf, b, self.det_cfg, step_idx, pelvis_pos, pelvis_vel,
                        pelvis_fwd, k_la, self.lookahead_k, observe=self.rule_cfg.kind == "belief_voi")
    want, (cb, cc, cw) = RU.evaluate(self.rule_cfg, self.rm, b, la)
    mem, brace = RU.step(self.rule_cfg, self.rm, c.mem, b, det.valid, step_idx, la,
                         capture_step=det.capture_step, want_override=want)
    delivery, capture, wait = C.next_delivery(step_idx, self.det_cfg.period, self.det_cfg.latency, jp)
    la_mean_ess = la_min_ess = la_fallback_probability = jp.asarray(0.0)
    if la is not None:
      ess = 1.0 / jp.maximum(jp.sum(la.posterior_weights ** 2, axis=1), 1e-12)
      active_mass = jp.sum(la.branch_probability)
      la_mean_ess = jp.sum(la.branch_probability * ess) / jp.maximum(active_mass, 1e-12)
      la_min_ess = jp.where(active_mass > 0, jp.min(jp.where(la.branch_probability > 1e-9, ess,
                                                          self.pf_cfg.n_particles)), 0.0)
      la_fallback_probability = jp.sum(la.branch_probability * la.posterior_fallback)
    logs = {"p_hit": b.p_hit, "t_hit": b.t_hit, "t_hit_std": b.t_hit_std,
            "momentum_mean": b.momentum_mean, "ess": b.ess, "n_obs": b.n_obs,
            "det_valid": det.valid, "capture_step": det.capture_step,
            "delivery_step": jp.asarray(step_idx, jp.int32), "pe_t_hit": b.pe_t_hit,
            "t_pass": b.t_pass, "empty_observation": empty,
            "empty_update_applied": empty_applied, "empty_model_probability": empty_probability,
            "pe_pos": b.pe_pos, "pe_vel": b.pe_vel,
            "brace_started": brace & ~c.mem.braced, "brace_released": c.mem.braced & ~brace,
            "release_reason": mem.release_reason, "release_step": mem.release_step,
            "last_capture_step": mem.last_capture_step, "onset_step": mem.onset_step,
            "completed": mem.completed, "request": want,
            "next_delivery_step": delivery, "next_capture_step": capture, "wait_steps": wait,
            "lookahead_mean_ess": la_mean_ess, "lookahead_min_ess": la_min_ess,
            "lookahead_posterior_fallback_probability": la_fallback_probability,
            "cost_brace": cb, "cost_continue": cc, "cost_wait": cw}
    return pf, mem, brace, logs

  def _step_one(self, c: Carry, obj_pos, pelvis_pos, pelvis_fwd, pelvis_vel, launched, step_idx, rng):
    k_det, k_analyse = jax.random.split(rng)
    det_st, det = D.step(self.det_cfg, c.det, obj_pos, pelvis_pos, pelvis_fwd, launched, k_det)
    pf, mem, brace, logs = self._analyse(c, det, pelvis_pos, pelvis_fwd, pelvis_vel, step_idx, k_analyse)
    return Carry(det_st, pf, mem), brace, logs

  def _step_one_det(self, c: Carry, det_valid, det_pos, det_sigma, capture_step, pelvis_pos,
                    pelvis_fwd, pelvis_vel, step_idx, rng):
    det = D.Detection(valid=det_valid, pos=det_pos, sigma=det_sigma, capture_step=capture_step)
    pf, mem, brace, logs = self._analyse(c, det, pelvis_pos, pelvis_fwd, pelvis_vel, step_idx, rng)
    return Carry(c.det, pf, mem), brace, logs

  def step_with_detections(self, carry: Carry, det_valid, det_pos, det_sigma, capture_step, pelvis_pos,
                           pelvis_vel, step_idx, rng, pelvis_fwd=None):
    n = pelvis_pos.shape[0]
    if pelvis_fwd is None:
      speed = jp.linalg.norm(pelvis_vel[:, :2], axis=-1, keepdims=True)
      fallback = jp.broadcast_to(jp.array([1.0, 0.0]), (n, 2))
      heading = jp.where(speed > 1e-6, pelvis_vel[:, :2] / jp.maximum(speed, 1e-6), fallback)
      pelvis_fwd = jp.concatenate((heading, jp.zeros((n, 1))), axis=-1)
    keys = jax.random.split(rng, n)
    f = jax.vmap(self._step_one_det, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, None, 0))
    return f(carry, det_valid, det_pos, det_sigma, capture_step, pelvis_pos, pelvis_fwd,
             pelvis_vel, step_idx, keys)

  def step(self, carry: Carry, info, pelvis_pos, pelvis_fwd, pelvis_vel, step_idx, rng):
    n = pelvis_pos.shape[0]
    keys = jax.random.split(rng, n)
    f = jax.vmap(self._step_one, in_axes=(0, 0, 0, 0, 0, 0, None, 0))
    return f(carry, info["obj_pos"], pelvis_pos, pelvis_fwd, pelvis_vel, info["proj_launched"],
             step_idx, keys)
