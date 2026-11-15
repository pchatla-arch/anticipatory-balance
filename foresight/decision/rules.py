# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Decision rules with an explicit single-threat commitment contract.

Corrected default: at first BRACE, lock a release deadline from the current predicted
arrival plus hold, round to the control grid, and enforce minimum/maximum duration.
New observations cannot slide this deadline. One preparation is allowed per projectile
episode by default. Stale captures cannot initiate a new preparation. Historical
stale-driven retention is available only as release_mode='legacy'.

The response model remains an empirical surrogate, not an exact dynamics model.
Corrected lead/wait/VOI rules share its per-sample lead and independent mass integration.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from foresight.belief.particle_filter import BeliefStats
from foresight.decision import contracts as C


class ResponseModel(NamedTuple):
  a_cont: float
  b_cont: float
  a_brace: float
  b_brace: float
  a_brace_lead: float = 0.0
  b_brace_lead: float = 0.0
  c_lead: float = 0.0
  lead_max: float = 0.5
  has_lead: bool = False

  def p_fall(self, J, brace: bool, lead=None):
    logJ = jp.log(jp.maximum(J, 1e-3))
    if not brace:
      return jax.nn.sigmoid(self.a_cont + self.b_cont * logJ)
    if lead is None or not self.has_lead:
      return jax.nn.sigmoid(self.a_brace + self.b_brace * logJ)
    lead = jp.clip(lead, 0.0, self.lead_max)
    return jax.nn.sigmoid(self.a_brace_lead + self.b_brace_lead * logJ + self.c_lead * lead)


class RuleConfig(NamedTuple):
  kind: str = "belief_commit"
  theta: float = 0.5
  p_min: float = 0.5
  c_fall: float = 1.0
  c_prep: float = 0.2
  c_track: float = 0.1  # per second of BRACE, which interrupts walking
  tau_brace: float = 0.22  # historical heuristic; not validated readiness-at-impact
  hold: float = 0.5
  commit_steps: int = 15
  release_after: float = 0.3  # legacy only
  stale_steps: int = 50
  release_on_unwant: bool = False  # legacy ablation only
  wait_steps: int = 5  # legacy metadata; corrected waiting uses detector timestamps
  dt: float = 0.02
  release_mode: str = "deadline"  # deadline | legacy
  max_hold_s: float = 4.0
  single_commit: bool = True
  horizon_steps: int = 0  # 0 means unspecified; evaluators supply their actual horizon


class CtrlMemory(NamedTuple):
  braced: jax.Array
  brace_steps: jax.Array
  last_det_step: jax.Array
  n_switches: jax.Array
  last_capture_step: jax.Array = -1000
  release_step: jax.Array = -1
  onset_step: jax.Array = -1
  completed: jax.Array = False
  release_reason: jax.Array = 0  # event: 1 deadline,2 maxhold,3 legacy stale,4 legacy request,5 legacy passed
  deadline_capped: jax.Array = False


def validate_config(cfg: RuleConfig):
  if cfg.release_mode not in ("deadline", "legacy"):
    raise ValueError("release_mode must be deadline or legacy")
  if cfg.kind not in ("reactive", "always", "ttc_threshold", "belief_commit", "belief_lead", "belief_wait", "belief_voi"):
    raise ValueError("unknown decision rule: " + cfg.kind)
  if min(cfg.c_fall, cfg.c_prep, cfg.c_track) < 0 or cfg.stale_steps < 0 or cfg.horizon_steps < 0:
    raise ValueError("negative cost, staleness, or horizon")
  C.locked_duration(0.0, cfg.hold, cfg.commit_steps, cfg.dt, cfg.max_hold_s)


def init_memory() -> CtrlMemory:
  return CtrlMemory(jp.array(False), jp.int32(0), jp.int32(-1000), jp.int32(0),
                    jp.int32(-1000), jp.int32(-1), jp.int32(-1), jp.array(False),
                    jp.int32(0), jp.array(False))


def _risks(rm: ResponseModel, b: BeliefStats, use_lead=True):
  """Average over the independent mass prior BEFORE choosing an action.

  All posterior observation branches retain the same mass prior. Only geometry
  particle weights are updated by position observations.
  """
  lead = b.t_hit_samples if b.t_hit_samples is not None else jp.full_like(b.weights, b.t_hit)
  lead = jp.maximum(jp.where(jp.isfinite(lead), lead, 0.0), 0.0)
  if b.state_samples is None:
    return rm.p_fall(b.momentum_samples, False), rm.p_fall(b.momentum_samples, True, lead if use_lead else None)
  speed = jp.linalg.norm(b.state_samples[:, 3:], axis=-1)
  momentum = speed[:, None] * b.mass_samples[None, :]
  pc = jp.mean(rm.p_fall(momentum, False), axis=1)
  pb = jp.mean(rm.p_fall(momentum, True, lead[:, None] if use_lead else None), axis=1)
  return pc, pb


def _duration(cfg, t):
  return C.locked_duration(jp.where(jp.isfinite(t), t, 0.0), cfg.hold,
                           cfg.commit_steps, cfg.dt, cfg.max_hold_s, jp)


def _costs(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats, use_lead=True):
  pc, pb = _risks(rm, b, use_lead)
  wh = b.weights * b.hit_samples
  cost_cont = jp.sum(wh * pc) * cfg.c_fall
  cost_brace = jp.sum(wh * pb) * cfg.c_fall + (cfg.c_prep + cfg.c_track) * _duration(cfg, b.t_hit)
  return cost_brace, cost_cont


def decision_values(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats, la=None):
  cb, cc = _costs(cfg, rm, b, cfg.kind != "belief_commit")
  cw = jp.asarray(jp.inf)
  if la is not None:
    # Each observation has an analytic point posterior (or moment-matched empty
    # posterior), with fresh geometry samples. Average mass/geometry costs first,
    # then choose one future action for that observable branch.
    later_brace, later_cont = jax.vmap(lambda bk: _costs(cfg, rm, bk))(la.future)
    pc_now, _ = _risks(rm, b)
    prefix = jp.sum(la.prefix_hit_weights * pc_now) * cfg.c_fall
    # Continuing to walk during the wait has no lost-tracking/preparation charge.
    cw = prefix + jp.sum(la.branch_probability * jp.minimum(later_cont, later_brace))
    if cfg.horizon_steps > 0:
      cw = jp.where(la.delivery_step >= cfg.horizon_steps, cc, cw)
  return cb, cc, cw


def evaluate(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats, la=None):
  cb, cc, cw = decision_values(cfg, rm, b, la)
  finite_hit = jp.isfinite(b.t_hit) & (b.p_hit > 0)
  if cfg.kind == "reactive":
    want = jp.array(False)
  elif cfg.kind == "always":
    want = b.p_hit > cfg.p_min
  elif cfg.kind == "ttc_threshold":
    want = b.pe_hit & (b.pe_t_hit < cfg.theta)
  elif cfg.kind == "belief_commit":
    if cfg.release_mode == "legacy":
      # Preserve the original hard-gate request for release-only ablations.
      dP = rm.p_fall(b.momentum_samples, False) - rm.p_fall(b.momentum_samples, True)
      wh = b.weights * b.hit_samples
      benefit = jp.sum(wh * dP) * cfg.c_fall
      t = jp.where(jp.isfinite(b.t_hit), b.t_hit, 0.0)
      want = (benefit > cfg.c_prep * (t + cfg.hold) + cfg.c_track * t) & finite_hit & (t > cfg.tau_brace)
    else:
      want = (cb < cc) & finite_hit & (b.t_hit > cfg.tau_brace)
  elif cfg.kind == "belief_lead":
    want = (cb < cc) & finite_hit
  elif cfg.kind in ("belief_wait", "belief_voi"):
    if la is None:
      raise ValueError("waiting rules require a forecast")
    want = (cb < jp.minimum(cc, cw)) & finite_hit
  else:
    raise ValueError(cfg.kind)
  return want, (cb, cc, cw)


def request(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats, la=None):
  return evaluate(cfg, rm, b, la)[0]


def step(cfg: RuleConfig, rm: ResponseModel, mem: CtrlMemory, b: BeliefStats, det_valid,
         step_idx, la=None, capture_step=None, want_override=None) -> tuple[CtrlMemory, jax.Array]:
  step_idx = jp.asarray(step_idx, jp.int32)
  capture_step = step_idx if capture_step is None else capture_step
  usable = det_valid & (capture_step >= 0) & (capture_step <= step_idx)
  last_det = jp.where(usable, step_idx, mem.last_det_step)
  last_capture = jp.where(usable, jp.maximum(capture_step, mem.last_capture_step), mem.last_capture_step)
  want = request(cfg, rm, b, la) if want_override is None else want_override
  committed = mem.braced & (mem.brace_steps < cfg.commit_steps)
  if cfg.release_mode == "legacy":
    passed = jp.isfinite(b.t_hit) & (b.t_hit < -cfg.release_after)
    stale = (step_idx - last_det) > cfg.stale_steps
    keep = ~passed & ~stale
    if cfg.release_on_unwant:
      keep &= want
    stay = committed | (mem.braced & keep)
    start = ~mem.braced & want
    braced = stay | start
    reason = jp.where(stale, 3, jp.where(passed, 5, 4))
    deadline, capped = mem.release_step, mem.deadline_capped
  else:
    stale = (step_idx - last_capture) > cfg.stale_steps
    eligible = ~mem.completed if cfg.single_commit else jp.array(True)
    start = ~mem.braced & eligible & ~stale & want & jp.isfinite(b.t_hit)
    duration = _duration(cfg, b.t_hit)
    n_hold = jp.rint(duration / cfg.dt).astype(jp.int32)
    deadline = jp.where(start, step_idx + n_hold, mem.release_step)
    capped = jp.where(start, b.t_hit + cfg.hold > cfg.max_hold_s, mem.deadline_capped)
    stay = mem.braced & (committed | (step_idx < mem.release_step))
    braced = stay | start
    reason = jp.where(mem.deadline_capped, 2, 1)
  released = mem.braced & ~braced
  completed = mem.completed | released
  brace_steps = jp.where(braced, jp.where(mem.braced, mem.brace_steps + 1, 1), 0)
  n_switches = mem.n_switches + (braced != mem.braced).astype(jp.int32)
  onset = jp.where(start, step_idx, mem.onset_step)
  updated = CtrlMemory(braced, brace_steps, last_det, n_switches, last_capture, deadline,
                       onset, completed, jp.where(released, reason, 0).astype(jp.int32), capped)
  return updated, braced
