# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Decision rules for CONTINUE vs BRACE on top of the particle belief, with commitment.

Every rule maps (belief stats, controller memory) -> brace request. The common wrapper
applies the commitment: once BRACE starts it is held for at least `commit_steps`, and
released only when the belief says the object has passed (expected impact time negative by
more than `release_after`) or no detection has arrived for `stale_steps`. The request
decides when to START; it is not re-evaluated to decide whether to stay (an earlier version
did, and the belief rule's `t_hit > tau_brace` start condition then released the brace
0.22 s before contact once the commitment had expired: a mode switch at the worst moment).

Rules
  reactive        never brace
  always          brace as soon as any particle predicts a hit (p_hit > p_min)
  ttc_threshold   brace when the point-estimate (mean particle) says hit and TTC < theta
  belief_commit   the myopic commit rule:
                    p_hit * E_b[dP(J)] * c_fall  >  c_prep * (E_b[t_hit] + hold) + c_track * E_b[t_hit]
                    and E_b[t_hit] > tau_brace
                  dP(J) = P(fall | CONTINUE, J) - P(fall | BRACE, J) from the fitted
                  response model (logistic in log J), averaged over hit particles so the
                  mass mixture enters through J = m|v|.
All JAX, per environment.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jp

from foresight.belief.particle_filter import BeliefStats


class ResponseModel(NamedTuple):
  """P(fall | J) = sigmoid(a + b * log J) for each action; for BRACE, optionally conditioned
  on the lead with which the brace was started: sigmoid(a_l + b_l log J + c_l min(lead, lead_max))."""
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
    l = jp.clip(lead, 0.0, self.lead_max)
    return jax.nn.sigmoid(self.a_brace_lead + self.b_brace_lead * logJ + self.c_lead * l)


class RuleConfig(NamedTuple):
  kind: str = "belief_commit"      # reactive | always | ttc_threshold | belief_commit | oracle
  theta: float = 0.5               # ttc_threshold: brace when TTC < theta (s)
  p_min: float = 0.5               # always / ttc_threshold: minimum hit probability
  c_fall: float = 1.0
  c_prep: float = 0.2              # per second braced
  c_track: float = 0.1             # per second of lost tracking (approximation)
  tau_brace: float = 0.22          # s, measured sustained readiness
  hold: float = 0.5                # s kept braced after the expected impact
  commit_steps: int = 15           # 0.3 s at 50 Hz
  release_after: float = 0.3       # s past expected impact
  stale_steps: int = 50            # release if no detection for 1 s
  release_on_unwant: bool = False  # old behaviour: release when the start condition fails
  wait_steps: int = 5              # belief_voi: the wait is one detection period
  dt: float = 0.02


class CtrlMemory(NamedTuple):
  braced: jax.Array        # bool
  brace_steps: jax.Array   # int32, steps since brace start
  last_det_step: jax.Array # int32
  n_switches: jax.Array    # int32


def init_memory() -> CtrlMemory:
  return CtrlMemory(jp.array(False), jp.int32(0), jp.int32(-1000), jp.int32(0))


def _costs(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats, t_shift: float = 0.0):
  """Expected cost of BRACE-now and of CONTINUE-forever under belief b, with the expected
  time to contact shifted by t_shift (negative: the belief as it will be t_shift later)."""
  t = jp.where(jp.isfinite(b.t_hit), b.t_hit, 0.0) + t_shift
  t = jp.maximum(t, 0.0)
  wh = b.weights * b.hit_samples
  pf_c = rm.p_fall(b.momentum_samples, False)
  pf_b = rm.p_fall(b.momentum_samples, True, lead=t)
  pf_c = jp.sum(wh * pf_c) / (jp.sum(wh) + 1e-9)
  pf_b = jp.sum(wh * pf_b) / (jp.sum(wh) + 1e-9)
  cost_cont = b.p_hit * pf_c * cfg.c_fall
  cost_brace = b.p_hit * pf_b * cfg.c_fall + cfg.c_prep * (t + cfg.hold) + cfg.c_track * t
  return cost_brace, cost_cont


def request(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats, la=None) -> jax.Array:
  """Does the rule want BRACE now (ignoring commitment)? `la`: lookahead BeliefStats
  (leading axis K = hypothetical next detections) for belief_voi."""
  if cfg.kind == "belief_lead":
    # Myopic rule with the lead-conditioned response model and no time-to-form gate: a
    # late brace is worth whatever the oracle data say it is worth.
    cb, cc = _costs(cfg, rm, b)
    return (cb < cc) & jp.isfinite(b.t_hit) & (b.p_hit > 0)
  if cfg.kind == "belief_voi":
    # One-step lookahead with an explicit observation branch: brace now, or walk for one
    # more detection period, observe, and then take the better of brace/continue.
    cb, cc = _costs(cfg, rm, b)
    dt_wait = cfg.wait_steps * cfg.dt
    cb_k, cc_k = jax.vmap(lambda bk: _costs(cfg, rm, bk))(la)
    later = jp.mean(jp.minimum(cb_k, cc_k))
    # If contact is expected inside the wait, waiting means continuing.
    hit_during_wait = jp.isfinite(b.t_hit) & (b.t_hit < dt_wait)
    cost_wait = jp.where(hit_during_wait, cc, cfg.c_track * dt_wait + later)
    return (cb < jp.minimum(cc, cost_wait)) & jp.isfinite(b.t_hit) & (b.p_hit > 0)
  if cfg.kind == "reactive":
    return jp.array(False)
  if cfg.kind == "always":
    return b.p_hit > cfg.p_min
  if cfg.kind == "ttc_threshold":
    return b.pe_hit & (b.pe_t_hit < cfg.theta)
  if cfg.kind == "belief_commit":
    dP = rm.p_fall(b.momentum_samples, False) - rm.p_fall(b.momentum_samples, True)
    wh = b.weights * b.hit_samples
    dP_mean = jp.sum(wh * dP) / (jp.sum(wh) + 1e-9)
    t = jp.where(jp.isfinite(b.t_hit), b.t_hit, 0.0)
    benefit = b.p_hit * dP_mean * cfg.c_fall
    cost = cfg.c_prep * (t + cfg.hold) + cfg.c_track * t
    return (benefit > cost) & (b.t_hit > cfg.tau_brace) & jp.isfinite(b.t_hit)
  raise ValueError(cfg.kind)


def step(cfg: RuleConfig, rm: ResponseModel, mem: CtrlMemory, b: BeliefStats, det_valid,
         step_idx, la=None) -> tuple[CtrlMemory, jax.Array]:
  """Apply the rule with commitment; returns (memory, brace_active)."""
  last_det = jp.where(det_valid, step_idx, mem.last_det_step)
  want = request(cfg, rm, b, la)
  passed = jp.isfinite(b.t_hit) & (b.t_hit < -cfg.release_after)
  stale = (step_idx - last_det) > cfg.stale_steps
  committed = mem.braced & (mem.brace_steps < cfg.commit_steps)
  # Stay braced while committed; afterwards until the object has passed or gone stale.
  # (cfg.release_on_unwant restores the old behaviour for the ablation.)
  keep = ~passed & ~stale
  if cfg.release_on_unwant:
    keep = keep & want
  stay = committed | (mem.braced & keep)
  start = ~mem.braced & want
  braced = stay | start
  n_sw = mem.n_switches + (braced != mem.braced).astype(jp.int32)
  brace_steps = jp.where(braced, jp.where(mem.braced, mem.brace_steps + 1, 1), 0)
  return CtrlMemory(braced, brace_steps, last_det, n_sw), braced
