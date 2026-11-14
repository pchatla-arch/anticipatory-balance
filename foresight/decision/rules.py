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
  """P(fall | J) = sigmoid(a + b * log J) for each action."""
  a_cont: float
  b_cont: float
  a_brace: float
  b_brace: float

  def p_fall(self, J, brace: bool):
    a, b = (self.a_brace, self.b_brace) if brace else (self.a_cont, self.b_cont)
    return jax.nn.sigmoid(a + b * jp.log(jp.maximum(J, 1e-3)))


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
  dt: float = 0.02


class CtrlMemory(NamedTuple):
  braced: jax.Array        # bool
  brace_steps: jax.Array   # int32, steps since brace start
  last_det_step: jax.Array # int32
  n_switches: jax.Array    # int32


def init_memory() -> CtrlMemory:
  return CtrlMemory(jp.array(False), jp.int32(0), jp.int32(-1000), jp.int32(0))


def request(cfg: RuleConfig, rm: ResponseModel, b: BeliefStats) -> jax.Array:
  """Does the rule want BRACE now (ignoring commitment)?"""
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
         step_idx) -> tuple[CtrlMemory, jax.Array]:
  """Apply the rule with commitment; returns (memory, brace_active)."""
  last_det = jp.where(det_valid, step_idx, mem.last_det_step)
  want = request(cfg, rm, b)
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
