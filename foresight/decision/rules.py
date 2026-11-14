# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Decision rules for CONTINUE vs BRACE on top of the particle belief, with commitment.

Every rule generates an entry request. The default single-projectile protocol latches
a bounded release deadline from the entry forecast, holds for at least `commit_steps`,
and permits one commitment per episode. Later forecasts do not revise the deadline.
The `legacy` and `stale_only` modes reproduce the historical release behaviors for
comparison. See commitment.py for the exact contract and reason codes.

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
from foresight.decision import commitment as C
from foresight.decision.commitment import CtrlMemory


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
  tau_brace: float = 0.22          # s, assumed heuristic; not validated readiness
  hold: float = 0.5                # post-forecast margin for cost and default deadline
  commit_steps: int = 15           # 0.3 s at 50 Hz
  release_after: float = 0.3       # retained for config compatibility; unused
  stale_steps: int = 50            # entry freshness / historical release timeout
  dt: float = 0.02
  release_mode: str = "deadline"  # deadline | legacy | stale_only
  max_brace_seconds: float = 3.0  # finite duration cap; minimum commitment wins


def init_memory() -> CtrlMemory:
  return C.init_memory(jp)


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
  mem = C.transition(cfg, mem, request(cfg, rm, b), C.forecast_ttc(cfg.kind, b),
                     det_valid, step_idx, xp=jp)
  return mem, mem.braced
