# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Shared NumPy/JAX transition for the single-projectile commitment protocol.

The default protocol latches one bounded release deadline from the forecast available
at BRACE entry. It does not estimate passage from the future-hit-only TTC statistic.
Later forecasts, requests, and detector staleness cannot move that deadline. A consumed
flag prevents a second commitment in the same episode; reset memory for a new threat.
This is a deliberate one-threat protocol change, not an isolated one-line ablation.

``legacy`` reproduces request-dependent release, and ``stale_only`` reproduces the
Session D partial fix. Both retain historical retriggering/stale-start behavior. Their
old negative-TTC passage predicate is omitted because the production belief statistic
is positive or infinity and cannot satisfy it.

No simulator state or ground-truth contact enters this transition. ``xp`` is either
numpy or jax.numpy, so CPU tests execute the same transition used by the controller.
"""
from __future__ import annotations

import math
from typing import Any, NamedTuple

RELEASE_NONE = 0
RELEASE_DEADLINE = 1
RELEASE_REQUEST_FALSE = 2
RELEASE_STALE = 3
RELEASE_REASONS = {
    RELEASE_NONE: "none",
    RELEASE_DEADLINE: "latched_deadline",
    RELEASE_REQUEST_FALSE: "request_false",
    RELEASE_STALE: "detector_stale",
}
RELEASE_MODES = ("deadline", "legacy", "stale_only")


class CtrlMemory(NamedTuple):
  braced: Any
  brace_steps: Any
  last_det_step: Any
  n_switches: Any
  release_deadline_step: Any
  has_committed: Any
  release_reason: Any  # event code for this step; zero on steps without a release


def init_memory(xp) -> CtrlMemory:
  return CtrlMemory(xp.asarray(False), xp.int32(0), xp.int32(-1000), xp.int32(0),
                    xp.int32(-1), xp.asarray(False), xp.int32(RELEASE_NONE))


def validate_config(cfg) -> None:
  if cfg.release_mode not in RELEASE_MODES:
    raise ValueError(f"release_mode must be one of {RELEASE_MODES}")
  for name in ("dt", "max_brace_seconds"):
    if not math.isfinite(getattr(cfg, name)) or getattr(cfg, name) <= 0:
      raise ValueError(f"{name} must be finite and positive")
  if not math.isfinite(cfg.hold) or cfg.hold < 0:
    raise ValueError("hold must be finite and nonnegative")
  if isinstance(cfg.commit_steps, bool) or int(cfg.commit_steps) != cfg.commit_steps or cfg.commit_steps < 1:
    raise ValueError("commit_steps must be a positive integer")
  if isinstance(cfg.stale_steps, bool) or int(cfg.stale_steps) != cfg.stale_steps or cfg.stale_steps < 0:
    raise ValueError("stale_steps must be a nonnegative integer")


def forecast_ttc(kind: str, belief):
  """Use the same forecast as the corresponding entry rule."""
  return belief.pe_t_hit if kind == "ttc_threshold" else belief.t_hit


def transition(cfg, mem: CtrlMemory, want, estimated_ttc, det_valid, step_idx, *, xp) -> CtrlMemory:
  """Advance one control tick, returning memory with per-step release reason.

  Fresh means at least one valid detection was delivered within ``stale_steps``;
  a new detection on this exact tick is not required. Deadline duration in ticks is
  max(commit_steps, min(ceil((TTC + hold) / dt), floor(max_brace_seconds / dt))).
  Thus the finite horizon is conservative on the control grid, except that an
  explicitly longer minimum commitment takes precedence. The first active tick is
  the entry tick; the policy releases at the latched absolute deadline, exclusively.
  """
  validate_config(cfg)
  want = xp.asarray(want, dtype=bool)
  det_valid = xp.asarray(det_valid, dtype=bool)
  step_idx = xp.asarray(step_idx, dtype=xp.int32)
  last_det = xp.where(det_valid, step_idx, mem.last_det_step)
  stale = (step_idx - last_det) > cfg.stale_steps

  if cfg.release_mode == "deadline":
    ttc = xp.asarray(estimated_ttc)
    finite_positive = xp.isfinite(ttc) & (ttc > 0)
    fresh = (last_det >= 0) & ~stale
    start = ~mem.braced & ~mem.has_committed & want & fresh & finite_positive
    # Bound before integer conversion, including on invalid forecasts, because JAX
    # evaluates both operands of where. The cap never overrides minimum commitment.
    safe_ttc = xp.where(finite_positive, ttc, 0.0)
    requested = xp.ceil(xp.minimum(safe_ttc + cfg.hold, cfg.max_brace_seconds) / cfg.dt).astype(xp.int32)
    cap_steps = math.floor(cfg.max_brace_seconds / cfg.dt)
    duration = xp.maximum(cfg.commit_steps, xp.minimum(requested, cap_steps))
    deadline = xp.where(start, step_idx + duration, mem.release_deadline_step)
    stay = mem.braced & (step_idx < mem.release_deadline_step)
    braced = stay | start
    reason = xp.where(mem.braced & ~braced, RELEASE_DEADLINE, RELEASE_NONE)
  else:
    committed = mem.braced & (mem.brace_steps < cfg.commit_steps)
    keep = ~stale
    if cfg.release_mode == "legacy":
      keep = keep & want
    stay = committed | (mem.braced & keep)
    start = ~mem.braced & want
    braced = stay | start
    deadline = mem.release_deadline_step
    reason = xp.where(mem.braced & ~braced,
                      xp.where(stale, RELEASE_STALE, RELEASE_REQUEST_FALSE), RELEASE_NONE)

  brace_steps = xp.where(braced, xp.where(mem.braced, mem.brace_steps + 1, 1), 0)
  switches = mem.n_switches + (braced != mem.braced).astype(xp.int32)
  return CtrlMemory(braced, brace_steps.astype(xp.int32), last_det, switches,
                    deadline.astype(xp.int32), mem.has_committed | start, reason.astype(xp.int32))
