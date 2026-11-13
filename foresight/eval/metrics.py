# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""NumPy-only outcome metrics for one rollout without autoreset.

``traj[t]`` is recorded after control transition ``t`` (zero based), at
time ``(t + 1) * dt``. Contact and termination indices use that same convention;
``-1`` means the event was not observed. No reset-state sample is assumed.
"""
from __future__ import annotations

import numpy as np


def _finite_scalar(value, name, *, positive=False):
  value = float(value)
  if not np.isfinite(value) or value < 0 or (positive and value == 0):
    qualifier = "positive" if positive else "nonnegative"
    raise ValueError(f"{name} must be finite and {qualifier}")
  return value


def _step_indices(values, name, num_steps, num_envs=None):
  values = np.asarray(values, dtype=float)
  if values.ndim != 1 or (num_envs is not None and len(values) != num_envs):
    raise ValueError(f"{name} must have shape (N,)")
  if (not np.all(np.isfinite(values)) or
      np.any(values != np.floor(values)) or
      np.any((values < -1) | (values >= num_steps))):
    raise ValueError(f"{name} must contain -1 or indices in [0, T - 1]")
  return values.astype(np.int64)


def _boolean(values, shape, name):
  values = np.asarray(values)
  if values.shape != shape:
    raise ValueError(f"{name} must have shape {shape}")
  try:
    valid = np.isfinite(values) & ((values == 0) | (values == 1))
  except TypeError as exc:
    raise ValueError(f"{name} must contain booleans or finite 0/1 values") from exc
  if not np.all(valid):
    raise ValueError(f"{name} must contain booleans or finite 0/1 values")
  return values.astype(bool)


def _pelvis(values):
  values = np.asarray(values, dtype=float)
  if values.ndim != 3 or values.shape[2] != 3 or values.shape[0] < 1:
    raise ValueError("pelvis_pos must have shape (T, N, 3), with T >= 1")
  return values


def _horizontal_heading(values, shape):
  values = np.asarray(values, dtype=float)
  if values.shape != shape:
    raise ValueError(f"pelvis_fwd must have shape {shape}")
  xy = values[..., :2]
  norm = np.linalg.norm(xy, axis=-1, keepdims=True)
  unit = np.full_like(xy, np.nan)
  np.divide(xy, norm, out=unit, where=np.isfinite(norm) & (norm > 1e-12))
  return unit


def post_contact_survival(contact_step, term_step, *, num_steps, dt, window_s):
  """Classify survival through a fixed duration after the detected contact.

  Returns per-episode arrays. ``survived_post_contact_window`` is 0 for a
  failure at contact or within the inclusive window, 1 for observed survival
  through its end, and NaN for missing contact, pre-contact failure, or censoring.
  An observed failure within the window is conclusive even if the rollout is
  shorter than the requested window. A later failure does not undo survival of
  an earlier window. Failure-free short follow-up is censored, never a success.

  Available contact-to-horizon time is ``(T - 1 - contact_step) * dt``;
  ``post_contact_followup_s`` stops earlier at a detected failure. Failure at
  the same recorded step as contact counts as time-zero failure. Substep event
  ordering is unavailable. Invalid indices (including term_step >= T) raise.
  """
  if isinstance(num_steps, bool) or not isinstance(num_steps, (int, np.integer)) or num_steps < 1:
    raise ValueError("num_steps must be a positive integer")
  dt = _finite_scalar(dt, "dt", positive=True)
  window_s = _finite_scalar(window_s, "window_s")
  contact = _step_indices(contact_step, "contact_step", num_steps)
  term = _step_indices(term_step, "term_step", num_steps, len(contact))
  has_contact = contact >= 0
  pre_failure = has_contact & (term >= 0) & (term < contact)
  eligible = has_contact & ~pre_failure
  has_failure = eligible & (term >= contact) & (term >= 0)
  failure_time = (term - contact) * dt
  horizon_time = (num_steps - 1 - contact) * dt
  # Inclusive boundaries must not depend on roundoff in e.g. 3 * 0.1.
  tol = 16 * np.finfo(float).eps * max(1.0, window_s)
  failed_in_window = has_failure & (failure_time <= window_s + tol)
  full_window = eligible & (horizon_time >= window_s - tol)
  observed = eligible & (failed_in_window | full_window)
  survival = np.full(len(contact), np.nan)
  survival[observed] = (~failed_in_window[observed]).astype(float)
  followup = np.where(has_failure, failure_time, horizon_time)
  return {
      "post_contact_eligible": eligible,
      "pre_contact_failure": pre_failure,
      "post_contact_observed": observed,
      "post_contact_censored": eligible & ~observed,
      "survived_post_contact_window": survival,
      "post_contact_followup_s": np.where(eligible, followup, np.nan),
  }


def summarize_trajectory(pelvis_pos, pelvis_fwd, brace_active, term_step, dt):
  """Report observed forward progress and executed brace exposure before failure.

  Progress projects each horizontal displacement onto the pelvis heading at
  its starting sample. It excludes intervals ending in or after the first
  failed sample, and cannot include reset-to-first-sample displacement. Thus it
  measures forward path progress, not displacement along a fixed world axis.
  Nonfinite data in retained progress intervals yields NaN; invalid data after
  censoring does not contaminate earlier observations.

  brace_active[t] selects the action for transition t, before its outcome is
  observed. Brace duration counts that entire dt interval, including the first
  failure transition, but none after it. Durations describe sampled control
  exposure, not substep-accurate failure time. A failure on step 0 therefore
  permits zero observed progress but one executed control interval.
  """
  dt = _finite_scalar(dt, "dt", positive=True)
  pelvis = _pelvis(pelvis_pos)
  T, N, _ = pelvis.shape
  heading = _horizontal_heading(pelvis_fwd, pelvis.shape)
  active = _boolean(brace_active, (T, N), "brace_active")
  term = _step_indices(term_step, "term_step", T, N)
  first_failed = np.where(term >= 0, term, T)
  interval_ok = np.arange(1, T)[:, None] < first_failed[None, :]
  projected = np.sum(np.diff(pelvis[..., :2], axis=0) * heading[:-1], axis=-1)
  progress = np.sum(np.where(interval_ok, projected, 0.0), axis=0)
  last_executed = np.where(term >= 0, term, T - 1)
  executed = np.arange(T)[:, None] <= last_executed[None, :]
  return {
      "progress_before_failure_m": progress,
      "progress_observed_s": interval_ok.sum(axis=0) * dt,
      "brace_duration_through_failure_s": (active & executed).sum(axis=0) * dt,
      "control_duration_through_failure_s": executed.sum(axis=0) * dt,
  }


def brace_readiness(pelvis_pos, torso_up, feet_pos, feet_contact, brace_active,
                    term_step, *, dt, target_pelvis_height, height_tolerance,
                    min_torso_up, min_foot_separation, dwell_s, pelvis_fwd=None):
  """Find the first sustained brace posture; unobserved readiness times are NaN.

  The height target is pelvis/base z, NOT center-of-mass height. A sample must
  meet height, uprightness, bilateral floor contact, foot separation, active
  brace, and no-failure criteria. If pelvis_fwd is supplied, separation is the
  lateral component in that heading; otherwise it is horizontal foot distance.
  Supply pelvis_fwd when interpreting separation as lateral stance width.

  Dwell is a sampled criterion: each qualifying post-step sample counts one
  completed control interval. At least ceil(dwell_s / dt) consecutive samples
  are required (minimum one). Five good samples at dt=.02 give .10 s of sampled
  dwell. This is not a claim about unobserved continuous motion between samples.
  Any bad sample or inactive brace resets dwell. Failure and all later samples
  are ineligible. Nonfinite posture cannot qualify; nonbinary masks raise.

  Readiness time is (ready_step + 1)*dt. Latency is measured from the start of
  the active-brace bout achieving readiness: (ready_step - bout_start + 1)*dt.
  This records the FIRST readiness event, not maintained readiness at impact.
  """
  dt = _finite_scalar(dt, "dt", positive=True)
  target = _finite_scalar(target_pelvis_height, "target_pelvis_height", positive=True)
  height_tolerance = _finite_scalar(height_tolerance, "height_tolerance")
  min_foot_separation = _finite_scalar(min_foot_separation, "min_foot_separation")
  dwell_s = _finite_scalar(dwell_s, "dwell_s")
  min_torso_up = float(min_torso_up)
  if not np.isfinite(min_torso_up) or not -1 <= min_torso_up <= 1:
    raise ValueError("min_torso_up must be finite and in [-1, 1]")
  pelvis = _pelvis(pelvis_pos)
  T, N, _ = pelvis.shape
  up = np.asarray(torso_up, dtype=float)
  feet = np.asarray(feet_pos, dtype=float)
  if up.shape != (T, N) or feet.shape != (T, N, 2, 3):
    raise ValueError("torso_up must have shape (T, N); feet_pos must have shape (T, N, 2, 3)")
  contact = _boolean(feet_contact, (T, N, 2), "feet_contact")
  active = _boolean(brace_active, (T, N), "brace_active")
  term = _step_indices(term_step, "term_step", T, N)
  foot_delta = feet[:, :, 0, :2] - feet[:, :, 1, :2]
  if pelvis_fwd is None:
    separation = np.linalg.norm(foot_delta, axis=-1)
  else:
    heading = _horizontal_heading(pelvis_fwd, pelvis.shape)
    lateral = np.stack((-heading[..., 1], heading[..., 0]), axis=-1)
    separation = np.abs(np.sum(foot_delta * lateral, axis=-1))
  sample = np.arange(T)[:, None]
  before_failure = (term[None, :] < 0) | (sample < term[None, :])
  tol = 16 * np.finfo(float).eps
  good = (active & before_failure & contact.all(axis=-1) &
          np.isfinite(pelvis).all(axis=-1) & np.isfinite(feet).all(axis=(-2, -1)) &
          np.isfinite(up) & (up >= -1 - 1e-6) & (up <= 1 + 1e-6) &
          (np.abs(pelvis[..., 2] - target) <= height_tolerance + tol) &
          (up >= min_torso_up - tol) &
          np.isfinite(separation) & (separation >= min_foot_separation - tol))
  ratio = dwell_s / dt
  required = max(1, int(np.ceil(ratio - tol * max(1.0, ratio))))
  run = np.zeros(N, dtype=np.int64)
  bout_start = np.zeros(N, dtype=np.int64)
  previous_active = np.zeros(N, dtype=bool)
  ready_step = np.full(N, np.nan)
  latency = np.full(N, np.nan)
  for t in range(T):
    bout_start = np.where(active[t] & ~previous_active, t, bout_start)
    run = np.where(good[t], run + 1, 0)
    first_ready = np.isnan(ready_step) & (run >= required)
    ready_step[first_ready] = t
    latency[first_ready] = (t - bout_start[first_ready] + 1) * dt
    previous_active = active[t]
  return {
      "brace_ready": np.isfinite(ready_step),
      "brace_ready_step": ready_step,
      "brace_ready_time_s": (ready_step + 1) * dt,
      "brace_ready_latency_s": latency,
  }
