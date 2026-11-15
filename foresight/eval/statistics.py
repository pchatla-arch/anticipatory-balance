"""CPU-only evidence utilities for the reviewed evaluation protocol.

The simulator calls these after a rollout. No JAX or simulator import is needed.
Rates are pooled from episode numerators/denominators, never means of batch rates.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

SCHEMA_VERSION = "foresight.review_rerun.v1"


def json_safe(value):
  if isinstance(value, dict):
    return {str(k): json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [json_safe(v) for v in value]
  if isinstance(value, np.ndarray):
    return json_safe(value.tolist())
  if isinstance(value, np.generic):
    return json_safe(value.item())
  if isinstance(value, float) and not math.isfinite(value):
    return None
  if isinstance(value, Path):
    return str(value)
  return value


def write_json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temp = path.with_name(path.name + ".tmp")
  temp.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + "\n")
  os.replace(temp, path)


def protocol_metadata(paths: dict, config: dict):
  """Record source revision and content hashes; never reads credential files."""
  root = Path(__file__).resolve().parents[2]
  def git(*args):
    p = subprocess.run(["git", "-C", str(root), *args], text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    return p.stdout.strip() if p.returncode == 0 else None
  manifests = {}
  for name, p in paths.items():
    if p is None:
      continue
    p = Path(p).resolve()
    if not p.exists():
      raise FileNotFoundError(p)
    files = sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p]
    entries = []
    for f in files:
      h = hashlib.sha256()
      with f.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
          h.update(block)
      entries.append({"path": str(f.relative_to(p)) if p.is_dir() else f.name,
                      "bytes": f.stat().st_size, "sha256": h.hexdigest()})
    manifests[name] = {"path": str(p), "files": entries}
  status = git("status", "--porcelain", "--untracked-files=no")
  return {"source_commit": git("rev-parse", "HEAD"),
          "source_dirty": None if status is None else bool(status),
          "argv": list(sys.argv), "artifacts": manifests,
          "config_sha256": hashlib.sha256(json.dumps(json_safe(config), sort_keys=True).encode()).hexdigest()}


def concatenate_tables(tables):
  if not tables:
    raise ValueError("No episode tables")
  keys = set(tables[0])
  if any(set(t) != keys for t in tables):
    raise ValueError("Episode table schemas differ")
  return {k: np.concatenate([np.asarray(t[k]) for t in tables]) for k in tables[0]}


def validate_response_protocol(model: dict, expected: dict, allow_legacy: bool = False):
  """Reject a historical surrogate used as if fitted under the reviewed contract."""
  if model.get("schema_version") != "foresight.review_response.v1":
    if allow_legacy:
      return {"matched": False, "legacy_override": True, "reason": "explicitly permitted historical response"}
    raise ValueError("Response model is not a reviewed protocol fit; fit fresh oracle data or explicitly allow legacy response")
  protocol = model.get("protocol", {})
  mismatches = []
  for key, want in expected.items():
    got = protocol.get(key)
    if isinstance(want, (float, int)) and not isinstance(want, bool):
      try:
        same = math.isfinite(float(got)) and math.isclose(float(got), float(want), rel_tol=1e-7, abs_tol=1e-8)
      except (TypeError, ValueError):
        same = False
    else:
      same = got == want
    if not same:
      mismatches.append({"field": key, "model": got, "evaluation": want})
  if mismatches:
    raise ValueError("Response/evaluation protocol mismatch: " + json.dumps(mismatches))
  return {"matched": True, "legacy_override": False, "checked": expected}


def _rate(numerator, denominator):
  return float(numerator / denominator) if denominator else None


def summarize_episodes(table: dict, dt: float = 0.02):
  """Pooled counts/rates; absent subgroups return null, not a fabricated zero."""
  fell, miss, contact = (np.asarray(table[k], bool) for k in ("fell", "is_miss", "contact"))
  n = len(fell)
  if n == 0 or miss.shape != fell.shape or contact.shape != fell.shape:
    raise ValueError("Need equally sized nonempty episode arrays")
  brace = np.asarray(table["brace_steps_total"], float)
  progress = np.asarray(table["progress_m"], float)
  if brace.shape != fell.shape or progress.shape != fell.shape:
    raise ValueError("Bad brace/progress shape")
  hit = ~miss
  c_hit = hit & contact
  censored = np.asarray(table.get("censored", np.zeros(n, bool)), bool)
  counts = {"n_total": n, "n_hits": int(hit.sum()), "n_misses": int(miss.sum()),
            "n_contact_hits": int(c_hit.sum()), "n_falls_hits": int((fell & hit).sum()),
            "n_falls_contact_hits": int((fell & c_hit).sum()),
            "n_false_prep_miss": int(((brace > 0) & miss).sum()),
            "n_braced_hits": int(((brace > 0) & hit).sum()),
            "n_falls_miss": int((fell & miss).sum()),
            "n_censored_hits": int((censored & hit).sum()),
            "n_censored": int(censored.sum())}
  return {**counts,
          "fall_rate_hits": _rate(counts["n_falls_hits"], counts["n_hits"]),
          "fall_rate_given_contact": _rate(counts["n_falls_contact_hits"], counts["n_contact_hits"]),
          "contact_rate_hits": _rate(counts["n_contact_hits"], counts["n_hits"]),
          "false_prep_rate": _rate(counts["n_false_prep_miss"], counts["n_misses"]),
          "braced_on_hits": _rate(counts["n_braced_hits"], counts["n_hits"]),
          "fall_rate_miss": _rate(counts["n_falls_miss"], counts["n_misses"]),
          "censored_rate": float(censored.mean()),
          "braced_s": float(brace.mean() * dt), "progress_m": float(progress.mean())}


def paired_difference(a, b):
  """a minus b with a normal 95% paired interval, conditional on this skill pair.

  This interval is descriptive and is not corrected for multiple comparisons or
  development-grid selection. Raw discordant counts permit alternative analyses.
  """
  a, b = np.asarray(a, float), np.asarray(b, float)
  if a.shape != b.shape or a.ndim != 1 or not np.all(np.isfinite(a + b)):
    raise ValueError("Need aligned finite one-dimensional arrays")
  n = len(a)
  if not n:
    return {"n": 0, "difference": None, "se": None, "normal95": None}
  d = a - b
  se = float(d.std(ddof=1) / np.sqrt(n)) if n > 1 else None
  out = {"n": n, "mean_a": float(a.mean()), "mean_b": float(b.mean()),
         "difference": float(d.mean()), "se": se,
         "normal95": None if se is None else [float(d.mean() - 1.96 * se), float(d.mean() + 1.96 * se)]}
  if np.isin(a, [0, 1]).all() and np.isin(b, [0, 1]).all():
    out.update({"a_only": int(((a == 1) & (b == 0)).sum()),
                "b_only": int(((a == 0) & (b == 1)).sum()),
                "both": int(((a == 1) & (b == 1)).sum()),
                "neither": int(((a == 0) & (b == 0)).sum())})
    if out["normal95"] is not None:
      out["normal95"] = [max(-1.0, out["normal95"][0]), min(1.0, out["normal95"][1])]
  return out


def paired_comparison(table_a, table_b, dt: float = 0.02):
  """Align by explicit IDs; return a-minus-b effects on the same scenario sample."""
  ia, ib = (np.asarray(t["scenario_id"]).astype(str) for t in (table_a, table_b))
  if len(set(ia)) != len(ia) or len(set(ib)) != len(ib) or set(ia) != set(ib):
    raise ValueError("Paired comparison needs unique, identical scenario IDs")
  index_b = {s: i for i, s in enumerate(ib)}
  order = np.array([index_b[s] for s in ia], int)
  a = {k: np.asarray(v) for k, v in table_a.items()}
  b = {k: np.asarray(v)[order] for k, v in table_b.items()}
  if not np.array_equal(a["is_miss"], b["is_miss"]):
    raise ValueError("Paired scenarios disagree about hit/miss labels")
  for k in ("mass", "speed", "t_hit_nominal"):
    if k in a and k in b and not np.allclose(a[k], b[k], rtol=1e-7, atol=1e-7):
      raise ValueError("Paired scenarios disagree about " + k)
  miss = a["is_miss"].astype(bool)
  hit = ~miss
  common_contact = hit & a["contact"].astype(bool) & b["contact"].astype(bool)
  return {"effect_direction": "condition_a minus condition_b",
          "uncertainty": "approximate normal 95%, scenario-level within one skill pair; no selection correction",
          "failure_hits": paired_difference(a["fell"][hit], b["fell"][hit]),
          "false_preparation_misses": paired_difference(a["brace_steps_total"][miss] > 0, b["brace_steps_total"][miss] > 0),
          "contact_hits": paired_difference(a["contact"][hit], b["contact"][hit]),
          "failure_common_contact": paired_difference(a["fell"][common_contact], b["fell"][common_contact]),
          "common_contact_caution": "post-intervention selected subset; descriptive only",
          "progress_m": paired_difference(a["progress_m"], b["progress_m"]),
          "braced_s": paired_difference(a["brace_steps_total"] * dt, b["brace_steps_total"] * dt)}


def readiness_metrics(traj: dict, dt: float, z_target: float, stable_s: float = 0.2,
                      height_tolerance: float = 0.03):
  """Posture evidence from pre-action samples; never uses post-impact state as readiness.

  A window requires active BRACE and a valid, nonterminated posture at every
  sample, with last_time-first_time >= stable_s. Ready at contact requires the
  window to end at the PRE-action sample of the first contacting control step.
  That state's posture precedes the physics step in which contact is registered.
  The height test intentionally remains the original one-sided lower-posture
  threshold, z <= target+tolerance; it is named explicitly in metadata.
  """
  active = np.asarray(traj["brace_active"], bool)
  z = np.asarray(traj["pre_pelvis_pos"])[..., 2]
  up = np.asarray(traj["pre_torso_up"])
  feet = np.asarray(traj["pre_feet_contact"], bool).all(axis=-1)
  terminated = np.asarray(traj["pre_terminated"], bool)
  contact = np.asarray(traj["impact_contact"], bool)
  T, n = active.shape
  if dt <= 0 or stable_s <= 0:
    raise ValueError("dt and stable_s must be positive")
  # Endpoints represent physical samples; ten intervals require eleven samples.
  intervals = int(np.ceil(stable_s / dt - 1e-10))
  ok = active & (z <= z_target + height_tolerance) & (up > 0.9) & feet & ~terminated
  ok &= np.isfinite(z) & np.isfinite(up)
  out = {"readiness_ever_after_switch": np.zeros(n, bool),
         "ready_at_contact": np.zeros(n, bool), "readiness_contact_observed": contact.any(axis=0),
         "readiness_window_start_s": np.full(n, np.nan),
         "readiness_window_complete_s": np.full(n, np.nan),
         "readiness_precontact_start_s": np.full(n, np.nan),
         "readiness_precontact_complete_s": np.full(n, np.nan)}
  for i in range(n):
    if not active[:, i].any():
      continue
    onset = int(np.flatnonzero(active[:, i])[0])
    ci = int(np.flatnonzero(contact[:, i])[0]) if contact[:, i].any() else -1
    run = 0
    for t in range(T):
      run = run + 1 if ok[t, i] else 0
      if run < intervals + 1:
        continue
      if not out["readiness_ever_after_switch"][i]:
        out["readiness_ever_after_switch"][i] = True
        out["readiness_window_start_s"][i] = (t - intervals - onset) * dt
        out["readiness_window_complete_s"][i] = (t - onset) * dt
      if ci >= 0 and t <= ci and not np.isfinite(out["readiness_precontact_complete_s"][i]):
        out["readiness_precontact_start_s"][i] = (t - intervals - onset) * dt
        out["readiness_precontact_complete_s"][i] = (t - onset) * dt
      if t == ci:
        out["ready_at_contact"][i] = True
  return out


def reference_entry_times(traj: dict, dt: float, radius: float = 0.45,
                          z_range=(0.3, 1.4)):
  """First reference cylinder entry predicted within a control interval.

  Uses each pre-action position and relative velocity, solving the same local
  constant-velocity entry geometry as the belief. Fractional entry in [0,dt]
  avoids discarding fast events that cross the boundary and contact in one step.
  It is a privileged, locally interpolated CONTINUE-reference event, not the
  action-dependent physical contact time. Samples after contact/termination are
  excluded. Missing events remain NaN and must be reported by mass/speed.
  """
  p = np.asarray(traj["pre_proj_pos"])
  pel = np.asarray(traj["pre_pelvis_pos"])
  valid = np.asarray(traj["pre_proj_launched"], bool).copy()
  valid &= ~np.asarray(traj["pre_terminated"], bool)
  valid &= ~np.asarray(traj["pre_impact_contact"], bool)
  rel = p[..., :2] - pel[..., :2]
  v = np.asarray(traj["pre_proj_vel"]) - np.asarray(traj["pre_pelvis_vel"])
  a = np.sum(v[..., :2] ** 2, axis=-1)
  b = np.sum(rel * v[..., :2], axis=-1)
  c = np.sum(rel ** 2, axis=-1) - radius ** 2
  disc = b * b - a * c
  enter = (-b - np.sqrt(np.maximum(disc, 0.0))) / np.maximum(a, 1e-12)
  enter = np.where(c <= 0.0, 0.0, enter)
  height = p[..., 2] + np.asarray(traj["pre_proj_vel"])[..., 2] * enter
  ok = valid & ((a > 1e-12) | (c <= 0)) & (disc >= 0) & (enter >= 0) & (enter <= dt + 1e-9)
  ok &= (height > z_range[0]) & (height < z_range[1])
  times = np.arange(p.shape[0])[:, None] * dt + enter
  result = np.min(np.where(ok, times, np.inf), axis=0)
  return np.where(np.isfinite(result), result, np.nan)


def episode_table(final_info: dict, traj: dict, dt: float) -> dict:
  """Per-episode scalars from a rollout: scenario, outcome, progress, pre-impact state."""
  n = final_info["fell"].shape[0]
  T = traj["pelvis_pos"].shape[0]
  # Each displacement includes its initial state; stop accruing after termination.
  dp = traj["pelvis_pos"][..., :2] - traj["pre_pelvis_pos"][..., :2]
  fwd = traj["pre_pelvis_fwd"][..., :2]
  fwd = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)
  step_prog = np.sum(dp * fwd, axis=-1)                              # (T-1, N)
  alive = ~traj["pre_terminated"].astype(bool)
  progress = np.sum(step_prog * alive, axis=0)                        # (N,) until first failure
  cs = final_info["contact_step"]
  # Censored: contact happened with less than `window` seconds of observation left.
  window = 2.5
  # Contact is somewhere inside interval cs; using interval end is conservative.
  recovery_observed = np.maximum((T - cs - 1) * dt, 0.0)
  censored = (cs >= 0) & (recovery_observed < window) & ~final_info["fell"]
  pre = np.clip(cs, 0, T - 1)
  idx = np.arange(n)
  com_pre = traj["pre_com"][pre, idx]
  com_prev = traj["pre_com"][np.clip(pre - 1, 0, T - 1), idx]
  com_vel = (com_pre - com_prev) / dt
  feet_pre = traj["pre_feet_pos"][pre, idx]
  pvel = traj["pre_proj_vel"][pre, idx]
  observed = cs >= 0
  com_pre = np.where(observed[:, None], com_pre, np.nan)
  com_vel = np.where(observed[:, None], com_vel, np.nan)
  feet_pre = np.where(observed[:, None, None], feet_pre, np.nan)
  pvel = np.where(observed[:, None], pvel, np.nan)
  brace = traj["brace_active"].astype(bool)
  onset = np.where(brace.any(axis=0), np.argmax(brace, axis=0), -1)
  released = np.zeros_like(brace)
  released[1:] = brace[:-1] & ~brace[1:]
  release_step = np.where(released.any(axis=0), np.argmax(released, axis=0), -1)
  out = {
      "mass": final_info["proj_mass"], "speed": final_info["proj_speed"],
      "momentum": final_info["proj_momentum"], "is_miss": final_info["proj_is_miss"],
      "t_hit_nominal": final_info["t_hit_nominal"],
      "launched": final_info["proj_launched"], "contact": final_info["impact_contact"],
      "contact_step": cs, "fell": final_info["fell"],
      "fell_before_launch": final_info["fell_before_launch"],
      "term_reason": final_info["term_reason"], "term_step": final_info["term_step"],
      "progress_m": progress, "censored": censored,
      "recovery_observed_s": np.where(cs >= 0, recovery_observed, np.nan),
      "contact_time_lower_s": np.where(cs >= 0, cs * dt, np.nan),
      "contact_time_upper_s": np.where(cs >= 0, (cs + 1) * dt, np.nan),
      "launch_step": final_info["proj_launch_step"],
      "launch_time_s": final_info["proj_launch_step"] * dt,
      "evaluated_steps": np.full(n, T),
      "miss_offset_m": final_info["proj_lateral"],
      "brace_steps_total": np.sum(traj["brace_active"], axis=0),
      "brace_onset_step": onset,
      "actual_onset_time_s": np.where(onset >= 0, onset * dt, np.nan),
      "brace_release_step": release_step,
      "actual_release_time_s": np.where(release_step >= 0, release_step * dt, np.nan),
      "brace_releases": released.sum(axis=0),
      "brace_at_contact": (cs >= 0) & brace[pre, idx],
      "onset_to_contact_lower_s": np.where((cs >= 0) & (onset >= 0), (cs - onset) * dt, np.nan),
      "onset_to_contact_upper_s": np.where((cs >= 0) & (onset >= 0), (cs + 1 - onset) * dt, np.nan),
      "com_x_pre": com_pre[:, 0], "com_y_pre": com_pre[:, 1], "com_z_pre": com_pre[:, 2],
      "com_vx_pre": com_vel[:, 0], "com_vy_pre": com_vel[:, 1],
      "lfoot_x_pre": feet_pre[:, 0, 0], "lfoot_y_pre": feet_pre[:, 0, 1],
      "rfoot_x_pre": feet_pre[:, 1, 0], "rfoot_y_pre": feet_pre[:, 1, 1],
      "proj_vx_pre": pvel[:, 0], "proj_vy_pre": pvel[:, 1], "proj_vz_pre": pvel[:, 2],
  }
  if "ctrl_p_hit" in traj:
    pre_p = np.where(observed, traj["ctrl_p_hit"][pre, idx], np.nan)
    out["p_hit_pre_contact"] = pre_p
    out["n_detections"] = np.sum(traj["ctrl_det_valid"], axis=0)
    out["false_prep"] = (final_info["proj_is_miss"] & traj["brace_active"].any(axis=0))
  if "ctrl_release_reason" in traj:
    out["controller_release_reason"] = np.max(traj["ctrl_release_reason"], axis=0)
  if "ctrl_release_step" in traj:
    out["controller_deadline_step"] = traj["ctrl_release_step"][-1]
  if "ctrl_last_capture_step" in traj:
    out["last_capture_step"] = traj["ctrl_last_capture_step"][-1]
  return {k: np.asarray(v) for k, v in out.items()}


def write_csv(path: str | Path, table: dict, extra: dict | None = None):
  import csv
  keys = list(table.keys())
  n = len(table[keys[0]])
  if any(len(table[k]) != n for k in keys):
    raise ValueError("CSV columns have different episode counts")
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  extra = extra or {}
  with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(list(extra.keys()) + keys)
    for i in range(n):
      w.writerow([extra[k] for k in extra] + [_scalar(table[k][i]) for k in keys])


def _scalar(v):
  v = np.asarray(v)
  if v.dtype == bool:
    return int(v)
  if np.issubdtype(v.dtype, np.floating):
    return repr(float(v))
  if np.issubdtype(v.dtype, np.str_) or v.dtype == object:
    return str(v.item())
  return int(v)
