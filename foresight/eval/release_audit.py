"""Simulator-free traces and contract checks for the paired release diagnostic."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from foresight.eval import frontier_output as O


_SCENARIO_FIELDS = ("mass", "speed", "is_miss", "launch_pos_x", "launch_pos_y",
                    "launch_pos_z", "launch_vel_x", "launch_vel_y", "launch_vel_z")
_DEADLINE_FIELDS = ("ctrl_release_deadline_step", "ctrl_brace_start", "ctrl_brace_release",
                    "ctrl_release_reason", "ctrl_has_committed")


def write_trace(path, traj, table, *, seed, batch, method, param, latency_steps,
                period, release_mode, dt):
  """Write scalar controller logs through the first terminating transition only.

  Step zero is the first action, matching term_step in the environment. Controller
  vectors are deliberately omitted; every scalar ctrl_* stream is retained.
  """
  active = np.asarray(traj["brace_active"], dtype=bool)
  if active.ndim != 2 or not all(active.shape):
    raise ValueError("brace_active must have nonempty shape (steps, episodes)")
  steps, n = active.shape
  term = np.asarray(table["term_step"])
  contact = np.asarray(table["contact_step"])
  if term.shape != (n,) or contact.shape != (n,):
    raise ValueError("term_step and contact_step must have shape (episodes,)")
  if not math.isfinite(dt) or dt <= 0:
    raise ValueError("dt must be positive and finite")
  logs = {}
  for name in sorted(traj):
    if not name.startswith("ctrl_"):
      continue
    value = np.asarray(traj[name])
    if value.shape[:2] != active.shape:
      raise ValueError(f"{name} must start with shape {active.shape}, got {value.shape}")
    if value.ndim == 2:
      logs[name] = value
  rows = []
  n_braced = n_onsets = n_releases = n_contact = n_braced_contact = n_unobserved_contact = 0
  for i in range(n):
    stop = min(steps, int(term[i]) + 1) if term[i] >= 0 else steps
    previous = False
    n_braced += bool(active[:stop, i].any())
    for step in range(stop):
      braced = bool(active[step, i])
      n_onsets += braced and not previous
      n_releases += previous and not braced
      previous = braced
      row = dict(scenario_id=f"seed{seed}_episode{i:06d}", batch=batch, seed=seed,
                 episode_index=i, method=method, param=param, latency_steps=latency_steps,
                 period=period, release_mode=release_mode, step=step, time_s=step * dt,
                 brace_active=braced, contact_step=int(contact[i]), term_step=int(term[i]))
      row.update({name: value[step, i].item() for name, value in logs.items()})
      rows.append(row)
    if 0 <= contact[i] < stop:
      n_contact += 1
      n_braced_contact += bool(active[int(contact[i]), i])
    elif contact[i] >= 0:
      n_unobserved_contact += 1
  O.atomic_rows(path, rows)
  return dict(n_trace_rows=len(rows), n_trace_episodes=n, n_braced_episodes=int(n_braced),
              n_brace_onsets=int(n_onsets), n_brace_releases=int(n_releases),
              n_contact_observed=n_contact, n_braced_at_contact=int(n_braced_contact),
              n_contact_outside_observed_trace=n_unobserved_contact,
              braced_at_contact_fraction=n_braced_contact / n_contact if n_contact else None,
              scalar_controller_fields=list(logs))


def _integer(value, label):
  try:
    number = float(value)
  except (TypeError, ValueError) as exc:
    raise RuntimeError(f"Invalid integer {label}: {value!r}") from exc
  if not math.isfinite(number) or not number.is_integer():
    raise RuntimeError(f"Invalid integer {label}: {value!r}")
  return int(number)


def _boolean(value, label):
  if str(value).lower() in ("true", "1", "1.0"):
    return True
  if str(value).lower() in ("false", "0", "0.0"):
    return False
  raise RuntimeError(f"Invalid boolean {label}: {value!r}")


def _csv(path):
  with Path(path).open(newline="") as stream:
    rows = list(csv.DictReader(stream))
  if not rows:
    raise RuntimeError(f"Empty audit CSV: {path}")
  return rows


def _condition(row):
  return (row["method"], float(row["param"]), int(row["latency_steps"]), row["release_mode"])


def _scenario(row):
  missing = [name for name in _SCENARIO_FIELDS if name not in row]
  if missing:
    raise RuntimeError(f"Episode CSV missing pairing fields: {missing}")
  values = {}
  for name in _SCENARIO_FIELDS:
    value = _boolean(row[name], name) if name == "is_miss" else float(row[name])
    if name != "is_miss" and not math.isfinite(value):
      raise RuntimeError(f"Nonfinite paired scenario field {name}")
    values[name] = value
  return values


def validate_release_run(path):
  """Validate pairing and the observed deadline contract, without judging outcomes.

  An episode ending before its deadline is a censored observation, not a failed
  contract. Missing trace rows before termination are errors, not censoring.
  """
  path = Path(path)
  config = json.loads((path / "config.json").read_text())
  batches = json.loads((path / "batches.json").read_text())
  rows = batches["rows"]
  if config.get("status") != "complete":
    raise RuntimeError("Release run is incomplete according to config.json")
  steps = _integer(config["steps"], "config steps")
  dt = float(config["dt"])
  if steps <= 0 or not math.isfinite(dt) or dt <= 0:
    raise RuntimeError("Invalid configured horizon or dt")
  expected = {(b, _condition(c)) for b in range(int(config["batches"]))
              for c in config["conditions"]}
  actual = [(int(r["batch"]), _condition(r)) for r in rows]
  if len(actual) != len(set(actual)) or set(actual) != expected:
    raise RuntimeError("Missing, duplicate, or unexpected condition batches")
  if batches.get("completed_condition_batches", len(rows)) != len(rows):
    raise RuntimeError("batches.json completed-condition count does not match rows")
  paired = {}
  reports = []
  total_deadline_onsets = 0
  for condition in rows:
    batch = int(condition["batch"])
    mode = condition["release_mode"]
    label = f"batch {batch}, {_condition(condition)}"
    episodes = _csv(path / condition["episode_file"])
    by_id = {r["scenario_id"]: r for r in episodes}
    if len(by_id) != len(episodes) or len(episodes) != int(config["episodes"]):
      raise RuntimeError(f"{label}: duplicate IDs or wrong episode count")
    scenarios = {sid: _scenario(ep) for sid, ep in by_id.items()}
    if batch not in paired:
      paired[batch] = scenarios
    elif scenarios != paired[batch]:
      if set(scenarios) != set(paired[batch]):
        raise RuntimeError(f"{label}: pairing mismatch in scenario IDs")
      bad = next(sid for sid in scenarios if scenarios[sid] != paired[batch][sid])
      changed = [k for k in _SCENARIO_FIELDS if scenarios[bad][k] != paired[batch][bad][k]]
      raise RuntimeError(f"{label}: pairing mismatch for {bad}, fields {changed}")
    trace = _csv(path / condition["trace_file"])
    grouped = {}
    for row in trace:
      grouped.setdefault(row["scenario_id"], []).append(row)
    if set(grouped) != set(by_id):
      raise RuntimeError(f"{label}: trace/episode scenario ID mismatch")
    report = dict(batch=batch, method=condition["method"], param=condition["param"],
                  latency_steps=condition["latency_steps"], release_mode=mode,
                  n_episodes=len(episodes), n_trace_rows=len(trace), n_braced_episodes=0,
                  n_brace_onsets=0, n_brace_releases=0, n_deadlines_observed=0,
                  n_deadline_censored=0, n_deadline_censored_termination=0,
                  n_deadline_censored_horizon=0, n_contact_observed=0, n_braced_at_contact=0,
                  n_contact_outside_observed_trace=0)
    for sid, episode in by_id.items():
      seq = grouped[sid]
      seq.sort(key=lambda r: _integer(r["step"], "step"))
      term = _integer(episode["term_step"], "term_step")
      contact = _integer(episode["contact_step"], "contact_step")
      stop = min(steps, term + 1) if term >= 0 else steps
      if [_integer(r["step"], "step") for r in seq] != list(range(stop)):
        raise RuntimeError(f"{label}, {sid}: trace missing steps or contains post-termination rows")
      for row in seq:
        if (_integer(row["term_step"], "term_step") != term
            or _integer(row["contact_step"], "contact_step") != contact):
          raise RuntimeError(f"{label}, {sid}: trace/episode event mismatch")
        if _condition(row) != _condition(condition):
          raise RuntimeError(f"{label}, {sid}: trace condition metadata mismatch")
        if not math.isclose(float(row["time_s"]), int(row["step"]) * dt, abs_tol=1e-10):
          raise RuntimeError(f"{label}, {sid}: trace time does not match step and dt")
      active = [_boolean(r["brace_active"], "brace_active") for r in seq]
      starts = [i for i, value in enumerate(active) if value and (i == 0 or not active[i-1])]
      releases = [i for i in range(1, len(active)) if active[i-1] and not active[i]]
      report["n_braced_episodes"] += bool(starts)
      report["n_brace_onsets"] += len(starts)
      report["n_brace_releases"] += len(releases)
      if 0 <= contact < stop:
        report["n_contact_observed"] += 1
        report["n_braced_at_contact"] += active[contact]
      elif contact >= 0:
        report["n_contact_outside_observed_trace"] += 1
      if mode != "deadline":
        continue
      if any(name not in seq[0] for name in _DEADLINE_FIELDS):
        raise RuntimeError(f"{label}: missing deadline controller trace fields")
      if len(starts) > 1:
        raise RuntimeError(f"{label}, {sid}: deadline controller rearmed")
      onset = starts[0] if starts else None
      deadline = _integer(seq[onset]["ctrl_release_deadline_step"], "deadline") if starts else -1
      if starts and deadline <= onset:
        raise RuntimeError(f"{label}, {sid}: deadline must follow brace onset")
      for step, row in enumerate(seq):
        committed = onset is not None and step >= onset
        expected_deadline = deadline if committed else -1
        if _integer(row["ctrl_release_deadline_step"], "deadline") != expected_deadline:
          raise RuntimeError(f"{label}, {sid}: deadline drift or premature deadline at step {step}")
        expected_active = committed and step < deadline
        if active[step] != expected_active:
          raise RuntimeError(f"{label}, {sid}: deadline contract broken at step {step}; "
                             f"expected brace_active={expected_active}, deadline={deadline}")
        if _boolean(row["ctrl_has_committed"], "has_committed") != committed:
          raise RuntimeError(f"{label}, {sid}: commitment latch inconsistent at step {step}")
        is_start = onset is not None and step == onset
        is_release = committed and step == deadline
        if (_boolean(row["ctrl_brace_start"], "brace_start") != is_start
            or _boolean(row["ctrl_brace_release"], "brace_release") != is_release):
          raise RuntimeError(f"{label}, {sid}: start/release event inconsistent at step {step}")
        if _integer(row["ctrl_release_reason"], "release_reason") != (1 if is_release else 0):
          raise RuntimeError(f"{label}, {sid}: release reason inconsistent at step {step}")
      if starts:
        total_deadline_onsets += 1
        if deadline < stop:
          report["n_deadlines_observed"] += 1
        else:
          report["n_deadline_censored"] += 1
          cause = "termination" if 0 <= term < steps else "horizon"
          report[f"n_deadline_censored_{cause}"] += 1
    report["braced_at_contact_fraction"] = (report["n_braced_at_contact"] / report["n_contact_observed"]
                                              if report["n_contact_observed"] else None)
    reports.append(report)
  if total_deadline_onsets == 0:
    raise RuntimeError("No deadline condition ever braces; release contract was not exercised")
  return dict(status="passed", pairing="exact", n_condition_batches=len(rows),
              n_deadline_onsets=total_deadline_onsets, git_revision=config.get("git_revision"),
              interpretation="Observed release contract only; no fall-rate or contact-timing performance gate.",
              conditions=reports)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", required=True)
  args = parser.parse_args()
  run_dir = Path(args.run_dir)
  try:
    result = validate_release_run(run_dir)
  except (RuntimeError, KeyError, ValueError, OSError) as exc:
    result = dict(status="failed", error=str(exc))
    O.atomic_json(run_dir / "release_audit.json", result)
    raise SystemExit(f"Release audit FAILED: {exc}") from exc
  O.atomic_json(run_dir / "release_audit.json", result)
  print(json.dumps({k: v for k, v in result.items() if k != "conditions"}))


if __name__ == "__main__":
  main()
