"""Small, simulator-free persistence helpers for paired frontier experiments."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def prepare_output(path, overwrite=False):
  """Reject accidental reuse of a result directory; only clear known files on opt-in."""
  path = Path(path)
  if path.exists() and not path.is_dir():
    raise ValueError(f"Output is not a directory: {path}")
  if path.exists() and any(path.iterdir()):
    if not overwrite:
      raise ValueError(f"Output directory is not empty: {path}. Choose a new --out.")
    known = {"config.json", "batches.json", "summary.json", "frontier.csv", "frontier.png", "episodes"}
    unknown = [p.name for p in path.iterdir() if p.name not in known]
    if unknown:
      raise ValueError(f"Refusing overwrite of unrelated output entries: {unknown}")
    # Validate the entire directory before deleting anything.
    for item in path.iterdir():
      if item.name == "episodes" and item.is_dir() and not item.is_symlink():
        entries = list(item.iterdir())
        if any(not p.is_file() or p.is_symlink() or not p.name.endswith('.csv') for p in entries):
          raise ValueError("Unexpected content in episodes/; use a fresh --out")
      elif not item.is_file() or item.is_symlink():
        raise ValueError(f"Refusing unusual output entry: {item}")
    for item in path.iterdir():
      if item.is_dir():
        for entry in item.iterdir():
          entry.unlink()
        item.rmdir()
      else:
        item.unlink()
  path.mkdir(parents=True, exist_ok=True)
  (path / "episodes").mkdir(exist_ok=True)
  return path


def _atomic_write(path, write):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
  try:
    with os.fdopen(fd, "w", newline="") as stream:
      write(stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(name, path)
  except BaseException:
    Path(name).unlink(missing_ok=True)
    raise


def atomic_json(path, value):
  _atomic_write(path, lambda stream: json.dump(value, stream, indent=2))


def atomic_rows(path, rows):
  rows = list(rows)
  if not rows:
    raise ValueError("Cannot write an empty CSV without a schema")
  fields = list(rows[0])
  def write(stream):
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
  _atomic_write(path, write)


def write_episodes(path, table, *, batch, seed, method, param, latency_steps, period):
  """Full precision scalar data with stable IDs shared across methods/rates/delays."""
  values = {key: np.asarray(value) for key, value in table.items()}
  n = len(next(iter(values.values())))
  if any(value.shape != (n,) for value in values.values()):
    raise ValueError("Episode columns must all have shape (episodes,)")
  rows = []
  for i in range(n):
    row = dict(scenario_id=f"seed{seed}_episode{i:06d}", batch=batch, seed=seed,
               episode_index=i, method=method, param=param,
               latency_steps=latency_steps, period=period)
    row.update({key: value[i].item() for key, value in values.items()})
    rows.append(row)
  atomic_rows(path, rows)


def outcome_counts(table):
  """Integer numerators and denominators, avoiding average-of-batches ambiguity."""
  miss = np.asarray(table["is_miss"], dtype=bool)
  hit = ~miss
  contact = np.asarray(table["contact"], dtype=bool)
  fell = np.asarray(table["fell"], dtype=bool)
  prep = np.asarray(table["false_prep"], dtype=bool)
  braced = np.asarray(table["brace_steps_total"]) > 0
  counts = {
      "n_hits": int(hit.sum()), "n_misses": int(miss.sum()),
      "n_contact_hits": int((contact & hit).sum()),
      "n_falls_hits": int((fell & hit).sum()),
      "n_falls_contact_hits": int((fell & contact & hit).sum()),
      "n_false_prep_misses": int((prep & miss).sum()),
      "n_braced_hits": int((braced & hit).sum()),
      "n_falls_misses": int((fell & miss).sum()),
      "n_censored": int(np.asarray(table.get("censored", np.zeros_like(miss)), bool).sum()),
  }
  if "false_prep_until_termination" in table:
    counts["n_false_prep_misses_until_termination"] = int((table["false_prep_until_termination"] & miss).sum())
    counts["n_braced_hits_until_termination"] = int(((table["brace_steps_until_termination"] > 0) & hit).sum())
  return counts


def pooled_rates(counts):
  """Pooled rates accompany, rather than silently replace, legacy batch means."""
  ratio = lambda a, b: counts[a] / counts[b] if counts[b] else None
  rates = {
      "fall_rate_hits_pooled": ratio("n_falls_hits", "n_hits"),
      "fall_rate_given_contact_pooled": ratio("n_falls_contact_hits", "n_contact_hits"),
      "contact_rate_hits_pooled": ratio("n_contact_hits", "n_hits"),
      "false_prep_rate_pooled": ratio("n_false_prep_misses", "n_misses"),
      "braced_on_hits_pooled": ratio("n_braced_hits", "n_hits"),
      "fall_rate_miss_pooled": ratio("n_falls_misses", "n_misses"),
  }
  if "n_false_prep_misses_until_termination" in counts:
    rates["false_prep_rate_until_termination_pooled"] = ratio("n_false_prep_misses_until_termination", "n_misses")
    rates["braced_on_hits_until_termination_pooled"] = ratio("n_braced_hits_until_termination", "n_hits")
  return rates


def preparation_until_termination(brace_active, term_step, is_miss, detections=None):
  """Include the action causing first termination, exclude all later scan actions.

  The environment latches term_step before incrementing ep_step, so term_step=0
  means the first transition failed. A negative value means no termination.
  """
  active = np.asarray(brace_active, dtype=bool)
  termination = np.asarray(term_step)
  alive = (termination[None, :] < 0) | (np.arange(active.shape[0])[:, None] <= termination[None, :])
  before = active & alive
  any_before = before.any(axis=0)
  values = {
      "brace_steps_until_termination": before.sum(axis=0),
      "false_prep_until_termination": np.asarray(is_miss, bool) & any_before,
      "brace_onset_step_until_termination": np.where(any_before, before.argmax(axis=0), -1),
  }
  if detections is not None:
    values["n_detections_until_termination"] = (np.asarray(detections, bool) & alive).sum(axis=0)
  return values
