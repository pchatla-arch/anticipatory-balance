"""CPU-only evidence utilities for the Session H follow-up protocol.

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

SCHEMA_VERSION = "foresight.session_h.v1"


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
