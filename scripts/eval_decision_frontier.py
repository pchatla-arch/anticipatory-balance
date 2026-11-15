# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Paired reviewed decision experiment with explicit timing and retained episodes.

Every condition sees the same scenario IDs and replayed external projectile path.
The default operating points are fixed, not tuned on this evaluation. Counts and
rates are pooled across batches, and paired differences retain joint outcomes.
"""
import csv
import itertools
import json
from pathlib import Path
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R
from foresight.eval import statistics as ST
from foresight.perception import detections as D
from foresight.belief import particle_filter as PF
from foresight.decision import rules as RU
from foresight.decision.controller import AnticipationController

_NOM = flags.DEFINE_string("nominal", "checkpoints/nominal_s1", "nominal checkpoint")
_BR = flags.DEFINE_string("brace", "checkpoints/brace2_s1", "brace checkpoint")
_RM = flags.DEFINE_string("response_model", "results/response_model.json", "fitted response model")
_ALLOW_LEGACY = flags.DEFINE_bool("allow_legacy_response", False, "explicitly permit a historical unmatched response model")
_IMPL = flags.DEFINE_string("impl", "warp", "warp | jax")
_N = flags.DEFINE_integer("episodes", 256, "episodes per batch shared by all methods")
_BATCHES = flags.DEFINE_integer("batches", 2, "independent scenario seeds")
_STEPS = flags.DEFINE_integer("steps", 360, "control steps including recovery")
_MISS = flags.DEFINE_float("miss_fraction", 0.3, "scheduled near-miss probability")
_OFFSET = flags.DEFINE_float("miss_offset", 0.6, "fixed lateral offset for labeled near misses, m")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command")
_MASS = flags.DEFINE_list("mass_range", ["4", "16"], "kg")
_SPEED = flags.DEFINE_list("speed_range", ["3", "8"], "m/s")
_AZ = flags.DEFINE_float("azimuth_deg", 60.0, "launch azimuth half range")
_LAT = flags.DEFINE_list("latencies", ["0", "5", "15"], "added latency in control steps")
_PERIOD = flags.DEFINE_integer("period", 5, "detector period in control steps")
_THETAS = flags.DEFINE_list("thetas", ["0.3"], "fixed TTC threshold(s), s")
_CPREPS = flags.DEFINE_list("c_preps", ["0.1"], "fixed preparation cost(s) per s")
_PMIN = flags.DEFINE_float("p_min", 0.3, "always-brace trigger")
_OLEAD = flags.DEFINE_float("oracle_lead", 0.5, "privileged switch lead")
_OTIMING = flags.DEFINE_enum("oracle_timing", "reference_entry", ["reference_entry", "reference_contact", "nominal"], "oracle event definition")
_METHODS = flags.DEFINE_list("methods", ["reactive", "oracle", "ttc_threshold", "belief_lead", "belief_wait", "belief_voi"], "methods")
_OUT = flags.DEFINE_string("out", "results/review_frontier", "output directory")
_SEED = flags.DEFINE_integer("seed", 9001, "scenario seed; disjoint from model development")
_SPLIT = flags.DEFINE_enum("split_label", "test", ["gate", "development", "test"], "declared evaluation role")
_SIG0 = flags.DEFINE_float("sigma0", 0.03, "detector base position noise, m")
_SIG1 = flags.DEFINE_float("sigma1", 0.0, "detector position noise per range, m/m")
_MISSP = flags.DEFINE_float("miss_prob", 0.055, "injected IID missed detections")
_FOV = flags.DEFINE_float("fov_deg", 120.0, "pelvis-horizontal-cone FOV, deg")
_MAXRANGE = flags.DEFINE_float("max_range", 6.0, "detector range, m")
_COMMIT = flags.DEFINE_integer("commit_steps", 15, "minimum committed control intervals")
_TAUB = flags.DEFINE_float("tau_brace", 0.22, "legacy myopic gate; not a validated physical readiness time")
_CTRACK = flags.DEFINE_float("c_track", 0.1, "cost per second of executed preparation")
_HOLD = flags.DEFINE_float("hold", 0.5, "post-predicted-entry hold, seconds")
_MAXHOLD = flags.DEFINE_float("max_hold_s", 4.0, "maximum preparation duration")
_PRIOR = flags.DEFINE_list("mass_prior", [], "belief prior support, kg; defaults to scenario support")
_RCONT = flags.DEFINE_float("r_contact", 0.45, "decision/reference entry cylinder radius, m")
_RELEASE = flags.DEFINE_bool("release_on_unwant", False, "legacy initiation-based retention ablation")
_RELEASE_MODE = flags.DEFINE_enum("release_mode", "deadline", ["deadline", "legacy"], "explicit hold protocol")
_INFERENCE = flags.DEFINE_enum("inference_mode", "deterministic", ["deterministic", "legacy"], "belief inference contract")
_ENVSKILL = flags.DEFINE_string("env_skill", "nominal", "nominal | aware; aware supports reactive only")
_CALIB = flags.DEFINE_list("calib_params", [], "compatibility: belief-commit parameters to retain full calibration traces")
_TRACES = flags.DEFINE_bool("save_traces", True, "retain compact per-step controller evidence")


def _fmt(x):
  return "NA" if x is None else f"{x:.3f}"


def _id(method, param, lat):
  return f"{method}:p={param:g}:lat={lat}"


def _trace(path, traj, table, dt, period, lat):
  fields = ("action_step", "brace_active", "impact_contact", "fell", "pre_terminated")
  data = {k: traj[k] for k in fields}
  for k, v in traj.items():
    if k.startswith("ctrl_"):
      data[k] = v
  for k in ("scenario_seed", "scenario_index", "contact_step", "term_step", "is_miss"):
    data[k] = table[k]
  data.update(dt=np.asarray(dt), period=np.asarray(period), latency=np.asarray(lat))
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, **data)


def main(_):
  outdir = Path(_OUT.value)
  if (outdir / "summary.json").exists():
    raise SystemExit("Output already contains a summary; choose fresh --out (no silent cache reuse)")
  outdir.mkdir(parents=True, exist_ok=True)
  known = {"reactive", "oracle", "always", "ttc_threshold", "belief_commit", "belief_lead", "belief_wait", "belief_voi"}
  if not _METHODS.value or set(_METHODS.value) - known or len(set(_METHODS.value)) != len(_METHODS.value):
    raise SystemExit("Unknown or duplicate method")
  if _ENVSKILL.value not in ("nominal", "aware") or (_ENVSKILL.value == "aware" and _METHODS.value != ["reactive"]):
    raise SystemExit("Object-aware policy supports only --methods reactive")
  if _N.value <= 0 or _BATCHES.value <= 0 or _STEPS.value <= 0 or _PERIOD.value <= 0:
    raise SystemExit("Counts, horizon and period must be positive")
  if not 0 <= _MISS.value <= 1 or not 0 <= _MISSP.value <= 1:
    raise SystemExit("Probabilities must lie in [0,1]")
  lats = list(map(int, _LAT.value)); thetas = list(map(float, _THETAS.value)); cpreps = list(map(float, _CPREPS.value))
  if not lats or min(lats) < 0:
    raise SystemExit("Nonnegative latencies required")
  if _ENVSKILL.value == "aware" and len(lats) != 1:
    raise SystemExit("Aware evaluation requires exactly one latency per invocation")
  rmj = json.loads(Path(_RM.value).read_text())
  response_check = ST.validate_response_protocol(rmj, {
      "timing_reference": "reference_entry", "reference_radius": _RCONT.value,
      "hold_s": _HOLD.value, "commit_steps": _COMMIT.value, "max_hold_s": _MAXHOLD.value,
      "release_mode": _RELEASE_MODE.value, "single_commit": True, "dt": 0.02}, _ALLOW_LEGACY.value)
  rm = RU.ResponseModel(rmj["a_cont"], rmj["b_cont"], rmj["a_brace"], rmj["b_brace"],
                        rmj.get("a_brace_lead", 0.0), rmj.get("b_brace_lead", 0.0), rmj.get("c_lead", 0.0),
                        rmj.get("lead_max", 0.5), "c_lead" in rmj)
  if "c_lead" not in rmj and any(m in _METHODS.value for m in ("belief_lead", "belief_wait", "belief_voi")):
    raise SystemExit("Lead/wait/lookahead methods require a fitted lead response model")
  nominal_path = R.latest_checkpoint(_NOM.value)
  brace_path = R.latest_checkpoint(_BR.value) if _ENVSKILL.value != "aware" else None
  nominal = R.load_policy(nominal_path)
  brace = R.load_policy(brace_path) if brace_path is not None else None
  ov = {}
  if _ENVSKILL.value == "aware":
    ov = {"aware.period": _PERIOD.value, "aware.latency": lats[0], "aware.sigma0": _SIG0.value,
          "aware.sigma1": _SIG1.value, "aware.miss_prob": _MISSP.value,
          "aware.fov_deg": _FOV.value, "aware.max_range": _MAXRANGE.value}
  env = R.make_env(skill=_ENVSKILL.value, impl=_IMPL.value, azimuth_range_deg=_AZ.value,
                   range_dist=[3.0, 5.0], overrides=ov)
  if _COMMIT.value < 0 or _HOLD.value < 0 or _MAXHOLD.value < _COMMIT.value * env.dt:
    raise SystemExit("Invalid commitment/hold bounds")
  n = _N.value; mr = tuple(map(float, _MASS.value)); sr = tuple(map(float, _SPEED.value))
  prior = tuple(map(float, _PRIOR.value)) if _PRIOR.value else mr
  conds = []
  for method in _METHODS.value:
    if method == "reactive": conds.append((method, 0.0, lats[0] if _ENVSKILL.value == "aware" else 0))
    elif method == "oracle": conds.append((method, _OLEAD.value, 0))
    elif method == "always": conds += [(method, _PMIN.value, lat) for lat in lats]
    elif method == "ttc_threshold": conds += [(method, p, lat) for p, lat in itertools.product(thetas, lats)]
    else: conds += [(method, p, lat) for p, lat in itertools.product(cpreps, lats)]
  if len(set(conds)) != len(conds):
    raise SystemExit("Duplicate evaluated condition")
  config = {"schema_version": ST.SCHEMA_VERSION, "kind": "frontier", "seed": _SEED.value,
            "split_label": _SPLIT.value, "episodes_per_batch": n, "batches": _BATCHES.value,
            "steps": _STEPS.value, "dt": env.dt, "period": _PERIOD.value, "latencies_steps": lats,
            "mass_range": mr, "speed_range": sr, "mass_prior": prior, "azimuth_deg": _AZ.value,
            "range_dist": [3.0, 5.0], "miss_fraction": _MISS.value, "miss_offset": _OFFSET.value,
            "command_vx": _VX.value, "env_skill": _ENVSKILL.value, "r_contact": _RCONT.value,
            "detector": {"sigma0": _SIG0.value, "sigma1": _SIG1.value, "miss_prob": _MISSP.value,
                         "fov_deg": _FOV.value, "max_range": _MAXRANGE.value,
                         "visibility": "pelvis horizontal cone, capture-time validity"},
            "rule": {"commit_steps": _COMMIT.value, "hold": _HOLD.value, "max_hold_s": _MAXHOLD.value,
                     "release_mode": _RELEASE_MODE.value, "single_commit": True, "tau_brace": _TAUB.value,
                     "c_track": _CTRACK.value, "release_on_unwant": _RELEASE.value,
                     "horizon_steps": _STEPS.value}, "inference_mode": _INFERENCE.value,
            "oracle_timing": _OTIMING.value, "conditions": [{"method": m, "param": p, "latency_steps": lat} for m,p,lat in conds],
            "fixed_operating_points": len(thetas) == 1 and len(cpreps) == 1,
            "robot_velocity": "free-base world qvel" if _RELEASE_MODE.value != "legacy" else "commanded approximation",
            "save_traces": _TRACES.value, "survival_window_s": 2.5}
  config["response_protocol_check"] = response_check
  provenance = ST.protocol_metadata({"nominal": nominal_path, "brace": brace_path, "response_model": _RM.value}, config)
  ST.write_json(outdir / "config.json", {"config": config, "provenance": provenance})
  print(f"{len(conds)} conditions x {n} episodes x {_BATCHES.value} batches", flush=True)
  tables = {c: [] for c in conds}
  reference_quality = []
  for b in range(_BATCHES.value):
    seed = _SEED.value + b
    rs = np.random.RandomState(seed)
    masses = rs.uniform(*mr, n); speeds = rs.uniform(*sr, n)
    hits = rs.uniform(*env._config.impact.hit_time_range, n)
    is_miss = rs.uniform(size=n) < _MISS.value
    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(jax.random.fold_in(key, 1), n)
    wenv = R.batched_env(env, masses)
    base = dict(speed=speeds, hit_time=hits, is_miss=is_miss,
                command=[_VX.value, 0, 0], miss_offset=_OFFSET.value)
    t0 = time.time()
    ref_info, ref_traj = R.rollout(env, wenv, nominal, key, _STEPS.value,
                                  scenario_fn=lambda s: R.set_scenario(env, s, **base), reset_keys=reset_keys)
    lp, lv = ref_info["proj_launch_pos"], ref_info["proj_launch_vel"]
    targets = (ST.reference_entry_times(ref_traj, env.dt, _RCONT.value) if _OTIMING.value == "reference_entry"
               else np.where(ref_info["contact_step"] >= 0, ref_info["contact_step"] * env.dt, np.nan)
               if _OTIMING.value == "reference_contact" else np.asarray(ref_info["t_hit_nominal"]))
    reference_quality.append({"batch": b, "seed": seed, "n": n, "n_hits": int((~is_miss).sum()),
                              "n_misses": int(is_miss.sum()),
                              "n_reference_events_hits": int((np.isfinite(targets) & ~is_miss).sum()),
                              "missing_hit_events": [{"scenario_id": f"seed{seed}_env{i}", "mass": float(masses[i]),
                                                       "speed": float(speeds[i]), "contact": bool(ref_info["impact_contact"][i])}
                                                      for i in np.flatnonzero(~np.isfinite(targets) & ~is_miss)]})
    print(f"batch {b}: reference {time.time()-t0:.1f}s", flush=True)
    scn = lambda s: R.set_scenario(env, s, launch_pos=lp, launch_vel=lv, **base)
    for ci, (method, param, lat) in enumerate(conds):
      t0 = time.time()
      if method == "oracle":
        decide = R.oracle_brace(jp.full((n,), param), _HOLD.value, env.dt,
                                target_time_s=jp.asarray(targets), commit_steps=_COMMIT.value,
                                max_hold_s=_MAXHOLD.value)
        info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                               decide=decide, scenario_fn=scn, reset_keys=reset_keys)
      else:
        rc = RU.RuleConfig(kind=method, theta=param if method == "ttc_threshold" else .3,
                           p_min=param if method == "always" else _PMIN.value,
                           c_prep=param if method.startswith("belief_") else .1,
                           commit_steps=_COMMIT.value, tau_brace=_TAUB.value,
                           c_track=_CTRACK.value, hold=_HOLD.value, max_hold_s=_MAXHOLD.value,
                           release_mode=_RELEASE_MODE.value, single_commit=True,
                           release_on_unwant=_RELEASE.value, wait_steps=_PERIOD.value,
                           horizon_steps=_STEPS.value)
        pf_kw = dict(mass_range=prior, r_contact=_RCONT.value, inference_mode=_INFERENCE.value)
        if _INFERENCE.value == "legacy": pf_kw["process_vel_sigma"] = .02
        ctrl = AnticipationController(D.DetectorConfig(period=_PERIOD.value, latency=lat,
                     buffer_len=max(16, lat + 1),
                     sigma0=_SIG0.value, sigma1=_SIG1.value, miss_prob=_MISSP.value,
                     fov_deg=_FOV.value, max_range=_MAXRANGE.value), PF.PFConfig(**pf_kw), rc, rm)
        info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                               scenario_fn=scn, reset_keys=reset_keys, controller=ctrl)
      tab = R.episode_table(info, traj, env.dt)
      tab.update({"scenario_id": np.asarray([f"seed{seed}_env{i}" for i in range(n)]),
                  "scenario_seed": np.full(n, seed), "scenario_index": np.arange(n),
                  "batch": np.full(n, b), "method": np.full(n, method), "param": np.full(n, param),
                  "latency_steps": np.full(n, lat), "period": np.full(n, _PERIOD.value),
                  "condition_id": np.full(n, _id(method,param,lat)),
                  "release_mode": np.full(n, _RELEASE_MODE.value),
                  "reference_event_time_s": targets, "reference_event_observed": np.isfinite(targets),
                  "actual_reference_lead_s": targets - tab["actual_onset_time_s"]})
      tables[(method,param,lat)].append(tab)
      R.write_csv(outdir / f"episodes_b{b:03d}_c{ci:03d}.csv", tab)
      save_calib = method == "belief_commit" and any(abs(param-float(x))<1e-9 for x in _CALIB.value if x)
      if _TRACES.value or save_calib:
        _trace(outdir / "traces" / f"frontier_b{b:03d}_c{ci:03d}.npz", traj, tab, env.dt, _PERIOD.value, lat)
      row = ST.summarize_episodes(tab, env.dt)
      print(f"  {method:14} p={param:g} lat={lat} {time.time()-t0:.1f}s "
            f"fall|hits {_fmt(row['fall_rate_hits'])} FP {_fmt(row['false_prep_rate'])}", flush=True)
  pooled = {c: ST.concatenate_tables(t) for c,t in tables.items()}
  # Preserve Session G's compact numeric outcome archive as an additional
  # interchange format. CSV scenario IDs and paired.json remain authoritative.
  archive = {}
  aliases = {"fell": "fell", "contact": "contact", "is_miss": "is_miss",
             "brace_steps": "brace_steps_total", "brace_onset": "brace_onset_step",
             "contact_step": "contact_step", "progress_m": "progress_m",
             "scenario_seed": "scenario_seed", "scenario_index": "scenario_index"}
  for (method, param, lat), table in pooled.items():
    tag = f"{method}__p{param:g}__lat{lat}"
    for name, column in aliases.items():
      archive[f"{tag}__{name}"] = table[column]
    archive[f"{tag}__release_reason"] = table.get("controller_release_reason", np.zeros(len(table["fell"]), int))
  np.savez_compressed(outdir / "episodes.npz", **archive)
  # Schema differs only for privileged rows without controller diagnostics; fill
  # absent scalar diagnostics, keeping one rectangular, explicit episode table.
  union = set().union(*(t.keys() for t in pooled.values()))
  for table in pooled.values():
    size = len(table["fell"])
    for name in union - table.keys(): table[name] = np.full(size, np.nan)
  all_tab = ST.concatenate_tables(list(pooled.values()))
  R.write_csv(outdir / "episodes.csv", all_tab)
  rows = []
  react = next((t for c,t in pooled.items() if c[0] == "reactive"), None)
  for c, table in pooled.items():
    method,param,lat = c
    row = {"method": method, "param": param, "latency_steps": lat,
           **ST.summarize_episodes(table, env.dt)}
    row["progress_loss_m"] = None if react is None else float(react["progress_m"].mean()-table["progress_m"].mean())
    rows.append(row)
  comparisons = []
  for (ca, ta), (cb, tb) in itertools.combinations(pooled.items(), 2):
    if ca[2] != cb[2] and ca[0] not in ("reactive", "oracle") and cb[0] not in ("reactive", "oracle"):
      continue
    comparisons.append({"condition_a": _id(*ca), "condition_b": _id(*cb), **ST.paired_comparison(ta,tb,env.dt)})
  ST.write_json(outdir / "paired.json", comparisons)
  with (outdir / "frontier.csv").open("w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
  ST.write_json(outdir / "summary.json", {"schema_version": ST.SCHEMA_VERSION, "status": "complete", "config": config,
                "provenance": provenance, "nominal": str(nominal_path), "brace": str(brace_path),
                "episodes_per_condition": n*_BATCHES.value, "latencies_steps":lats,"period":_PERIOD.value,
                "detector":config['detector'],"rule":config['rule'],"reference_quality":reference_quality,
                "aggregation":"pooled episode numerators/denominators", "rows":rows})
  print(json.dumps(ST.json_safe({"rows": rows})),flush=True)


if __name__ == "__main__":
  app.run(main)
