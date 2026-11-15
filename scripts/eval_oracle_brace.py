# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Reviewed oracle experiment: explicit event timing, readiness and paired evidence.

The default schedule refers to local within-step entry into the CONTINUE reference's
0.45 m cylinder. This is a privileged reference event, not actual action-dependent
contact. Nominal-arrival timing remains available as an explicitly labeled baseline.
All episode outcomes and pre-action readiness traces are retained.
"""
import csv
import json
import os
from pathlib import Path
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R
from foresight.eval import statistics as ST

_NOM = flags.DEFINE_string("nominal", None, "nominal checkpoint", required=True)
_BR = flags.DEFINE_string("brace", None, "brace checkpoint", required=True)
_IMPL = flags.DEFINE_string("impl", "warp", "warp | jax")
_LEADS = flags.DEFINE_list("leads", ["-1", "0.1", "0.2", "0.3", "0.5", "0.75"], "seconds before reference event; -1=never")
_TIMING = flags.DEFINE_enum("timing_reference", "reference_entry", ["reference_entry", "reference_contact", "nominal"], "privileged timing event")
_RADIUS = flags.DEFINE_float("reference_radius", 0.45, "reference cylinder radius, m")
_HOLD = flags.DEFINE_float("hold", 0.5, "seconds held after reference event")
_COMMIT = flags.DEFINE_integer("commit_steps", 15, "minimum committed control intervals")
_MAXHOLD = flags.DEFINE_float("max_hold_s", 4.0, "maximum uninterrupted preparation duration")
_EPL = flags.DEFINE_integer("episodes_per_lead", 128, "scenarios per lead per batch")
_BATCHES = flags.DEFINE_integer("batches", 2, "independent scenario batches")
_STEPS = flags.DEFINE_integer("steps", 360, "control steps, including recovery")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command, m/s")
_MASS = flags.DEFINE_list("mass_range", ["4", "16"], "uniform projectile mass range, kg")
_SPEED = flags.DEFINE_list("speed_range", ["3", "8"], "uniform projectile speed range, m/s")
_AZ = flags.DEFINE_float("azimuth_deg", 60.0, "launch azimuth half range")
_MODES = flags.DEFINE_list("modes", ["brace", "zerocmd"], "brace | zerocmd")
_CONTROLS = flags.DEFINE_bool("no_launch_controls", True, "also run transition controls without launch")
_SHARED = flags.DEFINE_bool("shared_trajectory", True, "replay CONTINUE reference's projectile trajectories")
_TRACES = flags.DEFINE_bool("save_traces", True, "save compact pre-action timing/readiness evidence")
_OUT = flags.DEFINE_string("out", "results/review_oracle", "output directory")
_SEED = flags.DEFINE_integer("seed", 7001, "scenario seed; reserve disjoint test seeds")
_SPLIT = flags.DEFINE_enum("split_label", "development", ["gate", "development", "test"], "declared evaluation role")


def readiness(traj, dt, z_target):
  """Explicit reviewed metrics, not the historical mislabeled scalar."""
  return ST.readiness_metrics(traj, dt, z_target)


def _mean(x):
  x = np.asarray(x)
  return float(x.mean()) if x.size else None


def _finite_mean(x):
  x = np.asarray(x)
  return _mean(x[np.isfinite(x)])


def _trace(path, traj, table):
  fields = ["action_step", "brace_active", "impact_contact", "fell", "pre_pelvis_pos",
            "pre_torso_up", "pre_feet_contact", "pre_terminated", "pre_impact_contact",
            "pre_proj_launched", "pre_proj_pos", "pre_proj_vel", "pre_pelvis_vel"]
  payload = {k: traj[k] for k in fields}
  for k in ("scenario_seed", "scenario_index", "lead", "reference_event_time_s", "contact_step", "term_step"):
    payload[k] = table[k]
  Path(path).parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, **payload)


def main(_):
  outdir = Path(_OUT.value)
  if (outdir / "summary.json").exists():
    raise SystemExit("Output already contains a summary; choose a fresh --out (no silent cache reuse)")
  outdir.mkdir(parents=True, exist_ok=True)
  leads = [float(x) for x in _LEADS.value]
  if len(set(leads)) != len(leads) or not leads or any(x < 0 and x != -1 for x in leads):
    raise SystemExit("Leads must be distinct nonnegative values or -1")
  if _EPL.value <= 0 or _BATCHES.value <= 0 or _STEPS.value <= 0:
    raise SystemExit("Episode, batch and horizon counts must be positive")
  if _TIMING.value != "nominal" and not _SHARED.value:
    raise SystemExit("Reference-event timing requires --shared_trajectory")
  if any(m not in ("brace", "zerocmd") for m in _MODES.value):
    raise SystemExit("Unknown mode")
  epl, nl = _EPL.value, len(leads)
  n = epl * nl
  lead_arr = np.repeat(np.array(leads, np.float32), epl)
  nominal_path, brace_path = R.latest_checkpoint(_NOM.value), R.latest_checkpoint(_BR.value)
  nominal, brace = R.load_policy(nominal_path), R.load_policy(brace_path)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=_AZ.value, range_dist=[3.0, 5.0])
  if _COMMIT.value < 0 or _HOLD.value < 0 or _MAXHOLD.value < _COMMIT.value * env.dt:
    raise SystemExit("Invalid commitment/hold bounds")
  z_target = float(env._config.brace.base_height_target)
  config = {"schema_version": ST.SCHEMA_VERSION, "kind": "oracle", "seed": _SEED.value,
            "split_label": _SPLIT.value,
            "episodes_per_lead_per_batch": epl, "batches": _BATCHES.value, "steps": _STEPS.value,
            "dt": env.dt, "leads": leads, "timing_reference": _TIMING.value,
            "reference_radius": _RADIUS.value, "hold_s": _HOLD.value,
            "commit_steps": _COMMIT.value, "max_hold_s": _MAXHOLD.value,
            "release_mode": "deadline", "single_commit": True,
            "mass_range": list(map(float, _MASS.value)), "speed_range": list(map(float, _SPEED.value)),
            "azimuth_deg": _AZ.value, "range_dist": [3.0, 5.0], "command_vx": _VX.value,
            "modes": _MODES.value, "no_launch_controls": _CONTROLS.value,
            "shared_trajectory": _SHARED.value, "readiness_height_target": z_target,
            "readiness_definition": "BRACE active, pre-action z<=target+.03, upright>.9, both feet, not terminated, sustained over >=.2s between sample endpoints",
            "contact_time_resolution_s": env.dt,
            "reference_entry_definition": "first local constant-relative-velocity entry in [0,dt] from launched pre-contact reference states",
            "survival_window_s": 2.5, "save_traces": _TRACES.value}
  provenance = ST.protocol_metadata({"nominal": nominal_path, "brace": brace_path}, config)
  ST.write_json(outdir / "config.json", {"config": config, "provenance": provenance})
  conds = [(m, True) for m in _MODES.value]
  if _CONTROLS.value:
    conds += [(m, False) for m in _MODES.value]
  tables, reference_quality = [], []
  for b in range(_BATCHES.value):
    seed = _SEED.value + b
    rs = np.random.RandomState(seed)
    m_block = rs.uniform(*config["mass_range"], epl)
    s_block = rs.uniform(*config["speed_range"], epl)
    h_block = rs.uniform(*env._config.impact.hit_time_range, epl)
    key = jax.random.PRNGKey(seed)
    keys_block = jax.random.split(jax.random.fold_in(key, 1), epl)
    masses, speeds, hits = (np.tile(x, nl) for x in (m_block, s_block, h_block))
    reset_keys = jp.tile(keys_block, (nl, 1))
    wref = R.batched_env(env, m_block)
    ref_scn = lambda s: R.set_scenario(env, s, speed=s_block, hit_time=h_block,
                                       is_miss=np.zeros(epl, bool), command=[_VX.value, 0, 0])
    t0 = time.time()
    ref_info, ref_traj = R.rollout(env, wref, nominal, key, _STEPS.value,
                                  scenario_fn=ref_scn, reset_keys=keys_block)
    if _TIMING.value == "reference_entry":
      targets = ST.reference_entry_times(ref_traj, env.dt, _RADIUS.value)
    elif _TIMING.value == "reference_contact":
      targets = np.where(ref_info["contact_step"] >= 0, ref_info["contact_step"] * env.dt, np.nan)
    else:
      targets = np.asarray(ref_info["t_hit_nominal"])
    reference_quality.append({"batch": b, "seed": seed, "n": epl,
                              "n_reference_events": int(np.isfinite(targets).sum()),
                              "missing": [{"scenario_id": f"seed{seed}_env{i}", "mass": float(m_block[i]),
                                           "speed": float(s_block[i]), "contact": bool(ref_info["impact_contact"][i])}
                                          for i in np.flatnonzero(~np.isfinite(targets))]})
    lp = np.tile(ref_info["proj_launch_pos"], (nl, 1)) if _SHARED.value else None
    lv = np.tile(ref_info["proj_launch_vel"], (nl, 1)) if _SHARED.value else None
    target_arr = np.tile(targets, nl)
    decide = R.oracle_brace(jp.asarray(lead_arr), _HOLD.value, env.dt,
                            target_time_s=jp.asarray(target_arr), commit_steps=_COMMIT.value,
                            max_hold_s=_MAXHOLD.value)
    print(f"batch {b}: reference {time.time()-t0:.1f}s, valid events {np.isfinite(targets).sum()}/{epl}", flush=True)
    wenv = R.batched_env(env, masses)
    for mode, launch in conds:
      scn = lambda s, launch=launch: R.set_scenario(
          env, s, speed=speeds, hit_time=hits, is_miss=np.zeros(n, bool),
          command=[_VX.value, 0, 0], launch=launch, launch_pos=lp, launch_vel=lv)
      t0 = time.time()
      info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value,
                             brace_policy=brace if mode == "brace" else nominal,
                             decide=decide, scenario_fn=scn, reset_keys=reset_keys)
      tab = R.episode_table(info, traj, env.dt)
      tab.update({"scenario_id": np.tile([f"seed{seed}_env{i}" for i in range(epl)], nl),
                  "scenario_seed": np.full(n, seed), "scenario_index": np.tile(np.arange(epl), nl),
                  "lead": lead_arr, "batch": np.full(n, b), "mode": np.full(n, mode),
                  "mode_brace": np.full(n, mode == "brace"), "launch": np.full(n, int(launch)),
                  "timing_reference": np.full(n, _TIMING.value), "reference_event_time_s": target_arr,
                  "reference_event_observed": np.isfinite(target_arr),
                  "actual_reference_lead_s": target_arr - tab["actual_onset_time_s"],
                  "scheduled_switch_time_s": np.where((lead_arr >= 0) & np.isfinite(target_arr),
                      np.ceil(np.maximum(target_arr - lead_arr, 0) / env.dt - 1e-6) * env.dt, np.nan),
                  "hold_s": np.full(n, _HOLD.value), "commit_steps": np.full(n, _COMMIT.value),
                  "max_hold_s": np.full(n, _MAXHOLD.value), "reference_radius_m": np.full(n, _RADIUS.value)})
      tab["release_mode"] = np.full(n, "deadline")
      tab["single_commit"] = np.full(n, True)
      tab.update(readiness(traj, env.dt, z_target))
      tables.append(tab)
      # Flush evidence after each completed rollout, including before later failure.
      R.write_csv(outdir / f"episodes_b{b:03d}_{mode}_launch{int(launch)}.csv", tab)
      if _TRACES.value:
        _trace(outdir / "traces" / f"oracle_b{b:03d}_{mode}_launch{int(launch)}.npz", traj, tab)
      print(f"batch {b} mode={mode} launch={int(launch)} {time.time()-t0:.1f}s", flush=True)
  all_tab = ST.concatenate_tables(tables)
  R.write_csv(outdir / "episodes.csv", all_tab)
  curve, paired = [], []
  for mode, launch in conds:
    basemask = (all_tab["mode"] == mode) & (all_tab["launch"] == int(launch)) & (all_tab["lead"] == -1)
    baseline = {k: v[basemask] for k, v in all_tab.items()} if basemask.any() else None
    for lead in leads:
      mask = (all_tab["mode"] == mode) & (all_tab["launch"] == int(launch)) & (all_tab["lead"] == lead)
      tab = {k: v[mask] for k, v in all_tab.items()}
      contact = tab["contact"].astype(bool)
      row = {"mode": mode, "launch": int(launch), "lead_s": lead, **ST.summarize_episodes(tab, env.dt),
             "n": int(mask.sum()), "fall_rate": _mean(tab["fell"]), "contact_rate": _mean(contact),
             "switched_s": _mean(tab["brace_steps_total"]) * env.dt,
             "progress_loss_m": None if baseline is None else _mean(baseline["progress_m"]) - _mean(tab["progress_m"]),
             "n_reference_events": int(tab["reference_event_observed"].sum()),
             "readiness_ever_after_switch_rate": _mean(tab["readiness_ever_after_switch"]),
             "ready_at_contact_count": int(tab["ready_at_contact"].sum()),
             "ready_at_contact_given_contact": _mean(tab["ready_at_contact"][contact]),
             "brace_at_contact_given_contact": _mean(tab["brace_at_contact"][contact]),
             "readiness_window_start_mean_s": _finite_mean(tab["readiness_window_start_s"]),
             "readiness_window_complete_mean_s": _finite_mean(tab["readiness_window_complete_s"]),
             "readiness_precontact_start_mean_s": _finite_mean(tab["readiness_precontact_start_s"]),
             "readiness_precontact_complete_mean_s": _finite_mean(tab["readiness_precontact_complete_s"])}
      curve.append(row)
      if baseline is not None and lead != -1:
        paired.append({"mode": mode, "launch": int(launch), "condition_a": "continue", "condition_b": f"lead_{lead:g}",
                       **ST.paired_comparison(baseline, tab, env.dt)})
  with (outdir / "lead_curve.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(curve[0]))
    writer.writeheader(); writer.writerows(curve)
  ST.write_json(outdir / "paired.json", paired)
  ST.write_json(outdir / "summary.json", {"schema_version": ST.SCHEMA_VERSION, "status": "complete", "config": config,
                "provenance": provenance, "nominal": str(nominal_path), "brace": str(brace_path),
                "hold_s": _HOLD.value, "shared_trajectory": _SHARED.value,
                "episodes_per_condition": epl * _BATCHES.value, "reference_quality": reference_quality, "curve": curve})
  for row in curve:
    print(json.dumps(ST.json_safe(row)), flush=True)


if __name__ == "__main__":
  app.run(main)
