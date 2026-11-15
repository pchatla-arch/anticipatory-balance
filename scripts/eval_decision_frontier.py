# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E10: the decision frontier. Fall rate vs preparation cost for every decision rule, on the
same scenarios (paired reset keys, shared replayed projectile trajectories), at several
perception latencies.

Scenario mix per batch: (1 - miss_fraction) true hits and miss_fraction near-misses
(0.6 m lateral), masses U(mass_range), speeds U(speed_range), launch azimuth within
+-azimuth_deg of the heading, range 3-5 m.

Methods and sweeps (one rollout each):
  reactive                       never brace
  always                         brace on first credible detection (p_hit > p_min)
  ttc_threshold  theta in --thetas         point-estimate time-to-contact threshold
  belief_commit  c_prep in --c_preps       myopic commit rule on the belief (c_fall = 1)
  oracle         privileged switch at --oracle_lead before nominal arrival
Latencies in --latencies (control steps; 5 = 100 ms) apply to the detector for the three
non-privileged methods.

  python scripts/eval_decision_frontier.py --nominal checkpoints/nominal_s1 \
      --brace checkpoints/brace2_s1 --impl warp --out results/E10_frontier
"""
import itertools
import csv
from pathlib import Path
import json
import os
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R
from foresight.eval import session_h_evidence as ST
from foresight.perception import detections as D
from foresight.belief import particle_filter as PF
from foresight.decision import rules as RU
from foresight.decision.controller import AnticipationController

_NOM = flags.DEFINE_string("nominal", "checkpoints/nominal_s1", "nominal checkpoint")
_BR = flags.DEFINE_string("brace", "checkpoints/brace2_s1", "brace checkpoint")
_RM = flags.DEFINE_string("response_model", "results/response_model.json", "fitted response model")
_IMPL = flags.DEFINE_string("impl", "warp", "warp | jax")
_N = flags.DEFINE_integer("episodes", 1024, "episodes per batch (shared by all methods)")
_BATCHES = flags.DEFINE_integer("batches", 1, "batches (seeds)")
_STEPS = flags.DEFINE_integer("steps", 300, "control steps")
_MISS = flags.DEFINE_float("miss_fraction", 0.3, "fraction of near-miss scenarios")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command")
_MASS = flags.DEFINE_list("mass_range", ["4", "16"], "kg")
_SPEED = flags.DEFINE_list("speed_range", ["3", "8"], "m/s")
_AZ = flags.DEFINE_float("azimuth_deg", 60.0, "launch azimuth half-range about the heading")
_LAT = flags.DEFINE_list("latencies", ["0", "5", "15"], "detector latency in control steps")
_PERIOD = flags.DEFINE_integer("period", 5, "detector period, control steps (5 = 10 Hz)")
_THETAS = flags.DEFINE_list("thetas", ["0.2", "0.3", "0.4", "0.6", "0.8"], "TTC thresholds, s")
_CPREPS = flags.DEFINE_list("c_preps", ["0.02", "0.05", "0.1", "0.2", "0.4", "0.8"], "c_prep per s braced")
_PMIN = flags.DEFINE_float("p_min", 0.3, "always-brace trigger")
_OLEAD = flags.DEFINE_float("oracle_lead", 0.5, "oracle switch lead, s")
_OHOLD = flags.DEFINE_float("oracle_hold", 0.5, "oracle nominal-arrival hold, independent of release_after")
_METHODS = flags.DEFINE_list("methods", ["reactive", "always", "ttc_threshold", "belief_commit", "oracle"], "subset")
_OUT = flags.DEFINE_string("out", "results/E10_frontier", "output dir")
_SEED = flags.DEFINE_integer("seed", 200, "seed")
# Detector model (E12 perfect-sensing ladder; E8 measured values)
_SIG0 = flags.DEFINE_float("sigma0", 0.03, "detector noise at zero range, m")
_SIG1 = flags.DEFINE_float("sigma1", 0.02, "detector noise per m of range")
_MISSP = flags.DEFINE_float("miss_prob", 0.05, "random missed-detection probability")
_FOV = flags.DEFINE_float("fov_deg", 120.0, "detector field of view, deg")
_MAXRANGE = flags.DEFINE_float("max_range", 6.0, "detector range, m")
# Rule internals (E13 ablations)
_COMMIT = flags.DEFINE_integer("commit_steps", 15, "minimum commitment, control steps")
_TAUB = flags.DEFINE_float("tau_brace", 0.22, "s the posture needs to form")
_CTRACK = flags.DEFINE_float("c_track", 0.1, "tracking cost per s")
_HOLD = flags.DEFINE_float("hold", 0.5, "s held after expected impact (cost model)")
_RELAFTER = flags.DEFINE_float("release_after", 0.3, "s past the belief's closest approach before BRACE is released")
_PRIOR = flags.DEFINE_list("mass_prior", [], "belief mass prior range, kg (default: the scenario range)")
_RCONT = flags.DEFINE_float("r_contact", 0.45, "belief decision radius about the pelvis axis, m")
_RELEASE = flags.DEFINE_bool("release_on_unwant", False, "old rule: release BRACE when the start condition fails")
# End-to-end baseline (E14): run the object-aware walker itself, no decision layer
_ENVSKILL = flags.DEFINE_string("env_skill", "nominal", "nominal | aware (the policy in --nominal must match)")
_CALIB = flags.DEFINE_list("calib_params", ["0.1"], "belief_commit c_prep values whose per-step belief "
                           "logs are dumped to <out>/calib/ for E9 calibration (empty: none)")


def _episode_evidence(tab, traj, seed, dt):
  """Account for actions up to the first terminal transition; preserve legacy totals.

  This changes reporting only: the Session G controller and physics are untouched.
  Release diagnostics count actual controller transitions before termination, so
  terminating an episode is not relabeled as a stale or passed release.
  """
  tab = {k: np.asarray(v).copy() for k, v in tab.items()}
  requested = np.asarray(traj["brace_active"], bool)
  T, n = requested.shape
  term = np.asarray(tab["term_step"], int)
  active_before = np.arange(T)[:, None] <= np.where(term < 0, T, term)[None]
  active = requested & active_before
  previous = np.concatenate([np.zeros((1, n), bool), requested[:-1]], axis=0)
  starts = requested & ~previous & active_before
  releases = previous & ~requested & active_before
  first = lambda mask: np.where(mask.any(axis=0), np.argmax(mask, axis=0), -1)
  tab.update({"scenario_seed": np.full(n, seed, np.int64), "scenario_index": np.arange(n),
              "scenario_id": np.array([f"seed{seed}_env{i:05d}" for i in range(n)]),
              "brace_steps_total_legacy": requested.sum(axis=0),
              "brace_steps_total": active.sum(axis=0), "brace_onset_step": first(active),
              "brace_release_step": first(releases), "brace_segments": starts.sum(axis=0),
              "brace_releases": releases.sum(axis=0),
              "brace_steps_after_termination": (requested & ~active_before).sum(axis=0),
              "brace_horizon_truncated": active[-1] & (term < 0)})
  cs = np.asarray(tab["contact_step"], int)
  valid_contact = (cs >= 0) & (cs < T)
  tab["contact_before_termination"] = valid_contact & ((term < 0) | (cs <= term))
  tab["brace_at_contact"] = valid_contact & active[np.clip(cs, 0, T-1), np.arange(n)]
  tab["false_prep_legacy"] = np.asarray(tab.get("false_prep", tab["is_miss"] & requested.any(axis=0))).copy()
  tab["false_prep"] = np.asarray(tab["is_miss"], bool) & active.any(axis=0)
  tab["censored_legacy"] = tab["censored"].copy()
  tab["censored"] = valid_contact & ((T-cs-1)*dt < 2.5) & ~tab["fell"]
  reason = np.asarray(traj.get("ctrl_release_reason", np.zeros_like(requested, dtype=int)))
  tab["release_reason"] = reason[-1]
  tab["passed_releases"] = (releases & (reason == RU.RELEASE_PASSED)).sum(axis=0)
  tab["stale_releases"] = (releases & (reason == RU.RELEASE_STALE)).sum(axis=0)
  tab["other_releases"] = (releases & (reason == RU.RELEASE_NONE)).sum(axis=0)
  if "ctrl_t_pass" in traj:
    tp = np.asarray(traj["ctrl_t_pass"])
    idx = np.clip(tab["brace_release_step"], 0, T-1)
    tab["t_pass_at_first_release"] = np.where(releases.any(axis=0), tp[idx, np.arange(n)], np.nan)
  else:
    tab["t_pass_at_first_release"] = np.full(n, np.nan)
  return tab


def _summarize(tab, dt):
  row = ST.summarize_episodes(tab, dt)
  row.update({"braced_s_legacy": float(np.mean(tab["brace_steps_total_legacy"])*dt),
              "n_releases_passed": int(np.sum(tab["passed_releases"])),
              "n_releases_stale": int(np.sum(tab["stale_releases"])),
              "n_releases_other": int(np.sum(tab["other_releases"])),
              "n_episodes_rebraced": int(np.sum(np.asarray(tab["brace_segments"]) > 1)),
              "n_horizon_truncated_braces": int(np.sum(tab["brace_horizon_truncated"])),
              "n_contact_before_termination": int(np.sum(tab["contact_before_termination"]))})
  return row


def _write_episodes(path, tab, method, param, latency, batch):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(["method", "param", "latency_steps", "batch", *tab])
    for i in range(len(tab["fell"])):
      writer.writerow([method, param, latency, batch, *[v[i].item() if isinstance(v[i], np.generic) else v[i] for v in tab.values()]])


def main(_):
  outdir = Path(_OUT.value)
  if any((outdir/name).exists() for name in ("config.json", "summary.json", "episodes.npz", "episodes")):
    raise ValueError("Output already contains evidence; use a new directory rather than overwrite/resume stale data")
  if _N.value < 1 or _BATCHES.value < 1 or _STEPS.value < 1 or _PERIOD.value < 1:
    raise ValueError("Episodes, batches, steps and detector period must be positive")
  if not np.isfinite(_RELAFTER.value) or _RELAFTER.value < 0:
    raise ValueError("release_after must be finite and nonnegative")
  supported = {"reactive", "oracle", "always", "ttc_threshold", "belief_commit", "belief_lead", "belief_wait", "belief_voi"}
  if not _METHODS.value or set(_METHODS.value)-supported or len(set(_METHODS.value)) != len(_METHODS.value):
    raise ValueError("Methods must be nonempty, supported and unique")
  outdir.mkdir(parents=True, exist_ok=True)
  with open(_RM.value) as stream:
    rmj = json.load(stream)
  rm = RU.ResponseModel(rmj["a_cont"], rmj["b_cont"], rmj["a_brace"], rmj["b_brace"],
                        rmj.get("a_brace_lead", 0.0), rmj.get("b_brace_lead", 0.0), rmj.get("c_lead", 0.0),
                        rmj.get("lead_max", 0.5), "c_lead" in rmj)
  if "c_lead" not in rmj and any(m in _METHODS.value for m in ("belief_lead", "belief_wait", "belief_voi")):
    raise SystemExit("Lead-aware methods need the lead-conditioned response model: rerun fit_response_model.py")
  nominal = R.load_policy(_NOM.value)
  # The object-aware walker has no BRACE skill: its obs differ, and only "reactive" applies.
  brace = R.load_policy(_BR.value) if _ENVSKILL.value != "aware" else None
  ov = {}
  if _ENVSKILL.value == "aware":
    ov = {"aware.period": _PERIOD.value, "aware.latency": int(_LAT.value[0]), "aware.sigma0": _SIG0.value,
          "aware.sigma1": _SIG1.value, "aware.miss_prob": _MISSP.value, "aware.fov_deg": _FOV.value,
          "aware.max_range": _MAXRANGE.value}
    print("object-aware env:", ov)
  env = R.make_env(skill=_ENVSKILL.value, impl=_IMPL.value, azimuth_range_deg=_AZ.value, range_dist=[3.0, 5.0],
                   overrides=ov)
  n = _N.value
  lats = [int(x) for x in _LAT.value]
  thetas = [float(x) for x in _THETAS.value]; cpreps = [float(x) for x in _CPREPS.value]
  mr = (float(_MASS.value[0]), float(_MASS.value[1]))
  prior = tuple(float(x) for x in _PRIOR.value) if _PRIOR.value else mr

  conds = []   # (method, param, latency)
  for m in _METHODS.value:
    if m == "reactive":
      conds.append(("reactive", 0.0, int(_LAT.value[0]) if _ENVSKILL.value == "aware" else 0))
    elif m == "oracle":
      conds.append(("oracle", _OLEAD.value, 0))
    elif m == "always":
      conds += [("always", _PMIN.value, L) for L in lats]
    elif m == "ttc_threshold":
      conds += [("ttc_threshold", th, L) for th, L in itertools.product(thetas, lats)]
    elif m in ("belief_commit", "belief_lead", "belief_voi", "belief_wait"):
      conds += [(m, c, L) for c, L in itertools.product(cpreps, lats)]
  if not conds or any(L < 0 for L in lats):
    raise ValueError("Need at least one condition and nonnegative observation latencies")
  print(f"{len(conds)} conditions x {n} episodes x {_BATCHES.value} batches")

  config = {"schema": "foresight.session_h.v1", "episodes": n, "batches": _BATCHES.value,
            "steps": _STEPS.value, "dt": float(env.dt), "seed": _SEED.value,
            "nominal": str(R.latest_checkpoint(_NOM.value)), "brace": str(R.latest_checkpoint(_BR.value)),
            "response_model": str(Path(_RM.value).resolve()), "impl": _IMPL.value,
            "period": _PERIOD.value, "latencies_steps": lats, "conditions": [list(x) for x in conds],
            "mass_range": list(mr), "speed_range": [float(x) for x in _SPEED.value],
            "mass_prior": list(prior), "miss_fraction": _MISS.value, "azimuth_deg": _AZ.value,
            "command_vx": _VX.value, "env_skill": _ENVSKILL.value,
            "detector": {"sigma0": _SIG0.value, "sigma1": _SIG1.value, "miss_prob": _MISSP.value,
                         "fov_deg": _FOV.value, "max_range": _MAXRANGE.value},
            "rule": {"commit_steps": _COMMIT.value, "tau_brace": _TAUB.value, "c_track": _CTRACK.value,
                     "hold": _HOLD.value, "release_after": _RELAFTER.value,
                     "release_on_unwant": _RELEASE.value, "stale_steps": RU.RuleConfig().stale_steps,
                     "mass_prior": list(prior), "r_contact": _RCONT.value},
            "oracle": {"lead_s": _OLEAD.value, "hold_s": _OHOLD.value, "timing": "nominal_arrival",
                       "release_after_does_not_apply": True},
            "protocol": "Session G adaptive closest-approach release; only requested release_after varies in H",
            "response_model_refit_for_hold": False,
            "response_model_caution": "Response and cost hold remain frozen; this tests deployed-policy sensitivity, not optimal decisions under a refitted hold-dependent model.",
            "preparation_metric": "Applied intervals through the first terminal transition; legacy unmasked count retained separately.",
            "recovery_window_s": 2.5, "env_config": env._config.to_dict()}
  provenance = ST.protocol_metadata({"nominal": R.latest_checkpoint(_NOM.value),
                                    "brace": R.latest_checkpoint(_BR.value), "response_model": _RM.value}, config)
  ST.write_json(outdir/"config.json", {**config, "provenance": provenance})
  rows = []
  per_ep = {}
  for b in range(_BATCHES.value):
    rs = np.random.RandomState(_SEED.value + b)
    masses = rs.uniform(*mr, n)
    speeds = rs.uniform(float(_SPEED.value[0]), float(_SPEED.value[1]), n)
    hits = rs.uniform(*env._config.impact.hit_time_range, n)
    is_miss = rs.uniform(size=n) < _MISS.value
    key = jax.random.PRNGKey(_SEED.value + b)
    reset_keys = jax.random.split(jax.random.fold_in(key, 1), n)
    wenv = R.batched_env(env, masses)
    base = dict(speed=speeds, hit_time=hits, is_miss=is_miss, command=[_VX.value, 0, 0])

    t0 = time.time()
    ref_info, _ = R.rollout(env, wenv, nominal, key, _STEPS.value,
                            scenario_fn=lambda s: R.set_scenario(env, s, **base), reset_keys=reset_keys)
    lp, lv = ref_info["proj_launch_pos"], ref_info["proj_launch_vel"]
    print(f"batch {b}: reference {time.time()-t0:.0f}s  contact(hits) {ref_info['impact_contact'][~is_miss].mean():.3f}  "
          f"contact(miss) {ref_info['impact_contact'][is_miss].mean():.3f}  fell(hits) {ref_info['fell'][~is_miss].mean():.3f}")
    scn = lambda s: R.set_scenario(env, s, launch_pos=lp, launch_vel=lv, **base)

    for method, param, lat in conds:
      t0 = time.time()
      if method == "oracle":
        decide = R.oracle_brace(jp.full((n,), param, jp.float32), _OHOLD.value, env.dt)
        info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                               decide=decide, scenario_fn=scn, reset_keys=reset_keys)
      else:
        common = dict(commit_steps=_COMMIT.value, tau_brace=_TAUB.value, c_track=_CTRACK.value, hold=_HOLD.value,
                      release_on_unwant=_RELEASE.value, release_after=_RELAFTER.value)
        rc = {"reactive": RU.RuleConfig(kind="reactive", **common),
              "always": RU.RuleConfig(kind="always", p_min=param, **common),
              "ttc_threshold": RU.RuleConfig(kind="ttc_threshold", theta=param, **common),
              "belief_commit": RU.RuleConfig(kind="belief_commit", c_prep=param, c_fall=1.0, **common),
              "belief_lead": RU.RuleConfig(kind="belief_lead", c_prep=param, c_fall=1.0, **common),
              "belief_voi": RU.RuleConfig(kind="belief_voi", c_prep=param, c_fall=1.0, wait_steps=_PERIOD.value,
                                          **common),
              "belief_wait": RU.RuleConfig(kind="belief_wait", c_prep=param, c_fall=1.0, wait_steps=_PERIOD.value,
                                           **common)}[method]
        ctrl = AnticipationController(D.DetectorConfig(period=_PERIOD.value, latency=lat, sigma0=_SIG0.value,
                                                       sigma1=_SIG1.value, miss_prob=_MISSP.value, fov_deg=_FOV.value,
                                                       max_range=_MAXRANGE.value),
                                      PF.PFConfig(mass_range=prior, r_contact=_RCONT.value), rc, rm)
        info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                               scenario_fn=scn, reset_keys=reset_keys, controller=ctrl)
      tab = _episode_evidence(R.episode_table(info, traj, env.dt), traj, _SEED.value+b, env.dt)
      if method == "belief_commit" and _CALIB.value and any(abs(param - float(c)) < 1e-9 for c in _CALIB.value if c):
        # E9: per-step belief against the eventual outcome (compressed; p_hit is mostly zero).
        cdir = os.path.join(_OUT.value, "calib"); os.makedirs(cdir, exist_ok=True)
        np.savez_compressed(os.path.join(cdir, f"belief_c{param:g}_lat{lat}_b{b}.npz"),
                            p_hit=traj["ctrl_p_hit"].astype(np.float16), t_hit=traj["ctrl_t_hit"].astype(np.float16),
                            pe_t_hit=traj["ctrl_pe_t_hit"].astype(np.float16), det_valid=traj["ctrl_det_valid"],
                            n_obs=traj["ctrl_n_obs"].astype(np.int16), brace_active=traj["brace_active"],
                            contact_step=tab["contact_step"], is_miss=tab["is_miss"], fell=tab["fell"],
                            contact=tab["contact"], t_hit_nominal=tab["t_hit_nominal"], mass=tab["mass"],
                            speed=tab["speed"], launch_step=info["proj_launch_step"],
                            period=_PERIOD.value, latency=lat, dt=env.dt)
      if "false_prep" not in tab:
        tab["false_prep"] = info["proj_is_miss"] & traj["brace_active"].any(axis=0)
      hit_s = ~tab["is_miss"]; c = tab["contact"] & hit_s
      # Keep scenario identifiers and sufficient statistics for pairing across hold arms.
      per_ep.setdefault((method, param, lat), []).append(tab)
      cond_i = conds.index((method, param, lat))
      _write_episodes(outdir/"episodes"/f"batch{b:03d}_condition{cond_i:03d}.csv", tab,
                      method, param, lat, b)
      row = {"method": method, "param": param, "latency_steps": lat, "batch": b, "n": n,
             **_summarize(tab, env.dt)}
      rows.append(row)
      shown = {k: (float("nan") if v is None else v) for k,v in row.items()}
      print(f"  {method:14} p={param:<5g} lat={lat:<3} {time.time()-t0:4.0f}s  fall|hits {shown['fall_rate_hits']:.3f}  "
            f"fall|contact {shown['fall_rate_given_contact']:.3f}  false-prep {shown['false_prep_rate']:.3f}  "
            f"braced {shown['braced_s']:.2f}s  progress {shown['progress_m']:.2f}")

  # Pool episode numerators/denominators, not averages of conditional batch rates.
  pooled = {key: ST.concatenate_tables(parts) for key, parts in per_ep.items()}
  ep_out = {}
  for (meth, par, lat), tab in pooled.items():
    tag = f"{meth}__p{par:g}__lat{lat}"
    for name, values in tab.items():
      if np.asarray(values).dtype.kind in "biufc":
        ep_out[f"{tag}__{name}"] = values
    # Backward-compatible numeric aliases for existing analysis scripts.
    ep_out[f"{tag}__brace_steps"] = tab["brace_steps_total"]
    ep_out[f"{tag}__brace_onset"] = tab["brace_onset_step"]
  np.savez_compressed(outdir/"episodes.npz", **ep_out)
  out = []
  for (meth, par, lat), tab in pooled.items():
    out.append({"method": meth, "param": par, "latency_steps": lat, **_summarize(tab, env.dt)})
  reactive = [r for r in out if r["method"] == "reactive"]
  base_prog = reactive[0]["progress_m"] if reactive else None
  for row in out:
    row["progress_loss_m"] = None if base_prog is None else base_prog-row["progress_m"]
  with (outdir/"frontier.csv").open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(out[0]))
    writer.writeheader(); writer.writerows(out)
  ST.write_json(outdir/"batch_summaries.json", rows)
  paired = []
  reference = next((k for k in pooled if k[0] == "reactive"), None)
  if reference is not None:
    for key, tab in pooled.items():
      if key == reference:
        continue
      comparison = ST.paired_comparison(tab, pooled[reference], env.dt)
      comparison["braced_s_legacy"] = ST.paired_difference(
          tab["brace_steps_total_legacy"]*env.dt, pooled[reference]["brace_steps_total_legacy"]*env.dt)
      paired.append({"condition_a": list(key), "condition_b": list(reference), **comparison})
  ST.write_json(outdir/"paired.json", paired)

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(lats), figsize=(5.2 * len(lats), 4.4), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    col = {"ttc_threshold": "#2b6f8e", "belief_commit": "#d9772b", "belief_lead": "#b5179e", "belief_voi": "#7209b7",
           "always": "#6a7179", "reactive": "#14213d", "oracle": "#3a9d5d"}
    col["belief_wait"] = "#4cc9f0"
    for ax, L in zip(axes, lats):
      for m in ("ttc_threshold", "belief_commit", "belief_lead", "belief_wait", "belief_voi"):
        pts = sorted([o for o in out if o["method"] == m and o["latency_steps"] == L], key=lambda o: o["false_prep_rate"])
        if pts:
          ax.plot([o["false_prep_rate"] for o in pts], [o["fall_rate_hits"] for o in pts], "o-", color=col[m], label=m)
      for m in ("reactive", "oracle"):
        pts = [o for o in out if o["method"] == m]
        if pts:
          ax.plot([pts[0]["false_prep_rate"]], [pts[0]["fall_rate_hits"]], "*", ms=14, color=col[m], label=m)
      pts = [o for o in out if o["method"] == "always" and o["latency_steps"] == L]
      if pts:
        ax.plot([pts[0]["false_prep_rate"]], [pts[0]["fall_rate_hits"]], "s", ms=9, color=col["always"], label="always")
      ax.set_xlabel("false preparation rate (braced on near-misses)"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(0, 1)
      ax.set_title(f"detector latency {L*env.dt*1000:.0f} ms", loc="left", fontsize=10)
      ax.grid(True, color="#e3e6ea", lw=0.6)
    axes[0].set_ylabel("failure rate on scheduled hits"); axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Decision frontier: lower-left is better", fontsize=11)
    fig.savefig(os.path.join(_OUT.value, "frontier.png"), dpi=170)
  except Exception as e:
    print("plot skipped:", e)

  ST.write_json(outdir/"summary.json", {"schema": "foresight.session_h.v1", "status": "complete",
                "nominal": config["nominal"], "brace": config["brace"],
                "episodes_per_condition": n*_BATCHES.value, "latencies_steps": lats,
                "period": _PERIOD.value, "env_skill": _ENVSKILL.value,
                "detector": config["detector"], "rule": config["rule"],
                "miss_fraction": _MISS.value, "rows": out, "config": config, "provenance": provenance})


if __name__ == "__main__":
  app.run(main)
