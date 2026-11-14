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
import json
import os
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R
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
_HOLD = flags.DEFINE_float("hold", 0.5, "s held after expected impact")
_PRIOR = flags.DEFINE_list("mass_prior", [], "belief mass prior range, kg (default: the scenario range)")
# End-to-end baseline (E14): run the object-aware walker itself, no decision layer
_ENVSKILL = flags.DEFINE_string("env_skill", "nominal", "nominal | aware (the policy in --nominal must match)")
_CALIB = flags.DEFINE_list("calib_params", ["0.1"], "belief_commit c_prep values whose per-step belief "
                           "logs are dumped to <out>/calib/ for E9 calibration (empty: none)")


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  rmj = json.load(open(_RM.value))
  rm = RU.ResponseModel(rmj["a_cont"], rmj["b_cont"], rmj["a_brace"], rmj["b_brace"])
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
    elif m == "belief_commit":
      conds += [("belief_commit", c, L) for c, L in itertools.product(cpreps, lats)]
  print(f"{len(conds)} conditions x {n} episodes x {_BATCHES.value} batches")

  rows = []
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
        decide = R.oracle_brace(jp.full((n,), param, jp.float32), 0.5, env.dt)
        info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                               decide=decide, scenario_fn=scn, reset_keys=reset_keys)
      else:
        common = dict(commit_steps=_COMMIT.value, tau_brace=_TAUB.value, c_track=_CTRACK.value, hold=_HOLD.value)
        rc = {"reactive": RU.RuleConfig(kind="reactive", **common),
              "always": RU.RuleConfig(kind="always", p_min=param, **common),
              "ttc_threshold": RU.RuleConfig(kind="ttc_threshold", theta=param, **common),
              "belief_commit": RU.RuleConfig(kind="belief_commit", c_prep=param, c_fall=1.0, **common)}[method]
        ctrl = AnticipationController(D.DetectorConfig(period=_PERIOD.value, latency=lat, sigma0=_SIG0.value,
                                                       sigma1=_SIG1.value, miss_prob=_MISSP.value, fov_deg=_FOV.value,
                                                       max_range=_MAXRANGE.value),
                                      PF.PFConfig(mass_range=prior), rc, rm)
        info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                               scenario_fn=scn, reset_keys=reset_keys, controller=ctrl)
      tab = R.episode_table(info, traj, env.dt)
      if method == "belief_commit" and any(abs(param - float(c)) < 1e-9 for c in _CALIB.value):
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
      row = {"method": method, "param": param, "latency_steps": lat, "batch": b, "n": n,
             "fall_rate_hits": float(tab["fell"][hit_s].mean()),
             "fall_rate_given_contact": float(tab["fell"][c].mean()) if c.any() else float("nan"),
             "contact_rate_hits": float(tab["contact"][hit_s].mean()),
             "false_prep_rate": float(tab["false_prep"][tab["is_miss"]].mean()) if tab["is_miss"].any() else 0.0,
             "braced_on_hits": float(np.mean(tab["brace_steps_total"][hit_s] > 0)),
             "braced_s": float(tab["brace_steps_total"].mean() * env.dt),
             "progress_m": float(tab["progress_m"].mean()),
             "fall_rate_miss": float(tab["fell"][tab["is_miss"]].mean()) if tab["is_miss"].any() else 0.0}
      rows.append(row)
      print(f"  {method:14} p={param:<5g} lat={lat:<3} {time.time()-t0:4.0f}s  fall|hits {row['fall_rate_hits']:.3f}  "
            f"fall|contact {row['fall_rate_given_contact']:.3f}  false-prep {row['false_prep_rate']:.3f}  "
            f"braced {row['braced_s']:.2f}s  progress {row['progress_m']:.2f}")

  # Aggregate over batches.
  keys = ["method", "param", "latency_steps"]
  agg = {}
  for r in rows:
    k = tuple(r[x] for x in keys)
    agg.setdefault(k, []).append(r)
  react = [r for r in rows if r["method"] == "reactive"]
  base_prog = float(np.mean([r["progress_m"] for r in react])) if react else float("nan")
  out = []
  for k, rs_ in agg.items():
    o = {x: k[i] for i, x in enumerate(keys)}
    for f in ("fall_rate_hits", "fall_rate_given_contact", "contact_rate_hits", "false_prep_rate",
              "braced_on_hits", "braced_s", "progress_m", "fall_rate_miss"):
      o[f] = float(np.nanmean([r[f] for r in rs_]))
    o["progress_loss_m"] = base_prog - o["progress_m"]
    o["n_total"] = int(sum(r["n"] for r in rs_))
    out.append(o)
  cols = list(out[0].keys())
  with open(os.path.join(_OUT.value, "frontier.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for o in out:
      f.write(",".join(f"{o[c]:.4g}" if isinstance(o[c], float) else str(o[c]) for c in cols) + "\n")
  json.dump({"nominal": str(R.latest_checkpoint(_NOM.value)), "brace": str(R.latest_checkpoint(_BR.value)),
             "episodes_per_condition": n * _BATCHES.value, "latencies_steps": lats, "period": _PERIOD.value,
             "env_skill": _ENVSKILL.value, "detector": {"sigma0": _SIG0.value, "sigma1": _SIG1.value, "miss_prob": _MISSP.value,
                                                        "fov_deg": _FOV.value, "max_range": _MAXRANGE.value},
             "rule": {"commit_steps": _COMMIT.value, "tau_brace": _TAUB.value, "c_track": _CTRACK.value, "hold": _HOLD.value,
                      "mass_prior": list(prior)},
             "miss_fraction": _MISS.value, "rows": out},
            open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(lats), figsize=(5.2 * len(lats), 4.4), constrained_layout=True, sharey=True)
    axes = np.atleast_1d(axes)
    col = {"ttc_threshold": "#2b6f8e", "belief_commit": "#d9772b", "always": "#6a7179", "reactive": "#14213d", "oracle": "#3a9d5d"}
    for ax, L in zip(axes, lats):
      for m in ("ttc_threshold", "belief_commit"):
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


if __name__ == "__main__":
  app.run(main)
