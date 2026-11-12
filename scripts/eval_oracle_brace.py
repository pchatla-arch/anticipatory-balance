# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E6 (oracle): does a well-timed BRACE help, and what does it cost?

Privileged rule: for true hits, switch to the BRACE skill from t_hit - lead until
t_hit + hold, then return to walking. lead = -1 is the CONTINUE baseline (never brace);
a large lead (e.g. 9) braces from the start of the episode ("always prepare").
Scenarios are paired across leads: same seed, same (mass, speed, hit time, azimuth).

Measures per lead: fall rate, forward progress, time spent braced, and tau_brace (time from
brace onset until the CoM is within 5 cm of the brace height target).

  python scripts/eval_oracle_brace.py --nominal logs/nominal_s1/<run>/checkpoints \
      --brace logs/brace_s1/<run>/checkpoints --impl warp --out results/E6_oracle
"""
import json
import os
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R

_NOM = flags.DEFINE_string("nominal", None, "nominal checkpoint", required=True)
_BR = flags.DEFINE_string("brace", None, "brace checkpoint", required=True)
_IMPL = flags.DEFINE_string("impl", "warp", "warp | jax")
_LEADS = flags.DEFINE_list("leads", ["-1", "0.1", "0.2", "0.3", "0.5", "0.75", "1.0", "9"],
                           "brace lead times before nominal hit, s (-1 = never)")
_HOLD = flags.DEFINE_float("hold", 0.5, "keep bracing this long after nominal hit, s")
_EPL = flags.DEFINE_integer("episodes_per_lead", 256, "episodes per lead per batch")
_BATCHES = flags.DEFINE_integer("batches", 2, "batches (seeds)")
_STEPS = flags.DEFINE_integer("steps", 300, "control steps")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command, m/s")
_MASS = flags.DEFINE_list("mass_range", ["1", "8"], "uniform projectile mass range, kg")
_OUT = flags.DEFINE_string("out", "results/E6_oracle", "output dir")
_SEED = flags.DEFINE_integer("seed", 100, "seed")


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  leads = [float(x) for x in _LEADS.value]
  epl = _EPL.value
  n = len(leads) * epl
  lead_arr = np.repeat(np.array(leads, np.float32), epl)
  nominal = R.load_policy(_NOM.value)
  brace = R.load_policy(_BR.value)
  env = R.make_env(impl=_IMPL.value)
  z_target = float(env._config.brace.base_height_target)
  decide = R.oracle_brace(jp.asarray(lead_arr), _HOLD.value, env.dt)

  rows = []
  for b in range(_BATCHES.value):
    rs = np.random.RandomState(_SEED.value + b)
    # Paired scenarios: the same draws for every lead block.
    m_block = rs.uniform(float(_MASS.value[0]), float(_MASS.value[1]), epl)
    masses = np.tile(m_block, len(leads))
    wenv = R.batched_env(env, masses)
    speeds = np.tile(rs.uniform(*env._config.impact.speed_range, epl), len(leads))
    hits = np.tile(rs.uniform(*env._config.impact.hit_time_range, epl), len(leads))
    scenario = lambda s: R.set_scenario(env, s, speed=speeds, hit_time=hits,
                                        is_miss=np.zeros(n, bool), command=[_VX.value, 0, 0])
    # One reset key per paired episode, repeated across lead blocks -> identical azimuth,
    # launch distance and initial pose.
    key = jax.random.PRNGKey(_SEED.value + b)
    reset_keys = jp.tile(jax.random.split(jax.random.fold_in(key, 1), epl), (len(leads), 1))
    t0 = time.time()
    info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value, brace_policy=brace,
                           decide=decide, scenario_fn=scenario, reset_keys=reset_keys)
    tab = R.episode_table(info, traj, env.dt)
    tab["lead"] = lead_arr
    tab["batch"] = np.full(n, b)
    # tau_brace: steps from brace onset to CoM z <= z_target + 0.05.
    onset = np.argmax(traj["brace_active"], axis=0)
    braced_any = traj["brace_active"].any(axis=0)
    low = traj["com"][..., 2] <= z_target + 0.05
    tau = np.full(n, np.nan)
    for i in np.where(braced_any)[0]:
      after = np.where(low[onset[i]:, i])[0]
      if len(after):
        tau[i] = after[0] * env.dt
    tab["tau_brace_s"] = tau
    rows.append(tab)
    print(f"batch {b}: {time.time()-t0:.0f}s")
    for L in leads:
      sel = lead_arr == L
      print(f"  lead {L:5.2f}: fall {tab['fell'][sel].mean():.3f}  contact {tab['contact'][sel].mean():.3f}  "
            f"progress {tab['progress_m'][sel].mean():.2f} m  braced {tab['brace_steps_total'][sel].mean()*env.dt:.2f} s  "
            f"tau_brace {np.nanmean(tau[sel]) if np.isfinite(tau[sel]).any() else float('nan'):.2f} s")
  tab = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
  R.write_csv(os.path.join(_OUT.value, "episodes.csv"), tab)

  curve = []
  base = tab["lead"] == -1
  for L in leads:
    sel = tab["lead"] == L
    curve.append({"lead_s": L, "n": int(sel.sum()), "fall_rate": float(tab["fell"][sel].mean()),
                  "contact_rate": float(tab["contact"][sel].mean()),
                  "progress_m": float(tab["progress_m"][sel].mean()),
                  "progress_loss_m": float(tab["progress_m"][base].mean() - tab["progress_m"][sel].mean()),
                  "braced_s": float(tab["brace_steps_total"][sel].mean() * env.dt),
                  "tau_brace_s": float(np.nanmean(tab["tau_brace_s"][sel])) if np.isfinite(tab["tau_brace_s"][sel]).any() else None})
  with open(os.path.join(_OUT.value, "lead_curve.csv"), "w") as f:
    keys = list(curve[0].keys()); f.write(",".join(keys) + "\n")
    for c in curve:
      f.write(",".join("" if c[k] is None else (f"{c[k]:.4g}" if isinstance(c[k], float) else str(c[k])) for k in keys) + "\n")
  summary = {"nominal": str(R.latest_checkpoint(_NOM.value)), "brace": str(R.latest_checkpoint(_BR.value)),
             "hold_s": _HOLD.value, "episodes_per_lead": epl * _BATCHES.value, "curve": curve}
  json.dump(summary, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  print(json.dumps(curve, indent=2))

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    xs = [c["lead_s"] for c in curve if 0 <= c["lead_s"] < 5]
    fr = [c["fall_rate"] for c in curve if 0 <= c["lead_s"] < 5]
    pl = [c["progress_loss_m"] for c in curve if 0 <= c["lead_s"] < 5]
    fig, ax1 = plt.subplots(figsize=(5.4, 3.8))
    ax1.plot(xs, fr, "o-", color="#b5361c", label="fall rate (oracle brace)")
    ax1.axhline(curve[0]["fall_rate"], color="#b5361c", ls="--", label="CONTINUE baseline")
    alw = [c for c in curve if c["lead_s"] >= 5]
    if alw:
      ax1.axhline(alw[0]["fall_rate"], color="#b5361c", ls=":", label="always brace")
    ax1.set_xlabel("brace lead time before impact (s)"); ax1.set_ylabel("P(fall)")
    ax1.set_ylim(0, 1)
    ax2 = ax1.twinx()
    ax2.plot(xs, pl, "s-", color="#2b5f9e", label="progress lost (m)")
    ax2.set_ylabel("forward progress lost vs CONTINUE (m)")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=7, loc="center right")
    ax1.set_title("Oracle brace: benefit vs lead time")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT.value, "oracle_lead_curve.png"), dpi=160)
  except Exception as e:
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
