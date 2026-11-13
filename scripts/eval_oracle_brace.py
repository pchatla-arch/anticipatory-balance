# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E6 (oracle): does a well-timed BRACE help, what does it cost, and is the switch itself safe?

Conditions, all on the same sampled scenarios (paired reset keys, speeds, hit times) and,
with --shared_trajectory, the same world-space projectile trajectories replayed from a
CONTINUE reference rollout (so a policy that moves differently still meets the same object):

  mode=brace    switch to the BRACE skill from t_hit - lead until t_hit + hold
  mode=zerocmd  same schedule, but the nominal walker with a zero velocity command
  launch=0      same schedule with the projectile never launched (transition control)

lead = -1 is CONTINUE (never switch); a large lead (9) switches from the episode start.

Readiness tau_brace: time from switch until pelvis height <= target + 0.03 m, torso up-vector
> 0.9 and both feet in contact, sustained for 0.2 s.

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
                           "switch lead times before nominal hit, s (-1 = never)")
_HOLD = flags.DEFINE_float("hold", 0.5, "keep the mode this long after nominal hit, s")
_EPL = flags.DEFINE_integer("episodes_per_lead", 256, "episodes per lead per batch")
_BATCHES = flags.DEFINE_integer("batches", 2, "batches (seeds)")
_STEPS = flags.DEFINE_integer("steps", 300, "control steps")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command, m/s")
_MASS = flags.DEFINE_list("mass_range", ["4", "16"], "uniform projectile mass range, kg")
_SPEED = flags.DEFINE_list("speed_range", ["3", "8"], "uniform projectile speed range, m/s")
_MODES = flags.DEFINE_list("modes", ["brace", "zerocmd"], "brace | zerocmd")
_CONTROLS = flags.DEFINE_bool("no_launch_controls", True, "also run every condition without the projectile")
_SHARED = flags.DEFINE_bool("shared_trajectory", True, "replay the CONTINUE reference's projectile trajectories")
_OUT = flags.DEFINE_string("out", "results/E6_oracle", "output dir")
_SEED = flags.DEFINE_integer("seed", 100, "seed")


def readiness(traj, dt, z_target):
  """tau from switch onset to a sustained braced posture; NaN if never reached."""
  T, n = traj["brace_active"].shape
  onset = np.argmax(traj["brace_active"], axis=0)
  any_on = traj["brace_active"].any(axis=0)
  # z_target is the pelvis-height target of the BRACE reward (pelvis z, not the CoM).
  ok = (traj["pelvis_pos"][..., 2] <= z_target + 0.03) & (traj["torso_up"] > 0.9) & traj["feet_contact"].all(axis=-1)
  tau = np.full(n, np.nan)
  k = int(round(0.2 / dt))
  for i in np.where(any_on)[0]:
    seq = ok[onset[i]:, i]
    run = 0
    for t, v in enumerate(seq):
      run = run + 1 if v else 0
      if run >= k:
        tau[i] = (t - k + 1) * dt
        break
  return tau


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  leads = [float(x) for x in _LEADS.value]
  epl, nl = _EPL.value, len(leads)
  n = nl * epl
  lead_arr = np.repeat(np.array(leads, np.float32), epl)
  nominal = R.load_policy(_NOM.value)
  brace = R.load_policy(_BR.value)
  env = R.make_env(impl=_IMPL.value)
  z_target = float(env._config.brace.base_height_target)
  decide = R.oracle_brace(jp.asarray(lead_arr), _HOLD.value, env.dt)
  conds = [(m, True) for m in _MODES.value]
  if _CONTROLS.value:
    conds += [(m, False) for m in _MODES.value]

  rows = []
  for b in range(_BATCHES.value):
    rs = np.random.RandomState(_SEED.value + b)
    m_block = rs.uniform(float(_MASS.value[0]), float(_MASS.value[1]), epl)
    s_block = rs.uniform(float(_SPEED.value[0]), float(_SPEED.value[1]), epl)
    h_block = rs.uniform(*env._config.impact.hit_time_range, epl)
    key = jax.random.PRNGKey(_SEED.value + b)
    keys_block = jax.random.split(jax.random.fold_in(key, 1), epl)
    masses, speeds, hits = (np.tile(x, nl) for x in (m_block, s_block, h_block))
    reset_keys = jp.tile(keys_block, (nl, 1))
    wenv = R.batched_env(env, masses)

    # Reference: CONTINUE on the block, to fix the projectile trajectories.
    lp = lv = None
    if _SHARED.value:
      wref = R.batched_env(env, m_block)
      ref_scn = lambda s: R.set_scenario(env, s, speed=s_block, hit_time=h_block,
                                         is_miss=np.zeros(epl, bool), command=[_VX.value, 0, 0])
      t0 = time.time()
      ref_info, _ = R.rollout(env, wref, nominal, key, _STEPS.value, scenario_fn=ref_scn,
                              reset_keys=keys_block)
      lp = np.tile(ref_info["proj_launch_pos"], (nl, 1)); lv = np.tile(ref_info["proj_launch_vel"], (nl, 1))
      print(f"batch {b}: reference CONTINUE rollout {time.time()-t0:.0f}s, "
            f"contact {ref_info['impact_contact'].mean():.3f}, fell {ref_info['fell'].mean():.3f}")

    for mode, launch in conds:
      scn = lambda s, launch=launch: R.set_scenario(
          env, s, speed=speeds, hit_time=hits, is_miss=np.zeros(n, bool),
          command=[_VX.value, 0, 0], launch=launch, launch_pos=lp, launch_vel=lv)
      t0 = time.time()
      info, traj = R.rollout(env, wenv, nominal, key, _STEPS.value,
                             brace_policy=(brace if mode == "brace" else nominal),
                             decide=decide, scenario_fn=scn, reset_keys=reset_keys)
      tab = R.episode_table(info, traj, env.dt)
      tab["lead"] = lead_arr; tab["batch"] = np.full(n, b)
      tab["mode"] = np.array([mode] * n); tab["launch"] = np.full(n, int(launch))
      tab["tau_ready_s"] = readiness(traj, env.dt, z_target)
      rows.append(tab)
      print(f"batch {b} mode={mode} launch={int(launch)}: {time.time()-t0:.0f}s")
      for L in leads:
        sel = lead_arr == L
        tr = tab["tau_ready_s"][sel]
        print(f"  lead {L:5.2f}: fall {tab['fell'][sel].mean():.3f}  contact {tab['contact'][sel].mean():.3f}  "
              f"progress {tab['progress_m'][sel].mean():.2f} m  switched {tab['brace_steps_total'][sel].mean()*env.dt:.2f} s  "
              f"ready {np.isfinite(tr).mean():.2f} tau {np.nanmean(tr) if np.isfinite(tr).any() else float('nan'):.2f} s")

  tab = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
  # write_csv wants numeric/bool columns; encode mode as a flag.
  tab["mode_brace"] = (tab["mode"] == "brace")
  R.write_csv(os.path.join(_OUT.value, "episodes.csv"), {k: v for k, v in tab.items() if k != "mode"})

  curve = []
  for mode, launch in conds:
    base = (tab["mode"] == mode) & (tab["launch"] == int(launch)) & (tab["lead"] == -1)
    for L in leads:
      sel = (tab["mode"] == mode) & (tab["launch"] == int(launch)) & (tab["lead"] == L)
      tr = tab["tau_ready_s"][sel]
      curve.append({"mode": mode, "launch": int(launch), "lead_s": L, "n": int(sel.sum()),
                    "fall_rate": float(tab["fell"][sel].mean()),
                    "contact_rate": float(tab["contact"][sel].mean()),
                    "censored_rate": float(tab["censored"][sel].mean()),
                    "progress_m": float(tab["progress_m"][sel].mean()),
                    "progress_loss_m": float(tab["progress_m"][base].mean() - tab["progress_m"][sel].mean()),
                    "switched_s": float(tab["brace_steps_total"][sel].mean() * env.dt),
                    "ready_frac": float(np.isfinite(tr).mean()),
                    "tau_ready_s": float(np.nanmean(tr)) if np.isfinite(tr).any() else None})
  keys = list(curve[0].keys())
  with open(os.path.join(_OUT.value, "lead_curve.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for c in curve:
      f.write(",".join("" if c[k] is None else (f"{c[k]:.4g}" if isinstance(c[k], float) else str(c[k])) for k in keys) + "\n")
  json.dump({"nominal": str(R.latest_checkpoint(_NOM.value)), "brace": str(R.latest_checkpoint(_BR.value)),
             "hold_s": _HOLD.value, "shared_trajectory": _SHARED.value,
             "episodes_per_condition": epl * _BATCHES.value, "curve": curve},
            open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  for c in curve:
    print(c)

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.9), constrained_layout=True)
    styles = {("brace", 1): ("#b5361c", "-", "BRACE, projectile"), ("brace", 0): ("#b5361c", ":", "BRACE, no projectile"),
              ("zerocmd", 1): ("#2b5f9e", "-", "zero command, projectile"), ("zerocmd", 0): ("#2b5f9e", ":", "zero command, no projectile")}
    for (mode, launch), (col, ls, lab) in styles.items():
      pts = [c for c in curve if c["mode"] == mode and c["launch"] == launch and 0 <= c["lead_s"] < 5]
      if not pts:
        continue
      xs = [c["lead_s"] for c in pts]
      a1.plot(xs, [c["fall_rate"] for c in pts], "o", ls=ls, color=col, label=lab)
      a2.plot(xs, [c["progress_loss_m"] for c in pts], "s", ls=ls, color=col, label=lab)
    cont = [c for c in curve if c["lead_s"] == -1 and c["launch"] == 1]
    if cont:
      a1.axhline(cont[0]["fall_rate"], color="#333", ls="--", lw=1, label="CONTINUE")
    a1.set_xlabel("switch lead before nominal impact (s)"); a1.set_ylabel("P(failure)"); a1.set_ylim(0, 1)
    a2.set_xlabel("switch lead before nominal impact (s)"); a2.set_ylabel("progress lost vs CONTINUE (m)")
    a1.legend(fontsize=7); a1.set_title("Failure vs lead time", loc="left", fontsize=10)
    a2.set_title("Cost vs lead time", loc="left", fontsize=10)
    fig.savefig(os.path.join(_OUT.value, "oracle_lead_curve.png"), dpi=170)
  except Exception as e:
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
