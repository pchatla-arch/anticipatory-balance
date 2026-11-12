# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E4: does the projectile hit where and when it is aimed, and how much impulse transfers?

Base pinned (hold_base), deterministic pose, no policy (zero action = PD holds the pose),
one env per (mass, speed) cell. Records, per episode: intended vs actual impact point,
arrival-time error vs t_hit_nominal, incoming momentum m*v and transferred impulse
m*|v_before - v_after|.

  python scripts/validate_contact.py --impl warp --out results/E4_contact
"""
import json
import os

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R

_IMPL = flags.DEFINE_string("impl", "warp", "mjx impl: warp | jax")
_MASSES = flags.DEFINE_list("masses", ["1", "2", "4", "8"], "kg")
_SPEEDS = flags.DEFINE_list("speeds", ["1", "2", "3", "4"], "m/s")
_REPS = flags.DEFINE_integer("reps", 1, "episodes per cell (different launch azimuths)")
_OUT = flags.DEFINE_string("out", "results/E4_contact", "output dir")


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  masses = [float(m) for m in _MASSES.value]; speeds = [float(s) for s in _SPEEDS.value]
  cells = [(m, s) for m in masses for s in speeds for _ in range(_REPS.value)]
  n = len(cells)
  env = R.make_env(impl=_IMPL.value, reset_perturb=0.0, hold_base=True)
  wenv = R.batched_env(env, np.array([m for m, _ in cells]))
  zero_policy = lambda obs, key: (jp.zeros(env.action_size), {})
  scenario = lambda s: R.set_scenario(env, s, speed=np.array([v for _, v in cells]),
                                      hit_time=np.full(n, 1.5), command=[0.0, 0.0, 0.0])
  steps = 150
  info, traj = R.rollout(env, wenv, zero_policy, jax.random.PRNGKey(0), steps, scenario_fn=scenario)
  dt = env.dt
  robot_mass = float(np.sum(np.asarray(env.mj_model.body_mass)) - float(np.asarray(env.mj_model.body_mass)[env._proj_body]))

  rows = []
  for i, (m, s) in enumerate(cells):
    cs = int(info["contact_step"][i]); ls = int(info["proj_launch_step"][i])
    launch_pelvis = traj["pelvis_pos"][max(ls - 1, 0), i]
    intended = launch_pelvis + np.array([0, 0, 0.25])
    r = {"mass": m, "speed": s, "momentum_in": m * s, "launched": int(info["proj_launched"][i]),
         "contact": int(info["impact_contact"][i]), "contact_step": cs,
         "t_hit_nominal": float(info["t_hit_nominal"][i])}
    if cs >= 0:
      pre = max(cs - 1, 0); post = min(cs + 2, steps - 1)
      p_hit = traj["proj_pos"][cs, i]
      v_pre = traj["proj_vel"][pre, i]; v_post = traj["proj_vel"][post, i]
      r.update({"t_contact": cs * dt, "t_err_s": cs * dt - r["t_hit_nominal"],
                "pos_err_m": float(np.linalg.norm(p_hit - intended)),
                "pos_err_xy_m": float(np.linalg.norm((p_hit - intended)[:2])),
                "pos_err_z_m": float((p_hit - intended)[2]),
                "speed_pre": float(np.linalg.norm(v_pre)),
                "impulse_transferred": float(m * np.linalg.norm(v_pre - v_post)),
                "momentum_pre": float(m * np.linalg.norm(v_pre[:2]))})
    rows.append(r)
  keys = ["mass", "speed", "momentum_in", "launched", "contact", "contact_step", "t_hit_nominal",
          "t_contact", "t_err_s", "pos_err_m", "pos_err_xy_m", "pos_err_z_m", "speed_pre",
          "momentum_pre", "impulse_transferred"]
  with open(os.path.join(_OUT.value, "contact_table.csv"), "w") as f:
    f.write(",".join(keys) + "\n")
    for r in rows:
      f.write(",".join(f"{r[k]:.4g}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")
  hit = [r for r in rows if r["contact"]]
  summary = {
      "n": n, "launched": int(sum(r["launched"] for r in rows)), "contacts": len(hit),
      "robot_mass_kg": robot_mass,
      "pos_err_m_mean": float(np.mean([r["pos_err_m"] for r in hit])) if hit else None,
      "t_err_s_mean": float(np.mean([r["t_err_s"] for r in hit])) if hit else None,
      "t_err_s_absmax": float(np.max([abs(r["t_err_s"]) for r in hit])) if hit else None,
      "impulse_over_momentum_mean": float(np.mean([r["impulse_transferred"] / r["momentum_pre"] for r in hit])) if hit else None,
  }
  json.dump(summary, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  print("mass speed  contact step  t_err(s)  pos_err(m)  J_in(N·s)  J_transferred(N·s)")
  for r in rows:
    if r["contact"]:
      print(f"{r['mass']:4g} {r['speed']:5g}  {r['contact_step']:5d}   {r['t_err_s']:+.3f}    "
            f"{r['pos_err_m']:.3f}    {r['momentum_pre']:6.1f}     {r['impulse_transferred']:6.1f}")
    else:
      print(f"{r['mass']:4g} {r['speed']:5g}   NO CONTACT (launched={r['launched']})")
  print(json.dumps(summary, indent=2))


if __name__ == "__main__":
  app.run(main)
