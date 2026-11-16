# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Side-by-side video: the same replayed impact, CONTINUE (walker only) vs BRACE (oracle
switch at a fixed lead before nominal arrival). Headless (MUJOCO_GL=egl), CPU or GPU.

  MUJOCO_GL=egl python scripts/render_compare.py --mass 12 --speed 6 --lead 0.5 \
      --out results/videos/brace_vs_continue.mp4
"""
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import jax
import jax.numpy as jp
import mujoco
import numpy as np
from absl import app, flags

from foresight.eval import rollout as R

_NOM = flags.DEFINE_string("nominal", "checkpoints/nominal_s1", "walker")
_BR = flags.DEFINE_string("brace", "checkpoints/brace2_s1", "BRACE skill")
_MASS = flags.DEFINE_float("mass", 12.0, "kg")
_SPEED = flags.DEFINE_float("speed", 6.0, "m/s")
_HIT = flags.DEFINE_float("hit_time", 2.5, "nominal arrival, s")
_LEAD = flags.DEFINE_float("lead", 0.5, "BRACE switch lead before nominal arrival, s")
_SECONDS = flags.DEFINE_float("seconds", 5.0, "duration")
_SEED = flags.DEFINE_integer("seed", 11, "reset seed")
_IMPL = flags.DEFINE_string("impl", "jax", "jax (CPU) or warp (GPU)")
_OUT = flags.DEFINE_string("out", "results/videos/brace_vs_continue.mp4", "output mp4")
_SEARCH = flags.DEFINE_bool("search", False, "evaluate a fixed grid of throws in one batch and render the FIRST one "
                            "(in grid order) where CONTINUE falls and BRACE stands; the rule is printed")


def main(_):
  os.makedirs(os.path.dirname(_OUT.value) or ".", exist_ok=True)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=30.0, range_dist=[3.0, 4.0])
  nominal, brace = R.load_policy(_NOM.value), R.load_policy(_BR.value)
  steps = int(_SECONDS.value / env.dt)
  key = jax.random.PRNGKey(_SEED.value)
  if _SEARCH.value:
    grid = [(m, v) for m in (6.0, 8.0, 10.0) for v in (4.0, 5.0, 6.0)]
  else:
    grid = [(_MASS.value, _SPEED.value)]
  n = len(grid)
  masses = np.array([g[0] for g in grid]); speeds = np.array([g[1] for g in grid])
  rk = jax.random.split(jax.random.fold_in(key, 1), n)
  wenv = R.batched_env(env, masses)
  base = dict(speed=speeds, hit_time=np.full(n, _HIT.value), is_miss=np.zeros(n, bool), command=[0.5, 0, 0])
  ref, _ = R.rollout(env, wenv, nominal, key, steps, reset_keys=rk,
                     scenario_fn=lambda s: R.set_scenario(env, s, **base))
  scn = lambda s: R.set_scenario(env, s, launch_pos=ref["proj_launch_pos"], launch_vel=ref["proj_launch_vel"], **base)
  out = {}
  for name, decide in (("CONTINUE", R.no_brace), ("BRACE", R.oracle_brace(jp.full((n,), _LEAD.value), 0.5, env.dt))):
    info, traj = R.rollout(env, wenv, nominal, key, steps, brace_policy=brace, decide=decide, scenario_fn=scn,
                           reset_keys=rk, record_qpos=True)
    out[name] = (info, traj)
  fc, fb = out["CONTINUE"][0]["fell"], out["BRACE"][0]["fell"]
  for i, (m_, v_) in enumerate(grid):
    print(f"  throw {i}: {m_:.0f} kg at {v_:.0f} m/s ({m_*v_:.0f} N s): CONTINUE fell {bool(fc[i])}, BRACE fell {bool(fb[i])}")
  pick = next((i for i in range(n) if fc[i] and not fb[i]), 0)
  if _SEARCH.value:
    print(f"selection rule: first throw in grid order where CONTINUE falls and BRACE stands -> throw {pick}")
  runs = {}
  for name in ("CONTINUE", "BRACE"):
    info, traj = out[name]
    runs[name] = (traj["qpos"][:, pick], bool(info["fell"][pick]), bool(info["impact_contact"][pick]), traj["brace_active"][:, pick])
    print(f"{name}: contact {runs[name][2]}, fell {runs[name][1]}")
  m = env.mj_model
  d = mujoco.MjData(m)
  r = mujoco.Renderer(m, 400, 480)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = m.body("pelvis").id
  cam.distance, cam.elevation, cam.azimuth = 3.4, -12.0, 140.0
  # The projectile is not in the model's body mass here; its geometry and position render from qpos.
  frames = []
  for t in range(steps):
    panes = []
    for name in ("CONTINUE", "BRACE"):
      qs, fell, _, active = runs[name]
      d.qpos[:] = qs[t]
      mujoco.mj_forward(m, d)
      r.update_scene(d, camera=cam)
      img = r.render().copy()
      if name == "BRACE" and active[t]:
        img[:8] = (220, 120, 30)          # orange bar while BRACE is active
      panes.append(img)
    frames.append(np.concatenate([panes[0], np.full((400, 16, 3), 255, np.uint8), panes[1]], axis=1))
  imageio.mimwrite(_OUT.value, frames, fps=int(round(1 / env.dt)), quality=8)
  print(f"wrote {_OUT.value}: left CONTINUE (fell={runs['CONTINUE'][1]}), right BRACE at {_LEAD.value}s lead "
        f"(fell={runs['BRACE'][1]}); {grid[pick][0]} kg at {grid[pick][1]} m/s")


if __name__ == "__main__":
  app.run(main)
