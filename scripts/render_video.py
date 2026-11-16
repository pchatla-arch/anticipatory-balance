# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Render a video of a frozen policy in the impact environment (headless, CPU or GPU).

Runs one environment, records qpos every control step, and renders with MuJoCo's
renderer (MUJOCO_GL=egl). No training-time rendering is involved, so this works on a pod
without a display and on a CPU-only machine.

  MUJOCO_GL=egl python scripts/render_video.py --checkpoint checkpoints/nominal_s1 \
      --seconds 6 --out results/videos/walker_s1.mp4
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

_CKPT = flags.DEFINE_string("checkpoint", "checkpoints/nominal_s1", "policy checkpoint dir")
_SECONDS = flags.DEFINE_float("seconds", 6.0, "duration")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command, m/s")
_SEED = flags.DEFINE_integer("seed", 3, "scenario seed")
_IMPL = flags.DEFINE_string("impl", "jax", "jax (CPU) or warp (GPU)")
_W = flags.DEFINE_integer("width", 640, "px")
_H = flags.DEFINE_integer("height", 400, "px")
_LAUNCH = flags.DEFINE_bool("launch", True, "launch the projectile")
_OUT = flags.DEFINE_string("out", "results/videos/walker.mp4", "output mp4")


def main(_):
  os.makedirs(os.path.dirname(_OUT.value) or ".", exist_ok=True)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=60.0, range_dist=[3.0, 5.0])
  policy = R.load_policy(_CKPT.value)
  reset = jax.jit(env.reset)
  step = jax.jit(env.step)
  act = jax.jit(policy)
  state = reset(jax.random.PRNGKey(_SEED.value))
  info = dict(state.info)
  cmd = jp.array([_VX.value, 0.0, 0.0])
  info["command"] = cmd
  info["command_nominal"] = cmd
  if not _LAUNCH.value:
    info["proj_launch_step"] = jp.int32(10**6)
  state = state.replace(info=info)
  n =int(_SECONDS.value / env.dt)
  qs, contact_t, fell_t = [], None, None
  key = jax.random.PRNGKey(0)
  for t in range(n):
    key, k = jax.random.split(key)
    obs = {**state.obs, "state": state.obs["state"].at[R.CMD_SLICE].set(cmd)}
    a, _ = act(obs, k)
    state = step(state, a)
    qs.append(np.asarray(state.data.qpos))
    if contact_t is None and bool(state.info["impact_contact"]):
      contact_t = t * env.dt
    if fell_t is None and bool(state.info["fell"]):
      fell_t = t * env.dt
  m = env.mj_model
  d = mujoco.MjData(m)
  r = mujoco.Renderer(m, _H.value, _W.value)
  cam = mujoco.MjvCamera()
  cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
  cam.trackbodyid = m.body("pelvis").id
  cam.distance, cam.elevation, cam.azimuth = 3.2, -15.0, 120.0
  frames = []
  for q in qs:
    d.qpos[:] = q
    mujoco.mj_forward(m, d)
    r.update_scene(d, camera=cam)
    frames.append(r.render())
  imageio.mimwrite(_OUT.value, frames, fps=int(round(1 / env.dt)), quality=8)
  print(f"wrote {_OUT.value}: {len(frames)} frames; contact at {contact_t} s, fell at {fell_t} s, "
        f"mass {float(state.info['proj_mass']):.1f} kg, speed {float(state.info['proj_speed']):.1f} m/s")


if __name__ == "__main__":
  app.run(main)
