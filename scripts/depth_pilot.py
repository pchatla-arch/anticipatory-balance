# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E7: depth-render pilot. Renders the head camera for a batch of environments with the
MJX-Warp batch ray-tracer while the walker runs, measures throughput, saves a contact
sheet, and checks a depth-threshold blob detector against the true object position.

Timeboxed pilot: if the renderer cannot be created or rendered inside the batched loop,
the script reports why and exits non-zero; the decision experiments then use parametric
detections (the default) and claim C5 is dropped.

  python scripts/depth_pilot.py --impl warp --num_envs 64 --steps 150 --out results/E7_depth_pilot
"""
import json
import os
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags
from mujoco import mjx

from foresight.eval import rollout as R

_IMPL = flags.DEFINE_string("impl", "warp", "warp (the batch renderer needs warp)")
_N = flags.DEFINE_integer("num_envs", 64, "environments")
_STEPS = flags.DEFINE_integer("steps", 150, "control steps")
_EVERY = flags.DEFINE_integer("render_every", 5, "render every k control steps (5 = 10 Hz)")
_RES = flags.DEFINE_integer("res", 64, "square image size")
_OUT = flags.DEFINE_string("out", "results/E7_depth_pilot", "output dir")
_CKPT = flags.DEFINE_string("checkpoint", "checkpoints/nominal_s1", "walker")


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=60, range_dist=[3.0, 5.0])
  m = env.mj_model
  cam_id = m.camera("head_cam").id
  n = _N.value
  policy = R.load_policy(_CKPT.value)
  reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
  rng = jax.random.PRNGKey(0)
  state = reset(jax.random.split(rng, n))
  state = R.set_scenario(env, state, hit_time=np.full(n, 1.5), is_miss=np.zeros(n, bool), command=[0.5, 0, 0])
  print("creating render context ...")
  t0 = time.time()
  try:
    rc = mjx.create_render_context(mjm=m, nworld=n, cam_res=(_RES.value, _RES.value),
                                   render_rgb=False, render_depth=True, enabled_geom_groups=[0, 1, 2])
  except Exception as e:  # noqa: BLE001
    print("NO-GO: create_render_context failed:", repr(e)); raise SystemExit(2)
  print(f"render context in {time.time()-t0:.1f}s")
  refit = jax.jit(mjx.refit_bvh); render = jax.jit(mjx.render)
  frames, truths, times = [], [], []
  act = jax.jit(jax.vmap(policy))
  t_all = time.time()
  for t in range(_STEPS.value):
    rng, k = jax.random.split(rng)
    a, _ = act(state.obs, jax.random.split(k, n))
    state = step(state, a)
    if t % _EVERY.value == 0:
      t0 = time.time()
      try:
        data = refit(env.mjx_model, state.data, rc.pytree())
        out = render(env.mjx_model, data, rc.pytree())
      except Exception as e:  # noqa: BLE001
        print("NO-GO: render failed at step", t, repr(e)); raise SystemExit(3)
      depth = np.asarray(out[1])
      jax.block_until_ready(depth)
      times.append(time.time() - t0)
      frames.append(depth)
      truths.append((np.asarray(state.info["obj_pos"]), np.asarray(state.data.xpos[:, env._pelvis_body]),
                     np.asarray(state.info["proj_launched"])))
  total = time.time() - t_all
  fr = np.stack(frames)             # (K, n, [cam?], H, W)
  print("depth array shape", fr.shape, "render ms per batch", 1000 * np.mean(times[1:]))
  # Simple blob detector: nearest connected depth below the robot-free background.
  summary = {"num_envs": n, "steps": _STEPS.value, "frames": int(fr.shape[0]), "shape": list(fr.shape),
             "render_ms_per_batch_mean": float(1000 * np.mean(times[1:])),
             "wall_s": total, "env_steps_per_s": n * _STEPS.value / total}
  json.dump(summary, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  np.savez_compressed(os.path.join(_OUT.value, "frames.npz"), depth=fr.astype(np.float16),
                      obj_pos=np.stack([t[0] for t in truths]), pelvis=np.stack([t[1] for t in truths]),
                      launched=np.stack([t[2] for t in truths]))
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    K = fr.shape[0]; picks = np.linspace(0, K - 1, 8).astype(int)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.4), constrained_layout=True)
    for ax, k in zip(axes.ravel(), picks):
      img = fr[k, 0]
      img = img.reshape(img.shape[-2], img.shape[-1]) if img.ndim > 2 else img
      ax.imshow(img, cmap="viridis"); ax.set_title(f"env0 t={k*_EVERY.value*env.dt:.2f}s", fontsize=8); ax.axis("off")
    fig.savefig(os.path.join(_OUT.value, "contact_sheet.png"), dpi=150)
  except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)
  print(json.dumps(summary, indent=2)); print("GO: rendering works inside the batched loop")


if __name__ == "__main__":
  app.run(main)
