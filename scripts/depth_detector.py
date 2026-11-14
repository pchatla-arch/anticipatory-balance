# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E8: an observation model measured from rendered depth instead of assumed.

Renders the head camera (MJX-Warp batch ray-tracer) while the walker runs with projectiles,
runs a geometric depth detector on every frame, and compares each detection with the true
object position. Outputs the detector's noise against range (sigma0 + sigma1 * range), its
detection and false-alarm rates, the camera's effective field of view, and the per-frame
render time (the perception latency a real pipeline would add), as the numbers that replace
DetectorConfig's defaults.

Detector (no learning, no object model): back-project every valid depth pixel to a world
point using the camera pose; pixels that lie above the ground, away from the robot's own
body and within range form the object blob; the detection is their median. The depth
convention (z-depth vs ray distance) and image row order are self-calibrated on ground pixels.

  python scripts/depth_detector.py --impl warp --num_envs 128 --steps 200 --render_every 2 \
      --res 96 --out results/E8_depth_detector
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
_N = flags.DEFINE_integer("num_envs", 128, "environments")
_STEPS = flags.DEFINE_integer("steps", 200, "control steps")
_EVERY = flags.DEFINE_integer("render_every", 2, "render every k control steps")
_RES = flags.DEFINE_integer("res", 96, "square image size")
_OUT = flags.DEFINE_string("out", "results/E8_depth_detector", "output dir")
_CKPT = flags.DEFINE_string("checkpoint", "checkpoints/nominal_s1", "walker")
_AZ = flags.DEFINE_float("azimuth_deg", 60.0, "launch azimuth half-range about the heading")
_MISS = flags.DEFINE_float("miss_fraction", 0.3, "near-miss fraction")
_SEED = flags.DEFINE_integer("seed", 7, "seed")
_MAXR = flags.DEFINE_float("max_range", 6.0, "m, detector range limit")
_RAD = flags.DEFINE_float("object_radius", 0.15, "m, assumed object radius (surface-to-centre correction)")


def pixel_dirs(res, fovy_deg, flip):
  t = np.tan(np.deg2rad(fovy_deg) / 2)
  u = (np.arange(res) + 0.5) / res * 2 - 1
  v = 1 - (np.arange(res) + 0.5) / res * 2
  if flip:
    v = -v
  X, Y = np.meshgrid(u * t, v * t)
  return np.stack([X, Y, -np.ones_like(X)], axis=-1)        # (H, W, 3) camera frame, looks down -z


def backproject(depth, cam_pos, cam_mat, dirs, ray):
  """depth (H, W) metres; cam_mat columns = camera axes in world. Returns world points (H, W, 3)."""
  if ray:
    d = dirs / np.linalg.norm(dirs, axis=-1, keepdims=True)
    pc = d * depth[..., None]
  else:
    pc = dirs * depth[..., None]                               # z-depth: scale so that -z = depth
  return cam_pos + pc @ cam_mat.T


def detect(depth, cam_pos, cam_mat, pelvis, dirs, ray, max_range, radius=0.15):
  valid = np.isfinite(depth) & (depth > 0.05) & (depth < 50.0)
  P = backproject(np.where(valid, depth, 1.0), cam_pos, cam_mat, dirs, ray)
  rng_xy = np.linalg.norm(P[..., :2] - pelvis[:2], axis=-1)
  cand = valid & (P[..., 2] > 0.3) & (P[..., 2] < 2.0) & (rng_xy > 0.6) & (rng_xy < max_range)
  n = int(cand.sum())
  if n < 3:
    return None, n, P, valid
  surf = np.median(P[cand], axis=0)
  # Depth sees the near surface; the centre lies one radius further along the view ray.
  view = surf - cam_pos
  return surf + radius * view / (np.linalg.norm(view) + 1e-9), n, P, valid


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=_AZ.value, range_dist=[3.0, 5.0])
  m = env.mj_model
  cam_id = m.camera("head_cam").id
  fovy = float(m.cam_fovy[cam_id])
  n = _N.value; res = _RES.value
  policy = R.load_policy(_CKPT.value)
  reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
  rng = jax.random.PRNGKey(_SEED.value)
  rs = np.random.RandomState(_SEED.value)
  state = reset(jax.random.split(rng, n))
  state = R.set_scenario(env, state, hit_time=rs.uniform(1.0, 3.0, n), is_miss=rs.uniform(size=n) < _MISS.value,
                         command=[0.5, 0, 0])
  print("creating render context ...")
  rc = mjx.create_render_context(mjm=m, nworld=n, cam_res=(res, res), render_rgb=False, render_depth=True,
                                 enabled_geom_groups=[0, 1, 2])
  refit = jax.jit(mjx.refit_bvh); render = jax.jit(mjx.render)
  act = jax.jit(jax.vmap(policy))
  frames, times, meta = [], [], []
  t_all = time.time()
  for t in range(_STEPS.value):
    rng, k = jax.random.split(rng)
    a, _ = act(state.obs, jax.random.split(k, n))
    state = step(state, a)
    if t % _EVERY.value == 0:
      t0 = time.time()
      data = refit(env.mjx_model, state.data, rc.pytree())
      out = render(env.mjx_model, data, rc.pytree())
      depth = np.asarray(out[1]); jax.block_until_ready(depth)
      times.append(time.time() - t0)
      # One camera's block of the packed buffer (all cameras share cam_res here).
      try:
        from mujoco.mjx.warp import render_context as _rcm
        adr = int(_rcm.get(rc.pytree()).depth_adr.numpy()[cam_id])
      except Exception:  # noqa: BLE001
        adr = cam_id * res * res
      frames.append(depth[:, adr:adr + res * res].reshape(n, res, res).astype(np.float32))
      meta.append(dict(obj=np.asarray(state.info["obj_pos"]), pelvis=np.asarray(state.data.xpos[:, env._pelvis_body]),
                       launched=np.asarray(state.info["proj_launched"]), cam_pos=np.asarray(state.data.cam_xpos[:, cam_id]),
                       cam_mat=np.asarray(state.data.cam_xmat[:, cam_id]).reshape(n, 3, 3),
                       fell=np.asarray(state.info["fell"]), t=t))
  wall = time.time() - t_all
  render_ms = float(1000 * np.mean(times[1:]))
  print(f"{len(frames)} frames x {n} envs, render {render_ms:.1f} ms per batch, {n*_STEPS.value/wall:.0f} env-steps/s")

  # Self-calibrate the depth convention and row order on ground pixels of the first frames.
  best = None
  for ray in (False, True):
    for flip in (False, True):
      dirs = pixel_dirs(res, fovy, flip)
      score = 0.0
      for f, md in zip(frames[:3], meta[:3]):
        for e in range(min(n, 8)):
          d = f[e]; valid = np.isfinite(d) & (d > 0.05) & (d < 50)
          P = backproject(np.where(valid, d, 1.0), md["cam_pos"][e], md["cam_mat"][e], dirs, ray)
          score += np.mean(valid & (np.abs(P[..., 2]) < 0.05))
      if best is None or score > best[0]:
        best = (score, ray, flip)
  _, ray, flip = best
  dirs = pixel_dirs(res, fovy, flip)
  print(f"depth convention: {'ray distance' if ray else 'z-depth'}, rows {'bottom-up' if flip else 'top-down'} "
        f"(ground-pixel score {best[0]/ (3*min(n,8)):.2f})")

  # Run the detector on every frame.
  recs = []
  hfov = np.deg2rad(fovy) / 2
  for f, md in zip(frames, meta):
    for e in range(n):
      if md["fell"][e]:
        continue
      det, npix, _, _ = detect(f[e], md["cam_pos"][e], md["cam_mat"][e], md["pelvis"][e], dirs, ray, _MAXR.value, _RAD.value)
      obj = md["obj"][e]; launched = bool(md["launched"][e])
      rel_c = md["cam_mat"][e].T @ (obj - md["cam_pos"][e])         # object in camera frame
      in_img = launched and rel_c[2] < -0.1 and abs(rel_c[0] / -rel_c[2]) < np.tan(hfov) and abs(rel_c[1] / -rel_c[2]) < np.tan(hfov)
      rng_ = float(np.linalg.norm((obj - md["pelvis"][e])[:2]))
      az = float(np.degrees(np.arctan2(rel_c[0], -rel_c[2]))) if launched else np.nan
      err = (det - obj) if det is not None else np.full(3, np.nan)
      recs.append(dict(t=md["t"], env=e, launched=launched, in_image=bool(in_img), range=rng_, azimuth_deg=az,
                       detected=det is not None, npix=npix, ex=err[0], ey=err[1], ez=err[2],
                       err=float(np.linalg.norm(err)) if det is not None else np.nan))
  r = {k: np.array([x[k] for x in recs]) for k in recs[0]}
  vis = r["launched"] & r["in_image"] & (r["range"] < _MAXR.value)
  tp = vis & r["detected"] & (r["err"] < 0.5)
  fn = vis & ~(r["detected"] & (r["err"] < 0.5))
  fa = (~vis) & r["detected"]
  det_rate = float(tp.sum() / max(vis.sum(), 1)); fa_rate = float(fa.sum() / max((~vis).sum(), 1))
  # Noise against range on true positives: per-axis RMS in range bins, then sigma0 + sigma1 * range.
  bins = np.arange(0.5, _MAXR.value + 0.01, 0.5)
  noise = []
  for i in range(len(bins) - 1):
    m_ = tp & (r["range"] >= bins[i]) & (r["range"] < bins[i + 1])
    if m_.sum() >= 10:
      e3 = np.stack([r["ex"][m_], r["ey"][m_], r["ez"][m_]], axis=-1)
      noise.append(dict(range_lo=float(bins[i]), range_hi=float(bins[i + 1]), n=int(m_.sum()),
                        rms_per_axis=float(np.sqrt(np.mean(e3 ** 2))), bias=[float(x) for x in e3.mean(axis=0)]))
  if len(noise) >= 2:
    x = np.array([(b["range_lo"] + b["range_hi"]) / 2 for b in noise]); y = np.array([b["rms_per_axis"] for b in noise])
    s1, s0 = np.polyfit(x, y, 1)
  else:
    s0, s1 = np.nan, np.nan
  # Field of view actually usable: azimuth range of launched objects that were detected.
  az_det = r["azimuth_deg"][tp]
  # Detection rate against range, for the summary and the plot.
  by_range = []
  for i in range(len(bins) - 1):
    m_ = vis & (r["range"] >= bins[i]) & (r["range"] < bins[i + 1])
    by_range.append(dict(range_lo=float(bins[i]), range_hi=float(bins[i + 1]), n=int(m_.sum()),
                         det_rate=float(tp[m_].mean()) if m_.any() else np.nan))
  summary = dict(num_envs=n, steps=_STEPS.value, render_every=_EVERY.value, res=res, fovy_deg=fovy,
                 render_ms_per_batch=render_ms, env_steps_per_s=n * _STEPS.value / wall,
                 depth_convention="ray" if ray else "z", rows_flipped=bool(flip),
                 frames_visible=int(vis.sum()), true_positives=int(tp.sum()), misses=int(fn.sum()),
                 false_alarms=int(fa.sum()), frames_without_visible_object=int((~vis).sum()),
                 detection_rate=det_rate, false_alarm_rate=fa_rate,
                 sigma0_fit=float(s0), sigma1_fit=float(s1), noise_by_range=noise, detection_by_range=by_range,
                 azimuth_detected_deg=[float(np.nanmin(az_det)), float(np.nanmax(az_det))] if tp.any() else None,
                 launched_out_of_image_frac=float(np.mean(~r["in_image"][r["launched"]])) if r["launched"].any() else np.nan,
                 suggested_detector_config=dict(sigma0=float(max(s0, 0.0)), sigma1=float(max(s1, 0.0)),
                                                fov_deg=fovy, miss_prob=float(1 - det_rate),
                                                latency_note=f"render adds {render_ms:.0f} ms per batch of {n}"))
  json.dump(summary, open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  with open(os.path.join(_OUT.value, "detections.csv"), "w") as f:
    keys = list(r.keys()); f.write(",".join(keys) + "\n")
    for i in range(len(recs)):
      f.write(",".join(str(r[k][i]) for k in keys) + "\n")
  print(json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)}, indent=2))

  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    xs = [(b["range_lo"] + b["range_hi"]) / 2 for b in noise]
    axes[0].plot(xs, [b["rms_per_axis"] for b in noise], "o-", color="#2b6f8e", label="measured RMS per axis")
    if np.isfinite(s0):
      xx = np.linspace(0.5, _MAXR.value, 20); axes[0].plot(xx, s0 + s1 * xx, "--", color="#d9772b", label=f"fit {s0:.3f} + {s1:.3f}·range")
    axes[0].plot(xx if np.isfinite(s0) else xs, 0.03 + 0.02 * (xx if np.isfinite(s0) else np.array(xs)), ":", color="#6a7179", label="assumed 0.03 + 0.02·range")
    axes[0].set_xlabel("range (m)"); axes[0].set_ylabel("position noise (m)"); axes[0].legend(fontsize=8, frameon=False)
    axes[1].plot([(b["range_lo"] + b["range_hi"]) / 2 for b in by_range], [b["det_rate"] for b in by_range], "o-", color="#2b6f8e")
    axes[1].set_xlabel("range (m)"); axes[1].set_ylabel("detection rate (object in image)"); axes[1].set_ylim(0, 1.02)
    az_all = r["azimuth_deg"][r["launched"]]
    axes[2].hist(az_all[np.isfinite(az_all)], bins=36, range=(-90, 90), color="#c9ced3", label="launched objects")
    axes[2].hist(az_det, bins=36, range=(-90, 90), color="#2b6f8e", label="detected")
    axes[2].axvline(-fovy / 2, color="#d9772b", ls="--"); axes[2].axvline(fovy / 2, color="#d9772b", ls="--")
    axes[2].set_xlabel("azimuth from camera axis (deg)"); axes[2].legend(fontsize=8, frameon=False)
    for ax in axes:
      ax.grid(True, color="#e3e6ea", lw=0.6)
    fig.suptitle(f"E8 depth detector at {res}x{res}, fovy {fovy:.0f}°: detection {det_rate:.0%}, false alarms {fa_rate:.1%}", fontsize=11)
    fig.savefig(os.path.join(_OUT.value, "detector.png"), dpi=170)
    # Contact sheet with detections.
    picks = np.linspace(0, len(frames) - 1, 8).astype(int)
    fig, axs = plt.subplots(2, 4, figsize=(10, 5.4), constrained_layout=True)
    for ax, k in zip(axs.ravel(), picks):
      f, md = frames[k], meta[k]
      d = f[0]; ax.imshow(np.where(np.isfinite(d) & (d < 50), d, np.nan), cmap="viridis")
      det, npix, _, _ = detect(d, md["cam_pos"][0], md["cam_mat"][0], md["pelvis"][0], dirs, ray, _MAXR.value, _RAD.value)
      ax.set_title(f"t={md['t']*env.dt:.2f}s " + (f"det {npix}px err {np.linalg.norm(det-md['obj'][0]):.2f}m" if det is not None else "no det"), fontsize=7)
      ax.axis("off")
    fig.savefig(os.path.join(_OUT.value, "contact_sheet.png"), dpi=150)
  except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
