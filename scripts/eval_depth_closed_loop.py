# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E17: the decision layer closed over rendered depth.

Every detection period the head camera is rendered for all environments (MJX-Warp batch
ray-tracer), the geometric depth detector of E8 (in JAX) turns each frame into a world-frame
detection or nothing, the belief absorbs it, and the rule decides. The parametric detector
with the same field of view runs on the same paired scenarios for comparison, so the only
difference between the two columns is where the detections come from.

A Python loop (not lax.scan) so the renderer can be called between steps, as in E7/E8.

  python scripts/eval_depth_closed_loop.py --impl warp --episodes 256 --batches 2 --out results/E17_depth_closed_loop
"""
import json
import os
import time

import jax
import jax.numpy as jp
import numpy as np
from absl import app, flags
from mujoco import mjx

from foresight.belief import particle_filter as PF
from foresight.decision import rules as RU
from foresight.decision.controller import AnticipationController
from foresight.eval import rollout as R
from foresight.perception import detections as D

_NOM = flags.DEFINE_string("nominal", "checkpoints/nominal_s1", "walker")
_BR = flags.DEFINE_string("brace", "checkpoints/brace2_s1", "BRACE skill")
_RM = flags.DEFINE_string("response_model", "results/response_model.json", "response model")
_IMPL = flags.DEFINE_string("impl", "warp", "warp")
_N = flags.DEFINE_integer("episodes", 256, "environments per batch")
_BATCHES = flags.DEFINE_integer("batches", 2, "batches")
_STEPS = flags.DEFINE_integer("steps", 300, "control steps")
_PERIOD = flags.DEFINE_integer("period", 5, "detection period, control steps")
_EXTRA_LAT = flags.DEFINE_integer("extra_latency", 0, "control steps added to the rendered detection's age")
_RES = flags.DEFINE_integer("res", 96, "image size")
_AZ = flags.DEFINE_float("azimuth_deg", 40.0, "launch azimuth half-range (inside the 90-deg camera)")
_MISS = flags.DEFINE_float("miss_fraction", 0.3, "near-miss fraction")
_VX = flags.DEFINE_float("command_vx", 0.5, "forward command")
_SIG = flags.DEFINE_float("det_sigma", 0.03, "noise std assumed for a depth detection (E8: 0.03 m)")
_RAD = flags.DEFINE_float("object_radius", 0.15, "m")
_METHODS = flags.DEFINE_list("methods", ["ttc_threshold:0.3", "belief_commit:0.1", "belief_lead:0.1", "belief_voi:0.1"],
                             "kind:param")
_OUT = flags.DEFINE_string("out", "results/E17_depth_closed_loop", "output dir")
_SEED = flags.DEFINE_integer("seed", 300, "seed")
_SOURCES = flags.DEFINE_list("sources", ["parametric", "depth"], "detection sources to run")


def pixel_dirs(res, fovy_deg):
  t = np.tan(np.deg2rad(fovy_deg) / 2)
  u = (np.arange(res) + 0.5) / res * 2 - 1
  v = 1 - (np.arange(res) + 0.5) / res * 2
  X, Y = np.meshgrid(u * t, v * t)
  return jp.asarray(np.stack([X, Y, -np.ones_like(X)], axis=-1), dtype=jp.float32)   # z-depth convention (E8)


def make_detector(dirs, radius, max_range=6.0):
  def one(depth, cam_pos, cam_mat, pelvis):
    valid = jp.isfinite(depth) & (depth > 0.05) & (depth < 50.0)
    P = cam_pos + (dirs * jp.where(valid, depth, 1.0)[..., None]) @ cam_mat.T
    rxy = jp.linalg.norm(P[..., :2] - pelvis[:2], axis=-1)
    cand = valid & (P[..., 2] > 0.3) & (P[..., 2] < 2.0) & (rxy > 0.6) & (rxy < max_range)
    n = jp.sum(cand)
    surf = jp.sum(P * cand[..., None], axis=(0, 1)) / jp.maximum(n, 1)
    view = surf - cam_pos
    centre = surf + radius * view / (jp.linalg.norm(view) + 1e-9)
    return n >= 3, centre
  return jax.jit(jax.vmap(one))


def main(_):
  os.makedirs(_OUT.value, exist_ok=True)
  rmj = json.load(open(_RM.value))
  rm = RU.ResponseModel(rmj["a_cont"], rmj["b_cont"], rmj["a_brace"], rmj["b_brace"],
                        rmj.get("a_brace_lead", 0.0), rmj.get("b_brace_lead", 0.0), rmj.get("c_lead", 0.0),
                        rmj.get("lead_max", 0.5), "c_lead" in rmj)
  nominal = R.load_policy(_NOM.value); brace = R.load_policy(_BR.value)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=_AZ.value, range_dist=[3.0, 5.0])
  m = env.mj_model
  cam_id = m.camera("head_cam").id
  fovy = float(m.cam_fovy[cam_id])
  n, res = _N.value, _RES.value
  pelvis_b = env._pelvis_body
  if "depth" in _SOURCES.value:
    rc = mjx.create_render_context(mjm=m, nworld=n, cam_res=(res, res), render_rgb=False, render_depth=True,
                                   enabled_geom_groups=[0, 1, 2])
    refit = jax.jit(mjx.refit_bvh); render = jax.jit(mjx.render)
    try:
      from mujoco.mjx.warp import render_context as _rcm
      adr = int(_rcm.get(rc.pytree()).depth_adr.numpy()[cam_id])
    except Exception:  # noqa: BLE001
      adr = cam_id * res * res
  detect = make_detector(pixel_dirs(res, fovy), _RAD.value)
  mr = (4.0, 16.0)
  det_cfg_param = D.DetectorConfig(period=_PERIOD.value, latency=_EXTRA_LAT.value, sigma0=_SIG.value, sigma1=0.0,
                                   fov_deg=fovy, miss_prob=0.055)

  @jax.jit
  def act_fn(obs, cmd_nom, keys, brace_flag):
    a_nom, _ = jax.vmap(nominal)(R._with_command(obs, cmd_nom), keys)
    a_br, _ = jax.vmap(brace)(R._with_command(obs, jp.zeros_like(cmd_nom)), keys)
    return jp.where(brace_flag[:, None], a_br, a_nom)

  rows = []
  for b in range(_BATCHES.value):
    rs = np.random.RandomState(_SEED.value + b)
    masses = rs.uniform(*mr, n); speeds = rs.uniform(3.0, 8.0, n)
    hits = rs.uniform(*env._config.impact.hit_time_range, n); is_miss = rs.uniform(size=n) < _MISS.value
    key = jax.random.PRNGKey(_SEED.value + b)
    reset_keys = jax.random.split(jax.random.fold_in(key, 1), n)
    wenv = R.batched_env(env, masses)
    step_fn = jax.jit(wenv.step); reset_fn = jax.jit(wenv.reset)
    base = dict(speed=speeds, hit_time=hits, is_miss=is_miss, command=[_VX.value, 0, 0])
    ref_info, _ = R.rollout(env, wenv, nominal, key, _STEPS.value,
                            scenario_fn=lambda s: R.set_scenario(env, s, **base), reset_keys=reset_keys)
    lp, lv = ref_info["proj_launch_pos"], ref_info["proj_launch_vel"]
    print(f"batch {b}: reference contact(hits) {ref_info['impact_contact'][~is_miss].mean():.3f} fell(hits) {ref_info['fell'][~is_miss].mean():.3f}")
    det_step = jax.jit(jax.vmap(lambda st, o, p, f, l, k: D.step(det_cfg_param, st, o, p, f, l, k)))

    for source in _SOURCES.value:
      for spec in _METHODS.value:
        kind, param = spec.split(":"); param = float(param)
        rc_rule = {"ttc_threshold": RU.RuleConfig(kind="ttc_threshold", theta=param),
                   "belief_commit": RU.RuleConfig(kind="belief_commit", c_prep=param),
                   "belief_lead": RU.RuleConfig(kind="belief_lead", c_prep=param),
                   "belief_voi": RU.RuleConfig(kind="belief_voi", c_prep=param, wait_steps=_PERIOD.value)}[kind]
        ctrl = AnticipationController(det_cfg_param, PF.PFConfig(mass_range=mr), rc_rule, rm)
        t0 = time.time()
        rng = jax.random.fold_in(key, 7)
        state = reset_fn(reset_keys)
        state = R.set_scenario(env, state, launch_pos=lp, launch_vel=lv, **base)
        carry = ctrl.init(n, jax.random.fold_in(rng, 1))
        dst = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (n,) + jp.shape(x)), D.init_state(det_cfg_param))
        brace_flag = jp.zeros((n,), bool)
        pel0 = np.asarray(state.data.xpos[:, pelvis_b]); fwd0 = np.asarray(state.data.xmat[:, pelvis_b].reshape(n, 3, 3)[:, :, 0])
        heading0 = fwd0[:, :2] / (np.linalg.norm(fwd0[:, :2], axis=-1, keepdims=True) + 1e-9)
        brace_steps = np.zeros(n, int); prog = np.zeros(n); alive = np.ones(n, bool); last_pel = pel0.copy()
        n_det = np.zeros(n, int); render_ms = []; queue = []   # (deliver_step, ok, centre, capture_step)
        for t in range(_STEPS.value):
          rng, k1, k2, k3 = jax.random.split(rng, 4)
          d0 = state.data
          pel = d0.xpos[:, pelvis_b]; fwd = d0.xmat[:, pelvis_b].reshape(n, 3, 3)[:, :, 0]
          fxy = fwd[:, :2] / (jp.linalg.norm(fwd[:, :2], axis=-1, keepdims=True) + 1e-6)
          pvel = jp.concatenate([fxy * state.info["command"][:, :1], jp.zeros((n, 1))], axis=-1)
          due = (t % _PERIOD.value == 0)
          if source == "depth":
            if due:
              tr = time.time()
              data = refit(env.mjx_model, d0, rc.pytree())
              out = render(env.mjx_model, data, rc.pytree())
              depth = out[1][:, adr:adr + res * res].reshape(n, res, res)
              ok, centre = detect(depth, d0.cam_xpos[:, cam_id], d0.cam_xmat[:, cam_id].reshape(n, 3, 3), pel)
              jax.block_until_ready(ok); render_ms.append(1000 * (time.time() - tr))
              # Captured now; delivered --extra_latency steps later (the filter absorbs the age exactly).
              queue.append((t + _EXTRA_LAT.value, ok & state.info["proj_launched"], centre, t))
            ready = [q for q in queue if q[0] == t]
            queue = [q for q in queue if q[0] > t]
            if ready:
              _, det_valid, det_pos, cs = ready[-1]; cap = jp.full((n,), cs, jp.int32)
            else:
              det_valid = jp.zeros((n,), bool); det_pos = jp.zeros((n, 3)); cap = jp.full((n,), t, jp.int32)
            det_sigma = jp.full((n,), _SIG.value)
          else:
            dst, det = det_step(dst, state.info["obj_pos"], pel, fwd, state.info["proj_launched"], jax.random.split(k1, n))
            det_valid, det_pos, det_sigma, cap = det.valid, det.pos, det.sigma, det.capture_step
          carry, brace_flag, logs = ctrl.step_with_detections(carry, det_valid, det_pos, det_sigma, cap, pel, pvel, t, k2)
          n_det += np.asarray(det_valid)
          info = dict(state.info); info["brace_active"] = brace_flag; state = state.replace(info=info)
          act = act_fn(state.obs, state.info["command_nominal"], jax.random.split(k3, n), brace_flag)
          state = step_fn(state, act)
          pel_np = np.asarray(state.data.xpos[:, pelvis_b])
          prog += np.where(alive, np.sum((pel_np[:, :2] - last_pel[:, :2]) * heading0, axis=-1), 0.0)
          last_pel = pel_np
          alive &= ~np.asarray(state.info["fell"])
          brace_steps += np.asarray(brace_flag)
        fi = jax.tree_util.tree_map(np.asarray, state.info)
        hit_s = ~is_miss; c = fi["impact_contact"] & hit_s
        row = {"source": source, "method": kind, "param": param, "batch": b, "n": n,
               "fall_rate_hits": float(fi["fell"][hit_s].mean()),
               "fall_rate_given_contact": float(fi["fell"][c].mean()) if c.any() else float("nan"),
               "contact_rate_hits": float(fi["impact_contact"][hit_s].mean()),
               "false_prep_rate": float((brace_steps[is_miss] > 0).mean()) if is_miss.any() else 0.0,
               "braced_on_hits": float((brace_steps[hit_s] > 0).mean()), "braced_s": float(brace_steps.mean() * env.dt),
               "progress_m": float(prog.mean()), "detections_per_episode": float(n_det.mean()),
               "render_ms_per_frame": float(np.mean(render_ms[1:])) if len(render_ms) > 1 else float("nan"),
               "wall_s": time.time() - t0}
        rows.append(row)
        print(f"  {source:10} {kind:14} p={param:<4g} {row['wall_s']:4.0f}s  fall|hits {row['fall_rate_hits']:.3f}  "
              f"fall|contact {row['fall_rate_given_contact']:.3f}  contact {row['contact_rate_hits']:.2f}  "
              f"false-prep {row['false_prep_rate']:.3f}  braced {row['braced_s']:.2f}s  det/ep {row['detections_per_episode']:.1f}  "
              f"render {row['render_ms_per_frame']:.0f} ms")
    # reactive reference for this batch
    rows.append({"source": "none", "method": "reactive", "param": 0.0, "batch": b, "n": n,
                 "fall_rate_hits": float(ref_info["fell"][~is_miss].mean()),
                 "fall_rate_given_contact": float(ref_info["fell"][ref_info["impact_contact"] & ~is_miss].mean()),
                 "contact_rate_hits": float(ref_info["impact_contact"][~is_miss].mean()), "false_prep_rate": 0.0,
                 "braced_on_hits": 0.0, "braced_s": 0.0, "progress_m": float("nan"), "detections_per_episode": 0.0,
                 "render_ms_per_frame": float("nan"), "wall_s": 0.0})

  agg = {}
  for r in rows:
    agg.setdefault((r["source"], r["method"], r["param"]), []).append(r)
  out = []
  for (src, meth, par), rs_ in agg.items():
    o = {"source": src, "method": meth, "param": par, "n_total": sum(r["n"] for r in rs_)}
    for f in ("fall_rate_hits", "fall_rate_given_contact", "contact_rate_hits", "false_prep_rate", "braced_on_hits",
              "braced_s", "progress_m", "detections_per_episode", "render_ms_per_frame"):
      o[f] = float(np.nanmean([r[f] for r in rs_]))
    out.append(o)
  cols = list(out[0].keys())
  with open(os.path.join(_OUT.value, "table.csv"), "w") as f:
    f.write(",".join(cols) + "\n")
    for o in out:
      f.write(",".join(f"{o[c]:.4g}" if isinstance(o[c], float) else str(o[c]) for c in cols) + "\n")
  json.dump({"period": _PERIOD.value, "extra_latency": _EXTRA_LAT.value, "res": res, "fovy_deg": fovy,
             "azimuth_deg": _AZ.value, "det_sigma": _SIG.value, "episodes_per_condition": n * _BATCHES.value, "rows": out},
            open(os.path.join(_OUT.value, "summary.json"), "w"), indent=2)
  try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meths = [s.split(":")[0] for s in _METHODS.value]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(meths)); w = 0.38
    for i, (src, col) in enumerate((("parametric", "#2b6f8e"), ("depth", "#d9772b"))):
      vals = [next((o for o in out if o["source"] == src and o["method"] == mm), None) for mm in meths]
      axes[0].bar(x + (i - 0.5) * w, [v["fall_rate_hits"] if v else np.nan for v in vals], w, color=col, label=src)
      axes[1].bar(x + (i - 0.5) * w, [v["false_prep_rate"] if v else np.nan for v in vals], w, color=col, label=src)
    react = next((o for o in out if o["method"] == "reactive"), None)
    if react:
      axes[0].axhline(react["fall_rate_hits"], color="#14213d", lw=1, ls="--", label="reactive")
    for ax, yl in zip(axes, ("failure rate on hits", "false-preparation rate")):
      ax.set_xticks(x); ax.set_xticklabels(meths, fontsize=8); ax.set_ylabel(yl); ax.set_ylim(0, 1); ax.grid(True, axis="y", color="#e3e6ea", lw=0.6)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Closed loop from rendered depth vs the parametric detector ({res}x{res}, {50/_PERIOD.value:.0f} Hz)", fontsize=11)
    fig.savefig(os.path.join(_OUT.value, "closed_loop.png"), dpi=170)
  except Exception as e:  # noqa: BLE001
    print("plot skipped:", e)


if __name__ == "__main__":
  app.run(main)
