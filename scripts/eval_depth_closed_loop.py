# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""E17: the decision layer closed over rendered depth.

Every detection period the head camera is rendered for all environments (MJX-Warp batch
ray-tracer), the geometric depth detector of E8 (in JAX) turns each frame into a world-frame
detection or nothing, the belief absorbs it, and the rule decides. The parametric detector
with the same nominal field of view runs on paired external-projectile scenarios.
Visibility and observation error distributions can differ; this compares the complete
sensor paths, not an isolated position-noise intervention.

A Python loop (not lax.scan) so the renderer can be called between steps, as in E7/E8.

  python scripts/eval_depth_closed_loop.py --impl warp --episodes 256 --batches 2 --out results/E17_depth_closed_loop
"""
import csv
import json
from pathlib import Path
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
from foresight.eval import session_h_evidence as SE
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
_METHODS = flags.DEFINE_list("methods", ["ttc_threshold:0.3", "belief_lead:0.05", "belief_wait:0.05", "belief_voi:0.05"],
                             "kind:param")
_OUT = flags.DEFINE_string("out", "results/E17_depth_closed_loop", "output dir")
_SEED = flags.DEFINE_integer("seed", 300, "seed")
_SOURCES = flags.DEFINE_list("sources", ["parametric", "depth"], "detection sources to run")
_RELEASE = flags.DEFINE_float("release_after", 0.3, "seconds after closest approach (Session G contract)")



_write_json = SE.write_json


def _pool(parts, dt):
  """Pool episode numerators, including absent-subgroup denominators explicitly."""
  a = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
  hit, miss, contact, fell = ~a["is_miss"].astype(bool), a["is_miss"].astype(bool), a["contact"].astype(bool), a["fell"].astype(bool)
  ch = contact & hit
  rate = lambda x, mask: float(np.mean(x[mask])) if mask.any() else None
  brace = a["brace_steps"]
  counts = {"n_total": len(hit), "n_hits": int(hit.sum()), "n_misses": int(miss.sum()),
            "n_contact_hits": int(ch.sum()), "n_falls_hits": int(fell[hit].sum()),
            "n_falls_contact_hits": int(fell[ch].sum()), "n_false_prep_miss": int((brace[miss] > 0).sum()),
            "prelaunch_frames": int(a["prelaunch_frames"].sum()),
            "prelaunch_detections": int(a["prelaunch_detections"].sum()),
            "prelaunch_frames_raw": int(a["prelaunch_frames_raw"].sum()),
            "prelaunch_detections_raw": int(a["prelaunch_detections_raw"].sum())}
  return a, {**counts, "fall_rate_hits": rate(fell, hit), "fall_rate_given_contact": rate(fell, ch),
             "contact_rate_hits": rate(contact, hit), "false_prep_rate": rate(brace > 0, miss),
             "braced_on_hits": rate(brace > 0, hit), "braced_s": float(brace.mean()*dt),
             "braced_s_raw": float(a["brace_steps_raw"].mean()*dt),
             # Match CSV/common evidence reductions even for float32 rollouts.
             "progress_m": float(np.mean(a["progress_m"], dtype=np.float64)),
             "detections_per_episode": float(a["n_detections"].mean()),
             "prelaunch_brace_rate": float(a["braced_prelaunch"].mean()),
             "false_alarm_rate": counts["prelaunch_detections"]/counts["prelaunch_frames"] if counts["prelaunch_frames"] else None,
             "false_alarm_rate_raw": counts["prelaunch_detections_raw"]/counts["prelaunch_frames_raw"] if counts["prelaunch_frames_raw"] else None,
             "n_censored": int(a["censored"].sum())}


def _paired(a, b):
  for name in ("scenario_seed", "scenario_index", "is_miss", "mass", "speed"):
    if not np.array_equal(a[name], b[name]): raise ValueError("Unpaired depth scenarios: " + name)
  hit = ~a["is_miss"].astype(bool)
  x, y = a["fell"][hit].astype(float), b["fell"][hit].astype(float)
  d = x-y; n = len(d)
  se = float(d.std(ddof=1)/np.sqrt(n)) if n > 1 else None
  return {"n_hits": n, "difference_a_minus_b": float(d.mean()) if n else None,
          "normal95": [max(-1., float(d.mean()-1.96*se)), min(1., float(d.mean()+1.96*se))] if se is not None else None,
          "a_only": int(((x==1)&(y==0)).sum()), "b_only": int(((x==0)&(y==1)).sum()),
          "uncertainty": "approximate paired normal interval, conditional on this frozen skill pair"}


def _episode_csv(path, tab, source, method, param):
  with open(path, "w", newline="") as stream:
    w = csv.writer(stream); keys = list(tab)
    w.writerow(["scenario_id", "source", "method", "param", *keys])
    for i in range(len(tab["fell"])):
      sid = f"seed{int(tab['scenario_seed'][i])}_env{int(tab['scenario_index'][i]):05d}"
      w.writerow([sid, source, method, param, *[np.asarray(tab[k][i]).item() for k in keys]])

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
    # Median of the candidate points per axis (as calibrated in E8), ignoring non-candidates.
    Pm = jp.where(cand[..., None], P, jp.nan).reshape(-1, 3)
    surf = jp.nanmedian(Pm, axis=0)
    surf = jp.where(jp.isnan(surf), 0.0, surf)
    view = surf - cam_pos
    centre = surf + radius * view / (jp.linalg.norm(view) + 1e-9)
    return n >= 3, centre
  return jax.jit(jax.vmap(one))


def main(_):
  outdir = Path(_OUT.value)
  if outdir.exists() and any(outdir.iterdir()):
    raise ValueError("Output directory is not empty; use a new output to preserve provenance")
  if _N.value < 1 or _BATCHES.value < 1 or _STEPS.value < 1 or _PERIOD.value < 1 or _EXTRA_LAT.value < 0:
    raise ValueError("Positive counts/cadence and nonnegative latency required")
  if not np.isfinite(_RELEASE.value) or _RELEASE.value < 0:
    raise ValueError("release_after must be finite and nonnegative")
  if _EXTRA_LAT.value % _PERIOD.value:
    raise ValueError("Session G comparison requires latency divisible by period (shared delivery phase)")
  if not _SOURCES.value or any(x not in ("depth", "parametric") for x in _SOURCES.value):
    raise ValueError("Unknown or empty observation sources")
  outdir.mkdir(parents=True, exist_ok=True); (outdir/"episodes").mkdir()
  with open(_RM.value) as stream:
    rmj = json.load(stream)
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
  # Parametric arm: same cadence, added latency, field of view and noise as the depth arm; its
  # random miss probability is E8's measured 5.5%. The controller's lookahead uses this config
  # for both arms (the depth arm's detector has no explicit miss model).
  det_cfg_param = D.DetectorConfig(period=_PERIOD.value, latency=_EXTRA_LAT.value, sigma0=_SIG.value, sigma1=0.0,
                                   fov_deg=fovy, miss_prob=0.055, buffer_len=max(16, _EXTRA_LAT.value+1))

  @jax.jit
  def act_fn(obs, cmd_nom, keys, brace_flag):
    a_nom, _ = jax.vmap(nominal)(R._with_command(obs, cmd_nom), keys)
    a_br, _ = jax.vmap(brace)(R._with_command(obs, jp.zeros_like(cmd_nom)), keys)
    return jp.where(brace_flag[:, None], a_br, a_nom)

  config = {"schema_version": SE.SCHEMA_VERSION, "protocol": "foresight.session_h.depth.v1", "steps": _STEPS.value, "dt": float(env.dt),
            "period": _PERIOD.value, "extra_latency": _EXTRA_LAT.value, "res": res, "fovy_deg": fovy,
            "azimuth_deg": _AZ.value, "det_sigma": _SIG.value, "seed": _SEED.value,
            "impl": _IMPL.value, "object_radius_m": _RAD.value, "miss_fraction": _MISS.value,
            "mass_range": mr, "speed_range": [3.,8.], "command_vx": _VX.value, "range_dist": [3.,5.],
            "env_config": env._config.to_dict(),
            "episodes": n, "batches": _BATCHES.value, "methods": _METHODS.value, "sources": _SOURCES.value,
            "release_after": _RELEASE.value, "release_contract": "Session G closest-approach OR stale release after minimum commitment",
            "parked_object_pos": np.asarray(env._init_q[env._proj_qpos_adr:env._proj_qpos_adr+3]),
            "estimator": "per-axis median with known sphere radius; no privileged launch gate",
            "kinematics_assumption": "Session G command-derived pelvis velocity retained",
            "future_visibility_assumption": "Session G fixed miss probability, no exact future image visibility forecast",
            "robot_pose_assumption": "exact simulator camera and pelvis pose",
            "wall_time_is_simulated_latency": False,
            "active_metric_convention": "exclude intervals after latched failure; *_raw metrics retain the Session G all-horizon counting convention"}
  provenance = SE.protocol_metadata({"nominal": R.latest_checkpoint(_NOM.value), "brace": R.latest_checkpoint(_BR.value), "response": _RM.value}, config)
  _write_json(outdir/"config.json", {**config, "provenance": provenance})
  rows = []
  per_ep = {}
  controller_cache = {}
  for b in range(_BATCHES.value):
    rs = np.random.RandomState(_SEED.value + b)
    masses = rs.uniform(*mr, n); speeds = rs.uniform(3.0, 8.0, n)
    hits = rs.uniform(*env._config.impact.hit_time_range, n); is_miss = rs.uniform(size=n) < _MISS.value
    key = jax.random.PRNGKey(_SEED.value + b)
    reset_keys = jax.random.split(jax.random.fold_in(key, 1), n)
    wenv = R.batched_env(env, masses)
    step_fn = jax.jit(wenv.step); reset_fn = jax.jit(wenv.reset)
    base = dict(speed=speeds, hit_time=hits, is_miss=is_miss, command=[_VX.value, 0, 0])
    ref_info, ref_traj = R.rollout(env, wenv, nominal, key, _STEPS.value,
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
                   "belief_voi": RU.RuleConfig(kind="belief_voi", c_prep=param, wait_steps=_PERIOD.value),
                   "belief_wait": RU.RuleConfig(kind="belief_wait", c_prep=param, wait_steps=_PERIOD.value)}[kind]
        rc_rule = rc_rule._replace(release_after=_RELEASE.value)
        if spec not in controller_cache:
          cached = AnticipationController(det_cfg_param, PF.PFConfig(mass_range=mr), rc_rule, rm)
          controller_cache[spec] = cached, jax.jit(cached.step_with_detections)
        ctrl, ctrl_step = controller_cache[spec]
        t0 = time.time()
        rng = jax.random.fold_in(key, 7)
        state = reset_fn(reset_keys)
        state = R.set_scenario(env, state, launch_pos=lp, launch_vel=lv, **base)
        carry = ctrl.init(n, jax.random.fold_in(rng, 1))
        dst = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (n,) + jp.shape(x)), D.init_state(det_cfg_param))
        brace_flag = jp.zeros((n,), bool)
        pel0 = np.asarray(state.data.xpos[:, pelvis_b]); fwd0 = np.asarray(state.data.xmat[:, pelvis_b].reshape(n, 3, 3)[:, :, 0])
        heading0 = fwd0[:, :2] / (np.linalg.norm(fwd0[:, :2], axis=-1, keepdims=True) + 1e-9)
        brace_steps_raw = np.zeros(n, int)
        braced_prelaunch = np.zeros(n, bool)
        first_brace = np.full(n, -1, int); first_capture = np.full(n, -1, int); first_delivery = np.full(n, -1, int)
        brace_steps = np.zeros(n, int); prog = np.zeros(n); alive = np.ones(n, bool); last_pel = pel0.copy()
        n_det = np.zeros(n, int); render_ms = []; queue = []   # (deliver_step, ok, centre, capture_step)
        fa_frames_raw = np.zeros(n, int); launched_frames_raw = np.zeros(n, int)
        fa_frames = np.zeros(n, int); launched_frames = np.zeros(n, int)   # false alarms: detections with no object in flight
        for t in range(_STEPS.value):
          rng, k1, k2, k3 = jax.random.split(rng, 4)
          d0 = state.data
          active_before = ~np.asarray(state.info["terminated"], bool)
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
              # No privileged gating: whatever the detector reports is delivered; detections while
              # no object is in flight are counted as false alarms.
              fa_frames_raw += np.asarray(ok & ~state.info["proj_launched"]).astype(int)
              launched_frames_raw += np.asarray(~state.info["proj_launched"]).astype(int)
              fa_frames += np.asarray(ok & ~state.info["proj_launched"]).astype(int) * active_before
              launched_frames += np.asarray(~state.info["proj_launched"]).astype(int) * active_before
              queue.append((t + _EXTRA_LAT.value, ok, centre, t))
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
          carry, brace_flag, logs = ctrl_step(carry, det_valid, det_pos, det_sigma, cap, pel, pvel, t, k2)
          n_det += np.asarray(det_valid) * active_before
          valid_np = np.asarray(det_valid, bool) & active_before
          first_capture = np.where((first_delivery < 0) & valid_np, np.asarray(cap), first_capture)
          first_delivery = np.where((first_delivery < 0) & valid_np, t, first_delivery)
          raw_brace = np.asarray(brace_flag, bool)
          applied_before_failure = raw_brace & active_before
          first_brace = np.where((first_brace < 0) & applied_before_failure, t, first_brace)
          braced_prelaunch |= applied_before_failure & ~np.asarray(state.info["proj_launched"], bool)
          brace_steps_raw += raw_brace
          info = dict(state.info); info["brace_active"] = brace_flag; state = state.replace(info=info)
          act = act_fn(state.obs, state.info["command_nominal"], jax.random.split(k3, n), brace_flag)
          state = step_fn(state, act)
          pel_np = np.asarray(state.data.xpos[:, pelvis_b])
          prog += np.where(alive, np.sum((pel_np[:, :2] - last_pel[:, :2]) * heading0, axis=-1), 0.0)
          last_pel = pel_np
          alive &= ~np.asarray(state.info["fell"])
          brace_steps += applied_before_failure
        fi = jax.tree_util.tree_map(np.asarray, state.info)
        hit_s = ~is_miss; c = fi["impact_contact"] & hit_s
        row = {"source": source, "method": kind, "param": param, "batch": b, "n": n,
               "fall_rate_hits": float(fi["fell"][hit_s].mean()),
               "fall_rate_given_contact": float(fi["fell"][c].mean()) if c.any() else float("nan"),
               "contact_rate_hits": float(fi["impact_contact"][hit_s].mean()),
               "false_prep_rate": float((brace_steps[is_miss] > 0).mean()) if is_miss.any() else 0.0,
               "braced_on_hits": float((brace_steps[hit_s] > 0).mean()), "braced_s": float(brace_steps.mean() * env.dt),
               "progress_m": float(prog.mean()), "detections_per_episode": float(n_det.mean()),
               "false_alarm_rate": float(fa_frames.sum() / max(launched_frames.sum(), 1)) if source == "depth" else float("nan"),
               "render_ms_per_frame": float(np.mean(render_ms[1:])) if len(render_ms) > 1 else float("nan"),
               "wall_s": time.time() - t0}
        rows.append(row)
        tab = {"fell": fi["fell"], "contact": fi["impact_contact"], "is_miss": is_miss,
               "brace_steps": brace_steps, "brace_steps_raw": brace_steps_raw, "progress_m": prog,
               "scenario_seed": np.full(n, _SEED.value+b), "scenario_index": np.arange(n),
               "mass": masses, "speed": speeds, "t_hit_nominal": fi["t_hit_nominal"],
               "contact_step": fi["contact_step"], "term_step": fi["term_step"], "term_reason": fi["term_reason"],
               "n_detections": n_det, "prelaunch_frames": launched_frames, "prelaunch_detections": fa_frames,
               "prelaunch_frames_raw": launched_frames_raw, "prelaunch_detections_raw": fa_frames_raw,
               "braced_prelaunch": braced_prelaunch, "brace_onset_step": first_brace,
               "first_detection_capture_step": first_capture, "first_detection_delivery_step": first_delivery,
               "censored": (fi["contact_step"] >= 0) & ((_STEPS.value-fi["contact_step"]-1)*env.dt < 2.5) & ~fi["fell"]}
        per_ep.setdefault((source, kind, param), []).append(tab)
        _episode_csv(outdir/"episodes"/f"{source}_{kind}_{param:g}_b{b:03d}.csv", tab, source, kind, param)
        print(f"  {source:10} {kind:14} p={param:<4g} {row['wall_s']:4.0f}s  fall|hits {row['fall_rate_hits']:.3f}  "
              f"fall|contact {row['fall_rate_given_contact']:.3f}  contact {row['contact_rate_hits']:.2f}  "
              f"false-prep {row['false_prep_rate']:.3f}  braced {row['braced_s']:.2f}s  det/ep {row['detections_per_episode']:.1f}  "
              f"render {row['render_ms_per_frame']:.0f} ms")
    # Reactive evidence is paired and retained, not only a marginal summary row.
    ref_tab = {"fell": ref_info["fell"], "contact": ref_info["impact_contact"], "is_miss": is_miss,
               "brace_steps": np.zeros(n, int), "brace_steps_raw": np.zeros(n, int),
               "progress_m": R.episode_table(ref_info, ref_traj, env.dt)["progress_m"],
               "scenario_seed": np.full(n, _SEED.value+b), "scenario_index": np.arange(n),
               "mass": masses, "speed": speeds, "t_hit_nominal": ref_info["t_hit_nominal"],
               "contact_step": ref_info["contact_step"], "term_step": ref_info["term_step"], "term_reason": ref_info["term_reason"],
               "n_detections": np.zeros(n, int), "prelaunch_frames": np.zeros(n, int), "prelaunch_detections": np.zeros(n, int),
               "prelaunch_frames_raw": np.zeros(n, int), "prelaunch_detections_raw": np.zeros(n, int),
               "braced_prelaunch": np.zeros(n, bool), "brace_onset_step": np.full(n, -1, int),
               "first_detection_capture_step": np.full(n, -1, int), "first_detection_delivery_step": np.full(n, -1, int),
               "censored": (ref_info["contact_step"] >= 0) & ((_STEPS.value-ref_info["contact_step"]-1)*env.dt < 2.5) & ~ref_info["fell"]}
    per_ep.setdefault(("none", "reactive", 0.), []).append(ref_tab)
    _episode_csv(outdir/"episodes"/f"none_reactive_0_b{b:03d}.csv", ref_tab, "none", "reactive", 0.)
    rows.append({"source": "none", "method": "reactive", "param": 0.0, "batch": b, "n": n,
                 "fall_rate_hits": float(ref_info["fell"][~is_miss].mean()),
                 "fall_rate_given_contact": float(ref_info["fell"][ref_info["impact_contact"] & ~is_miss].mean()),
                 "contact_rate_hits": float(ref_info["impact_contact"][~is_miss].mean()), "false_prep_rate": 0.0,
                 "braced_on_hits": 0.0, "braced_s": 0.0, "progress_m": float("nan"), "detections_per_episode": 0.0,
                 "render_ms_per_frame": float("nan"), "false_alarm_rate": float("nan"), "wall_s": 0.0})

  ep_out = {}
  for (src, meth, par), parts in per_ep.items():
    for f in parts[0]:
      ep_out[f"{src}__{meth}__p{par:g}__{f}"] = np.concatenate([p_[f] for p_ in parts])
  np.savez_compressed(os.path.join(_OUT.value, "episodes.npz"), **ep_out)
  out = []; pooled = {}
  for (src, meth, par), parts in per_ep.items():
    tab, metrics = _pool(parts, env.dt); pooled[(src, meth, par)] = tab
    timings = [r["render_ms_per_frame"] for r in rows if (r["source"], r["method"], r["param"]) == (src, meth, par) and np.isfinite(r["render_ms_per_frame"])]
    out.append({"source": src, "method": meth, "param": par, **metrics,
                "render_ms_per_frame": float(np.mean(timings)) if timings else None})
  with open(outdir/"table.csv", "w", newline="") as stream:
    w = csv.DictWriter(stream, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
  paired = []
  for condition, tab in pooled.items():
    if condition == ("none", "reactive", 0.): continue
    paired.append({"condition_a": condition, "condition_b": ("none", "reactive", 0.),
                   **_paired(tab, pooled[("none", "reactive", 0.)])})
    src, meth, par = condition
    if src == "depth" and ("parametric", meth, par) in pooled:
      paired.append({"condition_a": condition, "condition_b": ("parametric", meth, par),
                     **_paired(tab, pooled[("parametric", meth, par)])})
  _write_json(outdir/"paired.json", paired)
  _write_json(outdir/"batches.json", rows)
  _write_json(outdir/"summary.json", {"status": "complete", "schema_version": SE.SCHEMA_VERSION, "protocol": config["protocol"],
              "period": _PERIOD.value, "extra_latency": _EXTRA_LAT.value, "res": res, "fovy_deg": fovy,
              "azimuth_deg": _AZ.value, "det_sigma": _SIG.value, "episodes_per_condition": n*_BATCHES.value,
              "config": config, "provenance": provenance, "rows": out})
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
