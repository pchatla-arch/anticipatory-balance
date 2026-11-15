# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Reviewed online depth experiment with explicit timing and paired evidence.

Sources are deliberately distinguished:
  depth: image-derived centre/validity, with no launch metadata in its observations;
  parametric: delayed privileged centre, pelvis-cone visibility and IID dropout;
  parametric_matched: privileged noisy centre using the actual image-validity mask.
The last is a diagnostic comparator, not a deployable sensor. The renderer's wall
clock cost is reported separately and does not advance simulated observation age.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
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
from foresight.eval import statistics as ST
from foresight.perception import detections as D
from foresight.perception import depth_geometry as DG

_NOM = flags.DEFINE_string("nominal", "checkpoints/nominal_s1", "walker checkpoint")
_BR = flags.DEFINE_string("brace", "checkpoints/brace2_s1", "BRACE checkpoint")
_RM = flags.DEFINE_string("response_model", "results/response_model.json", "frozen response model")
_IMPL = flags.DEFINE_string("impl", "warp", "MuJoCo backend")
_N = flags.DEFINE_integer("episodes", 256, "environments per batch")
_BATCHES = flags.DEFINE_integer("batches", 2, "independent scenario batches")
_STEPS = flags.DEFINE_integer("steps", 360, "20 ms control steps; full test includes recovery")
_PERIOD = flags.DEFINE_integer("period", 5, "capture period in control steps")
_EXTRA_LAT = flags.DEFINE_integer("extra_latency", 0, "capture-to-delivery delay in control steps")
_RES = flags.DEFINE_integer("res", 96, "square image resolution")
_AZ = flags.DEFINE_float("azimuth_deg", 40.0, "launch half-angle; does not guarantee perpetual visibility")
_MISS = flags.DEFINE_float("miss_fraction", 0.3, "scheduled near-miss probability")
_VX = flags.DEFINE_float("command_vx", 0.5, "nominal forward command")
_SIG = flags.DEFINE_float("det_sigma", 0.03, "fixed assumed likelihood SD; measured errors saved separately")
_RAD = flags.DEFINE_float("object_radius", 0.15, "known sphere radius, metres")
_PARAM_MISS = flags.DEFINE_float("parametric_miss_prob", 0.055, "IID dropout only for parametric source")
_IMAGE_MISS = flags.DEFINE_float("image_miss_prob", 0.0, "optional shared IID dropout for depth/matched sources")
_METHODS = flags.DEFINE_list("methods", ["ttc_threshold:0.3", "belief_lead:0.1", "belief_wait:0.1", "belief_voi:0.1"], "kind:param")
_SOURCES = flags.DEFINE_list("sources", ["depth", "parametric", "parametric_matched"], "named observation sources")
_OUT = flags.DEFINE_string("out", "results/review_depth", "new output directory")
_SEED = flags.DEFINE_integer("seed", 10001, "fresh scenario seed; +batch")
_PHASE = flags.DEFINE_enum("phase", "test", ["smoke", "development", "test"], "evidence phase")
_RELEASE = flags.DEFINE_enum("release_mode", "deadline", ["deadline", "legacy"], "retention contract")
_HOLD = flags.DEFINE_float("hold", 0.5, "hold after locked predicted arrival")
_COMMIT = flags.DEFINE_integer("commit_steps", 15, "minimum BRACE duration")
_MAX_HOLD = flags.DEFINE_float("max_hold_s", 4.0, "absolute maximum brace duration")
_SINGLE = flags.DEFINE_bool("single_commit", True, "at most one preparation per single-object episode")
_TRACE = flags.DEFINE_integer("trace_episodes", 8, "first N scenarios per batch with full timing traces")
_ALLOW_LEGACY = flags.DEFINE_bool("allow_legacy_response", False, "explicitly allow a historical, unmatched response file for diagnostics")

PARK_POSITION = (1000.0, 1000.0, 100.0)
SENSOR_DESCRIPTIONS = {
    "depth": "actual image geometric validity and surface-median centre; known radius; no launch gate",
    "parametric": "capture-time launch validity, pelvis horizontal cone, true centre plus Gaussian noise and IID dropout",
    "parametric_matched": "privileged diagnostic: actual image validity, true centre plus Gaussian noise; shared image dropout",
}


def make_detector(res, fovy, radius):
  dirs = jp.asarray(DG.pixel_directions(res, fovy))
  fn = lambda depth, pos, mat, pelvis: DG.detect_centre(depth, pos, mat, pelvis, dirs, radius, xp=jp)
  return jax.jit(jax.vmap(fn))


def method_config(kind, param):
  common = dict(kind=kind, release_mode=_RELEASE.value, hold=_HOLD.value,
                commit_steps=_COMMIT.value, max_hold_s=_MAX_HOLD.value,
                single_commit=_SINGLE.value, horizon_steps=_STEPS.value,
                wait_steps=_PERIOD.value)
  if kind == "ttc_threshold":
    common["theta"] = param
  elif kind in ("belief_commit", "belief_lead", "belief_wait", "belief_voi"):
    common["c_prep"] = param
  else:
    raise ValueError("Unsupported depth controller: " + kind)
  return RU.RuleConfig(**common)


def parse_methods():
  parsed = []
  for spec in _METHODS.value:
    kind, value = spec.split(":", 1)
    value = float(value)
    if not np.isfinite(value) or value < 0:
      raise ValueError("Method parameter must be finite and nonnegative")
    method_config(kind, value)
    parsed.append((kind, value))
  if not parsed or len(set(parsed)) != len(parsed):
    raise ValueError("Need nonempty unique methods")
  return parsed


def _slug(source, method, param, batch):
  p = format(param, ".9g").replace(".", "p").replace("-", "m")
  return f"{source}_{method}_p{p}_batch{batch:03d}"


def _extra_columns(tab, source, kind, param, batch, seed, n):
  return {**tab, "scenario_id": np.array([f"seed{seed}_env{i:05d}" for i in range(n)]),
          "source": np.full(n, source), "method": np.full(n, kind),
          "param": np.full(n, param), "batch": np.full(n, batch), "scenario_seed": np.full(n, seed),
          "scenario_index": np.arange(n, dtype=int)}


def _log_value(logs, name, idx):
  if name not in logs:
    return ""
  a = np.asarray(logs[name])
  return ST.json_safe(a[idx] if a.ndim else a)


def _write_csv_rows(path, rows):
  if not rows:
    return
  names = list(dict.fromkeys(k for row in rows for k in row))
  with Path(path).open("w", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=names)
    writer.writeheader()
    writer.writerows(rows)


def main(_):
  if min(_N.value, _BATCHES.value, _STEPS.value, _PERIOD.value) <= 0 or _EXTRA_LAT.value < 0:
    raise ValueError("Positive episode/batch/step/period sizes and nonnegative latency required")
  if not 0 <= _MISS.value <= 1 or not 0 <= _PARAM_MISS.value <= 1 or not 0 <= _IMAGE_MISS.value <= 1:
    raise ValueError("Miss probabilities must lie in [0,1]")
  if _SIG.value <= 0 or _RAD.value <= 0 or _TRACE.value < 0:
    raise ValueError("Positive noise/radius and nonnegative trace size required")
  sources = list(_SOURCES.value)
  if not sources or len(set(sources)) != len(sources) or any(s not in SENSOR_DESCRIPTIONS for s in sources):
    raise ValueError("Sources must be unique members of depth,parametric,parametric_matched")
  methods = parse_methods()
  outdir = Path(_OUT.value)
  if (outdir / "config.json").exists() or (outdir / "summary.json").exists():
    raise FileExistsError("Choose a fresh output directory; existing runs are never reused implicitly")
  outdir.mkdir(parents=True, exist_ok=True)
  (outdir / "episodes").mkdir(); (outdir / "traces").mkdir()
  rmj = json.loads(Path(_RM.value).read_text())
  response_validation = ST.validate_response_protocol(rmj, {
      "timing_reference": "reference_entry", "reference_radius": 0.45,
      "hold_s": _HOLD.value, "commit_steps": _COMMIT.value,
      "max_hold_s": _MAX_HOLD.value, "release_mode": _RELEASE.value,
      "single_commit": _SINGLE.value, "dt": 0.02}, allow_legacy=_ALLOW_LEGACY.value)
  rm = RU.ResponseModel(rmj["a_cont"], rmj["b_cont"], rmj["a_brace"], rmj["b_brace"],
                        rmj.get("a_brace_lead", 0.0), rmj.get("b_brace_lead", 0.0), rmj.get("c_lead", 0.0),
                        rmj.get("lead_max", 0.5), "c_lead" in rmj)
  nominal = R.load_policy(_NOM.value); brace = R.load_policy(_BR.value)
  env = R.make_env(impl=_IMPL.value, azimuth_range_deg=_AZ.value, range_dist=[3.0, 5.0])
  # The original environment already keeps the free projectile at _init_q until
  # launch. Move that reset position geometrically out of the sensor range for
  # every arm, including the replay reference. No per-frame depth launch gate.
  qa = int(env._proj_qpos_adr)
  env._init_q = env._init_q.at[qa:qa + 3].set(jp.asarray(PARK_POSITION))
  m = env.mj_model
  cam_id = m.camera("head_cam").id
  fovy = float(m.cam_fovy[cam_id]); n, res = _N.value, _RES.value
  pelvis_b = env._pelvis_body
  needs_render = any(s in ("depth", "parametric_matched") for s in sources)
  if needs_render:
    rc = mjx.create_render_context(mjm=m, nworld=n, cam_res=(res, res), render_rgb=False, render_depth=True,
                                   enabled_geom_groups=[0, 1, 2])
    refit = jax.jit(mjx.refit_bvh); render = jax.jit(mjx.render)
    try:
      from mujoco.mjx.warp import render_context as _rcm
      adr = int(_rcm.get(rc.pytree()).depth_adr.numpy()[cam_id])
    except (ImportError, AttributeError, TypeError):
      adr = cam_id * res * res
    detect = make_detector(res, fovy, _RAD.value)
  mr = (4.0, 16.0)
  config = {"schema": ST.SCHEMA_VERSION, "phase": _PHASE.value, "episodes_per_batch": n,
            "batches": _BATCHES.value, "steps": _STEPS.value, "dt": float(env.dt),
            "seed": _SEED.value, "period": _PERIOD.value, "extra_latency": _EXTRA_LAT.value,
            "capture_schedule": "step % period == 0", "delivery_schedule": "capture + extra_latency",
            "simulated_delay_ms": 1000 * _EXTRA_LAT.value * float(env.dt),
            "renderer_wall_time_added_to_simulated_delay": False,
            "res": res, "fovy_deg": fovy, "azimuth_deg": _AZ.value, "miss_fraction": _MISS.value,
            "near_miss_lateral_offset_m": 0.6, "mass_range": mr, "speed_range": [3, 8], "range_dist": [3, 5],
            "command_vx": _VX.value, "park_position": PARK_POSITION,
            "det_sigma_assumed_m": _SIG.value, "known_radius_m": _RAD.value,
            "estimator": "per-axis surface median plus known-radius correction (latest upstream/E8 estimator)",
            "parametric_miss_prob": _PARAM_MISS.value, "image_miss_prob": _IMAGE_MISS.value,
            "sources": {s: SENSOR_DESCRIPTIONS[s] for s in sources}, "methods": [list(x) for x in methods],
            "rules": [method_config(*x)._asdict() for x in methods], "trace_episodes_per_batch": min(n, _TRACE.value),
            "robot_state_assumption": "exact simulated camera/robot pose and freejoint linear velocity",
            "lookahead_sensor_assumption": "simplified pelvis-cone forecast; empirical depth visibility is not forecast exactly",
            "depth_calibration": "diagnostic only; fixed assumed likelihood is not fitted on test frames",
            "response_validation": response_validation,
            "env_config": env._config.to_dict()}
  metadata = ST.protocol_metadata({"nominal": R.latest_checkpoint(_NOM.value), "brace": R.latest_checkpoint(_BR.value),
                                   "response_model": _RM.value}, config)
  ST.write_json(outdir / "config.json", {**config, "provenance": metadata})

  @jax.jit
  def act_fn(obs, cmd_nom, keys, brace_flag):
    a_nom, _ = jax.vmap(nominal)(R._with_command(obs, cmd_nom), keys)
    a_br, _ = jax.vmap(brace)(R._with_command(obs, jp.zeros_like(cmd_nom)), keys)
    return jp.where(brace_flag[:, None], a_br, a_nom)

  # Reuse compiled controllers across scenario batches; checkpoint and rule
  # parameters are fixed for this invocation, only episode state/keys change.
  detectors = {}; detector_steps = {}; controllers = {}; controller_steps = {}
  for source in sources:
    cfg = D.DetectorConfig(period=_PERIOD.value, latency=_EXTRA_LAT.value,
                           sigma0=_SIG.value, sigma1=0., fov_deg=fovy,
                           miss_prob=_PARAM_MISS.value if source == "parametric" else _IMAGE_MISS.value,
                           buffer_len=max(16, _EXTRA_LAT.value + 1))
    detectors[source] = cfg
    def one_detector(st, o, p, f, launched, key, cfg=cfg):
      return D.step(cfg, st, o, p, f, launched, key)
    detector_steps[source] = jax.jit(jax.vmap(one_detector))
    for kind, param in methods:
      ctrl = AnticipationController(cfg, PF.PFConfig(mass_range=mr), method_config(kind, param), rm)
      controllers[(source, kind, param)] = ctrl
      controller_steps[(source, kind, param)] = jax.jit(ctrl.step_with_detections)

  groups = {}; batch_rows = []; calibration = []; timing = {}
  for b in range(_BATCHES.value):
    seed = _SEED.value + b
    rs = np.random.RandomState(seed)
    masses = rs.uniform(*mr, n); speeds = rs.uniform(3.0, 8.0, n)
    hits = rs.uniform(*env._config.impact.hit_time_range, n); is_miss = rs.uniform(size=n) < _MISS.value
    key = jax.random.PRNGKey(seed)
    reset_keys = jax.random.split(jax.random.fold_in(key, 1), n)
    wenv = R.batched_env(env, masses)
    step_fn = jax.jit(wenv.step); reset_fn = jax.jit(wenv.reset)
    base = dict(speed=speeds, hit_time=hits, is_miss=is_miss, command=[_VX.value, 0, 0])
    ref_info, ref_traj = R.rollout(env, wenv, nominal, key, _STEPS.value,
                                  scenario_fn=lambda s: R.set_scenario(env, s, **base), reset_keys=reset_keys)
    lp, lv = ref_info["proj_launch_pos"], ref_info["proj_launch_vel"]
    ref_tab = _extra_columns(R.episode_table(ref_info, ref_traj, env.dt), "none", "reactive", 0., b, seed, n)
    groups.setdefault(("none", "reactive", 0.), []).append(ref_tab)
    R.write_csv(outdir / "episodes" / (_slug("none", "reactive", 0., b) + ".csv"), ref_tab)
    batch_rows.append({"source": "none", "method": "reactive", "param": 0., "batch": b,
                       **ST.summarize_episodes(ref_tab, env.dt)})
    print(f"batch {b}: {int((~is_miss).sum())} hits, {int(is_miss.sum())} misses; reference fall "
          f"{ref_info['fell'][~is_miss].mean() if (~is_miss).any() else float('nan'):.3f}", flush=True)

    for source in sources:
      det_cfg = detectors[source]
      det_step = detector_steps[source]
      for kind, param in methods:
        ctrl = controllers[(source, kind, param)]
        control_step = controller_steps[(source, kind, param)]
        t0 = time.perf_counter(); rng = jax.random.fold_in(key, 7)
        state = R.set_scenario(env, reset_fn(reset_keys), launch_pos=lp, launch_vel=lv, **base)
        carry = ctrl.init(n, jax.random.fold_in(rng, 1))
        dst = jax.tree_util.tree_map(lambda x: jp.broadcast_to(x, (n,) + jp.shape(x)), D.init_state(det_cfg))
        queue = DG.DetectionQueue(_EXTRA_LAT.value)
        n_det = np.zeros(n, int); det_before_fall = np.zeros(n, int)
        prelaunch_frames = np.zeros(n, int); prelaunch_detections = np.zeros(n, int)
        brace_steps = np.zeros(n, int); switches = np.zeros(n, int); prog = np.zeros(n)
        first_capture = np.full(n, -1, int); first_delivery = np.full(n, -1, int)
        first_brace = np.full(n, -1, int); first_release = np.full(n, -1, int)
        brace_at_contact = np.zeros(n, bool); previous_brace = np.zeros(n, bool); previous_contact = np.zeros(n, bool)
        alive = np.ones(n, bool)
        render_ms = []; err = DG.ErrorAccumulator()
        slug = _slug(source, kind, param, b)
        trace_fields = ["scenario_id", "step", "capture_step", "delivery_valid", "det_x", "det_y", "det_z",
                        "applied_brace", "requested_brace", "terminated_before", "contact_after",
                        "p_hit", "t_hit", "cost_brace", "cost_continue", "cost_wait", "next_delivery_step",
                        "next_capture_step", "release_step", "release_reason"]
        with (outdir / "traces" / (slug + ".csv")).open("w", newline="") as trace_stream:
          trace_writer = csv.DictWriter(trace_stream, fieldnames=trace_fields)
          trace_writer.writeheader()
          for t in range(_STEPS.value):
            rng, k1, k2, k3 = jax.random.split(rng, 4)
            d0 = state.data; pel = d0.xpos[:, pelvis_b]
            fwd = d0.xmat[:, pelvis_b].reshape(n, 3, 3)[:, :, 0]
            pvel = d0.qvel[:, :3]
            terminated_before = np.asarray(state.info["terminated"], bool)
            if source in ("depth", "parametric_matched"):
              if t % _PERIOD.value == 0:
                tr = time.perf_counter()
                data = refit(env.mjx_model, d0, rc.pytree())
                rendered = render(env.mjx_model, data, rc.pytree())
                depth = rendered[1][:, adr:adr + res * res].reshape(n, res, res)
                ok, image_centre, pixels = detect(depth, d0.cam_xpos[:, cam_id],
                                                  d0.cam_xmat[:, cam_id].reshape(n, 3, 3), pel)
                jax.block_until_ready(image_centre)
                render_ms.append(1000 * (time.perf_counter() - tr))
                kd, kn = jax.random.split(k1)
                valid = ok & (jax.random.uniform(kd, (n,)) >= _IMAGE_MISS.value)
                # Truth below is for an explicitly privileged diagnostic comparator
                # and error logging. The depth observation path uses image_centre.
                truth = d0.qpos[:, qa:qa + 3]
                pos = image_centre if source == "depth" else truth + _SIG.value * jax.random.normal(kn, (n, 3))
                sigma = jp.full((n,), _SIG.value)
                queue.capture(t, valid, pos, sigma)
                active = ~terminated_before
                prelaunch = active & ~np.asarray(state.info["proj_launched"], bool)
                prelaunch_frames += prelaunch
                prelaunch_detections += prelaunch & np.asarray(ok, bool)
                err.add(np.asarray(ok)[active], np.asarray(image_centre)[active], np.asarray(truth)[active],
                        np.asarray(pel)[active], np.asarray(state.info["proj_launched"])[active])
              delivered = queue.deliver(t)
              if delivered is None:
                det_valid, det_pos = jp.zeros(n, bool), jp.zeros((n, 3))
                cap, det_sigma = jp.full(n, -1, jp.int32), jp.full(n, _SIG.value)
              else:
                captured, det_valid, det_pos, det_sigma = delivered
                cap = jp.full(n, captured, jp.int32)
            else:
              dst, det = det_step(dst, state.info["obj_pos"], pel, fwd, state.info["proj_launched"], jax.random.split(k1, n))
              det_valid, det_pos, det_sigma, cap = det.valid, det.pos, det.sigma, det.capture_step
            carry, request_brace, logs = control_step(carry, det_valid, det_pos, det_sigma, cap,
                                                      pel, pvel, t, k2, pelvis_fwd=fwd)
            applied = request_brace & ~state.info["terminated"]
            valid_np = np.asarray(det_valid, bool); cap_np = np.asarray(cap)
            first_capture = np.where((first_delivery < 0) & valid_np, cap_np, first_capture)
            first_delivery = np.where((first_delivery < 0) & valid_np, t, first_delivery)
            n_det += valid_np; det_before_fall += valid_np & ~terminated_before
            brace_np = np.asarray(applied, bool)
            first_brace = np.where((first_brace < 0) & brace_np, t, first_brace)
            first_release = np.where((first_release < 0) & previous_brace & ~brace_np, t, first_release)
            switches += brace_np != previous_brace; previous_brace = brace_np
            info = dict(state.info); info["brace_active"] = applied; state = state.replace(info=info)
            state = step_fn(state, act_fn(state.obs, state.info["command_nominal"], jax.random.split(k3, n), applied))
            pel_np = np.asarray(state.data.xpos[:, pelvis_b])
            fwd_np = np.asarray(fwd)[:, :2]
            heading = fwd_np / (np.linalg.norm(fwd_np, axis=-1, keepdims=True) + 1e-9)
            prog += np.where(~terminated_before,
                             np.sum((pel_np[:, :2] - np.asarray(pel)[:, :2]) * heading, axis=-1), 0.)
            contact_np = np.asarray(state.info["impact_contact"], bool)
            brace_at_contact |= contact_np & ~previous_contact & brace_np
            previous_contact = contact_np
            alive &= ~np.asarray(state.info["fell"], bool)
            brace_steps += brace_np
            if alive.any() and not np.isfinite(pel_np[alive]).all():
              raise FloatingPointError("Nonfinite pelvis state in an unterminated episode")
            observed = np.asarray(det_pos)
            for i in range(min(n, _TRACE.value)):
              row = {"scenario_id": f"seed{seed}_env{i:05d}", "step": t, "capture_step": int(cap_np[i]),
                     "delivery_valid": int(valid_np[i]), "det_x": observed[i, 0], "det_y": observed[i, 1], "det_z": observed[i, 2],
                     "applied_brace": int(brace_np[i]), "requested_brace": int(np.asarray(request_brace)[i]),
                     "terminated_before": int(terminated_before[i]), "contact_after": int(contact_np[i])}
              row.update({k: _log_value(logs, k, i) for k in trace_fields[11:]})
              trace_writer.writerow(row)
            if t % 50 == 0:
              trace_stream.flush()
        fi = jax.tree_util.tree_map(np.asarray, state.info)
        cs = fi["contact_step"]
        tab = {"mass": masses, "speed": speeds, "momentum": masses * speeds, "is_miss": is_miss,
               "t_hit_nominal": fi["t_hit_nominal"], "launched": fi["proj_launched"], "contact": fi["impact_contact"],
               "contact_step": cs, "fell": fi["fell"], "fell_before_launch": fi["fell_before_launch"],
               "term_reason": fi["term_reason"], "term_step": fi["term_step"], "progress_m": prog,
               "censored": (cs >= 0) & ((_STEPS.value - cs - 1) * env.dt < 2.5) & ~fi["fell"],
               "brace_steps_total": brace_steps, "brace_onset_step": first_brace,
               "first_release_step": first_release, "switches": switches, "brace_at_contact": brace_at_contact,
               "n_detections": n_det, "n_detections_before_failure": det_before_fall,
               "prelaunch_image_frames": prelaunch_frames, "prelaunch_image_detections": prelaunch_detections,
               "first_detection_capture_step": first_capture, "first_detection_delivery_step": first_delivery}
        tab = _extra_columns(tab, source, kind, param, b, seed, n)
        R.write_csv(outdir / "episodes" / (slug + ".csv"), tab)
        groups.setdefault((source, kind, param), []).append(tab)
        wall = time.perf_counter() - t0
        timing.setdefault((source, kind, param), []).extend(render_ms[1:])
        row = {"source": source, "method": kind, "param": param, "batch": b,
               **ST.summarize_episodes(tab, env.dt), "wall_s": wall,
               "render_detect_batch_ms": float(np.mean(render_ms[1:])) if len(render_ms) > 1 else None}
        batch_rows.append(row)
        calibration.append({"source": source, "method": kind, "param": param, "batch": b, **err.summary()})
        ST.write_json(outdir / "batches.json", batch_rows)
        ST.write_json(outdir / "calibration.json", calibration)
        print(f"  {source} {kind}:{param:g}: {wall:.1f}s, hits={row['n_hits']} "
              f"fall={row['fall_rate_hits']} false_prep={row['false_prep_rate']}", flush=True)

  combined = {k: ST.concatenate_tables(v) for k, v in groups.items()}
  # Preserve Session G's compact archive in addition to the explicit-ID CSVs.
  # The old brace_steps key remains an alias for brace_steps_total.
  archive = {}
  for (src, meth, par), tab in combined.items():
    for name, values in tab.items():
      # The result publisher permits numeric/bool archives only. Text IDs and
      # condition labels stay in CSV; seed+index identify each archived scenario.
      if np.asarray(values).dtype.kind in "biufc":
        archive[f"{src}__{meth}__p{par:g}__{name}"] = values
    archive[f"{src}__{meth}__p{par:g}__brace_steps"] = tab["brace_steps_total"]
  np.savez_compressed(outdir / "episodes.npz", **archive)
  rows = []
  for key, tab in combined.items():
    src, meth, par = key
    times = timing.get(key, [])
    pre_n = int(np.asarray(tab.get("prelaunch_image_frames", [])).sum())
    pre_det = int(np.asarray(tab.get("prelaunch_image_detections", [])).sum())
    rows.append({"source": src, "method": meth, "param": par, **ST.summarize_episodes(tab, env.dt),
                 "n_prelaunch_image_frames": pre_n, "n_prelaunch_image_detections": pre_det,
                 "false_alarm_rate": pre_det / pre_n if pre_n else None,
                 "render_detect_batch_ms": float(np.mean(times)) if times else None})
  comparisons = []
  for key, tab in combined.items():
    if key[0] == "none":
      continue
    comparisons.append({"condition_a": list(key), "condition_b": ["none", "reactive", 0.],
                        **ST.paired_comparison(tab, combined[("none", "reactive", 0.)], env.dt)})
    if key[0] == "depth":
      for other in ("parametric", "parametric_matched"):
        other_key = (other, key[1], key[2])
        if other_key in combined:
          comparisons.append({"condition_a": list(key), "condition_b": list(other_key),
                              **ST.paired_comparison(tab, combined[other_key], env.dt)})
  ST.write_json(outdir / "paired.json", comparisons)
  _write_csv_rows(outdir / "table.csv", rows)
  ST.write_json(outdir / "batches.json", batch_rows)
  ST.write_json(outdir / "calibration.json", calibration)
  ST.write_json(outdir / "summary.json", {"schema": ST.SCHEMA_VERSION, "status": "complete", "phase": _PHASE.value,
                "period": _PERIOD.value, "extra_latency": _EXTRA_LAT.value, "res": res, "fovy_deg": fovy,
                "azimuth_deg": _AZ.value, "episodes_per_condition": n * _BATCHES.value,
                "aggregation": "pooled episode numerators/denominators", "rows": rows,
                "calibration_use": "post-hoc diagnostic; no test-set likelihood fitting",
                "provenance": metadata})
  print("Complete; per-episode evidence, paired comparisons and provenance saved.", flush=True)


if __name__ == "__main__":
  app.run(main)
