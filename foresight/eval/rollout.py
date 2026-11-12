# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Batched evaluation rollouts for the Foresight impact env.

Pieces:
  latest_checkpoint(path)      run dir / checkpoints dir / step dir -> step dir
  load_policy(path)            brax PPO checkpoint -> inference_fn(obs, key) -> (act, extras)
  make_env(skill, impl, ...)   ImpactJoystick with eval-friendly config
  batched_env(env, masses)     Playground DR-vmap wrapper with one projectile mass per env
  set_scenario(state, ...)     override speed / hit time / miss per env after reset
  rollout(...)                 lax.scan over the horizon with an optional BRACE policy and a
                               decision function; returns per-step records + final info

No autoreset: each env runs one episode for `steps` control steps. Outcome flags are
latched inside the env, so a terminated episode keeps its flags to the end.
"""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Callable, Optional

import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground._src import wrapper as pg_wrapper

from foresight.envs.g1_impact import ImpactJoystick, default_config
from foresight.envs.g1_impact.impact_joystick import brace_config

# Records taken every control step. All are (num_envs, ...) per step.
TRAJ_INFO_KEYS = (
    "proj_launched", "impact_contact", "fell", "terminated", "brace_active", "ep_step",
)


def latest_checkpoint(path: str | Path) -> Path:
  """Accepts a Playground run dir, its checkpoints/ dir, or a numeric step dir."""
  p = Path(path)
  if p.name.isdigit() and (p / "ppo_network_config.json").exists():
    return p
  if (p / "checkpoints").is_dir():
    p = p / "checkpoints"
  steps = [c for c in p.iterdir() if c.is_dir() and c.name.isdigit()]
  if not steps:
    raise FileNotFoundError(f"no numeric checkpoint dirs under {p}")
  return max(steps, key=lambda c: int(c.name))


def load_policy(path: str | Path, deterministic: bool = True):
  """Returns inference_fn(obs, rng) -> (action, extras) for one (unbatched) env.

  Re-implements brax 0.14.2's ppo checkpoint.load_policy: that one crashes on the
  `mean_kernel_init_fn: null` entry Playground's checkpoints contain (KeyError: None).
  """
  import json
  from brax.training import checkpoint as brax_checkpoint
  from brax.training import networks as brax_networks
  from ml_collections import config_dict

  step_dir = latest_checkpoint(path)
  d = json.loads((step_dir / "ppo_network_config.json").read_text())
  kw = d["network_factory_kwargs"]
  if kw.get("activation") is not None:
    kw["activation"] = brax_networks.ACTIVATION[kw["activation"]]
  for k in list(kw):
    if k.endswith("_kernel_init_fn"):
      kw[k] = brax_networks.KERNEL_INITIALIZER[kw[k]] if kw[k] is not None else None
  kw = {k: v for k, v in kw.items() if v is not None}
  d["network_factory_kwargs"] = kw
  cfg = config_dict.create(**d)
  network = brax_checkpoint.get_network(cfg, ppo_networks.make_ppo_networks)
  params = ppo_checkpoint.load(step_dir)
  make_inference_fn = ppo_networks.make_inference_fn(network)
  return make_inference_fn(params, deterministic=deterministic)


def make_env(skill: str = "nominal", impl: str = "warp", miss_fraction: float = 0.0,
             reset_perturb: float = 1.0, hold_base: bool = False,
             overrides: Optional[dict] = None) -> ImpactJoystick:
  cfg = brace_config() if skill == "brace" else default_config()
  cfg.impact.miss_fraction = miss_fraction
  cfg.impact.reset_perturb = reset_perturb
  cfg.impact.hold_base = hold_base
  cfg.impact.resample_on_autoreset = False   # no autoreset in eval
  ov = {"impl": impl}
  ov.update(overrides or {})
  return ImpactJoystick(config=cfg, config_overrides=ov)


def _mass_randomizer(proj_body_id: int, masses: np.ndarray):
  masses = jp.asarray(masses, dtype=jp.float32)

  def fn(model):
    n = masses.shape[0]
    m0 = model.body_mass[proj_body_id]
    bm = jp.tile(model.body_mass[None], (n, 1)).at[:, proj_body_id].set(masses)
    bi = jp.tile(model.body_inertia[None], (n, 1, 1))
    bi = bi.at[:, proj_body_id].set(
        model.body_inertia[proj_body_id][None] * (masses / m0)[:, None])
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({"body_mass": 0, "body_inertia": 0})
    model = model.tree_replace({"body_mass": bm, "body_inertia": bi})
    return model, in_axes

  return fn


def batched_env(env: ImpactJoystick, proj_masses: np.ndarray):
  """Vmapped env whose i-th world has projectile mass proj_masses[i]."""
  return pg_wrapper.BraxDomainRandomizationVmapWrapper(
      env, _mass_randomizer(env._proj_body, np.asarray(proj_masses)))


def set_scenario(env: ImpactJoystick, state, speed=None, hit_time=None, is_miss=None,
                 command=None):
  """Override per-env scenario fields after reset and recompute the derived ones."""
  info = dict(state.info)
  c = env._config.impact
  if speed is not None:
    info["proj_speed"] = jp.asarray(speed, jp.float32)
  if is_miss is not None:
    is_miss = jp.asarray(is_miss, bool)
    # Near-miss: fixed 0.6 m lateral offset (the side is irrelevant by symmetry).
    info["proj_lateral"] = jp.where(is_miss, 0.6, 0.0)
    info["proj_is_miss"] = is_miss
  if hit_time is None:
    hit_time = info["t_hit_nominal"]
  hit_time = jp.asarray(hit_time, jp.float32)
  flight = info["proj_dist"] / info["proj_speed"]
  t_launch = jp.maximum(hit_time - flight, c.min_launch_time)
  info["proj_launch_step"] = jp.round(t_launch / env.dt).astype(jp.int32)
  info["t_hit_nominal"] = t_launch + flight
  info["proj_momentum"] = info["proj_mass"] * info["proj_speed"]
  if command is not None:
    cmd = jp.broadcast_to(jp.asarray(command, jp.float32), info["command"].shape)
    info["command"] = cmd
    info["command_nominal"] = cmd
  return state.replace(info=info)


def no_brace(info, t):
  return jp.zeros_like(info["fell"])


def oracle_brace(lead_s: jax.Array, hold_s: float, dt: float):
  """Privileged rule: brace from t_hit - lead until t_hit + hold, only for true hits.
  lead_s is per-env; a negative lead means never brace."""

  def decide(info, t):
    tt = info["ep_step"].astype(jp.float32) * dt
    on = (lead_s >= 0) & (tt >= info["t_hit_nominal"] - lead_s) & \
         (tt <= info["t_hit_nominal"] + hold_s) & ~info["proj_is_miss"]
    return on

  return decide


def rollout(env: ImpactJoystick, wenv, nominal_policy, rng: jax.Array, steps: int,
            brace_policy=None, decide: Callable = no_brace,
            scenario_fn: Optional[Callable] = None, reset_keys: Optional[jax.Array] = None):
  """Run `steps` control steps for every env in `wenv` (built by batched_env).

  scenario_fn(state) -> state is applied right after reset (use set_scenario).
  reset_keys (n, 2): per-env reset keys; repeat a block to pair scenarios across conditions.
  Returns (final_info: dict of np arrays, traj: dict of np arrays (T, N, ...)).
  """
  n = wenv._mjx_model_v.body_mass.shape[0]
  pelvis = env._pelvis_body
  feet = env._feet_site_id
  torso = env._mj_model.body("torso_link").id

  def policy_act(obs, keys, brace):
    a_nom, _ = jax.vmap(nominal_policy)(obs, keys)
    if brace_policy is None:
      return a_nom
    a_br, _ = jax.vmap(brace_policy)(obs, keys)
    return jp.where(brace[:, None], a_br, a_nom)

  def body(carry, t):
    state, rng = carry
    rng, k = jax.random.split(rng)
    keys = jax.random.split(k, n)
    brace = decide(state.info, t)
    info = dict(state.info)
    info["brace_active"] = brace
    state = state.replace(info=info)
    act = policy_act(state.obs, keys, brace)
    state = wenv.step(state, act)
    d = state.data
    rec = {k: state.info[k] for k in TRAJ_INFO_KEYS}
    rec.update({
        "pelvis_pos": d.xpos[:, pelvis],
        "pelvis_fwd": d.xmat[:, pelvis].reshape(n, 3, 3)[:, :, 0],
        "com": d.subtree_com[:, pelvis],
        "torso_up": d.xmat[:, torso].reshape(n, 3, 3)[:, 2, 2],
        "feet_pos": d.site_xpos[:, feet],
        "proj_pos": state.info["obj_pos"],
        "proj_vel": state.info["obj_vel"],
        "done": state.done,
    })
    return (state, rng), rec

  @jax.jit
  def run(rng, reset_keys):
    rng, k = jax.random.split(rng)
    keys = jax.random.split(k, n) if reset_keys is None else reset_keys
    state = wenv.reset(keys)
    if scenario_fn is not None:
      state = scenario_fn(state)
    (state, _), traj = jax.lax.scan(body, (state, rng), jp.arange(steps))
    return state.info, traj

  final_info, traj = run(rng, reset_keys)
  to_np = lambda tree: jax.tree_util.tree_map(np.asarray, tree)
  return to_np(final_info), to_np(traj)


def episode_table(final_info: dict, traj: dict, dt: float) -> dict:
  """Per-episode scalars from a rollout: scenario, outcome, progress, pre-impact state."""
  n = final_info["fell"].shape[0]
  T = traj["pelvis_pos"].shape[0]
  # Forward progress: displacement projected on the pelvis heading at each step.
  dp = np.diff(traj["pelvis_pos"][..., :2], axis=0)                 # (T-1, N, 2)
  fwd = traj["pelvis_fwd"][:-1, :, :2]
  fwd = fwd / (np.linalg.norm(fwd, axis=-1, keepdims=True) + 1e-9)
  progress = np.sum(np.sum(dp * fwd, axis=-1), axis=0)              # (N,)
  # Progress until the first failure/contact window end is also useful; keep total.
  cs = final_info["contact_step"]
  pre = np.clip(cs - 1, 0, T - 1)
  idx = np.arange(n)
  com_pre = traj["com"][pre, idx]                                   # (N, 3)
  com_prev = traj["com"][np.clip(pre - 1, 0, T - 1), idx]
  com_vel = (com_pre - com_prev) / dt
  feet_pre = traj["feet_pos"][pre, idx]                             # (N, 2, 3)
  pvel = traj["proj_vel"][pre, idx]
  out = {
      "mass": final_info["proj_mass"], "speed": final_info["proj_speed"],
      "momentum": final_info["proj_momentum"], "is_miss": final_info["proj_is_miss"],
      "t_hit_nominal": final_info["t_hit_nominal"],
      "launched": final_info["proj_launched"], "contact": final_info["impact_contact"],
      "contact_step": cs, "fell": final_info["fell"],
      "fell_before_launch": final_info["fell_before_launch"],
      "term_reason": final_info["term_reason"], "term_step": final_info["term_step"],
      "progress_m": progress,
      "brace_steps_total": np.sum(traj["brace_active"], axis=0),
      "com_x_pre": com_pre[:, 0], "com_y_pre": com_pre[:, 1], "com_z_pre": com_pre[:, 2],
      "com_vx_pre": com_vel[:, 0], "com_vy_pre": com_vel[:, 1],
      "lfoot_x_pre": feet_pre[:, 0, 0], "lfoot_y_pre": feet_pre[:, 0, 1],
      "rfoot_x_pre": feet_pre[:, 1, 0], "rfoot_y_pre": feet_pre[:, 1, 1],
      "proj_vx_pre": pvel[:, 0], "proj_vy_pre": pvel[:, 1], "proj_vz_pre": pvel[:, 2],
  }
  return {k: np.asarray(v) for k, v in out.items()}


def write_csv(path: str | Path, table: dict, extra: Optional[dict] = None):
  import csv
  keys = list(table.keys())
  n = len(table[keys[0]])
  extra = extra or {}
  with open(path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(list(extra.keys()) + keys)
    for i in range(n):
      w.writerow([extra[k] for k in extra] + [_scalar(table[k][i]) for k in keys])


def _scalar(v):
  v = np.asarray(v)
  if v.dtype == bool:
    return int(v)
  if np.issubdtype(v.dtype, np.floating):
    return f"{float(v):.5g}"
  return int(v)
