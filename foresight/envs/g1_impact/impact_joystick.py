# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""G1 joystick locomotion with an approaching projectile (the 'Foresight' impact scenario).

Subclasses Playground's rss2025 G1 Joystick env. Adds:
  * a free, gravity-compensated projectile parked far away, launched toward the pelvis so
    that it arrives at a sampled *hit time* (so every episode keeps a recovery window);
  * a random lateral miss offset (a configurable fraction of episodes are near-misses);
  * per-episode projectile speed and (per-lane, via domain randomization) mass;
  * privileged object state in info (for the perception noise model outside the env);
  * contact detection via the projectile <-> body contact-found sensors;
  * a terminal reason that agrees with the parent env's termination;
  * a second skill, BRACE (config.skill = "brace"): zero velocity command, lower CoM,
    wider stance, both feet down;
  * an external mode switch info["brace_active"]: when True the velocity command seen by
    the policy and the reward is zero (the decision layer flips it; the env never does).

Scenario parameters vs measured outcomes (keep these apart in the analysis):
  scenario : proj_mass, proj_speed, proj_momentum (= m*v, the incoming momentum, NOT the
             transferred impulse), proj_is_miss (sampled path category), t_hit_nominal
             (straight-line arrival time), launch step/pos/vel
  outcome  : proj_launched, impact_contact (sensor detected), contact_step, fell,
             fell_before_launch, terminated, term_reason, term_step

Episode-reset semantics: Playground's autoreset wrapper restores physics but keeps info,
so latched flags would survive into the next episode. We therefore resample the scenario
and clear the latches whenever the wrapper's step counter reads zero (start of episode).

Playground's own velocity-kick pushes are disabled (push_config.enable=False), so the
projectile is the only disturbance.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import joystick as g1_joystick

from foresight.envs.g1_impact.parent_shim import JoystickRobotSliced

from foresight.envs.g1_impact import scene as impact_scene

TERM_NONE = 0          # episode still running / reached horizon upright
TERM_PARENT = 1        # Playground termination: torso inverted, self-contact, or NaN
TERM_TILT = 2          # sustained torso tilt (up-vector < 0.5 for >= 10 control steps)


def default_config() -> config_dict.ConfigDict:
  cfg = g1_joystick.default_config()
  cfg.episode_length = 300            # 6 s at 50 Hz
  cfg.push_config.enable = False      # projectile is the only disturbance
  # Constraint buffers: Playground sized njmax for its own pairs; add room for the
  # projectile's 5 condim-3 pairs (4 rows each) plus margin.
  cfg.njmax = 29 * 2 + 8 * 4 + 5 * 4 + 32
  cfg.naconmax = 8 * 8192 * 2
  cfg.impact = config_dict.create(
      mass_range=[1.0, 8.0],          # kg (per lane, via domain randomization)
      speed_range=[1.0, 4.0],         # m/s (per episode)
      range_dist=[2.0, 4.0],          # launch distance from pelvis, m
      height_range=[0.6, 1.1],        # launch height, m
      hit_time_range=[1.5, 3.5],      # nominal arrival time after episode start, s
      min_launch_time=0.3,            # never launch earlier than this, s
      miss_fraction=0.3,              # P(near-miss)
      miss_offset_range=[0.4, 0.8],   # lateral miss distance, m
      proj_radius=0.15,
      reset_perturb=1.0,              # 0.0 = deterministic initial pose (eval, smoke test)
      hold_base=False,                # True: pin the floating base (contact smoke test only)
      resample_on_autoreset=True,     # resample the scenario at every episode start
  )
  cfg.skill = "nominal"               # "nominal" (walk) or "brace"
  cfg.brace = config_dict.create(
      base_height_target=0.60,        # m; nominal keyframe pelvis height is 0.755
      feet_sep_target=0.40,           # m between foot sites (nominal stance ~0.2)
  )
  # Extra reward terms (zero weight for the nominal skill; metrics need the keys).
  cfg.reward_config.scales.feet_separation = 0.0
  cfg.reward_config.scales.feet_contact = 0.0
  return cfg


def brace_config() -> config_dict.ConfigDict:
  """Reward for the BRACE skill: stand still, low, wide, both feet down, survive hits."""
  cfg = default_config()
  cfg.skill = "brace"
  s = cfg.reward_config.scales
  s.tracking_lin_vel = 1.0      # command is zero -> rewards standing still
  s.tracking_ang_vel = 0.5
  s.feet_air_time = 0.0
  s.feet_phase = 0.0
  s.stand_still = 0.0           # would pull toward the default (tall, narrow) pose
  s.pose = 0.0
  s.joint_deviation_knee = 0.0
  s.joint_deviation_hip = 0.0
  s.base_height = -10.0
  s.feet_separation = -5.0
  s.feet_contact = 0.5
  s.orientation = -2.0
  s.action_rate = -0.1
  s.torques = -1e-4
  cfg.reward_config.base_height_target = cfg.brace.base_height_target
  return cfg


class ImpactJoystick(JoystickRobotSliced):
  """Joystick env + launched projectile (+ optional BRACE skill reward)."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      scene_dir: Optional[Path] = None,
  ):
    if task != "flat_terrain":
      raise NotImplementedError("impact scenario is flat-terrain only for now")
    scene_dir = Path(scene_dir or tempfile.mkdtemp(prefix="foresight_scene_"))
    scene_path, _ = impact_scene.build_scene(
        scene_dir, proj_radius=config.impact.proj_radius
    )
    # Bypass Joystick.__init__ (which hardcodes Playground's scene) and call G1Env directly.
    g1_joystick.g1_base.G1Env.__init__(
        self, xml_path=str(scene_path), config=config, config_overrides=config_overrides
    )
    self._post_init()
    m = self._mj_model
    self._proj_body = m.body("projectile").id
    self._proj_qpos_adr = m.joint("projectile_free").qposadr[0]
    self._proj_qvel_adr = m.joint("projectile_free").dofadr[0]
    self._pelvis_body = m.body("pelvis").id
    self._impact_found_adrs = [
        m.sensor_adr[m.sensor(n).id]
        for n in ("proj_torso_found", "proj_pelvis_found",
                  "proj_lthigh_found", "proj_rthigh_found")
    ]
    self._is_brace = self._config.skill == "brace"

  # ---- helpers -----------------------------------------------------------------
  def _proj_pos(self, data: mjx.Data) -> jax.Array:
    return data.qpos[self._proj_qpos_adr : self._proj_qpos_adr + 3]

  def _proj_vel(self, data: mjx.Data) -> jax.Array:
    return data.qvel[self._proj_qvel_adr : self._proj_qvel_adr + 3]

  def _pelvis_pos(self, data: mjx.Data) -> jax.Array:
    return data.xpos[self._pelvis_body]

  def sample_command(self, rng: jax.Array) -> jax.Array:
    if self._is_brace:
      return jp.zeros(3)
    return super().sample_command(rng)

  def _sample_scenario(self, rng: jax.Array) -> dict[str, jax.Array]:
    """One projectile scenario. Mass comes from the model (per-lane randomization)."""
    c = self._config.impact
    k_speed, k_dist, k_h, k_az, k_t, k_miss, k_off, k_side = jax.random.split(rng, 8)
    mass = self.mjx_model.body_mass[self._proj_body]
    speed = jax.random.uniform(k_speed, minval=c.speed_range[0], maxval=c.speed_range[1])
    dist = jax.random.uniform(k_dist, minval=c.range_dist[0], maxval=c.range_dist[1])
    height = jax.random.uniform(k_h, minval=c.height_range[0], maxval=c.height_range[1])
    azimuth = jax.random.uniform(k_az, maxval=2 * jp.pi)
    t_hit = jax.random.uniform(k_t, minval=c.hit_time_range[0], maxval=c.hit_time_range[1])
    flight = dist / speed
    t_launch = jp.maximum(t_hit - flight, c.min_launch_time)
    is_miss = jax.random.uniform(k_miss) < c.miss_fraction
    off = jax.random.uniform(k_off, minval=c.miss_offset_range[0], maxval=c.miss_offset_range[1])
    side = jp.where(jax.random.uniform(k_side) < 0.5, -1.0, 1.0)
    lateral = jp.where(is_miss, side * off, 0.0)
    dir_xy = jp.array([jp.cos(azimuth), jp.sin(azimuth)])
    perp = jp.array([-dir_xy[1], dir_xy[0]])
    return {
        # scenario
        "proj_mass": mass,
        "proj_speed": speed,
        "proj_momentum": mass * speed,
        "proj_dir_xy": dir_xy,
        "proj_perp_xy": perp,
        "proj_dist": dist,
        "proj_height": height,
        "proj_lateral": lateral,
        "proj_is_miss": is_miss,
        "proj_launch_step": jp.round(t_launch / self.dt).astype(jp.int32),
        "t_hit_nominal": t_launch + flight,
        "proj_launch_pos": jp.zeros(3),
        "proj_launch_vel": jp.zeros(3),
        # outcome latches
        "proj_launched": jp.array(False),
        "impact_contact": jp.array(False),
        "contact_step": jp.int32(-1),
        "fell": jp.array(False),
        "fell_before_launch": jp.array(False),
        "terminated": jp.array(False),
        "term_reason": jp.int32(TERM_NONE),
        "term_step": jp.int32(-1),
        "down_steps": jp.int32(0),
        "ep_step": jp.int32(0),
    }

  # ---- reset / step --------------------------------------------------------------
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_scn = jax.random.split(rng)
    state = super().reset(rng)
    state.info.update(self._sample_scenario(k_scn))
    state.info.update({
        "command_nominal": state.info["command"],
        "brace_active": jp.array(False),
        "brace_steps": jp.int32(0),
        # Privileged object state (present at reset so the info pytree is identical
        # across reset/step, as lax.scan requires).
        "obj_pos": self._proj_pos(state.data),
        "obj_vel": self._proj_vel(state.data),
        "obj_rel_pos": self._proj_pos(state.data) - self._pelvis_pos(state.data),
    })
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = dict(state.info)
    data = state.data

    # --- episode start under Playground's autoreset: resample scenario, clear latches.
    if self._config.impact.resample_on_autoreset and "steps" in info:
      new_ep = info["steps"] == 0
      info["rng"], k_scn = jax.random.split(info["rng"])
      fresh = self._sample_scenario(k_scn)
      for k, v in fresh.items():
        info[k] = jp.where(new_ep, v, info[k])
      info["brace_active"] = jp.where(new_ep, False, info["brace_active"])
      info["brace_steps"] = jp.where(new_ep, 0, info["brace_steps"])

    # --- mode switch: BRACE zeroes the velocity command seen by policy and reward.
    info["command"] = jp.where(info["brace_active"], jp.zeros(3), info["command_nominal"])
    info["brace_steps"] = jp.where(info["brace_active"], info["brace_steps"] + 1, 0)

    launch_now = (info["ep_step"] >= info["proj_launch_step"]) & (~info["proj_launched"])

    # Resolve launch pos/vel from the pelvis position *now* (so the aim is current).
    pelvis = self._pelvis_pos(data)
    launch_xy = pelvis[:2] + info["proj_dist"] * info["proj_dir_xy"] + info["proj_lateral"] * info["proj_perp_xy"]
    target_xy = pelvis[:2] + info["proj_lateral"] * info["proj_perp_xy"]
    d_xy = target_xy - launch_xy
    vel_xy = d_xy / (jp.linalg.norm(d_xy) + 1e-6) * info["proj_speed"]
    target_z = pelvis[2] + 0.25
    flight_t = info["proj_dist"] / info["proj_speed"]
    launch_pos = jp.array([launch_xy[0], launch_xy[1], info["proj_height"]])
    launch_vel = jp.array([vel_xy[0], vel_xy[1], (target_z - info["proj_height"]) / flight_t])
    info["proj_launch_pos"] = jp.where(launch_now, launch_pos, info["proj_launch_pos"])
    info["proj_launch_vel"] = jp.where(launch_now, launch_vel, info["proj_launch_vel"])

    # Place and release the projectile at launch; before that it stays parked.
    qpos = data.qpos
    qvel = data.qvel
    a, b = self._proj_qpos_adr, self._proj_qvel_adr
    qpos = jp.where(launch_now, qpos.at[a : a + 3].set(launch_pos), qpos)
    qvel = jp.where(launch_now, qvel.at[b : b + 3].set(launch_vel), qvel)
    parked = ~(info["proj_launched"] | launch_now)
    qpos = jp.where(parked, qpos.at[a : a + 3].set(self._init_q[a : a + 3]), qpos)
    qvel = jp.where(parked, qvel.at[b : b + 6].set(0.0), qvel)
    if self._config.impact.hold_base:
      # Pin the robot's floating base: reset its pose to the keyframe and zero its velocity.
      qpos = qpos.at[0:7].set(self._init_q[0:7])
      qvel = qvel.at[0:6].set(0.0)
    state = state.replace(data=data.replace(qpos=qpos, qvel=qvel), info=info)
    state = super().step(state, action)

    touch = jp.any(jp.array([state.data.sensordata[s] > 0 for s in self._impact_found_adrs]))
    info = state.info
    # Parent may resample the command (step > 500); keep the nominal copy in sync.
    info["command_nominal"] = jp.where(info["brace_active"], info["command_nominal"], info["command"])
    info["proj_launched"] = info["proj_launched"] | launch_now
    first_contact = touch & ~info["impact_contact"]
    info["contact_step"] = jp.where(first_contact, info["ep_step"], info["contact_step"])
    info["impact_contact"] = info["impact_contact"] | touch

    # Termination bookkeeping. Parent `done` = inverted torso | self-contact | NaN.
    parent_done = state.done > 0
    down = self.get_gravity(state.data, "torso")[-1] < 0.5
    info["down_steps"] = jp.where(down, info["down_steps"] + 1, 0)
    tilt_now = info["down_steps"] >= 10
    fell_now = parent_done | tilt_now
    new_term = fell_now & ~info["terminated"]
    info["term_reason"] = jp.where(
        new_term, jp.where(parent_done, TERM_PARENT, TERM_TILT), info["term_reason"])
    info["term_step"] = jp.where(new_term, info["ep_step"], info["term_step"])
    info["terminated"] = info["terminated"] | fell_now
    info["fell_before_launch"] = info["fell_before_launch"] | (fell_now & ~info["proj_launched"])
    info["fell"] = info["fell"] | fell_now
    info["ep_step"] = info["ep_step"] + 1

    # Privileged object state for the perception module.
    info["obj_pos"] = self._proj_pos(state.data)
    info["obj_vel"] = self._proj_vel(state.data)
    info["obj_rel_pos"] = info["obj_pos"] - self._pelvis_pos(state.data)
    return state.replace(info=info)

  # ---- BRACE reward terms ----------------------------------------------------------
  def _get_reward(self, data, action, info, metrics, done, first_contact, contact):
    r = super()._get_reward(data, action, info, metrics, done, first_contact, contact)
    feet = data.site_xpos[self._feet_site_id]                   # (2, 3)
    sep = jp.linalg.norm(feet[0, :2] - feet[1, :2])
    r["feet_separation"] = jp.square(sep - self._config.brace.feet_sep_target)
    r["feet_contact"] = jp.mean(contact.astype(jp.float32))
    return r
