# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""G1 joystick locomotion with an approaching projectile (the 'Foresight' impact scenario).

Subclasses Playground's rss2025 G1 Joystick env. Adds:
  * a free projectile body parked far away, launched at a random time toward the pelvis
    with a random lateral miss offset (a configurable fraction of episodes are near-misses);
  * per-episode projectile mass and speed (impulse class), written to state.info;
  * privileged object state in info (for the Stage-A perception noise model outside the env);
  * a torso touch sensor reading and the true time-to-impact in info;
  * termination that distinguishes fall from timeout.
Playground's own velocity-kick pushes are disabled by default here (push_config.enable=False),
so the projectile is the only disturbance.
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


def default_config() -> config_dict.ConfigDict:
  cfg = g1_joystick.default_config()
  cfg.episode_length = 300            # 6 s at 50 Hz; impacts need no more
  cfg.push_config.enable = False      # projectile is the only disturbance
  cfg.impact = config_dict.create(
      mass_range=[1.0, 8.0],          # kg
      speed_range=[1.0, 4.0],         # m/s
      range_dist=[2.0, 4.0],          # launch distance from pelvis, m
      height_range=[0.6, 1.1],        # launch height, m
      launch_time_range=[0.5, 2.0],   # s after reset
      miss_fraction=0.3,              # P(near-miss)
      miss_offset_range=[0.4, 0.8],   # lateral miss distance, m
      proj_radius=0.15,
  )
  return cfg


class ImpactJoystick(JoystickRobotSliced):
  """Joystick env + launched projectile."""

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

  # ---- helpers -----------------------------------------------------------------
  def _proj_pos(self, data: mjx.Data) -> jax.Array:
    return data.qpos[self._proj_qpos_adr : self._proj_qpos_adr + 3]

  def _proj_vel(self, data: mjx.Data) -> jax.Array:
    return data.qvel[self._proj_qvel_adr : self._proj_qvel_adr + 3]

  def _pelvis_pos(self, data: mjx.Data) -> jax.Array:
    return data.xpos[self._pelvis_body]

  # ---- reset / step --------------------------------------------------------------
  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, k_mass, k_speed, k_dist, k_h, k_az, k_t, k_miss, k_off, k_side = (
        jax.random.split(rng, 10)
    )
    state = super().reset(rng)
    c = self._config.impact
    mass = jax.random.uniform(k_mass, minval=c.mass_range[0], maxval=c.mass_range[1])
    speed = jax.random.uniform(k_speed, minval=c.speed_range[0], maxval=c.speed_range[1])
    dist = jax.random.uniform(k_dist, minval=c.range_dist[0], maxval=c.range_dist[1])
    height = jax.random.uniform(k_h, minval=c.height_range[0], maxval=c.height_range[1])
    azimuth = jax.random.uniform(k_az, maxval=2 * jp.pi)
    t_launch = jax.random.uniform(
        k_t, minval=c.launch_time_range[0], maxval=c.launch_time_range[1]
    )
    is_miss = jax.random.uniform(k_miss) < c.miss_fraction
    off = jax.random.uniform(k_off, minval=c.miss_offset_range[0], maxval=c.miss_offset_range[1])
    side = jp.where(jax.random.uniform(k_side) < 0.5, -1.0, 1.0)
    lateral = jp.where(is_miss, side * off, 0.0)

    # Launch point on a circle around the pelvis; aim at pelvis (+ lateral offset).
    pelvis = self._pelvis_pos(state.data)
    dir_xy = jp.array([jp.cos(azimuth), jp.sin(azimuth)])
    perp = jp.array([-dir_xy[1], dir_xy[0]])
    launch_xy = pelvis[:2] + dist * dir_xy + lateral * perp
    target_xy = pelvis[:2] + lateral * perp
    vel_xy = (target_xy - launch_xy) / jp.linalg.norm(target_xy - launch_xy) * speed
    launch_pos = jp.array([launch_xy[0], launch_xy[1], height])
    launch_vel = jp.array([vel_xy[0], vel_xy[1], 0.0])

    state.info.update({
        "proj_mass": mass,
        "proj_speed": speed,
        "proj_impulse": mass * speed,          # N*s if fully absorbed
        "proj_launch_pos": launch_pos,
        "proj_launch_vel": launch_vel,
        "proj_launch_step": jp.round(t_launch / self.dt).astype(jp.int32),
        "proj_launched": jp.array(False),
        "proj_is_miss": is_miss,
        "t_hit_true": t_launch + dist / speed,  # s from reset, straight-line estimate
        "impact_contact": jp.array(False),
        "fell": jp.array(False),
    })
    return state

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    info = state.info
    data = state.data
    launch_now = (info["step"] >= info["proj_launch_step"]) & (~info["proj_launched"])

    # Place and release the projectile at launch; before that it stays parked.
    qpos = data.qpos
    qvel = data.qvel
    a, b = self._proj_qpos_adr, self._proj_qvel_adr
    qpos = jp.where(launch_now, qpos.at[a : a + 3].set(info["proj_launch_pos"]), qpos)
    qvel = jp.where(launch_now, qvel.at[b : b + 3].set(info["proj_launch_vel"]), qvel)
    # Freeze the projectile while parked (zero velocity), so it does not fall.
    parked = ~(info["proj_launched"] | launch_now)
    qvel = jp.where(parked, qvel.at[b : b + 6].set(0.0), qvel)
    state = state.replace(data=data.replace(qpos=qpos, qvel=qvel))

    # Per-episode projectile mass: scale body_mass via the model is not per-env in MJX;
    # we emulate impulse class through speed only in v0 and keep mass in [1,8] via
    # domain randomization in a later commit (see TODO in README).
    state = super().step(state, action)

    touch = jp.any(jp.array([state.data.sensordata[a] > 0 for a in self._impact_found_adrs]))
    info = state.info
    info["proj_launched"] = info["proj_launched"] | launch_now
    info["impact_contact"] = info["impact_contact"] | touch
    info["fell"] = info["fell"] | (self.get_gravity(state.data, "torso")[-1] < 0.0)
    # Privileged object state for the perception module.
    info["obj_pos"] = self._proj_pos(state.data)
    info["obj_vel"] = self._proj_vel(state.data)
    info["obj_rel_pos"] = info["obj_pos"] - self._pelvis_pos(state.data)
    return state.replace(info=info)
