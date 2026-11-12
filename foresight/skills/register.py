# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Register the Foresight impact env with Playground's locomotion registry so the stock
train-jax-ppo pipeline (brax PPO, checkpoints, eval videos) can train on it unchanged."""
from __future__ import annotations

from mujoco_playground._src import locomotion
from foresight.envs.g1_impact import ImpactJoystick, default_config
from foresight.envs.g1_impact import randomize as fs_randomize

ENV_NAME = "G1Impact"


def register() -> str:
  if ENV_NAME not in locomotion._envs:
    locomotion.register_environment(ENV_NAME, ImpactJoystick, default_config)
    # Body ids are stable (projectile is appended after the robot); read them once.
    env = ImpactJoystick(config=default_config())
    cfg = default_config()
    locomotion._randomizer[ENV_NAME] = fs_randomize.make_domain_randomizer(
        torso_body_id=env._mj_model.body("torso_link").id,
        proj_body_id=env._proj_body,
        mass_range=tuple(cfg.impact.mass_range),
    )
  return ENV_NAME
