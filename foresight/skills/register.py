# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Register the Foresight impact envs with Playground's locomotion registry so the stock
train-jax-ppo pipeline (brax PPO, checkpoints) can train on them unchanged.

Two skills share one env class and differ only in config:
  G1Impact       nominal walking (joystick command), projectile as disturbance
  G1ImpactBrace  BRACE: zero command, low CoM, wide stance, both feet down
"""
from __future__ import annotations

from mujoco_playground._src import locomotion
from foresight.envs.g1_impact import ImpactJoystick, default_config
from foresight.envs.g1_impact.impact_joystick import brace_config
from foresight.envs.g1_impact import randomize as fs_randomize

ENV_NAME = "G1Impact"
BRACE_ENV_NAME = "G1ImpactBrace"
SKILL_ENVS = {"nominal": ENV_NAME, "brace": BRACE_ENV_NAME}
ENV_NAMES = tuple(SKILL_ENVS.values())


def register() -> str:
  if ENV_NAME not in locomotion._envs:
    locomotion.register_environment(ENV_NAME, ImpactJoystick, default_config)
    locomotion.register_environment(BRACE_ENV_NAME, ImpactJoystick, brace_config)
    # Body ids are stable (projectile is appended after the robot); read them once.
    env = ImpactJoystick(config=default_config())
    cfg = default_config()
    randomizer = fs_randomize.make_domain_randomizer(
        torso_body_id=env._mj_model.body("torso_link").id,
        proj_body_id=env._proj_body,
        mass_range=tuple(cfg.impact.mass_range),
    )
    locomotion._randomizer[ENV_NAME] = randomizer
    locomotion._randomizer[BRACE_ENV_NAME] = randomizer
  return ENV_NAME
