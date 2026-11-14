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
from foresight.envs.g1_impact.impact_joystick import brace_config, aware_config
from foresight.envs.g1_impact import randomize as fs_randomize

ENV_NAME = "G1Impact"
BRACE_ENV_NAME = "G1ImpactBrace"
AWARE_ENV_NAME = "G1ImpactAware"
SKILL_ENVS = {"nominal": ENV_NAME, "brace": BRACE_ENV_NAME, "aware": AWARE_ENV_NAME}
ENV_NAMES = tuple(SKILL_ENVS.values())


def register() -> str:
  if ENV_NAME not in locomotion._envs:
    locomotion.register_environment(ENV_NAME, ImpactJoystick, default_config)
    locomotion.register_environment(BRACE_ENV_NAME, ImpactJoystick, brace_config)
    locomotion.register_environment(AWARE_ENV_NAME, ImpactJoystick, aware_config)
    # Body ids are stable (projectile is appended after the robot); read them once.
    env = ImpactJoystick(config=default_config())
    torso = env._mj_model.body("torso_link").id
    # One randomizer per skill, with that skill's projectile mass range. (The committed
    # nominal_s1/brace_s1/brace2_s1 checkpoints predate this and were trained with the
    # nominal range, 1-8 kg, for every skill.)
    for name, cfg_fn in ((ENV_NAME, default_config), (BRACE_ENV_NAME, brace_config), (AWARE_ENV_NAME, aware_config)):
      locomotion._randomizer[name] = fs_randomize.make_domain_randomizer(
          torso_body_id=torso, proj_body_id=env._proj_body, mass_range=tuple(cfg_fn().impact.mass_range))
  return ENV_NAME
