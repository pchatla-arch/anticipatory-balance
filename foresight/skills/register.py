# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Register the Foresight impact env with Playground's locomotion registry so the stock
train-jax-ppo pipeline (brax PPO, checkpoints, eval videos) can train on it unchanged."""
from __future__ import annotations

from mujoco_playground._src import locomotion
from mujoco_playground._src.locomotion.g1 import randomize as g1_randomize

from foresight.envs.g1_impact import ImpactJoystick, default_config

ENV_NAME = "G1Impact"


def register() -> str:
  if ENV_NAME not in locomotion._envs:
    locomotion.register_environment(ENV_NAME, ImpactJoystick, default_config)
    locomotion._randomizer[ENV_NAME] = g1_randomize.domain_randomize
  return ENV_NAME
