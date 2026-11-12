# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Train a Foresight skill with Playground's stock brax-PPO script.

Registers the G1Impact env, patches the PPO config lookup so G1Impact reuses the tuned
G1 joystick hyperparameters, then hands off to learning/train_jax_ppo.py's main().

Examples (from the repo root, rss2025 venv active):
  # 20M-step runtime benchmark (nominal walking skill; projectile launches but no brace yet)
  python scripts/train_skill.py --env_name G1Impact --impl warp --domain_randomization \
      --num_timesteps 20000000 --num_envs 4096 --seed 1 --use_tb --logdir logs/bench_s1

  # full nominal skill
  python scripts/train_skill.py --env_name G1Impact --impl warp --domain_randomization \
      --num_timesteps 200000000 --num_envs 4096 --seed 1 --use_tb --logdir logs/nominal_s1
"""
import os
import sys

import jax
from absl import app

# brax 0.14.2 still calls jax.device_put_replicated / device_put_sharded, which JAX 0.12
# removed. Provide the drop-in replacements from JAX's own migration guide.
if not hasattr(jax, "device_put_replicated"):
  from jax.sharding import NamedSharding, PartitionSpec as P

  # The old functions returned arrays with a leading per-device axis (pmap layout):
  # replicated -> stack len(devices) copies; sharded -> stack the given shards.
  def _device_put_sharded(shards, devices):
    mesh = jax.make_mesh((len(devices),), ("i",), devices=devices)
    stacked = jax.tree_util.tree_map(lambda *xs: jax.numpy.stack(xs), *shards)
    return jax.device_put(stacked, NamedSharding(mesh, P("i")))

  def _device_put_replicated(x, devices):
    return _device_put_sharded([x] * len(devices), devices)

  jax.device_put_replicated = _device_put_replicated
  jax.device_put_sharded = _device_put_sharded

import mujoco_playground

from foresight.skills import register as fs_register

# Locate Playground's rss2025/learning dir from the installed package.
_PG = os.path.dirname(os.path.dirname(os.path.abspath(mujoco_playground.__file__)))
sys.path.insert(0, os.path.join(_PG, "learning"))

fs_register.register()

from mujoco_playground.config import locomotion_params  # noqa: E402

_orig_cfg = locomotion_params.brax_ppo_config


def _cfg(env_name, impl=None):
  if env_name == fs_register.ENV_NAME:
    cfg = _orig_cfg("G1JoystickFlatTerrain", impl)
    cfg.episode_length = mujoco_playground.locomotion.get_default_config(env_name).episode_length
    return cfg
  return _orig_cfg(env_name, impl)


locomotion_params.brax_ppo_config = _cfg

import train_jax_ppo  # noqa: E402  (Playground's script; flags are defined at import)

if __name__ == "__main__":
  app.run(train_jax_ppo.main)
