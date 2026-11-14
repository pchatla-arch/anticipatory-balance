# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Write a randomly initialised brax-PPO checkpoint in Playground's layout, so the
evaluation scripts can be exercised without a trained policy (CPU smoke tests).

  python tests/make_dummy_checkpoint.py --out /tmp/dummy_ckpt
"""
import functools

import jax
from absl import app, flags
from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint as ppo_checkpoint
from brax.training.agents.ppo import networks as ppo_networks

from foresight.envs.g1_impact import ImpactJoystick, default_config
from foresight.envs.g1_impact.impact_joystick import aware_config

_OUT = flags.DEFINE_string("out", None, "checkpoint root (a 000000000000/ dir is created)", required=True)
_SKILL = flags.DEFINE_string("skill", "nominal", "nominal | aware (sets the observation size)")


def main(_):
  env = ImpactJoystick(config=aware_config() if _SKILL.value == "aware" else default_config(),
                       config_overrides={"impl": "jax"})
  obs_size = env.observation_size
  factory = functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=(512, 256, 128), value_hidden_layer_sizes=(512, 256, 128),
      policy_obs_key="state", value_obs_key="privileged_state")
  cfg = ppo_checkpoint.network_config(obs_size, env.action_size, True, factory)
  net = factory(obs_size, env.action_size, preprocess_observations_fn=running_statistics.normalize)
  k1, k2 = jax.random.split(jax.random.PRNGKey(0))
  obs_spec = {k: specs.Array((v if isinstance(v, int) else v[0],), jax.numpy.float32)
              for k, v in obs_size.items()}
  params = (running_statistics.init_state(obs_spec),
            net.policy_network.init(k1), net.value_network.init(k2))
  ppo_checkpoint.save(_OUT.value, 0, params, cfg)
  print("wrote", _OUT.value)


if __name__ == "__main__":
  app.run(main)
