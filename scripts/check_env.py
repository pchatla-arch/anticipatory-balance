# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Day-1 smoke test: GPU, Playground G1, and the Foresight impact env.

Run from the repo root inside the rss2025 venv:
  uv --no-config run python scripts/check_env.py
Expected: backend gpu; 16 envs reset+step; projectile launches; at least one torso contact.
"""
import time

import jax
import jax.numpy as jp
from absl import app, flags

from foresight.envs.g1_impact import ImpactJoystick, default_config

_N = flags.DEFINE_integer("num_envs", 16, "parallel environments")
_STEPS = flags.DEFINE_integer("steps", 200, "control steps (50 Hz)")


def main(_):
  print("JAX backend:", jax.default_backend(), jax.devices())
  cfg = default_config()
  cfg.impact.miss_fraction = 0.0     # force hits for the smoke test
  cfg.impact.hit_time_range = [0.8, 1.0]
  cfg.impact.reset_perturb = 0.0   # zero action = default pose; keep the passive robot up
  cfg.noise_config.level = 0.0
  cfg.impact.hold_base = True      # pin the base: verifies projectile hits without a policy
  env = ImpactJoystick(config=cfg)
  print("obs size:", env.observation_size, "action size:", env.action_size)

  reset = jax.jit(jax.vmap(env.reset))
  step = jax.jit(jax.vmap(env.step))
  rng = jax.random.split(jax.random.PRNGKey(0), _N.value)
  t0 = time.time()
  state = reset(rng)
  print(f"reset jit+run: {time.time()-t0:.1f}s")

  zero = jp.zeros((_N.value, env.action_size))
  t0 = time.time()
  launched = contact = fell = 0
  for i in range(_STEPS.value):
    state = step(state, zero)
    launched = int(state.info["proj_launched"].sum())
    contact = int(state.info["impact_contact"].sum())
    fell = int(state.info["fell"].sum())
  dt = time.time() - t0
  print(f"{_STEPS.value} steps x {_N.value} envs: {dt:.1f}s "
        f"({_STEPS.value*_N.value/dt:.0f} env-steps/s incl. jit)")
  fbl = int(state.info["fell_before_launch"].sum())
  print(f"launched {launched}/{_N.value}, contact {contact}/{_N.value}, "
        f"fell {fell}/{_N.value} (of which before launch: {fbl})")
  print("incoming momentum m*v (N*s):", jp.round(state.info["proj_momentum"], 1))
  if contact == 0:
    print("WARNING: no torso contact; check capsule injection and launch aim.")


if __name__ == "__main__":
  app.run(main)
