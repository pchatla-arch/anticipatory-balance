# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""Domain randomization for the G1 impact env.

Same physics randomization as Playground's G1 randomizer, with slices bounded to the
robot's DOFs (the projectile free joint is appended), plus a per-environment projectile
mass drawn from config.impact.mass_range, which is how impulse class varies across envs.
"""
from __future__ import annotations

import jax
from mujoco import mjx

NV_ROBOT = 6 + 29
NQ_ROBOT = 7 + 29


def make_domain_randomizer(torso_body_id: int, proj_body_id: int,
                           mass_range: tuple[float, float]):
  """Return a Playground-style randomization_fn(model, rng) -> (model_v, in_axes)."""

  def domain_randomize(model: mjx.Model, rng: jax.Array):
    @jax.vmap
    def rand_dynamics(rng):
      rng, key = jax.random.split(rng)
      friction = jax.random.uniform(key, minval=0.4, maxval=1.0)
      pair_friction = model.pair_friction.at[0:2, 0:2].set(friction)  # foot-floor pairs are 0,1

      rng, key = jax.random.split(rng)
      fl = model.dof_frictionloss[6:NV_ROBOT] * jax.random.uniform(key, (29,), minval=0.5, maxval=2.0)
      dof_frictionloss = model.dof_frictionloss.at[6:NV_ROBOT].set(fl)

      rng, key = jax.random.split(rng)
      arm = model.dof_armature[6:NV_ROBOT] * jax.random.uniform(key, (29,), minval=1.0, maxval=1.05)
      dof_armature = model.dof_armature.at[6:NV_ROBOT].set(arm)

      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(key, (model.nbody,), minval=0.9, maxval=1.1)
      body_mass = model.body_mass * dmass

      rng, key = jax.random.split(rng)
      body_mass = body_mass.at[torso_body_id].set(
          body_mass[torso_body_id] + jax.random.uniform(key, minval=-1.0, maxval=1.0))

      # Projectile mass: U(mass_range) per environment (overrides the scene default).
      rng, key = jax.random.split(rng)
      pm = jax.random.uniform(key, minval=mass_range[0], maxval=mass_range[1])
      body_mass = body_mass.at[proj_body_id].set(pm)

      rng, key = jax.random.split(rng)
      qpos0 = model.qpos0.at[7:NQ_ROBOT].set(
          model.qpos0[7:NQ_ROBOT] + jax.random.uniform(key, (29,), minval=-0.05, maxval=0.05))

      return pair_friction, dof_frictionloss, dof_armature, body_mass, qpos0

    pf, fl, arm, bm, q0 = rand_dynamics(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({
        "pair_friction": 0, "dof_frictionloss": 0, "dof_armature": 0,
        "body_mass": 0, "qpos0": 0,
    })
    model = model.tree_replace({
        "pair_friction": pf, "dof_frictionloss": fl, "dof_armature": arm,
        "body_mass": bm, "qpos0": q0,
    })
    return model, in_axes

  return domain_randomize
