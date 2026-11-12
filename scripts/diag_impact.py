"""Per-env impact diagnostic: closest projectile–torso approach, contact step, fall step."""
import jax, jax.numpy as jp, numpy as np
from absl import app
from foresight.envs.g1_impact import ImpactJoystick, default_config


def main(_):
  cfg = default_config()
  cfg.impact.miss_fraction = 0.0
  cfg.impact.hit_time_range = [0.8, 1.0]
  cfg.impact.reset_perturb = 0.0
  cfg.noise_config.level = 0.0
  env = ImpactJoystick(config=cfg)
  n = 8
  reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
  st = reset(jax.random.split(jax.random.PRNGKey(0), n))
  zero = jp.zeros((n, env.action_size))
  torso_id = env._mj_model.body("torso_link").id
  mind = np.full(n, 1e9); c_step = np.full(n, -1); f_step = np.full(n, -1); l_step = np.full(n, -1)
  up0 = np.array(env.get_gravity(st.data, "torso")[:, -1]) if False else None
  for t in range(200):
    st = step(st, zero)
    pp = np.array(st.info["obj_pos"]); tp = np.array(st.data.xpos[:, torso_id])
    d = np.linalg.norm(pp - tp, axis=1); mind = np.minimum(mind, d)
    c = np.array(st.info["impact_contact"]); f = np.array(st.info["fell"]); l = np.array(st.info["proj_launched"])
    for i in range(n):
      if c[i] and c_step[i] < 0: c_step[i] = t
      if f[i] and f_step[i] < 0: f_step[i] = t
      if l[i] and l_step[i] < 0: l_step[i] = t
    if t in (0, 20, 40, 80):
      up = np.array(jax.vmap(lambda d: env.get_gravity(d, "torso"))(st.data))[:, -1]
      print(f"t={t:3d} torso_z={tp[:,2].round(2)} up={up.round(2)}")
  print("launch step :", l_step)
  print("min |proj-torso| (m):", mind.round(2), " (torso capsule r=0.09, proj r=0.15 -> contact if < ~0.24)")
  print("contact step:", c_step)
  print("fall step   :", f_step)
  print("speed       :", np.array(st.info["proj_speed"]).round(1), " dist/speed ->", (np.array(st.info["t_hit_nominal"]) * 50).round(0), "steps")


if __name__ == "__main__":
  app.run(main)
