# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""CPU check for proposal 0.3's decisive question: during healthy walking, how often does
the hold-aware force-error screen permit nonzero correction at all?

The uniform comparison permits correction only if the target error e = r - q keeps one
sign over the whole 20 ms hold (an error interval crossing zero forces zero correction,
proposal Table p.8 row 4). This rolls out the committed walker (healthy, 0.5 m/s), logs
r, q, qdot for the leg joints at each control step, and reports the fraction of
(step, joint) samples whose error interval [e - rho, e + rho] excludes zero, for two
envelope radii: rho = |qdot| H + a_max H^2 / 2 (measured speed) and rho = v99 H
(a global speed bound). Encoder error is ignored, which is optimistic for the method.

  python docs/adaptive_execution/reviews/check_authority.py --steps 250
"""
import json
import sys

import jax
import jax.numpy as jp
import numpy as np

from foresight.eval import rollout as R

STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 250
H, A_MAX = 0.02, 100.0   # hold (s); acceleration bound (rad/s^2), illustrative
env = R.make_env(impl="jax", azimuth_range_deg=60.0)
policy = R.load_policy("checkpoints/nominal_s1")
step = jax.jit(env.step)
act = jax.jit(policy)
state = jax.jit(env.reset)(jax.random.PRNGKey(5))
info = dict(state.info)
cmd = jp.array([0.5, 0.0, 0.0])
info.update(command=cmd, command_nominal=cmd, proj_launch_step=jp.int32(10 ** 6))
state = state.replace(info=info)
m = env.mj_model
jid = m.actuator_trnid[:, 0]
qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
legs = np.array([any(s in m.actuator(i).name for s in ("hip", "knee", "ankle")) for i in range(m.nu)])
names = [m.actuator(i).name for i in range(m.nu) if legs[i]]
lo, hi = m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1]
E, V = [], []
key = jax.random.PRNGKey(0)
for t in range(STEPS):
  key, k = jax.random.split(key)
  obs = {**state.obs, "state": state.obs["state"].at[R.CMD_SLICE].set(cmd)}
  a, _ = act(obs, k)
  r = np.clip(np.asarray(env._default_pose + a * env._config.action_scale), lo, hi)
  q = np.asarray(state.data.qpos)[qadr]
  qd = np.asarray(state.data.qvel)[dadr]
  if t >= 50:                       # skip the first second (gait start-up)
    E.append((r - q)[legs]); V.append(qd[legs])
  state = step(state, a)
  if bool(state.info["fell"]):
    print("fell at step", t); break
E, V = np.array(E), np.array(V)
v99 = np.percentile(np.abs(V), 99, axis=0)
rho_meas = np.abs(V) * H + 0.5 * A_MAX * H ** 2
rho_glob = v99[None] * H
ok_meas = np.abs(E) > rho_meas
ok_glob = np.abs(E) > rho_glob
groups = {"hip": [i for i, n in enumerate(names) if "hip" in n], "knee": [i for i, n in enumerate(names) if "knee" in n],
          "ankle_pitch": [i for i, n in enumerate(names) if "ankle_pitch" in n], "ankle_roll": [i for i, n in enumerate(names) if "ankle_roll" in n]}
out = {"samples": int(E.size), "steps": int(E.shape[0]), "hold_s": H, "a_max": A_MAX,
       "median_abs_error_rad": float(np.median(np.abs(E))), "median_abs_qdot": float(np.median(np.abs(V))),
       "permitted_fraction_measured_speed": float(ok_meas.mean()), "permitted_fraction_global_speed": float(ok_glob.mean()),
       "by_group": {g: {"median_abs_error": float(np.median(np.abs(E[:, ix]))),
                        "median_rho_measured": float(np.median(rho_meas[:, ix])),
                        "permitted_measured": float(ok_meas[:, ix].mean()), "permitted_global": float(ok_glob[:, ix].mean())}
                    for g, ix in groups.items()}}
print(json.dumps(out, indent=1))
