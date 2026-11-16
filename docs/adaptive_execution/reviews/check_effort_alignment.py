# Copyright 2026 Prasuna Chatla. Apache-2.0.
"""CPU check behind review findings R1 and R2: which joint position does MuJoCo's
actuator_force correspond to after a step, and is it the torque the joint receives?

  python docs/adaptive_execution/reviews/check_effort_alignment.py
"""
import mujoco
import numpy as np

from foresight.envs.g1_impact.impact_joystick import ImpactJoystick, default_config

env = ImpactJoystick(config=default_config(), config_overrides={"impl": "jax"})
m = env.mj_model
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, m.keyframe("knees_bent").id)
jid = m.actuator_trnid[:, 0]
qadr = m.jnt_qposadr[jid]
dadr = m.jnt_dofadr[jid]
kp = m.actuator_gainprm[:, 0]
legs = np.array([any(s in m.actuator(i).name for s in ("hip", "knee", "ankle")) for i in range(m.nu)])
rng = np.random.default_rng(0)
d.ctrl[:] = d.qpos[qadr] + rng.normal(0, 0.1, m.nu)
q_pre = d.qpos[qadr].copy()
mujoco.mj_step(m, d)
f = d.actuator_force.copy()
q_post = d.qpos[qadr]
err = lambda q: np.abs(f / (kp * (d.ctrl - q)) - 1)[legs]
print("healthy gain error, force paired with pre-step q : median %.4f max %.4f" % (np.median(err(q_pre)), err(q_pre).max()))
print("healthy gain error, force paired with post-step q: median %.4f max %.4f" % (np.median(err(q_post)), err(q_post).max()))
applied = d.qfrc_actuator[dadr]
clipped = np.abs(applied - f) > 1e-6
print("joint actuator-force limits enabled on leg joints:", bool(m.jnt_actfrclimited[jid][legs].all()))
print("leg joints where applied torque (qfrc_actuator) differs from actuator_force:", int(clipped[legs].sum()), "of", int(legs.sum()))

# Saturating command: actuator_force is computed before joint-level actuator-force limits.
mujoco.mj_resetDataKeyframe(m, d, m.keyframe("knees_bent").id)
d.ctrl[:] = np.clip(d.qpos[qadr] + 3.0, m.actuator_ctrlrange[:, 0], m.actuator_ctrlrange[:, 1])
mujoco.mj_step(m, d)
f, applied = d.actuator_force.copy(), d.qfrc_actuator[dadr]
sel = (np.abs(applied - f) > 1e-6) & legs
print("saturating command: leg joints where applied torque differs:", int(sel.sum()), "of", int(legs.sum()))
print("  actuator_force -> applied (N m):", [(round(float(x), 1), round(float(y), 1)) for x, y in zip(f[sel][:4], applied[sel][:4])])
print("leg proportional gains kp:", m.actuator_gainprm[legs, 0].tolist())
