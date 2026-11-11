# anticipatory-balance (Foresight)

Belief-based anticipation of external impacts for a learned Unitree G1 controller.
AA228/CS238 final project, Stanford, Autumn 2026. Simulation only (MuJoCo Playground rss2025, MJX).

## Layout
```
foresight/envs/g1_impact/   ImpactJoystick env: Playground G1 joystick + launched projectile
foresight/lip/              linear inverted pendulum / capture point (J_max, predicted fall)
foresight/perception/       Stage-A noise+delay model (later: rendered-camera CNN)
foresight/belief/           particle filter with delayed-observation replay
foresight/decision/         threshold, myopic commit rule, QMDP, learned gate
foresight/skills/           PPO training scripts for nominal / brace / step-aside skills
foresight/eval/             sweeps, paired seeds, frontier plots
scripts/check_env.py        Day-1 smoke test
```

## Install (on the GPU box)
```bash
git clone https://github.com/google-deepmind/mujoco_playground.git
cd mujoco_playground/rss2025
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -U "jax[cuda12]" --index-url https://pypi.org/simple
uv --no-config sync --all-extras
cd /workspace && git clone https://github.com/pchatla-arch/anticipatory-balance.git
cd anticipatory-balance && uv pip install -e ".[dev]"
uv --no-config run python scripts/check_env.py
```

## Status
- [x] LIP module + tests
- [x] Impact env v0 (projectile launch, torso capsules, contact/fall flags)
- [ ] Per-episode projectile mass (v0 uses speed only; mass fixed at scene build)
- [ ] Nominal skill training script
- [ ] Reactive fall-rate curve
