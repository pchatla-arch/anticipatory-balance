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
- [x] Per-env projectile mass via domain randomization
- [ ] Nominal skill training script
- [ ] Reactive fall-rate curve

## Saving results
Small artifacts (eval curves, plots, configs, console logs) are committed under `results/<run>/`
via `scripts/save_run.py`; checkpoints stay on the pod volume and final skills are uploaded to a
GitHub Release (`gh release upload`).

## Session A (experiments E1–E6)

All Session A code is tested on CPU against the real env (MJX backend) before the GPU run.

| Script | Experiment | Output |
|---|---|---|
| `scripts/validate_contact.py` | E4 contact validation (base pinned): aim, timing, transferred impulse | `results/E4_contact/contact_table.csv` |
| `scripts/train_skill.py --skill nominal/brace` | E1 nominal ×3 seeds, E5 BRACE skill | `logs/<tag>/.../checkpoints`, `results/<tag>/` via `save_run.py` |
| `scripts/eval_fall_rate.py` | E2 reactive fall-rate surface over (mass, speed) + near-miss control | `results/E2_fall_rate/{episodes,grid}.csv`, heatmap |
| `scripts/lip_overlay.py` | E3 LIP / capture-point prediction vs E2 | `results/E3_lip_overlay/` |
| `scripts/eval_oracle_brace.py` | E6 oracle brace: fall rate and progress loss vs lead time, τ_brace | `results/E6_oracle/lead_curve.csv`, plot |
| `scripts/run_session_a.sh` | runs all of the above in order, saves, commits, pushes | — |

Env semantics worth knowing (see `impact_joystick.py` docstring): scenario fields
(`proj_mass`, `proj_speed`, `proj_momentum = m·v`, `proj_is_miss`, `t_hit_nominal`) are
sampled inputs; outcome fields (`impact_contact`, `contact_step`, `fell`, `term_reason`,
`term_step`) are measured. The scenario is resampled at every episode start under
Playground's autoreset, the launch is timed so the nominal arrival falls in
`hit_time_range` (default 1.5–3.5 s of a 6 s episode, leaving a recovery window), and
`info["brace_active"]` is the external mode switch (zero command + BRACE policy).
