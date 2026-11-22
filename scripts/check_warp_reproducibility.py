#!/usr/bin/env python3
"""CPU check of the gate's control arms on the pod's MuJoCo-Warp backend.

Executes the healthy, identity and healthy_repeat arms of the adaptive gate with the
Warp physics backend on CPU (slow, about 1.5 s per control step for 8 walkers) and
reports whether the arms agree bitwise, with and without MuJoCo-Warp's determinism
mask. On CPU the backend runs sequentially, so a difference between healthy and
identity here would be a code-path difference, not scheduling noise. A GPU can still
differ run to run; the pod's healthy_repeat arm measures that.

  python scripts/check_warp_reproducibility.py --steps 160 --envs 8 --out results/warp_cpu_check
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

KEYS = ('term_step', 'first_nonfinite_step', 'pre_fault_failed', 'loss_sum', 'progress',
        'bound_sum', 'control_clip_sum', 'exposed', 'max_correction', 'max_slew', 'max_force')


def run_arms(G, env, policy, steps, n, seed, impl):
    import jax
    import jax.numpy as jp
    from foresight.eval import rollout as R
    reset = jax.jit(jax.vmap(env.reset))
    execute = G.build_rollout(env, policy, steps=steps)
    sc = G.scenarios(seed, n)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(jp.arange(0, n, dtype=jp.int32))
    initial = reset(keys)
    speed = jp.asarray(sc['speed'])
    command = jp.stack([speed, jp.zeros_like(speed), jp.zeros_like(speed)], axis=-1)
    initial = R.set_scenario(env, initial, launch=False, command=command)
    initial = initial.replace(obs=R._with_command(initial.obs, command))
    out = {}
    for label, gain, oracle in (('healthy', 1., False), ('identity', 1., True), ('healthy_repeat', 1., False)):
        info = dict(initial.info)
        g = np.where(env.leg_actuator_mask, gain, 1.).astype(np.float32)
        info['adaptive_gain'] = jp.broadcast_to(jp.asarray(g), (n, env.mjx_model.nu))
        info['adaptive_onset_step'] = jp.asarray(sc['onset'])
        info['adaptive_oracle'] = jp.full(n, oracle, dtype=bool)
        started = time.monotonic()
        result = jax.device_get(execute(initial.replace(info=info)))
        out[label] = {k: np.asarray(result[k]) for k in KEYS}
        print(f'{impl} {label}: {time.monotonic()-started:.0f}s, falls={int((out[label]["term_step"] >= 0).sum())}, '
              f'loss_sum mean={out[label]["loss_sum"].mean():.5f}, progress mean={out[label]["progress"].mean():.4f}, '
              f'max_force max={out[label]["max_force"].max():.4f}', flush=True)
    return out


def compare(a, b):
    diffs = {k: float(np.max(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)))) for k in KEYS}
    return {'bitwise_equal': all(np.array_equal(a[k], b[k]) for k in KEYS), 'max_abs_difference': diffs}


TRACE_UNITS = {'ctrl': 'rad (actuator position target)', 'qpos': 'm for base position, unit quaternion, rad for joints',
               'actuator_force': 'N.m (MuJoCo actuator_force, before joint force limits)'}


def trace_arms(G, env, policy, steps, n, seed, impl, arms):
    """Per-step applied commands and physics state for the requested control arms.

    Same reset, commands, onset times and policy as the gate; the rollout here keeps every lane
    simulating (no finished-lane freezing), so it is a divergence probe, not the gate metric.
    Returns {arm: {'ctrl': (steps, n, nu), 'qpos': (steps, n, nq), 'actuator_force': (steps, n, nu)}}.
    """
    import jax
    import jax.numpy as jp
    from foresight.eval import rollout as R
    reset = jax.jit(jax.vmap(env.reset))
    infer = jax.vmap(policy)
    advance = jax.vmap(env.step)

    @jax.jit
    def run(initial):
        def one_step(state, t):
            keys = jax.vmap(lambda key: jax.random.fold_in(key, t))(state.info['rng'])
            action, _ = infer(state.obs, keys)
            nxt = advance(state, action)
            return nxt, {'ctrl': nxt.data.ctrl, 'qpos': nxt.data.qpos, 'actuator_force': nxt.data.actuator_force}
        _, ys = jax.lax.scan(one_step, initial, jp.arange(steps))
        return ys

    sc = G.scenarios(seed, n)
    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(jp.arange(0, n, dtype=jp.int32))
    initial = reset(keys)
    speed = jp.asarray(sc['speed'])
    command = jp.stack([speed, jp.zeros_like(speed), jp.zeros_like(speed)], axis=-1)
    initial = R.set_scenario(env, initial, launch=False, command=command)
    initial = initial.replace(obs=R._with_command(initial.obs, command))
    out = {}
    for label in arms:
        oracle = label.startswith('identity')
        info = dict(initial.info)
        info['adaptive_gain'] = jp.ones((n, env.mjx_model.nu), jp.float32)
        info['adaptive_onset_step'] = jp.asarray(sc['onset'])
        info['adaptive_oracle'] = jp.full(n, oracle, dtype=bool)
        started = time.monotonic()
        ys = jax.device_get(run(initial.replace(info=info)))
        out[label] = {k: np.asarray(v) for k, v in ys.items()}
        print(f'{impl} trace {label}: {time.monotonic()-started:.0f}s', flush=True)
    return out, sc


def first_divergence(a, b, onsets):
    """First control step and channel where two traces differ, per recorded quantity."""
    report = {}
    for key in ('ctrl', 'qpos', 'actuator_force'):
        x, y = a[key].astype(np.float64), b[key].astype(np.float64)
        diff = np.abs(x - y)
        if not diff.any():
            report[key] = {'bitwise_equal': True, 'units': TRACE_UNITS[key]}
            continue
        step, env_i, chan = np.unravel_index(np.argmax(diff > 0), diff.shape)
        amax = np.unravel_index(np.argmax(diff), diff.shape)
        report[key] = {'bitwise_equal': False, 'units': TRACE_UNITS[key],
                       'first_divergence': {'step': int(step), 'env': int(env_i), 'channel': int(chan),
                                            'abs_difference': float(diff[step, env_i, chan]),
                                            'onset_step_of_that_env': int(onsets[env_i]),
                                            'before_onset': bool(step < onsets[env_i])},
                       'largest': {'step': int(amax[0]), 'env': int(amax[1]), 'channel': int(amax[2]),
                                   'abs_difference': float(diff[amax])}}
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'results/warp_cpu_check')
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--modes', default='default,deterministic')
    parser.add_argument('--trace', action='store_true',
                        help='Also record per-step ctrl, qpos and actuator_force for healthy, healthy_repeat, '
                             'identity, identity_repeat and report the first divergent step and channel.')
    parser.add_argument('--metrics', dest='metrics', action='store_true', default=True)
    parser.add_argument('--no-metrics', dest='metrics', action='store_false', help='Skip the gate-metric arms.')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    import eval_adaptive_gate as G
    import jax
    from foresight.adaptive.actuator_env import AdaptiveJoystick
    from foresight.envs.g1_impact import default_config
    from foresight.eval import rollout as R
    cfg = default_config()
    cfg.impact.resample_on_autoreset = False
    cfg.impact.miss_fraction = 0.
    cfg.impact.reset_perturb = 1.
    cfg.push_config.enable = False
    cfg.episode_length = G.HORIZON
    policy = R.load_policy(R.latest_checkpoint(ROOT / 'checkpoints/nominal_s1'), deterministic=True)
    import subprocess
    report = {'devices': [str(d) for d in jax.devices()], 'steps': args.steps, 'envs': args.envs,
              'seed': G.SCREEN_SEED, 'modes': {},
              'source_commit': subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                                              capture_output=True).stdout.strip(),
              'versions': {name: G.metadata.version(name) for name in ('jax', 'mujoco', 'mujoco-mjx', 'warp-lang')}}
    results = {}
    for mode in args.modes.split(','):
        env = AdaptiveJoystick(config=cfg, config_overrides={'impl': 'warp'})
        mask = None
        if mode == 'deterministic':
            env._mjx_model, mask = G.deterministic_model(env.mjx_model, 'warp')
        print(f'--- mode {mode}: opt._impl.deterministic={int(env.mjx_model.opt._impl.deterministic)}', flush=True)
        block = {'mujoco_warp_opt_deterministic': int(env.mjx_model.opt._impl.deterministic)}
        if args.metrics:
            arms = run_arms(G, env, policy, args.steps, args.envs, G.SCREEN_SEED, f'warp/{mode}')
            results[mode] = arms
            block.update({
                'all_finite': bool(all(np.isfinite(arms[a][k]).all() for a in arms for k in KEYS)),
                'identity_vs_healthy': compare(arms['healthy'], arms['identity']),
                'repeat_vs_healthy': compare(arms['healthy'], arms['healthy_repeat']),
                'healthy_summary': {'falls': int((arms['healthy']['term_step'] >= 0).sum()),
                                    'loss_sum_mean': float(arms['healthy']['loss_sum'].mean()),
                                    'progress_mean': float(arms['healthy']['progress'].mean())},
            })
        if args.trace:
            traces, sc = trace_arms(G, env, policy, args.steps, args.envs, G.SCREEN_SEED, f'warp/{mode}',
                                    ('healthy', 'healthy_repeat', 'identity', 'identity_repeat'))
            block['trace'] = {
                'healthy_vs_healthy_repeat': first_divergence(traces['healthy'], traces['healthy_repeat'], sc['onset']),
                'identity_vs_identity_repeat': first_divergence(traces['identity'], traces['identity_repeat'], sc['onset']),
                'healthy_vs_identity': first_divergence(traces['healthy'], traces['identity'], sc['onset']),
            }
            np.savez_compressed(args.out / f'trace_{mode}.npz', onsets=sc['onset'],
                                **{f'{arm}_{k}': v for arm, tr in traces.items() for k, v in tr.items()})
        report['modes'][mode] = block
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    if len(results) == 2:
        a, b = results.values()
        report['deterministic_vs_default_healthy'] = compare(a['healthy'], b['healthy'])
        (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    for mode, block in report['modes'].items():
        if 'identity_vs_healthy' in block:
            print(f"{mode}: identity bitwise={block['identity_vs_healthy']['bitwise_equal']} "
                  f"repeat bitwise={block['repeat_vs_healthy']['bitwise_equal']} finite={block['all_finite']}")
        for pair, rep in block.get('trace', {}).items():
            for key, r in rep.items():
                if r['bitwise_equal']:
                    print(f"{mode} trace {pair} {key}: bitwise equal")
                else:
                    f = r['first_divergence']
                    print(f"{mode} trace {pair} {key}: first divergence step {f['step']} env {f['env']} channel {f['channel']} "
                          f"|diff| {f['abs_difference']:.3g} ({'before' if f['before_onset'] else 'after'} that env's onset {f['onset_step_of_that_env']}); "
                          f"largest |diff| {r['largest']['abs_difference']:.3g} at step {r['largest']['step']}")
    if 'deterministic_vs_default_healthy' in report:
        print('deterministic vs default (healthy): bitwise', report['deterministic_vs_default_healthy']['bitwise_equal'],
              'max |diff|', {k: f'{v:.3g}' for k, v in report['deterministic_vs_default_healthy']['max_abs_difference'].items() if v})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
