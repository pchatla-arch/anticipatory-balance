#!/usr/bin/env python3
"""Known-gain compensation gate for a frozen G1 walking policy.

This is a feasibility experiment, not an online adaptation result. No PPO training,
projectile, depth camera, lag model, or learned residual is used. The pod worker
owns the wall-time limit, run.json, publication, and optional verified pod stop.
"""
from __future__ import annotations

import argparse
import csv
from importlib import metadata
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = 'adaptive-servo-gain-gate-v1'
HORIZON = 400
DT = 0.02
GAINS = (('gain_085', .85), ('gain_070', .70), ('gain_055', .55))
SCREEN_SEED = 73011
CONFIRM_SEED = 83011


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def conditions(selected=None):
    result = [('healthy', 1., False), ('identity', 1., True)]
    for code, gain in GAINS:
        if selected is None or selected == code:
            result.extend([(f'{code}_plain', gain, False), (f'{code}_oracle', gain, True)])
    return result


def scenarios(seed, count):
    rng = np.random.default_rng(seed)
    return {
        'ids': [f'seed{seed}_env{i:05d}' for i in range(count)],
        'onset': rng.integers(75, 126, size=count, dtype=np.int32),
        'speed': rng.choice(np.array([.4, .6], dtype=np.float32), size=count),
    }


def write_rows(path, rows):
    if not rows:
        return
    temporary = Path(str(path) + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def checkpoint_manifest(path):
    tree_hash = hashlib.sha256()
    files = []
    for entry in sorted(path.rglob('*')):
        if entry.is_file():
            relative = entry.relative_to(path).as_posix()
            digest = hashlib.sha256(entry.read_bytes()).hexdigest()
            tree_hash.update(relative.encode() + b'\0' + digest.encode() + b'\n')
            files.append({'path': relative, 'bytes': entry.stat().st_size, 'sha256': digest})
    return {'step': path.name, 'tree_sha256': tree_hash.hexdigest(), 'files': files}


def select_batched_rows(mask, new, old):
    """Per-environment select between two batched states.

    Only leaves batched along the environment axis are selected row by row. The
    MuJoCo-Warp backend also carries unbatched leaves in Data (for example
    zero-size (0,) buffers); those are shared by every environment, so they are
    taken from `new` unchanged. Selecting them row-wise crashed the first GPU
    gate runs with shapes (32,), (0,), (0,).
    """
    import jax
    import jax.numpy as jp

    n = mask.shape[0]

    def pick(x, y):
        if jp.ndim(x) == 0 or jp.shape(x)[0] != n:
            if jp.shape(x) != jp.shape(y):
                raise ValueError(f'unbatched leaf changed shape: {jp.shape(y)} -> {jp.shape(x)}')
            return x
        return jp.where(mask.reshape((n,) + (1,) * (jp.ndim(x) - 1)), x, y)

    return jax.tree_util.tree_map(pick, new, old)


def build_rollout(env, policy, steps=HORIZON):
    import jax
    import jax.numpy as jp

    infer = jax.vmap(policy)
    advance = jax.vmap(env.step)
    velocity = jax.vmap(lambda data: env.get_local_linvel(data, 'pelvis')[:2])
    pelvis = jax.vmap(env._pelvis_pos)

    def finite_rows(value):
        return jp.all(jp.isfinite(value), axis=tuple(range(1, value.ndim)))

    choose_rows = select_batched_rows

    @jax.jit
    def execute(initial):
        n = initial.done.shape[0]
        onset = initial.info['adaptive_onset_step']
        cmd = initial.info['command_nominal'][:, :2]
        # Invalid state is recorded separately from a finite physical termination.
        metrics = {
            'alive': jp.ones(n, bool),
            'term_step': jp.full(n, -1, jp.int32),
            'first_nonfinite_step': jp.full(n, -1, jp.int32),
            'pre_fault_failed': jp.zeros(n, bool),
            'loss_sum': jp.zeros(n), 'progress': jp.zeros(n),
            'bound_sum': jp.zeros(n), 'control_clip_sum': jp.zeros(n),
            'exposed': jp.zeros(n, jp.int32),
            'max_correction': jp.zeros(n), 'max_slew': jp.zeros(n),
            'max_force': jp.zeros(n),
        }

        def one_step(carry, t):
            state, acc = carry
            active = acc['alive']
            # vmap still executes every lane. Feed a known finite reset state
            # into completed lanes so a fallen body's later numerical explosion
            # cannot crash the batch inside the Warp callback.
            simulated = choose_rows(active, state, initial)
            keys = jax.vmap(lambda key: jax.random.fold_in(key, t))(simulated.info['rng'])
            action, _ = infer(simulated.obs, keys)
            action_finite = finite_rows(action)
            safe_action = jp.where(action_finite[:, None], action, jp.zeros_like(action))
            next_state = advance(simulated, safe_action)
            finite = (finite_rows(next_state.data.qpos)
                      & finite_rows(next_state.data.qvel)
                      & finite_rows(next_state.data.xpos)
                      & finite_rows(next_state.data.xmat)
                      & finite_rows(next_state.data.sensordata)
                      & finite_rows(next_state.data.actuator_force)
                      & finite_rows(next_state.obs['state'])
                      & action_finite)
            terminal = next_state.info['terminated'] | ~finite
            first_term = active & terminal
            invalid = active & ~finite
            after_onset = t >= onset
            observed = active & finite
            exposed = observed & after_onset
            # The fixed-horizon loss keeps failed episodes. A terminal transition
            # and every remaining step receive the declared maximum loss (4).
            error = velocity(next_state.data) - cmd
            loss = jp.minimum(jp.sum(jp.square(error), axis=-1), 4.)
            loss = jp.where(observed & ~terminal, loss, 4.)
            displacement = pelvis(next_state.data)[:, :2] - pelvis(state.data)[:, :2]
            forward = state.data.xmat[:, env._pelvis_body].reshape(n, 3, 3)[:, :2, 0]
            forward = forward / jp.maximum(jp.linalg.norm(forward, axis=-1, keepdims=True), 1e-6)
            increment = jp.sum(displacement * forward, axis=-1)
            # where, not multiplication by a mask: NaN * False is still NaN.
            progress = jp.where(exposed, increment, 0.)
            delta = next_state.info['adaptive_step_delta']
            prev_delta = simulated.info['adaptive_prev_delta']
            correction = jp.max(jp.abs(delta), axis=-1)
            slew = jp.max(jp.abs(delta - prev_delta), axis=-1) / env.dt
            force = jp.max(jp.abs(next_state.data.actuator_force), axis=-1)
            out = {
                'alive': active & ~terminal,
                'term_step': jp.where(first_term, t, acc['term_step']),
                'first_nonfinite_step': jp.where(invalid, t, acc['first_nonfinite_step']),
                'pre_fault_failed': acc['pre_fault_failed'] | (first_term & ~after_onset),
                'loss_sum': acc['loss_sum'] + jp.where(after_onset, loss, 0.),
                'progress': acc['progress'] + progress,
                'bound_sum': acc['bound_sum'] + jp.where(exposed, next_state.info['adaptive_bound_hit'], 0.),
                'control_clip_sum': acc['control_clip_sum'] + jp.where(exposed, next_state.info['adaptive_control_clip_fraction'], 0.),
                'exposed': acc['exposed'] + exposed.astype(jp.int32),
                'max_correction': jp.maximum(acc['max_correction'], jp.where(observed, correction, 0.)),
                'max_slew': jp.maximum(acc['max_slew'], jp.where(observed, slew, 0.)),
                'max_force': jp.maximum(acc['max_force'], jp.where(observed, force, 0.)),
            }
            # Keep the last finite outcome snapshot after an episode ends.
            # Inactive lanes use safe reset inputs at the next physics call.
            state = choose_rows(observed, next_state, state)
            return (state, out), None

        (_, result), _ = jax.lax.scan(one_step, (initial, metrics), jp.arange(steps))
        result['postfault_loss'] = result['loss_sum'] / (steps - onset)
        return result

    return execute


def render_report(summary, rows):
    lines = ['# Adaptive execution: known-gain feasibility gate', '',
             f"Scientific gate: **{summary['scientific_gate_status']}**.", '',
             'This tests a bounded, privileged correction for proportional-servo gain loss. '
             'It does not estimate faults online and is not evidence of learned adaptation.', '',
             '| Phase | Condition | Episodes | Failure rate | Mean fixed-horizon loss | Mean post-fault progress (m) |',
             '|---|---|---:|---:|---:|---:|']
    for phase in ('screen', 'confirm'):
        labels = list(dict.fromkeys(r['condition'] for r in rows if r['phase'] == phase))
        for label in labels:
            rr = [r for r in rows if r['phase'] == phase and r['condition'] == label]
            lines.append(f"| {phase} | {label} | {len(rr)} | {np.mean([r['fell'] for r in rr]):.3f} | "
                         f"{np.mean([r['postfault_loss'] for r in rr]):.4f} | {np.mean([r['progress_m'] for r in rr]):.3f} |")
    lines += ['', 'The same initial states, commands, onset times and noise streams are used within each phase. '
              'Confirmation uses new scenario seeds. All screened severities remain in episodes.csv.', '',
              'The loss is mean squared planar local-velocity error, capped at 4 (m/s)^2, over the fixed '
              'post-fault horizon. Failure receives that maximum loss for the terminal transition and all '
              'remaining steps. Progress stops at the last finite terminal transition. Numerical failures '
              'are retained and invalidate the scientific gate.', '',
              'See summary.json for paired uncertainty, selection, and threshold verdicts. '
              'A completed run with a failed scientific gate is still a useful result. '
              'No actuator lag, reduced torque limit, online estimator, vision loop, or real hardware is tested.', '']
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--profile', choices=('smoke', 'gate'), default='gate')
    parser.add_argument('--impl', choices=('warp', 'jax'), default='warp')
    parser.add_argument('--allow-cpu', action='store_true', help='Local engineering checks only; no scientific gate claim.')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    from foresight.adaptive.gate_analysis import analyze_phase, choose_candidate
    plan = {
        'protocol': PROTOCOL, 'profile': args.profile, 'horizon_steps': HORIZON, 'ctrl_dt_s': DT,
        'screen_episodes': 8 if args.profile == 'smoke' else 32,
        'confirmation_episodes': 0 if args.profile == 'smoke' else 128,
        'screen_seed': SCREEN_SEED, 'confirm_seed': CONFIRM_SEED,
        'gain_conditions': conditions(), 'onset_steps_inclusive': [75, 125],
        'forward_commands_mps': [.4, .6], 'correction_bound_rad': .20,
        'correction_rate_bound_rad_s': 2.,
        'fault': 'Scale proportional servo gain and position feedback bias together; damping and torque caps unchanged.',
        'oracle': 'Known gain; measured joint position; 50 Hz bounded position-target correction.',
        'hypothesis': 'Some moderate gain loss degrades walking and a bounded known-gain correction recovers at least half the loss.',
        'not_tested': ['online identification', 'lag', 'maximum torque loss', 'vision', 'learned residual', 'hardware'],
    }
    write_json(args.out / 'plan.json', plan)
    import jax
    import jax.numpy as jp
    from foresight.adaptive.actuator_env import AdaptiveJoystick
    from foresight.envs.g1_impact import default_config
    from foresight.eval import rollout as R
    started = time.monotonic()
    devices = jax.devices()
    if not args.allow_cpu and not any(d.platform == 'gpu' for d in devices):
        raise RuntimeError('GPU required for the pod experiment. --allow-cpu is reserved for local checks.')
    if args.allow_cpu and args.impl == 'warp':
        parser.error('--allow-cpu requires --impl jax')
    cfg = default_config()
    cfg.impact.resample_on_autoreset = False
    cfg.impact.miss_fraction = 0.
    cfg.impact.reset_perturb = 1.
    cfg.push_config.enable = False
    cfg.episode_length = HORIZON
    env = AdaptiveJoystick(config=cfg, config_overrides={'impl': args.impl})
    if not np.isclose(env.dt, DT) or env.mjx_model.nu != 29:
        raise RuntimeError('Unexpected control period or actor action interface.')
    if int(np.sum(env.leg_actuator_mask)) != 12:
        raise RuntimeError('Expected exactly twelve leg actuators; review joint mapping before running.')
    checkpoint = R.latest_checkpoint(ROOT / 'checkpoints/nominal_s1')
    policy = R.load_policy(checkpoint, deterministic=True)
    config = dict(plan, checkpoint=checkpoint_manifest(checkpoint),
                  affected_joints=[name for name, selected in zip(env.actuator_joint_names, env.leg_actuator_mask) if selected],
                  actor_frozen=True, observation_normalization_frozen=True,
                  previous_action='Nominal PPO actions with original parent observation timing.',
                  physics_dt_s=float(env._config.sim_dt), no_projectile=True,
                  correction_rate='Requested adapter delta; existing hardware-range clipping can change the effective delta more quickly.',
                  force_diagnostic='max_abs_actuator_force_nm records MuJoCo actuator_force before joint-level actuator force clipping; it is not a joint torque-limit audit.',
                  selection_rule='Least severe passing screen; independent confirmation; all conditions retained.')
    write_json(args.out / 'config.json', config)
    versions = {}
    for name in ('jax', 'jaxlib', 'brax', 'mujoco', 'mujoco-mjx', 'warp-lang', 'orbax-checkpoint'):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    preflight = {
        'devices': [str(d) for d in devices], 'versions': versions,
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'protocol': PROTOCOL, 'policy_checkpoint': config['checkpoint'],
    }
    write_json(args.out / 'preflight.json', preflight)
    reset = jax.jit(jax.vmap(env.reset))
    execute = build_rollout(env, policy)
    rows = []

    def evaluate_phase(phase, seed, count, selected=None):
        sc = scenarios(seed, count)
        batch_size = 8 if args.profile == 'smoke' else 32
        for start in range(0, count, batch_size):
            stop = min(start + batch_size, count)
            keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(
                jp.arange(start, stop, dtype=jp.int32))
            initial = reset(keys)
            speed = jp.asarray(sc['speed'][start:stop])
            command = jp.stack([speed, jp.zeros_like(speed), jp.zeros_like(speed)], axis=-1)
            initial = R.set_scenario(env, initial, launch=False, command=command)
            initial = initial.replace(obs=R._with_command(initial.obs, command))
            reset_values = jax.device_get((initial.data.qpos, initial.data.qvel,
                                          initial.data.xpos, initial.data.xmat,
                                          initial.data.sensordata, initial.obs['state']))
            reset_finite = np.ones(stop-start, dtype=bool)
            for value in reset_values:
                reset_finite &= np.isfinite(value).reshape(stop-start, -1).all(axis=1)
            if not reset_finite.all():
                preflight['invalid_reset_scenarios'] = [sc['ids'][start+i] for i in np.flatnonzero(~reset_finite)]
                preflight['invalid_reset_phase'] = phase
                write_json(args.out / 'preflight.json', preflight)
                raise RuntimeError('Nonfinite reset detected before policy/physics execution; see preflight.json.')
            for label, gain, oracle in conditions(selected):
                arm_started = time.monotonic()
                info = dict(initial.info)
                g = np.where(env.leg_actuator_mask, gain, 1.).astype(np.float32)
                info['adaptive_gain'] = jp.broadcast_to(jp.asarray(g), (stop-start, env.mjx_model.nu))
                info['adaptive_onset_step'] = jp.asarray(sc['onset'][start:stop])
                info['adaptive_oracle'] = jp.full(stop-start, oracle, dtype=bool)
                result = jax.device_get(execute(initial.replace(info=info)))
                for i in range(stop-start):
                    exposures = int(result['exposed'][i])
                    term = int(result['term_step'][i])
                    rows.append({
                        'phase': phase, 'condition': label, 'scenario_id': sc['ids'][start+i],
                        'gain': gain, 'oracle': oracle, 'speed_cmd_mps': float(sc['speed'][start+i]),
                        'onset_step': int(sc['onset'][start+i]), 'fell': term >= 0,
                        'nonfinite': int(result['first_nonfinite_step'][i]) >= 0,
                        'termination_step': term,
                        'first_nonfinite_step': int(result['first_nonfinite_step'][i]),
                        'pre_fault_failed': bool(result['pre_fault_failed'][i]),
                        'postfault_loss': float(result['postfault_loss'][i]),
                        'progress_m': float(result['progress'][i]),
                        'correction_bound_fraction': float(result['bound_sum'][i]) / max(exposures, 1),
                        'control_clip_fraction': float(result['control_clip_sum'][i]) / max(exposures, 1),
                        'max_correction_rad': float(result['max_correction'][i]),
                        'max_slew_rad_s': float(result['max_slew'][i]),
                        'fault_exposed_steps': exposures,
                        'max_abs_actuator_force_nm': float(result['max_force'][i]),
                    })
                write_rows(args.out / 'episodes.csv', rows)
                print(f'{phase} {label} scenarios {start}:{stop}: '
                      f'failure={np.mean(result["term_step"] >= 0):.3f}, '
                      f'loss={np.mean(result["postfault_loss"], dtype=np.float64):.4f}, '
                      f'time={time.monotonic()-arm_started:.1f}s', flush=True)
        return analyze_phase([r for r in rows if r['phase'] == phase], confirmation=(phase == 'confirm'))

    screen = evaluate_phase('screen', SCREEN_SEED, plan['screen_episodes'])
    summary = {'protocol': PROTOCOL, 'profile': args.profile, 'screen': screen,
               'confirmation': None, 'scientific_gate_status': 'inconclusive',
               'interpretation': 'Known-gain bounded compensation only; online adaptation has not been tested.'}
    selected = choose_candidate(screen)
    summary['selected_candidate'] = selected
    write_json(args.out / 'summary.json', summary)
    if args.profile == 'gate' and selected is not None:
        summary['confirmation'] = evaluate_phase('confirm', CONFIRM_SEED, 128, selected)
        summary['scientific_gate_status'] = summary['confirmation']['status']
    elif args.profile == 'gate':
        summary['scientific_gate_status'] = screen['status']
        if screen['status'] == 'pass':
            summary['scientific_gate_status'] = 'inconclusive'
    if args.profile == 'smoke' or args.allow_cpu:
        summary['scientific_gate_status'] = 'inconclusive'
        summary['interpretation'] = 'Engineering smoke only; run the GPU gate profile for independent confirmation.'
    summary['elapsed_seconds'] = round(time.monotonic()-started, 3)
    write_json(args.out / 'summary.json', summary)
    (args.out / 'report.md').write_text(render_report(summary, rows))
    print(f"Operational experiment completed. Scientific gate: {summary['scientific_gate_status']}. "
          'See summary.json; no online estimator was run.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
