"""Record two preselected paired G1 examples for the project page, using the course environment."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np
from foresight.belief.particle_filter import PFConfig
from foresight.decision.controller import AnticipationController
from foresight.decision.rules import ResponseModel, RuleConfig
from foresight.eval import rollout as R
from foresight.perception.detections import DetectorConfig


def record(output):
    seed, steps = 73132, 300
    output.mkdir(parents=True, exist_ok=True)
    env = R.make_env(skill='nominal', impl='jax', azimuth_range_deg=60.0, range_dist=[3., 5.])
    nominal = R.load_policy('checkpoints/nominal_s1/000202342400')
    brace = R.load_policy('checkpoints/brace2_s1/000202342400')
    fit = json.loads(Path('results/response_model.json').read_text())
    response = ResponseModel(**{k: fit[k] for k in ResponseModel._fields if k in fit}, has_lead=True)
    rs = np.random.RandomState(seed)
    masses = rs.uniform(4., 16., 2)
    scenario = dict(speed=rs.uniform(3., 8., 2),
                    hit_time=rs.uniform(*env._config.impact.hit_time_range, 2),
                    is_miss=np.array([False, True]), command=[.5, 0., 0.])
    world = R.batched_env(env, masses)
    key = jax.random.PRNGKey(seed)
    resets = jax.random.split(jax.random.fold_in(key, 1), 2)
    print('Generating the shared reference for the two preselected encounters.', flush=True)
    reference, _ = R.rollout(env, world, nominal, key, steps, reset_keys=resets,
                             scenario_fn=lambda s: R.set_scenario(env, s, **scenario))
    fixed = dict(scenario, launch_pos=reference['proj_launch_pos'], launch_vel=reference['proj_launch_vel'])
    data = {'scheduled_miss': scenario['is_miss'], 'launch_pos': fixed['launch_pos'],
            'launch_vel': fixed['launch_vel'], 'dt': np.array(env.dt), 'seed': np.array(seed)}
    summary = {'seed': seed, 'selection': 'one scheduled hit (index 0) and one scheduled near-miss (index 1), chosen before simulation',
               'steps': steps, 'dt_s': float(env.dt), 'backend': 'MJX/JAX CPU',
               'observation': 'parametric detector, period 5 steps, latency 5 steps, 120-degree FOV',
               'policies': ['nominal_s1/000202342400', 'brace2_s1/000202342400'], 'arms': {}}
    for arm, config in [('ttc', RuleConfig(kind='ttc_threshold', theta=.3)),
                        ('belief', RuleConfig(kind='belief_lead', c_prep=.05))]:
        controller = AnticipationController(DetectorConfig(period=5, latency=5, sigma0=.03, sigma1=.02,
                                                            miss_prob=.05, fov_deg=120., max_range=6.),
                                            PFConfig(mass_range=(4., 16.), r_contact=.45), config, response)
        print(f'Recording {arm}.', flush=True)
        info, trace = R.rollout(env, world, nominal, key, steps, brace_policy=brace,
                                reset_keys=resets, controller=controller, record_qpos=True,
                                scenario_fn=lambda s: R.set_scenario(env, s, **fixed))
        for field in ('qpos', 'brace_active', 'impact_contact', 'proj_launched', 'ctrl_p_hit', 'ctrl_det_valid'):
            data[f'{arm}_{field}'] = trace[field]
        data[f'{arm}_term_step'] = info['term_step']
        summary['arms'][arm] = []
        for i in range(2):
            term = int(info['term_step'][i])
            end = steps if term < 0 else term + 1
            if not np.isfinite(trace['qpos'][:end, i]).all():
                raise ValueError(f'Invalid pose in {arm}, encounter {i} before termination')
            summary['arms'][arm].append(dict(scheduled_miss=bool(scenario['is_miss'][i]),
                fell=bool(info['fell'][i]), term_step=term, contact_step=int(info['contact_step'][i]),
                applied_brace_s=round(float(env.dt)*int(trace['brace_active'][:end, i].sum()), 4)))
    np.savez_compressed(output/'example_traces.npz', **data)
    (output/'example_record.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    record(parser.parse_args().out)
