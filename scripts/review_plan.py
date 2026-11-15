"""Declarative bounded rerun plan; importable without JAX or MuJoCo."""
from __future__ import annotations

from pathlib import Path

BASE_SOURCE = 'cc17b7b1066a5694a4c9e8c5e4dddf0a8f5ecdcf'
PROTOCOL = 'foresight.review_rerun.v1'
METHODS = 'reactive,oracle,ttc_threshold,belief_lead,belief_wait,belief_voi'
DEPTH_METHODS = 'ttc_threshold:0.3,belief_lead:0.1,belief_wait:0.1,belief_voi:0.1'


def make_plan(output: Path, mode='full'):
    if mode not in ('full', 'smoke'):
        raise ValueError('Mode must be full or smoke')
    output = Path(output)
    model = output / 'response_model.json'
    common = ['--nominal', 'checkpoints/nominal_s1', '--brace', 'checkpoints/brace2_s1',
              '--impl', 'warp', '--steps', '360', '--hold', '0.5',
              '--commit_steps', '15', '--max_hold_s', '4.0']
    plan = []
    def oracle(name, n, batches, leads, seed, modes, controls, split):
        args = ['scripts/eval_oracle_brace.py', *common, '--episodes_per_lead', str(n),
                '--batches', str(batches), '--leads', leads, '--seed', str(seed),
                '--modes', modes, '--no_launch_controls='+str(controls).lower(),
                '--timing_reference', 'reference_entry', '--reference_radius', '0.45',
                '--azimuth_deg', '40', '--split_label', split,
                '--save_traces=true', '--out', str(output/name)]
        nc = len(leads.split(','))*len(modes.split(','))*(2 if controls else 1)
        plan.append(dict(name=name, kind='oracle', args=args,
                         episodes_per_condition=n*batches, conditions=nc,
                         seed=seed, batches=batches, split=split))
    def frontier(name, n, batches, period, latencies, seed, split):
        args = ['scripts/eval_decision_frontier.py', *common, '--episodes', str(n),
                '--batches', str(batches), '--period', str(period), '--latencies', latencies,
                '--seed', str(seed), '--methods', METHODS, '--thetas', '0.3', '--c_preps', '0.1',
                '--response_model', str(model), '--release_mode', 'deadline',
                '--inference_mode', 'deterministic', '--oracle_timing', 'reference_entry',
                '--r_contact', '0.45', '--sigma0', '0.03', '--sigma1', '0',
                '--miss_prob', '0.055', '--fov_deg', '90', '--azimuth_deg', '40',
                '--split_label', split, '--save_traces='+str(split=='gate').lower(),
                '--calib_params', '', '--out', str(output/name)]
        plan.append(dict(name=name, kind='frontier', args=args,
                         episodes_per_condition=n*batches,
                         conditions=2+4*len(latencies.split(',')),
                         seed=seed, batches=batches, split=split))
    def depth(name, n, batches, latency, seed, phase):
        args = ['scripts/eval_depth_closed_loop.py', *common, '--episodes', str(n),
                '--batches', str(batches), '--period', '5', '--extra_latency', str(latency),
                '--seed', str(seed), '--response_model', str(model),
                '--release_mode', 'deadline', '--single_commit=true',
                '--methods', DEPTH_METHODS, '--sources', 'depth,parametric,parametric_matched',
                '--trace_episodes', '4', '--phase', phase, '--out', str(output/name)]
        plan.append(dict(name=name, kind='depth', args=args,
                         episodes_per_condition=n*batches, conditions=13,
                         seed=seed, batches=batches, split='gate' if phase=='smoke' else 'test'))
    oracle('oracle_gate', 16 if mode=='smoke' else 32, 1, '-1,0.1,0.5',
           6001, 'brace,zerocmd', True, 'gate')
    oracle('oracle_development', 32 if mode=='smoke' else 128, 2,
           '-1,0.1,0.2,0.3,0.5,0.75', 7001, 'brace', False, 'development')
    plan.append(dict(name='response_fit', kind='fit',
                     args=['scripts/fit_review_response.py', '--episodes',
                           str(output/'oracle_development/episodes.csv'), '--oracle-summary',
                           str(output/'oracle_development/summary.json'), '--out', str(model)],
                     split='development'))
    frontier('frontier_gate', 16 if mode=='smoke' else 32, 1, 5, '0,5', 8001, 'gate')
    for latency in (0, 5):
        depth(f'depth_gate_l{latency}', 16, 1, latency, 8501, 'smoke')
    if mode=='full':
        for period in (5, 1):
            frontier(f'frontier_test_p{period}', 256, 2, period, '0,5,15', 9001, 'test')
        for latency in (0, 5):
            depth(f'depth_test_l{latency}', 256, 2, latency, 10001, 'test')
    return plan


def assert_disjoint_splits(plan):
    """Repeated method/rate/latency evaluations share IDs only within a split."""
    seeds = {}
    for stage in plan:
        if 'seed' not in stage:
            continue
        for seed in range(stage['seed'], stage['seed']+stage['batches']):
            old = seeds.setdefault(seed, stage['split'])
            if old != stage['split']:
                raise ValueError(f'Seed {seed} shared between {old} and {stage["split"]}')
    return True
