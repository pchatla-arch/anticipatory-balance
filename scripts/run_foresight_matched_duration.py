"""Prospective Foresight comparison on a prepared checkout; --plan uses only Python."""
import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import NamedTuple


ROOT = Path(__file__).resolve().parents[1]
TTC = 'ttc_threshold'
BELIEF = ('belief_lead', 0.05)
GRID = (0.06, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.24, 0.30)
DT = 0.02
MARGIN = 0.02
STEPS = 300
CAL_SEED = 21400
TEST_SEED = 21500
BATCH_SIZE = 512
TEST_BATCHES = 4
COMMON = [
    '--nominal=checkpoints/nominal_s1/000202342400', '--brace=checkpoints/brace2_s1/000202342400',
    '--response_model=results/response_model.json', '--methods=ttc_threshold,belief_lead',
    '--c_preps=0.05', '--impl=warp', '--latencies=5', '--period=5', '--steps=300',
    '--mass_range=4,16', '--speed_range=3,8', '--azimuth_deg=60', '--command_vx=0.5',
    '--miss_fraction=0.3', '--sigma0=0.03', '--sigma1=0.02', '--miss_prob=0.05',
    '--fov_deg=120', '--max_range=6', '--release_after=0.3', '--commit_steps=15',
    '--tau_brace=0.22', '--hold=0.5', '--c_track=0.1', '--r_contact=0.45',
    '--calib_params=',
]


def protocol():
    return {
        'experiment': 'Foresight deterministic TTC matched to belief preparation, revision 1',
        'calibration': {'seed': CAL_SEED, 'n': BATCH_SIZE, 'theta_grid_s': GRID,
                        'selection': 'minimize absolute difference in all-episode mean applied brace duration, ties use smaller theta; stop if gap > 0.02 s'},
        'confirmation': {'batch_seeds': list(range(TEST_SEED, TEST_SEED + TEST_BATCHES)),
                         'episodes_per_seed': BATCH_SIZE, 'frozen_belief': list(BELIEF),
                         'primary_risk': 'failure among scheduled hits over 300 steps',
                         'secondary_selectivity': 'probability of any applied BRACE on scheduled near-misses, TTC minus belief with a paired 95% interval',
                         'duration': 'seconds in BRACE through the first terminal transition, averaged over all scenarios',
                         'matched': 'entire paired approximate 95% duration interval within +/-0.02 s',
                         'direction': 'TTC minus belief; assess signed paired 95% risk interval only after matching',
                         'uncertainty': 'scenario-level normal approximation, conditional on one checkpoint pair and the frozen calibration result'},
        'evaluator_flags': COMMON,
        'required_checkout': 'G1 assets, restored checkpoints and the existing GPU-enabled Foresight Python environment',
    }


def utc():
    return datetime.now(timezone.utc).isoformat()


def write_record(path, record):
    with path.open('x') as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bool_value(field):
    if field not in ('True', 'False', '1', '0'):
        raise ValueError('Invalid binary endpoint: ' + repr(field))
    return field in ('True', '1')


class Episode(NamedTuple):
    scenario: tuple
    miss: bool
    fell: bool
    steps: int
    progress_ok: bool


def read_arms(directory, seed, n, batches, thetas):
    """Read raw applied-interval evidence; require every expected ID and scenario."""
    source = directory / 'episodes'
    methods = [(TTC, t) for t in thetas] + [BELIEF]
    arms = {key: {} for key in methods}
    for path in sorted(source.glob('*.csv')):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                arm = (row['method'], float(row['param']))
                batch, index = int(row['batch']), int(row['scenario_index'])
                sid = row['scenario_id']
                if (arm not in arms or not 0 <= batch < batches or not 0 <= index < n
                        or int(row['latency_steps']) != 5 or int(row['scenario_seed']) != seed + batch
                        or sid != f'seed{seed+batch}_env{index:05d}' or sid in arms[arm]):
                    raise ValueError(f'Unexpected or duplicate scenario, condition or latency in {path}: {sid}')
                fell, miss = bool_value(row['fell']), bool_value(row['is_miss'])
                term, steps = int(row['term_step']), int(row['brace_steps_total'])
                if not (-1 <= term < STEPS and fell == (term >= 0)
                        and 0 <= steps <= (term + 1 if fell else STEPS)):
                    raise ValueError(f'Invalid terminal state or applied brace count for {arm} {sid}')
                scenario = tuple(float(row[name]) for name in ('mass', 'speed', 't_hit_nominal'))
                if not all(map(math.isfinite, scenario)):
                    raise ValueError(f'Nonfinite scenario metadata for {arm} {sid}')
                try:
                    progress_ok = math.isfinite(float(row.get('progress_m', 'nan')))
                except (TypeError, ValueError):
                    progress_ok = False
                arms[arm][sid] = Episode(scenario, miss, fell, steps, progress_ok)
    expected = {f'seed{seed+b}_env{i:05d}' for b in range(batches) for i in range(n)}
    for arm, rows in arms.items():
        if set(rows) != expected:
            raise ValueError(f'Missing or unexpected scenario IDs for {arm} in {source}')
    for rows in arms.values():
        for sid, row in rows.items():
            if (row.scenario, row.miss) != (arms[BELIEF][sid].scenario, arms[BELIEF][sid].miss):
                raise ValueError(f'Scenario parameters or intended-hit label differ at {sid}')
    return arms


def duration(rows):
    return DT * statistics.fmean(row.steps for row in rows.values())


def choose_theta(arms):
    target = duration(arms[BELIEF])
    gaps = {par: duration(rows)-target for (method, par), rows in arms.items() if method == TTC}
    chosen = min(gaps, key=lambda t: (round(abs(gaps[t]), 12), t))
    return {'theta_s': chosen, 'belief_mean_s': target, 'ttc_minus_belief_s_by_theta': gaps,
            'matched_on_calibration': abs(gaps[chosen]) <= MARGIN + 1e-12,
            'nonfinite_or_missing_progress_by_arm': missing_progress(arms)}


def missing_progress(arms):
    return {f'{method}_{par:g}': sum(not row.progress_ok for row in rows.values())
            for (method, par), rows in arms.items()}


def paired_interval(differences):
    if len(differences) < 2:
        raise ValueError('A paired interval needs at least two observations')
    mean = statistics.fmean(differences)
    half = 1.96 * statistics.stdev(differences) / math.sqrt(len(differences))
    return {'n': len(differences), 'mean': mean, 'ci95': [mean-half, mean+half]}


def score(arms, theta):
    ttc, belief = arms[(TTC, theta)], arms[BELIEF]
    pairs = [(ttc[sid], belief[sid]) for sid in sorted(belief)]
    hits = [(x,y) for x,y in pairs if not x.miss]
    misses = [(x,y) for x,y in pairs if x.miss]
    if min(len(hits), len(misses)) < 2:
        raise ValueError('Unusable confirmation hit/miss split')
    delta_t = paired_interval([DT*(x.steps-y.steps) for x,y in pairs])
    delta_f = paired_interval([x.fell-y.fell for x,y in hits])
    delta_miss = paired_interval([(x.steps > 0) - (y.steps > 0) for x, y in misses])
    matched = delta_t['ci95'][0] >= -MARGIN and delta_t['ci95'][1] <= MARGIN
    verdict = 'held_out_durations_differ_or_equivalence_unresolved'
    if matched:
        lo, hi = delta_f['ci95']
        verdict = ('ttc_lower_failure' if hi < 0 else
                   'belief_lower_failure' if lo > 0 else 'risk_difference_unresolved')
    summaries = []
    for pos, (method, par) in enumerate(((TTC, theta), BELIEF)):
        summaries.append({'method':method, 'param':par, 'duration_s':duration(arms[(method,par)]),
                          'falls_on_hits':sum(pair[pos].fell for pair in hits),
                          'falls_on_misses':sum(pair[pos].fell for pair in misses),
                          'braces_on_misses':sum(pair[pos].steps > 0 for pair in misses)})
    return {'status':verdict, 'duration_equivalence':matched, 'n_hits':len(hits),
            'n_misses':len(misses), 'arms':summaries, 'duration_difference_s':delta_t,
            'near_miss_preparation_difference':delta_miss,
            'hit_risk_difference':delta_f, 'ttc_only_falls_on_hits':sum(x.fell and not y.fell for x,y in hits),
            'belief_only_falls_on_hits':sum(y.fell and not x.fell for x,y in hits),
            'nonfinite_or_missing_progress_by_arm':missing_progress(arms),
            'progress_handling':'Progress is secondary. Its missing/nonfinite count includes all flagged rows. Endpoint corruption or invalid pairing invalidates the phase.'}


def evaluate(directory, seed, batches, thetas):
    cmd = [sys.executable, 'scripts/eval_decision_frontier.py', *COMMON,
           f'--episodes={BATCH_SIZE}', f'--batches={batches}', f'--seed={seed}',
           '--thetas='+','.join(map(str, thetas)), '--out='+str(directory)]
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(filter(None, (str(ROOT), env.get('PYTHONPATH'))))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)
    config = json.loads((directory/'config.json').read_text())
    summary = json.loads((directory/'summary.json').read_text())
    if (summary.get('status') != 'complete' or config['seed'] != seed
            or config['batches'] != batches or config['episodes'] != BATCH_SIZE
            or config['steps'] != STEPS or not math.isclose(config['dt'], DT, abs_tol=1e-8)
            or config['latencies_steps'] != [5] or config['period'] != 5):
        raise ValueError('Run configuration or completion differs from the registered protocol')
    return read_arms(directory, seed, BATCH_SIZE, batches, thetas), config['provenance']


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    revision = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    write_record(out/'protocol.json', {**protocol(), 'source_commit':revision, 'started_utc':utc()})
    print(f'Paired evidence: {out}', flush=True)
    phase = 'calibration'
    try:
        calibration, provenance = evaluate(out/phase, CAL_SEED, 1, GRID)
        frozen = {**choose_theta(calibration), 'frozen_utc':utc(), 'source_commit':revision}
        write_record(out/'frozen.json', frozen)
        if frozen['matched_on_calibration']:
            phase = 'confirmation'
            theta = frozen['theta_s']
            arms, confirmed = evaluate(out/phase, TEST_SEED, TEST_BATCHES, [theta])
            if (confirmed['artifacts'] != provenance['artifacts'] or
                    confirmed['source_commit'] != provenance['source_commit']):
                raise ValueError('Checkpoint, response model, or source revision changed between phases')
            result = score(arms, theta)
        else:
            result = {'status':'no_calibration_match', 'confirmation_started':False}
        write_record(out/'verdict.json', {**result, 'completed_utc':utc(),
                                          'elapsed_s':time.monotonic()-start})
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        write_record(out/'invalid.json', {'status':'incomplete_or_invalid','phase':phase,
                     'error':str(exc),'ended_utc':utc(),'elapsed_s':time.monotonic()-start})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan', action='store_true', help='print the protocol without invoking the simulator')
    mode.add_argument('--run', action='store_true', help='execute the protocol with the prepared GPU Python')
    parser.add_argument('--out', type=Path, help='new output directory, defaults to timestamped results/')
    args = parser.parse_args()
    if args.plan:
        print(json.dumps(protocol(), indent=2))
    else:
        default = ROOT/'results'/('E20_matched_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
        run((args.out or default).resolve())
