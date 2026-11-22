"""WP 2.4. Run one approved development bench through the shared evaluator.

A release commit fills the two frozen_inputs records with {path, sha256}.
The launcher runs the CPU rehearsal and validates its pairing before starting Warp on GPU.
"""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = Path('docs/adaptive_execution/s1_dev_manifest.json')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def read(path):
    def reject(value):
        raise ValueError(f'{path} contains {value}')
    return json.loads(path.read_bytes(), parse_constant=reject)


def frozen_inputs(plan):
    if plan['split'] != 'dev' or plan['implementation'] != 'warp':
        raise ValueError('The S1 pod profile allows only development instances with Warp.')
    if (plan['attempts'] != 1 or plan['max_wall_minutes'] != 90
            or plan['work_cutoff_percent'] != 80 or plan['cleanup_reserve_minutes'] != 10):
        raise ValueError('The S1 attempt budget has changed.')
    if git('show', f'HEAD:{MANIFEST}') != (ROOT / MANIFEST).read_bytes():
        raise ValueError('Commit the S1 manifest before launching.')
    checkpoint = 'checkpoints/nominal_s1'
    if git('rev-parse', f'HEAD:{checkpoint}').decode().strip() != plan['checkpoint_tree']:
        raise ValueError('Frozen walker checkpoint tree does not match the manifest.')
    checkpoint_dirs = [p.name for p in (ROOT / checkpoint).iterdir() if p.is_dir() and p.name.isdigit()]
    if sorted(checkpoint_dirs, key=int)[-1:] != ['000202342400']:
        raise ValueError('The evaluator would select a different walker checkpoint.')
    if git('status', '--porcelain', '--untracked-files=all', '--', checkpoint):
        raise ValueError('Frozen walker checkpoint directory contains local changes.')
    files = {}
    for name in ('heuristic', 'candidate'):
        item = plan['frozen_inputs'][name]
        if not isinstance(item, dict) or not re.fullmatch('[0-9a-f]{64}', str(item.get('sha256'))):
            raise ValueError(f'The S1 {name} input has not been frozen.')
        path = Path(item['path'])
        if path.is_absolute() or '..' in path.parts or path.suffix != '.json':
            raise ValueError(f'Invalid repository-relative {name} config path.')
        data = (ROOT / path).read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256'] or data != git('show', f'HEAD:{path}'):
            raise ValueError(f'{name} config does not match its frozen commit and digest.')
        files[name] = ROOT / path
    return files


def dimensions(plan, smoke):
    if smoke:
        return ['healthy'], plan['telemetry'], ['heuristic_gate', 'candidate'], 2, 2, 2
    return (plan['families'], plan['telemetry'], plan['methods'], plan['episodes_per_cell'],
            plan['batch'], plan['steps'])


def command(plan, files, out, smoke):
    families, telemetry, methods, episodes, batch, steps = dimensions(plan, smoke)
    args = [sys.executable, '-u', str(ROOT / 'scripts/eval_adaptive_bench.py'), '--out', str(out),
            '--split', 'dev', '--impl', 'warp', '--families', ','.join(families), '--telemetry',
            ','.join(telemetry), '--methods', ','.join(methods), '--episodes', str(episodes),
            '--batch', str(batch), '--steps', str(steps), '--schedule', plan['schedule'],
            '--heuristic', str(files['heuristic']), '--candidate-config', str(files['candidate'])]
    if smoke:
        args.append('--allow-cpu')
    return args


def validate(out, plan, smoke):
    families, telemetry, methods, episodes, _, steps = dimensions(plan, smoke)
    config = read(out / 'config.json')
    reg = read(out / 'registration.json')
    for key, value in [('split', 'dev'), ('episodes', episodes), ('steps', steps),
                       ('families', families), ('methods', methods), ('telemetry', sorted(telemetry))]:
        if reg[key] != value:
            raise ValueError(f'Bench registration differs on {key}.')
    if config['impl'] != 'warp' or config['source_commit'] != git('rev-parse', 'HEAD').decode().strip():
        raise ValueError('Bench engine or source differs from the request.')
    candidate_digest = plan['frozen_inputs']['candidate']['sha256']
    if reg['candidate_config_sha256'] != candidate_digest:
        raise ValueError('The evaluator used a different candidate config.')
    devices = [str(device).lower() for device in config['devices']]
    if smoke and (not devices or not all('cpu' in device for device in devices)):
        raise ValueError('The engineering smoke must run entirely on CPU.')
    summary = read(out / 'summary.json')
    expected = set(itertools.product(families, telemetry, methods))
    groups = {cell: set() for cell in expected}
    with (out / 'episodes.csv').open(newline='') as stream:
        for row in csv.DictReader(stream):
            cell = (row['family'], row['telemetry'], row['method'])
            ids = groups[cell]
            if row['split'] != 'dev' or row['scenario_id'] in ids:
                raise ValueError('Duplicate or non-development episode row.')
            ids.add(row['scenario_id'])
    if set(summary['cells']) != {'|'.join(cell) for cell in expected}:
        raise ValueError('Incomplete summary matrix.')
    for cell, ids in groups.items():
        if len(ids) != episodes or ids != groups[cell[0], telemetry[0], methods[0]]:
            raise ValueError(f'Incomplete pairing for {cell}.')
        if summary['cells']['|'.join(cell)]['n'] != episodes:
            raise ValueError(f'Incorrect summary count for {cell}.')
    read(out / 'actuators.json')
    if not (out / 'report.md').is_file() or not (out / 'report.md').stat().st_size:
        raise ValueError('The bench report is empty.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['check', 'cpu-smoke', 'gpu'])
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    plan = read(ROOT / MANIFEST)
    files = frozen_inputs(plan)
    if args.mode == 'check':
        return
    if not args.out:
        parser.error('--out is required for a run')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    smoke = args.mode == 'cpu-smoke'
    destination = out / 'cpu_smoke' if smoke else out
    if not smoke:
        validate(out / 'cpu_smoke', plan, True)
    if (destination / 'registration.json').exists():
        raise ValueError('This attempt already started; use a fresh run directory.')
    (out / 'manifest.json').write_bytes((ROOT / MANIFEST).read_bytes())
    env = os.environ.copy()
    if smoke:
        env.update(JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='')
    subprocess.run(command(plan, files, destination, smoke), cwd=ROOT, env=env, check=True)
    validate(destination, plan, smoke)


if __name__ == '__main__':
    main()
