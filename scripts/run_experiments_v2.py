#!/usr/bin/env python3
"""Run a bounded checkpoint smoke test, then an optional rate/latency pilot.

Uses an existing venv. Writes a new run directory and publishes compact results.
The wall-time budget stops experiment processes; it does not stop pod billing.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROCESS = None


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_stage(path: Path, episodes: int, expected_conditions: int):
    """Validate execution evidence, not statistical superiority of either rule."""
    summary = json.loads((path / 'summary.json').read_text())
    config = json.loads((path / 'config.json').read_text())
    batches = json.loads((path / 'batches.json').read_text())
    if summary.get('status') != 'complete' or config.get('status') != 'complete':
        raise RuntimeError(f'{path.name}: computation did not finish')
    if batches['completed_condition_batches'] != expected_conditions:
        raise RuntimeError(f'{path.name}: incomplete condition batches')
    rows = summary['rows']
    if len(rows) != expected_conditions:
        raise RuntimeError(f'{path.name}: expected {expected_conditions} conditions, found {len(rows)}')
    if summary['episodes_per_condition'] != episodes:
        raise RuntimeError(f'{path.name}: incorrect scenario count')
    for row in rows:
        if row['n_total'] != episodes:
            raise RuntimeError(f'{path.name}: incomplete condition')
        for key in ('fall_rate_hits', 'contact_rate_hits', 'false_prep_rate', 'progress_m', 'braced_s'):
            if not math.isfinite(row[key]):
                raise RuntimeError(f'{path.name}: nonfinite {key}')
        for key in ('fall_rate_hits', 'contact_rate_hits', 'false_prep_rate'):
            if not 0 <= row[key] <= 1:
                raise RuntimeError(f'{path.name}: {key} outside [0, 1]')
    reactive = [r for r in rows if r['method'] == 'reactive']
    if not reactive or reactive[0]['contact_rate_hits'] <= 0:
        raise RuntimeError(f'{path.name}: no reactive scheduled-hit contact was observed')
    csv_paths = sorted((path / 'episodes').glob('*.csv'))
    if len(csv_paths) != expected_conditions:
        raise RuntimeError(f'{path.name}: missing per-condition episode tables')
    sensing_conditions = 0
    for source in csv_paths:
        with source.open(newline='') as handle:
            records = list(csv.DictReader(handle))
        if len(records) != episodes:
            raise RuntimeError(f'{source.name}: incorrect episode row count')
        if records[0]['method'] in ('ttc_threshold', 'belief_commit'):
            sensing_conditions += 1
            if sum(float(r.get('n_detections_until_termination', 0)) for r in records) <= 0:
                raise RuntimeError(f'{source.name}: no valid detections were delivered')
    if not sensing_conditions:
        raise RuntimeError(f'{path.name}: no sensing conditions were executed')
    return {'passed': True, 'conditions': len(rows), 'episodes_per_condition': episodes,
            'period_steps': config['period'], 'note': 'Operational checks only; not a scientific superiority gate.'}


def choose_python():
    explicit = os.environ.get('FORESIGHT_PYTHON')
    if explicit:
        candidate = Path(explicit).expanduser()
    else:
        venv = Path(os.environ.get('FORESIGHT_VENV', '/workspace/mujoco_playground/rss2025/.venv'))
        candidate = venv / 'bin/python'
        if not candidate.is_file() and os.environ.get('VIRTUAL_ENV'):
            candidate = Path(os.environ['VIRTUAL_ENV']) / 'bin/python'
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        raise RuntimeError('Existing Playground Python not found. Set FORESIGHT_VENV to the working rss2025 venv. No packages were installed.')
    return str(candidate.absolute())


def stop_process(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def run_child(command, env, log, timeout_seconds):
    global CURRENT_PROCESS
    if timeout_seconds <= 0:
        raise TimeoutError('Experiment wall-time budget was reached.')
    print('Running:', ' '.join(str(a) for a in command), flush=True)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1,
                               start_new_session=True)
    CURRENT_PROCESS = process

    def forward():
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)

    reader = threading.Thread(target=forward, daemon=True)
    reader.start()
    try:
        try:
            result = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            stop_process(process)
            raise TimeoutError('Stage timed out; completed condition files remain available.') from None
        if result:
            raise RuntimeError(f'Child process exited with status {result}. See console.log.')
    finally:
        if process.poll() is None:
            stop_process(process)
        reader.join(timeout=5)
        CURRENT_PROCESS = None


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('smoke', 'pilot'), default='pilot')
    parser.add_argument('--max-minutes', type=float, default=45)
    parser.add_argument('--no-publish', action='store_true', help='Keep results local for an offline/debug run.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_minutes) or args.max_minutes < 1:
        parser.error('--max-minutes must be finite and at least 1')
    (ROOT / 'logs').mkdir(exist_ok=True)
    lock = (ROOT / 'logs/experiments_v2.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another experiments-v2 runner is active. Attach to its tmux session.', file=sys.stderr)
        return 1
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'pod_v2_{stamp}_{uuid.uuid4().hex[:6]}'
    output = ROOT / 'results' / run_id
    output.mkdir(parents=True, exist_ok=False)
    revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                              capture_output=True, check=True).stdout.strip()
    record = {'run_id': run_id, 'source_commit': revision, 'mode': args.mode,
              'max_minutes': args.max_minutes, 'started_utc': stamp, 'status': 'running',
              'stages': [], 'interpretation': 'Operational smoke and exploratory pilot; synthetic detections; one frozen policy pair.'}
    atomic_json(output / 'run.json', record)
    print(f'Results: {output}\nBudget: {args.max_minutes:g} minutes for experiment processes.', flush=True)
    print('The runner does not stop the pod or its billing.', flush=True)
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    status = 0
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
    env['PYTHONUNBUFFERED'] = '1'
    env.pop('MUJOCO_GL', None)
    env.pop('LD_LIBRARY_PATH', None)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    try:
        python = choose_python()
        if shutil.which('nvidia-smi'):
            busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                  capture_output=True, text=True, timeout=15)
            if busy.returncode == 0 and any(line.strip().isdigit() for line in busy.stdout.splitlines()):
                raise RuntimeError('A GPU process is already active. Let that experiment finish before starting this runner.')
        with (output / 'console.log').open('a') as log:
            for pattern in ('test_detector_cadence.py', 'test_frontier_output.py'):
                run_child([python, '-m', 'unittest', 'discover', '-s', 'tests', '-p', pattern, '-v'],
                          env, log, min(180, deadline - time.monotonic()))
            run_child([python, 'scripts/preflight_experiments_v2.py', '--out', str(output / 'preflight.json')],
                      env, log, min(240, deadline - time.monotonic()))
            stages = [('smoke', 32, 2.5, '0.3', '0.2', 6)]
            if args.mode == 'pilot':
                stages += [(f'pilot_{hz}hz', 128, period, '0.2,0.6', '0.05,0.2', 10)
                           for hz, period in ((10, 5.0), (20, 2.5), (50, 1.0))]
            for name, episodes, period, thresholds, costs, condition_count in stages:
                stage = {'name': name, 'status': 'running', 'period_steps': period, 'episodes': episodes}
                record['stages'].append(stage)
                atomic_json(output / 'run.json', record)
                command = [python, '-u', 'scripts/eval_decision_frontier.py',
                           '--nominal', 'checkpoints/nominal_s1', '--brace', 'checkpoints/brace2_s1',
                           '--impl', 'warp', '--episodes', str(episodes), '--batches', '1', '--steps', '300',
                           '--period', str(period), '--latencies', '0,15', '--seed', '400',
                           '--methods', 'reactive,oracle,ttc_threshold,belief_commit',
                           '--thetas', thresholds, '--c_preps', costs, '--out', str(output / name)]
                run_child(command, env, log, min(600 if name == 'smoke' else 1200,
                                                deadline - time.monotonic()))
                stage['checks'] = validate_stage(output / name, episodes, condition_count)
                stage['status'] = 'passed'
                atomic_json(output / 'run.json', record)
                print(f'{name}: operational checks passed.', flush=True)
        record['status'] = 'completed'
    except (KeyboardInterrupt, Exception) as error:
        status = 130 if isinstance(error, KeyboardInterrupt) else 124 if isinstance(error, TimeoutError) else 1
        record['status'] = 'interrupted' if status == 130 else 'timed_out' if status == 124 else 'failed'
        record['error'] = str(error)
        if record['stages'] and record['stages'][-1]['status'] == 'running':
            record['stages'][-1]['status'] = record['status']
        print(f'Stopped: {error}', file=sys.stderr, flush=True)
    finally:
        record['elapsed_seconds'] = round(time.monotonic() - started, 3)
        record['exit_code'] = status
        atomic_json(output / 'run.json', record)
        archive = shutil.make_archive(str(ROOT / 'logs' / run_id), 'zip', root_dir=output)
        print(f'Local review ZIP: {archive}', flush=True)
        if not args.no_publish:
            command = [sys.executable, str(ROOT / 'scripts/publish_experiment_results.py'),
                       '--run-dir', str(output), '--log', str(output / 'console.log'), '--exit-code', str(status)]
            publication = subprocess.run(command, cwd=ROOT)
            if publication.returncode:
                print('Results remain local. Retry publication without rerunning GPU work:', flush=True)
                import shlex
                print(shlex.join(command), flush=True)
                if status == 0:
                    status = 2
        print(f'Experiment exit status: {record["exit_code"]}. The pod remains running; stop it when finished.', flush=True)
    return status


if __name__ == '__main__':
    raise SystemExit(main())
