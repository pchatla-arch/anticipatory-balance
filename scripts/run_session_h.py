#!/usr/bin/env python3
"""Run bounded Session H depth and execution-hold follow-up experiments.

Uses an existing venv. Writes a new run directory and publishes compact results.
The wall-time budget stops experiment processes; it does not stop pod billing.
"""
from __future__ import annotations

from session_h_plan import make_plan, assert_disjoint_splits, BASE_SOURCE, PROTOCOL

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid


ROOT = Path(__file__).resolve().parents[1]
CURRENT_PROCESS = None


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


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


def run_child(command, env, log, timeout_seconds, *, check=True):
    global CURRENT_PROCESS
    if timeout_seconds <= 0:
        raise TimeoutError('Experiment wall-time budget was reached.')
    message = 'Running: ' + ' '.join(str(a) for a in command)
    print(message, flush=True)
    log.write(message + '\n')
    log.flush()
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
        if result and check:
            raise RuntimeError(f'Child process exited with status {result}. See console.log.')
        return result
    finally:
        if process.poll() is None:
            stop_process(process)
        reader.join(timeout=5)
        CURRENT_PROCESS = None


def interrupt(signum, frame):
    raise KeyboardInterrupt(f'Received signal {signum}')


def verified_receipt(path, run_id, source_commit, experiment_exit_code):
    """Accept only the publisher's verification for this exact run and source."""
    receipt = json.loads(path.read_text())
    commit = receipt.get('commit')
    if (receipt.get('schema_version') != 1 or receipt.get('status') != 'verified'
            or receipt.get('verified') is not True or receipt.get('run_id') != run_id
            or receipt.get('run_source') != source_commit or not source_commit
            or type(receipt.get('run_exit_code')) is not int
            or receipt['run_exit_code'] != experiment_exit_code
            or not isinstance(commit, str) or re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', commit) is None):
        raise RuntimeError('Publication receipt does not verify this run and source revision.')
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('full', 'smoke'), default='full')
    parser.add_argument('--plan', action='store_true', help='Print commands without setup, GPU work, or publication.')
    parser.add_argument('--max-minutes', type=float, default=75)
    parser.add_argument('--no-publish', action='store_true', help='Keep results local for an offline/debug run.')
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_minutes) or args.max_minutes < 1:
        parser.error('--max-minutes must be finite and at least 1')
    if args.plan:
        plan = make_plan(Path('RESULTS'), args.mode)
        assert_disjoint_splits(plan)
        print(json.dumps(plan, indent=2))
        return 0
    (ROOT / 'logs').mkdir(exist_ok=True)
    lock = (ROOT / 'logs/session_h.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another review runner is active. Attach to its tmux session.', file=sys.stderr)
        return 1
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'session_h_{stamp}_{uuid.uuid4().hex[:6]}'
    output = ROOT / 'results' / run_id
    output.mkdir(parents=True, exist_ok=False)
    plan = make_plan(output, args.mode)
    assert_disjoint_splits(plan)
    record = {'run_id': run_id, 'source_commit': None, 'mode': args.mode,
              'max_minutes': args.max_minutes, 'started_utc': stamp, 'status': 'running',
              'stages': [], 'planned_stages': plan, 'protocol_version': PROTOCOL, 'base_source_commit': BASE_SOURCE, 'interpretation': 'Session G closest-approach release retained; same-code paired hold ablation and parked-object depth correction; frozen response and policies; operational gates are not superiority tests.'}
    started = time.monotonic()
    deadline = started + args.max_minutes * 60
    experiment_code = 0
    latest_path = ROOT / 'logs' / 'last_run.json'
    latest = {'schema_version': 1, 'run_id': run_id, 'run_dir': str(output),
              'invocation_id': os.environ.get('FORESIGHT_INVOCATION_ID'),
              'console_log': str(output / 'console.log'), 'source_commit': None,
              'mode': args.mode, 'started_utc': stamp, 'status': 'running',
              'experiment_status': 'running', 'experiment_exit_code': None,
              'archive_status': 'pending', 'archive_path': None,
              'publication_status': 'not_requested' if args.no_publish else 'pending',
              'publication_exit_code': None, 'publication_receipt': None,
              'overall_exit_code': None}
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
    env['PYTHONUNBUFFERED'] = '1'
    env.pop('MUJOCO_GL', None)
    env.pop('LD_LIBRARY_PATH', None)
    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGINT, interrupt)

    with (output / 'console.log').open('a') as log:
        def emit(message, *, error=False):
            print(message, file=sys.stderr if error else sys.stdout, flush=True)
            log.write(message + '\n')
            log.flush()

        def report_exception(label, error):
            emit(f'{label}: {error}', error=True)
            detail = traceback.format_exc()
            print(detail, file=sys.stderr, end='', flush=True)
            log.write(detail)
            log.flush()

        def save_latest():
            latest['updated_utc'] = datetime.now(timezone.utc).isoformat()
            atomic_json(latest_path, latest)

        try:
            atomic_json(output / 'run.json', record)
            save_latest()
            emit(f'Results: {output}\nBudget: {args.max_minutes:g} minutes for experiment processes.')
            emit('The runner does not stop the pod or its billing.')
            record['source_commit'] = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True,
                capture_output=True, check=True).stdout.strip()
            latest['source_commit'] = record['source_commit']
            atomic_json(output / 'run.json', record)
            save_latest()
            python = choose_python()
            if shutil.which('nvidia-smi'):
                busy = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                                      capture_output=True, text=True, timeout=15)
                if busy.returncode == 0 and any(line.strip().isdigit() for line in busy.stdout.splitlines()):
                    raise RuntimeError('A GPU process is already active. Let that experiment finish before starting this runner.')
            cpu_env = env.copy()
            cpu_env['JAX_PLATFORMS'] = 'cpu'
            cpu_env['OMP_NUM_THREADS'] = '1'
            run_child([python, 'scripts/run_session_h_tests.py', '--require-jax',
                       '--out', str(output / 'tests.json')],
                      cpu_env, log, min(300, deadline - time.monotonic()))
            run_child([python, 'scripts/preflight_session_h.py', '--out', str(output / 'preflight.json')],
                      env, log, min(240, deadline - time.monotonic()))
            atomic_json(output / 'plan.json', {'source_commit': record['source_commit'],
                                               'protocol': PROTOCOL, 'stages': plan})
            for spec in plan:
                name = spec['name']
                stage = {**spec, 'status': 'running', 'started_seconds': time.monotonic()-started}
                record['stages'].append(stage)
                atomic_json(output / 'run.json', record)
                run_child([python, '-u', *spec['args']], env, log, deadline - time.monotonic())
                command = [python, 'scripts/validate_session_h.py',
                           '--stage-dir', str(output/name), '--kind', spec['kind'],
                           '--episodes-per-condition', str(spec['episodes_per_condition']),
                           '--conditions', str(spec['conditions']),
                           '--expected-release', str(spec['release_after']),
                           '--source-commit', record['source_commit']]
                run_child(command, env, log, min(120, deadline-time.monotonic()))
                stage['checks'] = json.loads((output/name/'audit.json').read_text())
                stage['elapsed_seconds'] = round(time.monotonic()-started-stage['started_seconds'], 3)
                stage['status'] = 'passed'
                atomic_json(output / 'run.json', record)
                emit(f'{name}: completed and evidence checks passed.')
            record['status'] = 'completed'
        except (KeyboardInterrupt, Exception) as error:
            experiment_code = 130 if isinstance(error, KeyboardInterrupt) else 124 if isinstance(error, TimeoutError) else 1
            record['status'] = 'interrupted' if experiment_code == 130 else 'timed_out' if experiment_code == 124 else 'failed'
            record['error'] = str(error)
            if record['stages'] and record['stages'][-1]['status'] == 'running':
                record['stages'][-1]['status'] = record['status']
            report_exception('Experiment stopped', error)

        # Freeze the experiment record before publishing. Final publication state
        # lives outside that immutable snapshot, in logs/last_run.json and receipt.
        record['elapsed_seconds'] = round(time.monotonic() - started, 3)
        record['exit_code'] = experiment_code
        latest.update(experiment_status=record['status'], experiment_exit_code=experiment_code,
                      experiment_elapsed_seconds=record['elapsed_seconds'], status='finalizing')
        artifact_failed = False
        try:
            atomic_json(output / 'run.json', record)
            save_latest()
        except (KeyboardInterrupt, Exception) as error:
            artifact_failed = True
            latest['record_error'] = str(error)
            report_exception('Could not save final experiment state', error)
        try:
            run_child([choose_python(), 'scripts/summarize_session_h.py', '--run-dir', str(output)],
                      env, log, 120)
        except (KeyboardInterrupt, Exception) as error:
            artifact_failed = True
            report_exception('Postprocessing failed; raw evidence will still be published', error)
        try:
            archive = shutil.make_archive(str(ROOT / 'logs' / run_id), 'zip', root_dir=output)
            latest.update(archive_status='saved', archive_path=archive)
            emit(f'Local review ZIP: {archive}')
        except (KeyboardInterrupt, Exception) as error:
            artifact_failed = True
            latest.update(archive_status='failed', archive_error=str(error))
            report_exception('Review ZIP failed; publication will still be attempted', error)

        publication_failed = False
        if not args.no_publish:
            command = [sys.executable, str(ROOT / 'scripts/publish_experiment_results.py'),
                       '--run-dir', str(output), '--log', str(output / 'console.log'), '--exit-code', str(experiment_code)]
            try:
                # Publication is a separate bounded cleanup task, including when
                # the experiment used its entire requested wall-time budget.
                publication_code = run_child(command, env, log, 300, check=False)
                latest['publication_exit_code'] = publication_code
                if publication_code:
                    raise RuntimeError(f'Publisher exited with status {publication_code}.')
                receipt = verified_receipt(output / 'publication_receipt.json', run_id,
                                           record['source_commit'], experiment_code)
                latest.update(publication_status='verified', publication_receipt=receipt)
            except (KeyboardInterrupt, Exception) as error:
                publication_failed = True
                latest['publication_status'] = 'failed'
                if latest['publication_exit_code'] is None:
                    latest['publication_exit_code'] = 130 if isinstance(error, KeyboardInterrupt) else 124 if isinstance(error, TimeoutError) else 1
                latest['publication_error'] = str(error)
                report_exception('Publication failed', error)
                emit('Results remain local. Retry publication without rerunning GPU work:')
                import shlex
                emit(shlex.join(command))

        overall_code = experiment_code or (2 if publication_failed else 1 if artifact_failed else 0)
        latest.update(status='completed' if overall_code == 0 else 'failed',
                      overall_exit_code=overall_code, elapsed_seconds=round(time.monotonic() - started, 3))
        try:
            save_latest()
        except (KeyboardInterrupt, Exception) as error:
            overall_code = overall_code or 1
            report_exception('Could not save final launcher status', error)
        emit(f'Experiment status: {record["status"]} (exit {experiment_code}).')
        emit(f'Archive status: {latest["archive_status"]}.')
        emit(f'Publication status: {latest["publication_status"]} (exit {latest["publication_exit_code"]}).')
        emit(f'Overall runner exit status: {overall_code}. Status file: {latest_path}')
        emit('Experiment worker finished; the launcher now checks verified upload before requesting pod stop.')
    lock.close()
    return overall_code


if __name__ == '__main__':
    raise SystemExit(main())
