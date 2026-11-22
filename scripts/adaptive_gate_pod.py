#!/usr/bin/env python3
"""Bound setup, a privileged actuator gate, publication, and a verified pod stop.

Operational completion is separate from a scientific pass. This worker never
trains a policy or kills any process other than the child it started.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from run_session_h import atomic_json, interrupt, run_child, verified_receipt
from session_h_pod import setup_command

ROOT = Path(__file__).resolve().parents[1]
CLEANUP_RESERVE = 600


def parse_deadline(environ, now=None):
    """Translate the outer launcher's wall deadline once to a monotonic budget."""
    now = time.time() if now is None else now
    try:
        started = int(environ['ADAPTIVE_STARTED_EPOCH'])
        deadline = int(environ['ADAPTIVE_DEADLINE_EPOCH'])
    except (KeyError, ValueError):
        raise ValueError('Use start_adaptive_gate.sh: a valid original launch deadline is required.') from None
    if not 15 * 60 <= deadline - started <= 120 * 60 or started > now + 5:
        raise ValueError('The attempt must be 15..120 minutes from its original launch time.')
    return started, deadline, deadline - now


def remaining_timeout(deadline, cap, reserve=0):
    remaining = deadline - time.monotonic() - reserve
    if remaining <= 0:
        raise TimeoutError('The original launcher budget is exhausted; no further experiment will start.')
    return min(cap, remaining)


def require_idle_gpu():
    binary = shutil.which('nvidia-smi')
    if not binary:
        raise RuntimeError('nvidia-smi is required; no experiment started.')
    result = subprocess.run([binary, '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            text=True, capture_output=True, timeout=15)
    if result.returncode or result.stdout.strip():
        raise RuntimeError('The GPU is busy or its process list is unavailable. Existing work was left alone.')


def read_strict_json(path):
    def reject(value):
        raise ValueError(f'Non-finite JSON constant in {path.name}: {value}')
    data = json.loads(path.read_text(), parse_constant=reject)
    if not isinstance(data, dict):
        raise ValueError(f'{path.name} must be a JSON object.')
    return data


def validate_outputs(output):
    summary = read_strict_json(output / 'summary.json')
    if summary.get('scientific_gate_status') not in ('pass', 'fail', 'inconclusive'):
        raise ValueError('summary.json lacks a recognized scientific_gate_status.')
    for name in ('episodes.csv', 'report.md'):
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f'The evaluator did not produce {name}.')
    return summary['scientific_gate_status']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('smoke', 'gate', 'calibrate-identity'), default='gate')
    args = parser.parse_args(argv)
    started_epoch, deadline_epoch, seconds_left = parse_deadline(os.environ)
    calibration = args.profile == 'calibrate-identity'
    if calibration and deadline_epoch - started_epoch > 45 * 60:
        raise ValueError('Identity calibration budget must not exceed 45 minutes.')
    deadline = time.monotonic() + seconds_left
    invocation = uuid.uuid4().hex
    os.environ['FORESIGHT_INVOCATION_ID'] = invocation
    os.environ['PYTHONUNBUFFERED'] = '1'
    os.chdir(ROOT)
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    lock = (logs / 'adaptive_gate.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another adaptive gate is active; no work was started.', file=sys.stderr)
        return 1

    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'adaptive_gate_{stamp}_{invocation[:6]}'
    output = ROOT / 'results' / run_id
    output.mkdir(parents=True, exist_ok=False)
    setup_output = output / 'setup'
    setup_output.mkdir()
    record = {'run_id': run_id, 'source_commit': None, 'mode': args.profile,
              'protocol_version': 'adaptive-identity-calibration-v1' if calibration else 'adaptive-servo-gain-gate-v1',
              'status': 'running', 'started_utc': stamp, 'stage': 'setup',
              'launcher_started_epoch': started_epoch, 'launcher_deadline_epoch': deadline_epoch,
              'cleanup_reserve_seconds': CLEANUP_RESERVE,
              'interpretation': 'Known-parameter gain compensation over a frozen G1 walker. '
                  'An operational completion is not evidence that the scientific gate passed. '
                  'No online estimator, lag compensation, or learned residual is tested.'}
    if calibration:
        record['interpretation'] = 'Two repeats of the healthy walker on independent calibration seeds. G0 is undecided.'
    latest = {'schema_version': 1, 'run_id': run_id, 'run_dir': str(output),
              'invocation_id': invocation, 'source_commit': None, 'mode': args.profile,
              'status': 'running', 'experiment_status': 'running',
              'experiment_exit_code': None, 'publication_status': 'pending',
              'publication_exit_code': None, 'publication_receipt': None,
              'archive_status': 'not_created', 'overall_exit_code': None}
    env = os.environ.copy()
    setup_root = Path(env.get('FORESIGHT_SETUP_ROOT', '/workspace')).resolve()
    venv = Path(env.get('FORESIGHT_VENV', str(setup_root / 'mujoco_playground/rss2025/.venv'))).absolute()
    python = str(venv / 'bin/python')
    env['FORESIGHT_PYTHON'] = python
    env['PYTHONPATH'] = str(ROOT) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    env['JAX_DEFAULT_MATMUL_PRECISION'] = 'highest'
    env.pop('MUJOCO_GL', None)
    env.pop('LD_LIBRARY_PATH', None)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    code = 0
    publication_failed = False

    def save_latest():
        latest['updated_utc'] = datetime.now(timezone.utc).isoformat()
        atomic_json(logs / 'last_run.json', latest)

    with (output / 'console.log').open('a') as log:
        def emit(message):
            print(message, flush=True)
            log.write(message + '\n')
            log.flush()

        def note_error(label, error):
            emit(f'{label}: {error}')
            detail = traceback.format_exc()
            print(detail, end='', file=sys.stderr, flush=True)
            log.write(detail)
            log.flush()

        try:
            record['source_commit'] = subprocess.run(
                ['git', 'rev-parse', 'HEAD'], capture_output=True, text=True,
                timeout=15, check=True).stdout.strip()
            latest['source_commit'] = record['source_commit']
            atomic_json(output / 'run.json', record)
            save_latest()
            emit(f'Adaptive gate: {run_id}; profile={args.profile}.')
            emit('The attempt clock includes setup. Ten minutes are reserved for publication and stop checks.')
            remaining_timeout(deadline, 15, CLEANUP_RESERVE)
            require_idle_gpu()
            command, cap, message = setup_command(setup_root, venv, setup_output)
            # Keep environment provenance inside the publisher's allowlisted names.
            if command[1:2] == ['scripts/preflight_session_h.py']:
                command[-1] = str(setup_output / 'preflight.json')
            emit(message)
            run_child(command, env, log, remaining_timeout(deadline, cap, CLEANUP_RESERVE))
            if not (setup_output / 'preflight.json').exists():
                run_child([python, 'scripts/preflight_session_h.py', '--out', str(setup_output / 'preflight.json')],
                          env, log, remaining_timeout(deadline, 240, CLEANUP_RESERVE))
            require_idle_gpu()
            record['stage'] = 'evaluation'
            atomic_json(output / 'run.json', record)
            # ADAPTIVE_DETERMINISTIC=off is the escape hatch if the deterministic Warp kernels fail
            # on a GPU; the evaluator records the requested mode in preflight.json either way.
            determinism = os.environ.get('ADAPTIVE_DETERMINISTIC', 'auto')
            if determinism not in ('auto', 'on', 'off'):
                raise ValueError(f'ADAPTIVE_DETERMINISTIC must be auto, on or off, not {determinism!r}')
            if calibration:
                if determinism != 'auto':
                    raise ValueError('Identity calibration requires the pinned default Warp execution and auto XLA flags.')
                experiment = [python, '-u', 'scripts/calibrate_gate_identity.py', '--gpu-calibration', '--out', str(output)]
            else:
                experiment = [python, '-u', 'scripts/eval_adaptive_gate.py', '--out', str(output), '--profile', args.profile,
                              '--deterministic', determinism]
            run_child(experiment, env, log, remaining_timeout(deadline, 120 * 60, CLEANUP_RESERVE))
            scientific_status = validate_outputs(output)
            if calibration:
                result = read_strict_json(output / 'summary.json')
                if (result.get('profile') != 'gpu_calibration' or result.get('batches') != 64
                        or result.get('rows') != 4096 or scientific_status != 'inconclusive'):
                    raise ValueError('Incomplete calibration or unexpected scientific disposition.')
            record['scientific_gate_status'] = scientific_status
            record['status'] = 'completed'
            emit(f'Pipeline completed; scientific gate={record["scientific_gate_status"]}.')
        except (KeyboardInterrupt, Exception) as error:
            code = 130 if isinstance(error, KeyboardInterrupt) else 124 if isinstance(error, TimeoutError) else 1
            record['status'] = 'interrupted' if code == 130 else 'timed_out' if code == 124 else 'failed'
            record['error'] = str(error)
            note_error('Gate stopped', error)

        record['exit_code'] = code
        record['elapsed_from_launcher_seconds'] = round(time.time() - started_epoch, 3)
        latest.update(experiment_status=record['status'], experiment_exit_code=code,
                      scientific_gate_status=record.get('scientific_gate_status'), status='finalizing')
        atomic_json(output / 'run.json', record)
        if not (output / 'report.md').exists():
            (output / 'report.md').write_text(
                f'# Adaptive execution feasibility gate\n\nOperational status: **{record["status"]}**.\n\n'
                'No complete scientific conclusion is available. Partial evidence and the console log are retained.\n')
        save_latest()
        try:
            command = [sys.executable, 'scripts/publish_experiment_results.py',
                       '--run-dir', str(output), '--log', str(output / 'console.log'), '--exit-code', str(code)]
            pub_code = run_child(command, env, log, remaining_timeout(deadline, 300, 90), check=False)
            latest['publication_exit_code'] = pub_code
            if pub_code:
                raise RuntimeError(f'Publisher exited with status {pub_code}.')
            latest['publication_receipt'] = verified_receipt(
                output / 'publication_receipt.json', run_id, record['source_commit'], code)
            latest['publication_status'] = 'verified'
        except (KeyboardInterrupt, Exception) as error:
            publication_failed = True
            latest['publication_status'] = 'failed'
            latest['publication_exit_code'] = latest['publication_exit_code'] or 1
            latest['publication_error'] = str(error)
            note_error('Publication failed; local evidence is preserved', error)
            emit(f'Retry without GPU work: bash scripts/start_adaptive_gate.sh retry-publish {run_id}')
        overall = code or (2 if publication_failed else 0)
        latest.update(status='completed' if overall == 0 else 'failed', overall_exit_code=overall)
        save_latest()
        emit(f'Operational status: {record["status"]}; scientific gate: {record.get("scientific_gate_status", "not available")}.')
        emit(f'Publication: {latest["publication_status"]}. Results: {output}')

    # The shared finalizer checks the exact invocation, verified upload, current
    # pod ID and idle GPU. A timed-out/failed run may stop after verified upload;
    # an interrupt, stale receipt or failed upload cannot authorize a stop.
    try:
        stop_timeout = remaining_timeout(deadline, 90)
        stop_code = subprocess.run([sys.executable, 'scripts/finish_pod_session.py'],
                                   env=env, timeout=stop_timeout).returncode
    except (TimeoutError, subprocess.TimeoutExpired):
        print('Stop checks exceeded the attempt budget. Check RunPod; billing may continue.', flush=True)
        stop_code = 2
    lock.close()
    return overall or stop_code


if __name__ == '__main__':
    raise SystemExit(main())
