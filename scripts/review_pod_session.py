#!/usr/bin/env python3
"""Prepare/reuse the pod environment, run reviewed experiments, and finalize.

Uses only the standard library until the existing environment's preflight is
started. Setup errors get their own result record and upload attempt. This
worker never trains a policy or requests pod termination.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

from run_review_experiments import atomic_json, interrupt, run_child, verified_receipt


ROOT = Path(__file__).resolve().parents[1]


def setup_command(setup_root: Path, venv: Path, log_dir: Path):
    marker = setup_root / '.foresight-bootstrap/state.json'
    expected = setup_root / 'mujoco_playground/rss2025/.venv'
    if (venv / 'bin/python').is_file() and (venv != expected or not marker.exists()):
        return ([str(venv / 'bin/python'), 'scripts/preflight_review.py',
                 '--out', str(log_dir / 'reused_environment_preflight.json')], 300,
                'Reusing the existing environment; no Python packages will be installed.')
    if venv != expected:
        raise RuntimeError('Custom venv is missing or has a setup marker. Automatic fresh setup uses the default venv only.')
    return (['bash', 'scripts/bootstrap_fresh_pod.sh'], 1500,
            'Installing/verifying the pinned fresh environment; setup limit is 25 minutes.')


def save_setup_failure(run_id, invocation, source, code, error, log_path, elapsed):
    output = ROOT / 'results' / run_id
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(log_path, output / 'console.log')
    state = 'timed_out' if code == 124 else 'interrupted' if code == 130 else 'failed'
    atomic_json(output / 'run.json', {
        'run_id': run_id, 'source_commit': source, 'mode': 'setup',
        'status': state,
        'exit_code': code, 'error': str(error), 'elapsed_seconds': elapsed,
        'interpretation': 'Setup/preflight failed before the reviewed experiments. No experiment result exists.',
    })
    latest = {
        'schema_version': 1, 'run_id': run_id, 'run_dir': str(output),
        'invocation_id': invocation, 'source_commit': source, 'mode': 'setup',
        'status': 'failed', 'experiment_status': state, 'stage': 'setup',
        'experiment_exit_code': code, 'archive_status': 'not_created',
        'publication_status': 'pending', 'publication_exit_code': None,
        'publication_receipt': None, 'overall_exit_code': code,
    }
    atomic_json(ROOT / 'logs/last_run.json', latest)
    with (output / 'console.log').open('a') as log:
        try:
            command = [sys.executable, 'scripts/publish_experiment_results.py',
                       '--run-dir', str(output), '--log', str(output / 'console.log'),
                       '--exit-code', str(code)]
            pub_code = run_child(command, os.environ.copy(), log, 300, check=False)
            latest['publication_exit_code'] = pub_code
            if pub_code:
                raise RuntimeError('Setup-error upload failed; artifacts remain local.')
            latest['publication_receipt'] = verified_receipt(
                output / 'publication_receipt.json', run_id, source, code)
            latest['publication_status'] = 'verified'
        except (KeyboardInterrupt, Exception) as exc:
            latest['publication_status'] = 'failed'
            latest['publication_error'] = str(exc)
            if latest['publication_exit_code'] is None:
                latest['publication_exit_code'] = 1
            print(f'Publication: {exc}', flush=True)
    atomic_json(ROOT / 'logs/last_run.json', latest)
    print(f'Setup failure record: {output}', flush=True)
    return code


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('full', 'smoke'), default='full')
    parser.add_argument('--max-minutes', type=float,
                        help='Experiment budget after setup: full 90, smoke 30 minutes.')
    args = parser.parse_args(argv)
    if args.max_minutes is None:
        args.max_minutes = 30.0 if args.mode == 'smoke' else 90.0
    if not math.isfinite(args.max_minutes) or args.max_minutes < 1:
        parser.error('--max-minutes must be finite and at least 1')
    return args


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    invocation = uuid.uuid4().hex
    os.environ['FORESIGHT_INVOCATION_ID'] = invocation
    os.environ['PYTHONUNBUFFERED'] = '1'
    setup_root = Path(os.environ.get('FORESIGHT_SETUP_ROOT', '/workspace')).resolve()
    venv = Path(os.environ.get('FORESIGHT_VENV', str(setup_root / 'mujoco_playground/rss2025/.venv'))).absolute()
    os.environ['FORESIGHT_PYTHON'] = str(venv / 'bin/python')
    logs = ROOT / 'logs'
    logs.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    run_id = f'setup_{stamp}_{invocation[:6]}'
    log_path = logs / f'{run_id}.log'
    source = None
    code = 0
    started = time.monotonic()
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    try:
        with log_path.open('a') as log:
            try:
                source = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                                        capture_output=True, text=True, check=True).stdout.strip()
                command, budget, message = setup_command(setup_root, venv, logs)
                print(message, flush=True)
                log.write(message + '\n')
                log.flush()
                run_child(command, os.environ.copy(), log, budget)
            except (KeyboardInterrupt, Exception):
                detail = traceback.format_exc()
                log.write(detail)
                log.flush()
                print(detail, file=sys.stderr, end='', flush=True)
                raise
    except (KeyboardInterrupt, Exception) as error:
        code = 130 if isinstance(error, KeyboardInterrupt) else 124 if isinstance(error, TimeoutError) else 1
        save_setup_failure(run_id, invocation, source, code, error, log_path,
                           round(time.monotonic() - started, 3))
    else:
        print(f'Environment ready. Starting the {args.mode} reviewed plan with published '
              f'frozen checkpoints; no PPO training. Budget: {args.max_minutes:g} minutes.', flush=True)
        # The runner owns its process budget, process-group cleanup, and results.
        # If this worker is interrupted, ask the runner to save its failure record
        # and wait for that bounded cleanup before considering a pod stop.
        command = [sys.executable, 'scripts/run_review_experiments.py', '--mode', args.mode,
                   '--max-minutes', str(args.max_minutes)]
        process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy())
        try:
            code = process.wait()
        except KeyboardInterrupt:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=330)
            except subprocess.TimeoutExpired:
                print('Runner cleanup is still active. Pod stop is deferred; inspect logs/fresh_pod.log.', flush=True)
                return 130
            code = 130
    # The finalizer rejects stale invocation IDs and missing/failed receipts.
    finalizer = subprocess.run([sys.executable, 'scripts/finish_pod_session.py'],
                               cwd=ROOT, env=os.environ.copy()).returncode
    return code or finalizer


if __name__ == '__main__':
    raise SystemExit(main())
