"""Run the registered E20 experiment using existing pod infrastructure."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from finish_pod_session import finalize
from run_session_h import atomic_json, interrupt, run_child, verified_receipt
from session_h_pod import setup_command


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = ROOT / 'logs/e20_attempt.json'
LOG = ROOT / 'logs/e20_stage.log'


def execute(source):
    invocation = uuid.uuid4().hex
    run_id = 'E20_matched_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'results' / run_id
    record = {'run_id': run_id, 'source_commit': source, 'invocation_id': invocation}
    atomic_json(ATTEMPT, record)
    setup_root = Path(os.environ['FORESIGHT_SETUP_ROOT'])
    venv = Path(os.environ.get('FORESIGHT_VENV', str(setup_root / 'mujoco_playground/rss2025/.venv')))
    gpu_python = str(venv / 'bin/python')
    env = dict(os.environ, FORESIGHT_PYTHON=gpu_python)
    deadline = time.monotonic() + 70 * 60
    stage = 'setup'
    code = 0
    with LOG.open('w') as log:
        try:
            command, cap, message = setup_command(setup_root, venv, ROOT / 'logs')
            print(message, flush=True)
            run_child(command, env, log, cap)
            # Existing venv setup already restores both policies and checks JAX/GPU.
            if 'scripts/preflight_session_h.py' not in command:
                stage = 'preflight'
                run_child([gpu_python, 'scripts/preflight_session_h.py', '--out',
                           str(ROOT / 'logs/e20_preflight.json')], env, log, 300)
            stage = 'experiment'
            run_child([gpu_python, 'scripts/run_foresight_matched_duration.py', '--run',
                       '--out', str(out)], env, log, deadline - time.monotonic())
            if not (out / 'verdict.json').is_file():
                raise RuntimeError('E20 returned without recording a verdict.')
        except (Exception, KeyboardInterrupt) as exc:
            code = 124 if isinstance(exc, TimeoutError) else 130 if isinstance(exc, KeyboardInterrupt) else 1
            print(f'{stage} failed: {exc}', file=log, flush=True)
            print(f'{stage} failed: {exc}', flush=True)
    out.mkdir(exist_ok=True, parents=True)
    status = {0: 'completed', 124: 'timed_out', 130: 'failed'}.get(code, 'failed')
    atomic_json(out / 'run.json', dict(record, status=status, exit_code=code, stage=stage))
    for name in ['e20_preflight.json', 'reused_environment_preflight.json']:
        path = ROOT / 'logs' / name
        if path.is_file():
            atomic_json(out / 'preflight.json', json.loads(path.read_text()))
            break
    return record


def publish(record, source):
    run_id = record['run_id']
    out = ROOT / 'results' / run_id
    frozen = json.loads((out / 'run.json').read_text())
    if (record['source_commit'] != source
            or any(frozen.get(k) != record[k] for k in ('run_id', 'source_commit', 'invocation_id'))):
        raise RuntimeError('Recorded attempt, source checkout and frozen run must agree.')
    code = frozen['exit_code']
    os.environ['FORESIGHT_INVOCATION_ID'] = record['invocation_id']
    with LOG.open('a') as log:
        pub = run_child([sys.executable, 'scripts/publish_experiment_results.py',
                         '--run-dir', str(out), '--log', str(LOG), '--exit-code', str(code)],
                        os.environ.copy(), log, 300, check=False)
    if pub:
        raise RuntimeError('Upload failed. Retry with --retry-publish on this same pod; GPU simulation is skipped.')
    receipt = verified_receipt(out / 'publication_receipt.json', run_id, source, code)
    latest = dict(record, schema_version=1, run_dir=str(out), status='completed' if code == 0 else 'failed',
                  experiment_status=frozen['status'], experiment_exit_code=code, publication_exit_code=0,
                  publication_status='verified', publication_receipt=receipt)
    atomic_json(ROOT / 'logs/last_run.json', latest)
    stop_code = finalize()
    return code or stop_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--retry-publish', action='store_true')
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, interrupt)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    if args.run and ATTEMPT.exists():
        parser.error('An attempt already exists. Use --retry-publish for an upload error.')
    record = execute(source) if args.run else json.loads(ATTEMPT.read_text())
    return publish(record, source)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (Exception, KeyboardInterrupt) as exc:
        sys.exit(f'E20 could not finalize: {exc}. Check RunPod; the pod may still be billing.')
