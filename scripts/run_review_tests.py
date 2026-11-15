#!/usr/bin/env python3
"""Run the reviewed protocol tests on CPU and retain a machine-readable record."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--require-jax', action='store_true')
    p.add_argument('--out', type=Path)
    args = p.parse_args(argv)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    sys.path.insert(0, str(ROOT))
    record = {'schema_version': 'foresight.review_cpu_tests.v1',
              'python': sys.version, 'require_jax': args.require_jax,
              'test_pattern': 'test_review_*.py'}
    if args.require_jax:
        import jax
        if any(d.platform != 'cpu' for d in jax.devices()):
            raise RuntimeError('Protocol tests must use CPU; set JAX_PLATFORMS=cpu.')
        record.update(jax_version=jax.__version__, devices=[str(d) for d in jax.devices()])
    started = time.monotonic()
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_review_*.py')
    # Publication and verified-stop tests are included in every launch as well.
    for filename in ('test_experiment_publisher.py', 'test_finish_pod_session.py'):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern=filename))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    passed = result.wasSuccessful() and result.testsRun > 0 and (not args.require_jax or not result.skipped)
    record.update(passed=passed, tests_run=result.testsRun,
                  elapsed_seconds=round(time.monotonic()-started, 3),
                  failures=[{'test': t.id(), 'traceback': msg} for t, msg in result.failures],
                  errors=[{'test': t.id(), 'traceback': msg} for t, msg in result.errors],
                  skipped=[{'test': t.id(), 'reason': why} for t, why in result.skipped])
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: record[k] for k in ('passed', 'tests_run', 'elapsed_seconds', 'skipped')}, indent=2))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
