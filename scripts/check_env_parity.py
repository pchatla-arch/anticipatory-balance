#!/usr/bin/env python3
"""Refuse to call a CPU check valid unless the interpreter matches the pod's pinned versions.

The pod installs the MuJoCo Playground `rss2025` checkout with `uv sync --frozen`, so its
package versions are exactly the ones in that checkout's `uv.lock`. A CPU check run on a
different MuJoCo, MJX, Warp or JAX is a check of different software (AE-009: a determinism
option that exists in MuJoCo-Warp 4.0.1 was verified on CPU and crashed the pod's 3.16.0).

  python scripts/check_env_parity.py --lock /path/to/rss2025/uv.lock
  python scripts/check_env_parity.py --lock ... --json parity.json   # also write a record

Exit status 0 when every watched package matches, 1 otherwise. Run it as the first line of
every CPU checklist and store its JSON next to the evidence it protects.
"""
from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
import re
import sys

WATCHED = ('mujoco', 'mujoco-mjx', 'warp-lang', 'jax', 'jaxlib', 'brax', 'flax', 'optax',
           'orbax-checkpoint', 'numpy', 'ml-collections', 'etils')


def locked_versions(lock_path):
    """Parse `[[package]]` name/version pairs from a uv.lock without a TOML dependency."""
    text = Path(lock_path).read_text()
    versions = {}
    for block in text.split('[[package]]')[1:]:
        name = re.search(r'^name = "([^"]+)"', block, re.M)
        version = re.search(r'^version = "([^"]+)"', block, re.M)
        if name and version:
            versions[name.group(1).lower()] = version.group(1)
    if not versions:
        raise SystemExit(f'No packages parsed from {lock_path}')
    return versions


def compare(locked, watched=WATCHED):
    rows = []
    for name in watched:
        want = locked.get(name.lower())
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        rows.append({'package': name, 'pod_lock': want, 'installed': have,
                     'match': want is not None and have == want})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True, help="the pod's rss2025 uv.lock")
    parser.add_argument('--json', type=Path, help='write the comparison here as well')
    parser.add_argument('--watch', default=','.join(WATCHED), help='comma-separated package names')
    args = parser.parse_args(argv)
    rows = compare(locked_versions(args.lock), tuple(p for p in args.watch.split(',') if p))
    ok = all(r['match'] for r in rows)
    record = {'lock': str(args.lock), 'python': sys.version.split()[0], 'executable': sys.executable,
              'parity': ok, 'packages': rows}
    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + '\n')
    width = max(len(r['package']) for r in rows)
    for r in rows:
        flag = 'ok  ' if r['match'] else 'DIFF'
        print(f"{flag} {r['package']:<{width}}  pod {r['pod_lock']!s:<10} here {r['installed']!s}")
    print('environment parity with the pod lock:', 'OK' if ok else 'FAILED; CPU checks in this interpreter do not stand for the pod')
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
