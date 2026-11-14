#!/usr/bin/env python3
"""Restore the two published policies in the existing GPU environment."""
from pathlib import Path
from importlib import metadata
import argparse
import hashlib
import json
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    import jax
    import mujoco_playground
    from foresight.eval import rollout as R

    devices = jax.devices()
    print('Python:', sys.executable, flush=True)
    print('JAX devices:', devices, flush=True)
    if not any(device.platform == 'gpu' for device in devices):
        raise RuntimeError('JAX has no GPU device. The existing venv needs attention; nothing was installed.')
    if Path(R.__file__).resolve() != root / 'foresight/eval/rollout.py':
        raise RuntimeError('Python imported a different Foresight checkout.')
    record = {'python': sys.executable, 'python_version': sys.version,
              'playground': mujoco_playground.__file__, 'devices': [str(d) for d in devices],
              'versions': {}, 'checkpoints': {}}
    for name in ('jax', 'jaxlib', 'brax', 'mujoco', 'mujoco-mjx', 'warp-lang', 'orbax-checkpoint'):
        try:
            record['versions'][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            record['versions'][name] = None
    for label in ('nominal_s1', 'brace2_s1'):
        checkpoint = R.latest_checkpoint(root / 'checkpoints' / label)
        if not (checkpoint / 'manifest.ocdbt').is_file():
            raise RuntimeError(f'Checkpoint data missing: {checkpoint}')
        R.load_policy(checkpoint)
        record['checkpoints'][label] = {
            'path': str(checkpoint),
            'metadata_sha256': hashlib.sha256((checkpoint / 'ppo_network_config.json').read_bytes()).hexdigest(),
        }
        print(f'{label}: restored {checkpoint}', flush=True)
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print('GPU and both checkpoint restores passed.', flush=True)


if __name__ == '__main__':
    main()
