#!/usr/bin/env python3
"""CPU-only API reproduction for the b215b9c G1 gate; writes no rollout data.

Use the prepared Foresight Python environment with a separate b215b9c source checkout:
  python docs/adaptive_execution/reviews/probe_warp_determinism_api.py --source /path/to/b215b9c --out /tmp/warp-probe.json
"""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def capture(operation):
    try:
        operation()
    except (ImportError, AttributeError) as exc:
        # Remove machine-specific absolute package location while preserving API and type.
        return {'status': 'error', 'type': type(exc).__name__, 'message': str(exc).split(' (')[0]}
    return {'status': 'ok'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path.cwd())
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    expected = {
        'scripts/eval_adaptive_gate.py': 'd830aebe797a2a241e604e3ac3a586fc8509db6a',
        'scripts/check_warp_reproducibility.py': '2ee7e9fe5b9aa7ac5163331fbe65be578dc2620f',
        'foresight/adaptive/actuator_env.py': 'f2883ac670248fffad019529a14b27d8b29905ee',
    }
    if any(git_blob(source / name) != blob for name, blob in expected.items()):
        parser.error('Pass the evaluator, companion checker and G1 environment at commit b215b9c.')
    os.environ['JAX_PLATFORMS'] = 'cpu'
    sys.path[:0] = [str(source), str(source / 'scripts')]
    import jax
    import warp
    import eval_adaptive_gate as gate
    from foresight.adaptive.actuator_env import AdaptiveJoystick, default_config
    from mujoco.mjx.third_party.mujoco_warp._src import types

    if any(device.platform != 'cpu' for device in jax.devices()):
        raise RuntimeError('Expected CPU JAX devices only.')
    warp.set_device('cpu')
    cfg = default_config()
    cfg.impact.resample_on_autoreset = False
    cfg.impact.miss_fraction = 0.
    cfg.impact.reset_perturb = 1.
    cfg.push_config.enable = False
    cfg.episode_length = gate.HORIZON
    env = AdaptiveJoystick(config=cfg, config_overrides={'impl': 'warp'})
    model = env.mjx_model
    on = capture(lambda: gate.deterministic_model(model, 'warp'))
    # L357 of the gate and L171-L172 of its checker read exactly this option
    # even in "off"/"default". This isolates that operation from policy restore.
    default_read = capture(lambda: int(model.opt._impl.deterministic))
    record = {
        'scope': 'CPU G1 model construction and exact API operations before rollout',
        'reviewed_source_commit': 'b215b9c4b5d173a83cfb8df695e5a0f7c1636c5d',
        'verified_source_blobs': expected,
        'packages': {p: metadata.version(p) for p in ('mujoco', 'mujoco-mjx', 'warp-lang', 'jax', 'jaxlib')},
        'jax_platform': jax.default_backend(), 'warp_device': str(warp.get_device()),
        'model_actuators': int(model.nu), 'model_option_type': type(model.opt._impl).__qualname__,
        'physics_backend': 'warp', 'has_DeterminismType': hasattr(types, 'DeterminismType'),
        'has_model_option_deterministic': hasattr(model.opt._impl, 'deterministic'),
        'auto_enabled_smoke_l326': on, 'off_l357_and_checker_default_l171': default_read,
        'rollout_count': 0,
    }
    args.out.write_text(json.dumps(record, indent=2) + '\n')
    print(f'Wrote {args.out}')


if __name__ == '__main__':
    main()
