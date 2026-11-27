#!/usr/bin/env python3
"""CPU-only reanalysis of four saved S1a NPZ traces; never runs physics.

Requires Python 3 and numpy. Pass the published bench_dev directory:
  python reanalyze_s1a_segments.py --results-root /path/to/bench_dev --output segment_audit.json

Source: pchatla-arch/anticipatory-balance, result commit
896000b68aa9c7ea6efcd33ea33b4973ed629bb0,
results/g1_course_dev_20261127T032240Z_5fc7dd/bench_dev/.
Only healthy/none and gain_070/{none,oracle,heuristic_gate} are inspected.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

DT = 0.02
CASES = [('healthy', 'none'), ('gain_070', 'none'),
         ('gain_070', 'oracle'), ('gain_070', 'heuristic_gate')]

def fingerprints(path):
    raw = path.read_bytes()
    return {'sha256': hashlib.sha256(raw).hexdigest(),
            'git_blob_sha1': hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()}

def reanalyze(root):
    with (root / 'episodes.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    output = {
        'analysis': 'CPU reanalysis of existing saved traces; no new physics experiment',
        'result_commit': '896000b68aa9c7ea6efcd33ea33b4973ed629bb0',
        'physics_source_commit': '89b6d0dd31020f00032330c76684c727b1f697a0',
        'source_directory': 'results/g1_course_dev_20261127T032240Z_5fc7dd/bench_dev',
        'csv_fingerprints': fingerprints(root / 'episodes.csv'),
        'dt_s': DT,
        'caveats': [
            'Two paired scenario IDs per family; descriptive development observations only.',
            'NPZ velocity is local pelvis planar velocity after each simulation step.',
            'Integrated logged forward velocity is an approximation to forward travel.',
            'CSV progress_m sums actual planar displacement projected on the normalized previous pelvis heading after onset; it is not goal success.',
            'NPZ has commanded yaw rate but no measured yaw-rate series.',
            'Healthy scenarios have different commands/scenario IDs from gain cases and are contextual, not matched controls.',
        ],
        'cases': [],
    }
    for family, method in CASES:
        name = f'trace_{family}_declared_{method}.npz'
        path = root / name
        with np.load(path, allow_pickle=False) as data:
            c, v = data['command'], data['velocity']
            assert c.shape == (400, 2, 3) and v.shape == (400, 2, 2), name
            selected = [r for r in rows if r['family'] == family and r['method'] == method and r['telemetry'] == 'declared']
            assert len(selected) == 2, name
            scenario_output = []
            for env_idx in range(2):
                matches = [r for r in selected if r['scenario_id'].endswith(f'env{env_idx:05d}')]
                assert len(matches) == 1, name
                row = matches[0]
                onset = int(data['onset'][env_idx])
                assert onset == int(row['onset_step'])
                calculated_loss = float(data['loss'][onset:, env_idx].mean())
                reported_loss = float(row['postfault_loss'])
                assert abs(calculated_loss - reported_loss) < 1e-7, name
                cuts = np.r_[0, np.flatnonzero(np.any(c[1:, env_idx] != c[:-1, env_idx], axis=1)) + 1, len(c)]
                assert cuts.tolist() == [0, 100, 200, 275, 400], name
                phases = []
                for phase, (start, end) in zip(['walk', 'turn', 'stop', 'resume'], zip(cuts[:-1], cuts[1:])):
                    cmd, vel = c[start:end, env_idx], v[start:end, env_idx]
                    err = vel - cmd[:, :2]
                    phases.append({
                        'phase': phase, 'first_tick': int(start), 'end_tick_exclusive': int(end),
                        'n_ticks': int(end - start), 'duration_s': float((end - start) * DT),
                        'command_vx_vy_yawrate': cmd[0].tolist(),
                        'mean_vx_m_s': float(vel[:, 0].mean()),
                        'mean_vy_m_s': float(vel[:, 1].mean()),
                        'signed_forward_bias_m_s': float(err[:, 0].mean()),
                        'forward_mae_m_s': float(np.abs(err[:, 0]).mean()),
                        'forward_rmse_m_s': float(np.sqrt(np.square(err[:, 0]).mean())),
                        'planar_velocity_rmse_m_s': float(np.sqrt(np.square(err).sum(axis=1).mean())),
                        'approx_forward_travel_from_logged_velocity_m': float(vel[:, 0].sum() * DT),
                    })
                scenario_output.append({
                    'scenario_id': row['scenario_id'], 'environment_index': env_idx,
                    'onset_tick': onset, 'alive_ticks': int(data['alive'][:, env_idx].sum()),
                    'reported_postfault_loss': reported_loss, 'recomputed_postfault_loss': calculated_loss,
                    'loss_absolute_difference': abs(calculated_loss - reported_loss),
                    'reported_progress_m': float(row['progress_m']), 'phases': phases,
                })
            output['cases'].append({'family': family, 'method': method, 'trace': name,
                                    'fingerprints': fingerprints(path), 'scenarios': scenario_output})
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(reanalyze(args.results_root), indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end='')
