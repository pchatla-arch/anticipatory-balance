"""Reweight archived Foresight episodes across scheduled-hit frequencies. Requires NumPy."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


DATA = {
    'heldout': ('E18h_heldout_p5', '249e92e0ae0c084aff631ed092c78a50f77270d8', (0, 100, 300)),
    'transfer': ('E19_seed2_p5', '1b475315c11838a957263e8c3a06f48f1c5bdef7', (0, 100)),
}
ARMS = {'reactive': 'reactive__p0', 'ttc': 'ttc_threshold__p0.3', 'belief': 'belief_lead__p0.05'}
EXAMPLE_HIT_FREQUENCIES = (0.05, 0.25, 0.75)
EXAMPLE_PREPARATION_PRICE = 0.1
N_BOOT = 10000
SOURCE_COMMIT = 'e4dacfacc1c2330d48fdfabcb3a0f2e933e3117f'


def binary(array, name):
    if array.ndim != 1 or not np.isin(array, (0, 1)).all():
        raise ValueError(f'Expected a binary episode vector: {name}')
    return array.astype(np.int8)


def read_arms(archive, latency):
    arms = {}
    reference = None
    for arm, tag in ARMS.items():
        # Reactive is independent of observations. The evaluator archives one reactive vector.
        key = f"{tag}__lat{0 if arm == 'reactive' else latency // 20}__"
        miss = binary(archive[key + 'is_miss'], key + 'is_miss').astype(bool)
        fell = binary(archive[key + 'fell'], key + 'fell')
        steps = archive[key + 'brace_steps']
        if (steps.shape != miss.shape or fell.shape != miss.shape or
                not np.issubdtype(steps.dtype, np.integer) or (steps < 0).any()):
            raise ValueError('Invalid brace counts or episode shape for ' + key)
        if reference is None:
            reference = miss
        if not np.array_equal(reference, miss):
            raise ValueError('Inconsistent paired scheduled-miss labels for ' + key)
        arms[arm] = np.column_stack((fell, steps > 0)).astype(np.int8)
    if not 0 < int(reference.sum()) < reference.size:
        raise ValueError('Both scheduled strata must be present')
    return reference, arms


def counts(array):
    return {'n': len(array), 'falls': int(array[:, 0].sum()), 'preparations': int(array[:, 1].sum())}


def difference_table(delta):
    values, frequencies = np.unique(delta, axis=0, return_counts=True)
    return [{'fall': int(row[0]), 'preparation': int(row[1]), 'n': int(n)}
            for row, n in zip(values, frequencies)]


def sample_paired_means(table, rng):
    values = np.array([[row['fall'], row['preparation']] for row in table])
    counts_ = np.array([row['n'] for row in table])
    n = int(counts_.sum())
    # Multinomial resampling of identical paired rows equals resampling episodes with replacement.
    return rng.multinomial(n, counts_ / n, size=N_BOOT) @ values / n


def interval(array):
    return [float(value) for value in np.quantile(array, (0.025, 0.975))]


def loss_per_100(hit_rate, price, hit_stats, miss_stats):
    return 100 * (hit_rate * hit_stats[0] + (1 - hit_rate) * (miss_stats[0] + price * miss_stats[1]))


def describe_cell(miss, arms, seed):
    strata = {'hits': ~miss, 'misses': miss}
    raw = {arm: {name: counts(x[mask]) for name, mask in strata.items()} for arm, x in arms.items()}
    delta = arms['ttc'] - arms['belief']
    paired = {name: difference_table(delta[mask]) for name, mask in strata.items()}
    rng = np.random.default_rng(seed)
    hit_boot = sample_paired_means(paired['hits'], rng)
    miss_boot = sample_paired_means(paired['misses'], rng)
    hit_mean, miss_mean = delta[~miss].mean(axis=0), delta[miss].mean(axis=0)
    price = EXAMPLE_PREPARATION_PRICE

    def crossing_at_price(hit_diff, miss_diff):
        # L_TTC - L_belief is affine in prevalence with endpoint values a and b.
        a = miss_diff[..., 0] + price * miss_diff[..., 1]
        b = hit_diff[..., 0]
        crossed = a * b < 0
        ratio = np.divide(a, a - b, out=np.zeros_like(a, dtype=float), where=crossed)
        return crossed, ratio

    crossed, ratio = crossing_at_price(hit_mean, miss_mean)
    boot_cross, boot_ratio = crossing_at_price(hit_boot, miss_boot)
    crossing = {'preparation_price': price, 'hit_frequency': float(ratio) if crossed else None,
                'fraction_bootstrap_draws_with_interior_crossing': float(boot_cross.mean())}
    # A percentile ratio interval is reported only when *all* draws had an interior crossing.
    crossing['paired_95pct_hit_frequency'] = interval(boot_ratio) if boot_cross.all() else None

    examples = []
    empirical_freq = float((~miss).mean())
    for p in (*EXAMPLE_HIT_FREQUENCIES, empirical_freq):
        arm_scores = {name: float(loss_per_100(p, price, x[~miss].mean(axis=0), x[miss].mean(axis=0)))
                      for name, x in arms.items()}
        draws = 100 * (p * hit_boot[:, 0] + (1 - p) * (miss_boot[:, 0] + price * miss_boot[:, 1]))
        diff = arm_scores['ttc'] - arm_scores['belief']
        # An interior price at this prevalence exists only for a positive solution.
        slope = (1 - p) * miss_mean[1]
        intercept = p * hit_mean[0] + (1 - p) * miss_mean[0]
        price_root = -intercept / slope if slope != 0 else None
        examples.append({
            'hit_frequency': p, 'near_miss_preparation_price': price, 'loss_per_100': arm_scores,
            'ttc_minus_belief_loss_per_100': diff, 'paired_95pct': interval(draws),
            'ttc_belief_break_even_positive_price': float(price_root) if price_root is not None and price_root > 0 else None,
        })
    return {'arms': raw, 'paired_differences': paired, 'crossing': crossing, 'illustrations': examples}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=root / 'results')
    parser.add_argument('--output', type=Path, default=root / 'analysis/operating_conditions/operating_conditions.json')
    opts = parser.parse_args()
    record = {
        'schema_version': 1, 'source_commit': SOURCE_COMMIT,
        'estimand': 'Fall probability per encounter plus price times probability of any recorded brace on a scheduled near-miss.',
        'assumption': 'Scenario-type frequency may change. Conditional distributions and frozen controller behavior within each scenario type are held fixed.',
        'units': 'Price in fall-cost units per near-miss preparation. Reported loss scaled to 100 encounters.',
        'bootstrap': {'replicates': N_BOOT, 'base_seed': 228350, 'method': 'paired stratified percentile, conditional on the two observed strata, controller checkpoints and fixed parameter settings'},
        'splits': {},
    }
    for i, (key, (directory, expected, lags)) in enumerate(DATA.items()):
        path = opts.results / directory / 'episodes.npz'
        body = path.read_bytes()
        actual = hashlib.sha1(f'blob {len(body)}\0'.encode('ascii') + body).hexdigest()
        if actual != expected:
            raise ValueError(f'{path} differs from the registered Git blob {expected}')
        with np.load(path, allow_pickle=False) as archive:
            cells = {}
            for lag in lags:
                mask, arms = read_arms(archive, lag)
                cells[str(lag)] = describe_cell(mask, arms, 228350 + 1000 * i + lag)
        record['splits'][key] = {'directory': directory, 'git_blob': expected, 'cells': cells}
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    opts.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
