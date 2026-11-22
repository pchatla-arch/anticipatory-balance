"""Rebuild the reviewed Foresight outcome supplement from fingerprinted archives."""
import argparse
import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DT = 0.020
REACTION = 'reactive__p0__lat0'
THRESHOLD = 'ttc_threshold__p0.3__lat5'
BELIEF = 'belief_lead__p0.05__lat5'
FIELDS = ('fell', 'contact', 'is_miss', 'brace_steps', 'progress_m')
NA = 'NA'


def fresh(blob, payload):
    h = hashlib.sha1(b'blob %d\0' % len(payload) + payload).hexdigest()
    if h != blob:
        raise ValueError(f'Archive fingerprint {h} differs from {blob}')


def write_table(directory, name, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (directory / name).open('w', newline='') as f:
        w = csv.DictWriter(f, keys, restval=NA, lineterminator='\n')
        w.writeheader()
        for row in rows:
            w.writerow({k: f'{x:.9g}' if isinstance(x, float) else x for k, x in row.items()})


def average(values):
    return float(np.mean(values, dtype=np.float64))


def summarize(info, values, mask, stratum):
    f, c, _, b, p = (np.asarray(values[k])[mask] for k in FIELDS)
    n = len(f)
    if not (n and np.isfinite(p).all() and np.isfinite(b).all() and (b >= 0).all()
            and np.isin(f, [False, True]).all() and np.isin(c, [False, True]).all()):
        raise ValueError(f'Invalid archived outcomes in {info}')
    f, c = f.astype(bool), c.astype(bool)
    return dict(info, stratum=stratum, n=n, contact_n=int(c.sum()), failure_n=int(f.sum()),
                contact_and_failure_n=int((c & f).sum()), no_contact_and_failure_n=int((~c & f).sum()),
                failure_given_contact=(average(f[c]) if c.any() else NA),
                prepared_n=int((b > 0).sum()), recorded_brace_s=DT * average(b),
                progress_mean_m=average(p))


def records(info, values, no_projectile=False):
    miss = np.asarray(values['is_miss'])
    if not np.isin(miss, [False, True]).all():
        raise ValueError('Invalid scheduled-miss label')
    miss = miss.astype(bool)
    if any(np.asarray(v).shape != miss.shape for v in values.values()) or miss.ndim != 1:
        raise ValueError('Inconsistent shape among episode fields')
    if no_projectile:
        return [summarize(info, values, np.ones_like(miss), 'no_projectile')]
    groups = [('all', np.ones_like(miss)), ('scheduled_hit', ~miss), ('scheduled_miss', miss)]
    return [summarize(info, values, use, label) for label, use in groups if use.any()]


def npz_rows(path):
    arms = {}
    rows = []
    with np.load(path, allow_pickle=False) as data:
        name = path.parent.name
        for key in data.files:
            if not key.endswith('__fell'):
                continue
            prefix = key.removesuffix('__fell')
            a, b, c = prefix.split('__')
            if name.startswith('E17b'):
                source, method, parameter = a, b, float(c[1:])
                latency = int(name.split('lat')[-1])
            else:
                method, parameter, latency = a, float(b[1:]), int(c[3:]) * 20
                source = {'reactive': 'none', 'oracle': 'privileged'}.get(method, 'parametric')
            values = {k: data[prefix + '__' + k] for k in FIELDS}
            if arms and not np.array_equal(next(iter(arms.values()))['is_miss'], values['is_miss']):
                raise ValueError(f'Scenario-label ordering differs in {name}')
            arms[prefix] = values
            info = dict(archive=name, source=source, controller=method, parameter=parameter,
                        latency_ms=(NA if source in ('none', 'privileged') else latency))
            rows.extend(records(info, values))
    return rows, arms


def csv_arms(path):
    name = path.parent.name
    cells = defaultdict(lambda: defaultdict(list))
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if name.startswith('E2'):
                cell = ('reaction', NA, NA, float(row['mass']), float(row['speed']))
            else:
                mode = 'zero_command' if row.get('mode_brace') == '0' else 'brace'
                cell = (mode, float(row['lead']), int(row.get('launch', 1)), NA, NA)
            values = cells[cell]
            for key in FIELDS:
                raw = row['brace_steps_total' if key == 'brace_steps' else key]
                values[key].append(float(raw) if key == 'progress_m' else int(raw))
    rows = []
    for (mode, lead, enabled, mass, speed), values in cells.items():
        info = dict(archive=name, source='none' if mode == 'reaction' else 'privileged_timing',
                    controller=mode, parameter=lead, latency_ms=NA,
                    launch_enabled=enabled, grid_mass_kg=mass, grid_speed_m_s=speed)
        rows.extend(records(info, values, no_projectile=enabled == 0))
    return rows


def aggregate_rows(path):
    doc = json.loads(path.read_text())
    name = path.parent.name
    rows = []
    for row in doc['rows']:
        # E17b archives contain the individual sensor arms. The reference has summary metrics only.
        if name.startswith('E17b') and row['method'] != 'reactive':
            continue
        source = row.get('source', 'parametric')
        method = row['method']
        if name.startswith('E14_'):
            method = 'object_aware_PPO'
        elif method in ('reactive', 'oracle'):
            source = 'none' if method == 'reactive' else 'privileged'
        latency_steps = row.get('latency_steps', doc.get('extra_latency'))
        out = dict(archive=name, source=source, controller=method, parameter=row['param'],
                   latency_ms=(NA if source in ('none', 'privileged') else 20 * latency_steps),
                   observation_period_ms=20 * doc['period'], n_total=row['n_total'],
                   n_scheduled_hit=NA, n_scheduled_miss=NA, n_contacts_on_scheduled_hit=NA,
                   contact_rate_on_scheduled_miss=NA)
        for field in ('contact_rate_hits', 'fall_rate_hits', 'fall_rate_miss', 'fall_rate_given_contact',
                      'false_prep_rate', 'braced_s', 'progress_m'):
            value = row.get(field)
            out[field] = float(value) if value is not None and math.isfinite(value) else NA
        rows.append(out)
    return rows


def mixture_summary(archives):
    dev = archives['E18_rule_family_p5']
    length = lambda arm: average(arm['brace_steps'])
    w = (length(dev[BELIEF]) - length(dev[REACTION])) / (length(dev[THRESHOLD]) - length(dev[REACTION]))
    if not 0 <= w <= 1:
        raise ValueError('Unattainable randomized time match')
    rows = []
    for name in ('E18h_heldout_p5', 'E19_seed2_p5'):
        a = archives[name]
        miss = a[BELIEF]['is_miss'].astype(bool)
        for label, use in [('all', np.ones_like(miss)), ('scheduled_hit', ~miss), ('scheduled_miss', miss)]:
            def outcomes(arm):
                f, c, _, b, p = (a[arm][k][use] for k in FIELDS)
                return {'contact_probability': average(c), 'failure_probability': average(f),
                        'joint_failure_contact_probability': average(f & c),
                        'prepared_probability': average(b > 0), 'recorded_brace_s': DT*average(b),
                        'progress_mean_m': average(p)}
            t, r = outcomes(THRESHOLD), outcomes(REACTION)
            mix = {k: w*t[k] + (1-w)*r[k] for k in t}
            for policy, rates in [('mixture_expectation', mix), ('lead_belief', outcomes(BELIEF))]:
                denom = rates['contact_probability']
                rows.append(dict(archive=name, controller=policy, stratum=label, n=int(use.sum()),
                                 mixture_ttc_probability=w if policy == 'mixture_expectation' else NA,
                                 **rates, failure_given_contact=(rates['joint_failure_contact_probability']/denom
                                                                 if denom else NA)))
    return rows


def paired_comparisons(archives):
    rows = []
    for name, arms in archives.items():
        contrasts = []
        if name.startswith('E17b'):
            for arm in arms:
                if arm.startswith('depth__'):
                    contrasts.append((arm, arm.replace('depth__', 'parametric__', 1)))
        else:
            for arm in arms:
                if arm.startswith('belief_lead__p0.05__'):
                    other = 'ttc_threshold__p0.3__' + arm.split('__')[2]
                    if other in arms:
                        contrasts.append((arm, other))
        for treatment, baseline in contrasts:
            left, right = arms[treatment], arms[baseline]
            miss = left['is_miss'].astype(bool)
            def metrics(arm):
                return {'contact_probability': arm['contact'], 'failure_probability': arm['fell'],
                        'prepared_probability': arm['brace_steps'] > 0,
                        'recorded_brace_s': DT*arm['brace_steps'], 'progress_m': arm['progress_m']}
            a, b = metrics(left), metrics(right)
            for label, use in [('all', np.ones_like(miss)), ('scheduled_hit', ~miss), ('scheduled_miss', miss)]:
                for k in a:
                    d = a[k][use].astype(float) - b[k][use].astype(float)
                    n = len(d)
                    half = 1.96 * float(d.std(ddof=1)) / math.sqrt(n)
                    rows.append(dict(archive=name, arm_a=treatment, arm_b=baseline, stratum=label,
                                     n=n, metric=k, a_minus_b=average(d),
                                     paired_ci95_low=average(d)-half, paired_ci95_high=average(d)+half,
                                     a_higher_n=int((d > 0).sum()), b_higher_n=int((d < 0).sum())))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=ROOT)
    parser.add_argument('--out', type=Path, default=ROOT / 'analysis/controller_outcomes')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = ROOT / 'analysis/controller_outcomes/sources.sha1'
    exact, historic, archives = [], [], {}
    for entry in manifest.read_text().splitlines():
        blob, rel = entry.split(maxsplit=1)
        path = args.repo / rel
        fresh(blob, path.read_bytes())
        if path.suffix == '.npz':
            rows, arms = npz_rows(path)
            exact.extend(rows)
            archives[path.parent.name] = arms
        elif path.suffix == '.csv':
            exact.extend(csv_arms(path))
        else:
            historic.extend(aggregate_rows(path))
    write_table(args.out, 'episode_outcomes.csv', exact)
    write_table(args.out, 'historical_aggregate_outcomes.csv', historic)
    write_table(args.out, 'randomized_mixture_outcomes.csv', mixture_summary(archives))
    write_table(args.out, 'paired_differences.csv', paired_comparisons(archives))
    print(f'{len(exact)} episode-stratum rows; {len(historic)} aggregate-only rows; 12 mixture/lead rows')


if __name__ == '__main__':
    main()
