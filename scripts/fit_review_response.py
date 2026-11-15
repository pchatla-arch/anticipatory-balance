#!/usr/bin/env python3
"""Fit the reviewed response surrogate from NEW, protocol-matched oracle data.

Uses all scheduled launched hit scenarios with a finite reference entry event,
including noncontacts and failures before a scheduled switch. Contact is never
a row-selection criterion. Input is development data, not the held-out policy
test. NumPy is the only non-standard dependency.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def boolean(value):
    if str(value).lower() in ('1', 'true'):
        return True
    if str(value).lower() in ('0', 'false'):
        return False
    raise ValueError(f'Invalid boolean field: {value!r}')


def number(row, key):
    try:
        return float(row.get(key, 'nan'))
    except (TypeError, ValueError):
        return float('nan')


def sigmoid(z):
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def fit(x, y):
    """Damped, regularized Newton logistic fit with train-only standardization."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(y) < 16 or len(np.unique(y)) < 2:
        raise ValueError('Response fit needs at least 16 rows and both outcomes.')
    mu, sd = x.mean(0), x.std(0)
    sd[sd < 1e-10] = 1.0
    X = np.column_stack([np.ones(len(x)), (x-mu)/sd])
    penalty = np.r_[0.0, np.full(x.shape[1], 1e-3)]
    w = np.zeros(X.shape[1])
    def objective(v):
        z = X @ v
        return float(np.sum(np.logaddexp(0.0, z) - y*z) + .5*np.dot(penalty*v, v))
    for _ in range(100):
        p = sigmoid(X @ w)
        g = X.T @ (p-y) + penalty*w
        if np.max(np.abs(g)) < 1e-6:
            break
        H = (X * (p*(1-p))[:, None]).T @ X + np.diag(penalty + 1e-8)
        step = np.linalg.solve(H, g)
        old = objective(w)
        scale = 1.0
        while scale > 1e-8 and objective(w-scale*step) > old:
            scale *= .5
        if scale <= 1e-8:
            raise RuntimeError('Logistic line search failed.')
        w -= scale*step
    else:
        raise RuntimeError('Logistic fit did not converge.')
    coef = w[1:]/sd
    return np.r_[w[0] - np.dot(coef, mu), coef]


def score(y, p):
    p = np.clip(p, 1e-9, 1-1e-9)
    return {'n': len(y), 'observed_failure': float(y.mean()),
            'predicted_failure': float(p.mean()),
            'brier': float(np.mean((p-y)**2)),
            'log_loss': float(np.mean(-y*np.log(p)-(1-y)*np.log1p(-p)))}


def grouped_check(x, y, groups):
    """Hold out entire scenario batches, never different leads of one scenario."""
    folds = []
    preds = np.full(len(y), np.nan)
    batch = np.asarray([g.split('_env', 1)[0] for g in groups])
    unique = np.unique(batch)
    if len(unique) < 2:
        return {'status': 'unavailable', 'reason': 'Only one scenario batch.'}
    for held in unique:
        test, train = batch == held, batch != held
        w = fit(x[train], y[train])
        pred = sigmoid(np.column_stack([np.ones(test.sum()), x[test]]) @ w)
        preds[test] = pred
        folds.append({'held_batch': held, **score(y[test], pred)})
    return {'status': 'complete', 'pooled': score(y, preds), 'folds': folds,
            'interpretation': 'Development-data grouped prediction check, not policy-test evidence.'}


def fit_response(episodes: Path, summary_path: Path):
    summary = json.loads(summary_path.read_text())
    cfg = summary.get('config', {})
    if cfg.get('timing_reference') != 'reference_entry':
        raise ValueError('New response data must use timing_reference=reference_entry.')
    if cfg.get('release_mode') != 'deadline':
        raise ValueError('New response data must use the deadline hold protocol.')
    dt = float(cfg.get('dt', 0.02))
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError('Invalid control timestep.')
    with episodes.open(newline='') as stream:
        raw = list(csv.DictReader(stream))
    reasons = {k: 0 for k in ('not_launched_brace_family', 'missing_reference', 'censored', 'invalid_fields')}
    selected = []
    for row in raw:
        if not boolean(row['mode_brace']) or not boolean(row['launch']):
            reasons['not_launched_brace_family'] += 1
            continue
        event = number(row, 'reference_event_time_s')
        if not math.isfinite(event):
            reasons['missing_reference'] += 1
            continue
        if boolean(row['censored']):
            reasons['censored'] += 1
            continue
        lead = number(row, 'lead')
        J = number(row, 'momentum')
        y = number(row, 'fell')
        if not (math.isfinite(lead) and math.isfinite(J) and J > 0 and y in (0, 1)):
            reasons['invalid_fields'] += 1
            continue
        if lead < 0:
            modeled_lead = -1.0
        else:
            start = number(row, 'scheduled_switch_time_s')
            if not math.isfinite(start):
                reasons['invalid_fields'] += 1
                continue
            # Use the scheduled, discretized intervention even when failure
            # precedes its execution. Selecting executed switches would condition
            # on survival and remove real failures from the response target.
            planned_tick = max(0, math.ceil(start/dt - 1e-6))
            modeled_lead = max(0.0, event - planned_tick*dt)
        selected.append({**row, '_lead': modeled_lead, '_J': J, '_y': y})
    cont = [r for r in selected if r['_lead'] < 0]
    br = [r for r in selected if r['_lead'] >= 0]
    if not cont or not br:
        raise ValueError('Missing CONTINUE or BRACE rows after protocol checks.')
    lead_max = max(number(r, 'lead') for r in br)
    if lead_max <= 0:
        raise ValueError('BRACE data have no positive preparation lead.')
    xc = np.array([[math.log(r['_J'])] for r in cont])
    xb = np.array([[math.log(r['_J']), min(r['_lead'], lead_max)] for r in br])
    yc, yb = (np.array([r['_y'] for r in rows]) for rows in (cont, br))
    wc, wb = fit(xc, yc), fit(xb, yb)
    ref_lead = min(.5, lead_max)
    out = {'schema_version': 'foresight.review_response.v1',
           'a_cont': float(wc[0]), 'b_cont': float(wc[1]),
           'a_brace': float(wb[0]+wb[2]*ref_lead), 'b_brace': float(wb[1]),
           'brace_lead': ref_lead, 'a_brace_lead': float(wb[0]),
           'b_brace_lead': float(wb[1]), 'c_lead': float(wb[2]),
           'lead_max': lead_max, 'n_cont': len(cont), 'n_brace': len(br),
           'n_brace_lead': len(br),
           'source': str(episodes), 'episodes_sha256': hashlib.sha256(episodes.read_bytes()).hexdigest(),
           'oracle_summary_sha256': hashlib.sha256(summary_path.read_bytes()).hexdigest(),
           'source_provenance': summary.get('provenance'), 'protocol': cfg,
           'estimand': 'Failure on scheduled launched reference-entry scenarios, including noncontacts and pre-switch failures; censored rows excluded and counted.',
           'lead_definition': 'Reference .45m entry event minus scheduled controller tick; same intervention label for failures before execution.',
           'contact_conditioned': False, 'excluded_rows': reasons,
           'n_raw': len(raw), 'n_selected': len(selected),
           'scenario_groups_cont': len({r['scenario_id'] for r in cont}),
           'scenario_groups_brace': len({r['scenario_id'] for r in br}),
           'unexecuted_scheduled_braces_retained': sum(not math.isfinite(number(r, 'actual_onset_time_s')) for r in br),
           'support': {'scheduled_leads': sorted({number(r, 'lead') for r in br}),
                       'modeled_lead_min': float(xb[:, 1].min()), 'modeled_lead_max': float(xb[:, 1].max()),
                       'momentum_min': min(r['_J'] for r in selected),
                       'momentum_max': max(r['_J'] for r in selected)},
           'grouped_prediction_check': {'continue': grouped_check(xc, yc, [r['scenario_id'] for r in cont]),
                                        'brace': grouped_check(xb, yb, [r['scenario_id'] for r in br])},
           'limitations': ['Reduced log-momentum/lead surrogate; does not represent every posture or speed effect.',
                           'Lead clipped above observed maximum; values below observed minimum are extrapolation.',
                           'Reference entry uses local within-step geometry, not exact physical contact time.',
                           'Online predicted arrival can differ from the privileged reference event.',
                           'One frozen skill pair; prediction checks do not establish policy superiority.']}
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--episodes', required=True, type=Path)
    p.add_argument('--oracle-summary', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args(argv)
    if args.out.exists():
        p.error('Output already exists; use a new run directory to preserve provenance.')
    out = fit_response(args.episodes, args.oracle_summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: out[k] for k in ('n_cont', 'n_brace_lead', 'c_lead', 'excluded_rows', 'support')}, indent=2))
    print('Saved new development response model:', args.out)


if __name__ == '__main__':
    main()
