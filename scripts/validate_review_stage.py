#!/usr/bin/env python3
"""Validate execution/data contracts, not whether a controller performs well.

Run after each reviewed stage. audit.json is written on success AND failure.
Only standard Python plus the repository's NumPy evidence utilities are used.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
from foresight.eval import statistics as ST

REFERENCE_COVERAGE_MIN = 0.90


def number(value, label, *, integer=False):
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f'{label}: missing/non-numeric value') from None
    if not math.isfinite(x) or (integer and x != int(x)):
        raise ValueError(f'{label}: nonfinite or noninteger value')
    return int(x) if integer else x


def boolean(value, label):
    if value in (True, 1, '1', 'True', 'true'):
        return True
    if value in (False, 0, '0', 'False', 'false'):
        return False
    raise ValueError(f'{label}: invalid boolean {value!r}')


def key_for(row, kind):
    def parameter(name):
        # Requested leads are stored as float32 by the simulator. Normalize only
        # condition identifiers, never timing evidence or metric arithmetic.
        return round(number(row[name], name), 6)
    if kind == 'oracle':
        return (str(row['mode']), number(row['launch'], 'launch', integer=True),
                parameter('lead_s' if 'lead_s' in row else 'lead'))
    if kind == 'frontier':
        return (str(row['method']), parameter('param'), number(row['latency_steps'], 'latency', integer=True))
    return (str(row['source']), str(row['method']), parameter('param'))


def load_json(path):
    # Reject NaN/Infinity tokens even though Python's json accepts them by default.
    def invalid(value):
        raise ValueError(f'{path.name}: nonstandard JSON number {value}')
    return json.loads(path.read_text(), parse_constant=invalid)


def optional_number(row, key):
    value = row.get(key)
    if value in (None, '', 'nan', 'NaN'):
        return None
    return number(value, key)


def check_episode(row, config):
    for name in ('fell', 'is_miss', 'contact'):
        boolean(row[name], name)
    for name in ('progress_m',):
        number(row[name], name)
    steps = number(config['steps'], 'steps', integer=True)
    dt = number(config['dt'], 'dt')
    brace = number(row['brace_steps_total'], 'brace_steps_total', integer=True)
    if not 0 <= brace <= steps:
        raise ValueError('Preparation intervals outside episode horizon')
    onset = number(row['brace_onset_step'], 'brace_onset_step', integer=True)
    if (brace > 0) != (0 <= onset < steps):
        raise ValueError('Preparation count and onset disagree')
    release = optional_number(row, 'brace_release_step')
    if release is None:
        release = optional_number(row, 'first_release_step')
    if release is not None and release >= 0:
        if release != int(release) or not 0 <= onset < release < steps:
            raise ValueError('Impossible preparation release step')
    contact = boolean(row['contact'], 'contact')
    cs = number(row['contact_step'], 'contact_step', integer=True)
    if contact != (0 <= cs < steps):
        raise ValueError('Contact flag and contact step disagree')
    if 'brace_at_contact' in row and boolean(row['brace_at_contact'], 'brace_at_contact'):
        if not contact or not brace or onset > cs:
            raise ValueError('Brace-at-contact flag impossible')
    if 'ready_at_contact' in row and boolean(row['ready_at_contact'], 'ready_at_contact'):
        if not contact or not boolean(row['brace_at_contact'], 'brace_at_contact'):
            raise ValueError('Ready-at-contact requires a contacting active brace')
    term = number(row.get('term_step', -1), 'term_step', integer=True)
    if term < -1 or term >= steps:
        raise ValueError('Termination step outside horizon')
    if boolean(row['fell'], 'fell') and term < 0:
        raise ValueError('Failure without a termination step')
    rule = config.get('rule') or (config.get('rules') or [None])[0] or config
    release_mode = row.get('release_mode', rule.get('release_mode'))
    if release_mode == 'deadline' and rule.get('single_commit', True):
        commit = number(rule.get('commit_steps', 15), 'commit_steps', integer=True)
        maximum = int(math.floor(number(rule.get('max_hold_s', 4.), 'max_hold_s') / dt + 1e-6))
        if brace > maximum:
            raise ValueError('Single commitment exceeded maximum hold')
        if release is not None and release >= 0 and release - onset != brace:
            raise ValueError('Single commitment duration and applied intervals disagree')
        if 0 < brace < commit and not (onset <= term < onset + commit or steps < onset + commit):
            raise ValueError('Minimum commitment violated without termination/horizon truncation')
        for name, bound in (('brace_releases', 1), ('switches', 2)):
            value = optional_number(row, name)
            if value is not None and not 0 <= value <= bound:
                raise ValueError('Repeated switching under single-commit protocol')
        reason = optional_number(row, 'controller_release_reason')
        if reason is not None and reason not in (0, 1, 2):
            raise ValueError('Legacy release reason under deadline protocol')
        deadline = optional_number(row, 'controller_deadline_step')
        if brace and deadline is not None and deadline >= 0 and not onset + commit <= deadline <= onset + maximum:
            raise ValueError('Stored deadline outside commitment bounds')
    capture = optional_number(row, 'first_detection_capture_step')
    delivery = optional_number(row, 'first_detection_delivery_step')
    if capture is not None and delivery is not None and delivery >= 0:
        if not 0 <= capture <= delivery < steps:
            raise ValueError('Detection capture/delivery timestamp impossible')
        latency = number(row.get('latency_steps', config.get('extra_latency', 0)), 'latency', integer=True)
        if delivery - capture != latency:
            raise ValueError('Delivered observation age disagrees with configured latency')


def episode_table(records):
    required = ['scenario_id', 'fell', 'contact', 'is_miss', 'brace_steps_total', 'progress_m']
    out = {}
    for name in required + ['censored']:
        if name == 'scenario_id': out[name] = np.asarray([r[name] for r in records])
        elif name in ('fell', 'contact', 'is_miss', 'censored'):
            out[name] = np.asarray([boolean(r.get(name, False), name) for r in records])
        else: out[name] = np.asarray([number(r[name], name) for r in records])
    for name in ('mass', 'speed', 't_hit_nominal'):
        if all(name in r for r in records):
            out[name] = np.asarray([number(r[name], name) for r in records])
    return out


def comparison_key(result, side, kind):
    value = result['condition_' + side]
    if kind == 'oracle':
        lead = -1. if value == 'continue' else number(value.removeprefix('lead_'), 'paired lead')
        return str(result['mode']), int(result['launch']), round(lead, 6)
    if kind == 'depth':
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError('Malformed depth comparison condition')
        return str(value[0]), str(value[1]), round(number(value[2], 'paired param'), 6)
    match = re.fullmatch(r'([^:]+):p=([^:]+):lat=([0-9]+)', value)
    if match is None:
        raise ValueError('Malformed controller comparison condition')
    return match[1], round(number(match[2], 'paired param'), 6), int(match[3])


def compare_evidence(stored, actual, label):
    if isinstance(actual, dict):
        if not isinstance(stored, dict):
            raise ValueError(label + ': missing paired evidence')
        for name, value in actual.items():
            compare_evidence(stored.get(name), value, label + '.' + name)
    elif isinstance(actual, (list, tuple)):
        if not isinstance(stored, list) or len(stored) != len(actual):
            raise ValueError(label + ': invalid interval')
        for a, b in zip(stored, actual):
            compare_evidence(a, b, label)
    elif isinstance(actual, (int, float)):
        if not math.isclose(number(stored, label), actual, rel_tol=1e-8, abs_tol=1e-9):
            raise ValueError(label + ': disagrees with paired episode evidence')
    elif stored != actual:
        raise ValueError(label + ': disagrees with paired episode evidence')


def validate_stage(stage_dir, kind, episodes_per_condition, conditions, source_commit):
    path = Path(stage_dir)
    if kind not in ('oracle', 'frontier', 'depth'):
        raise ValueError('Unknown stage kind')
    if episodes_per_condition <= 0 or conditions <= 0 or re.fullmatch(r'(?:[a-f0-9]{40}|[a-f0-9]{64})', source_commit) is None:
        raise ValueError('Invalid expected stage contract')
    summary = load_json(path / 'summary.json')
    config_file = load_json(path / 'config.json')
    config = config_file.get('config') or {k:v for k,v in config_file.items() if k != 'provenance'}
    if summary.get('status') != 'complete':
        raise ValueError('Stage did not report complete')
    if summary.get('schema_version', summary.get('schema')) != ST.SCHEMA_VERSION:
        raise ValueError('Unexpected summary schema')
    if config.get('schema_version', config.get('schema')) != ST.SCHEMA_VERSION:
        raise ValueError('Unexpected configuration schema')
    provenance = summary.get('provenance', {})
    if provenance.get('source_commit') != source_commit or provenance.get('source_dirty') is not False:
        raise ValueError('Source provenance does not match the clean requested revision')
    if config_file.get('provenance') != provenance:
        raise ValueError('Configuration and summary provenance differ')
    digest = hashlib.sha256(json.dumps(ST.json_safe(config), sort_keys=True).encode()).hexdigest()
    if digest != provenance.get('config_sha256'):
        raise ValueError('Configuration content hash mismatch')
    if summary.get('config', config) != config:
        raise ValueError('Configuration changed between configuration and summary records')
    if not provenance.get('artifacts'):
        raise ValueError('Checkpoint/model content provenance is absent')
    for artifact in provenance['artifacts'].values():
        if not artifact.get('files'):
            raise ValueError('Checkpoint/model content manifest is empty')
        for item in artifact['files']:
            if re.fullmatch(r'[a-f0-9]{64}', item.get('sha256', '')) is None or number(item.get('bytes'), 'artifact bytes', integer=True) < 0:
                raise ValueError('Invalid checkpoint/model content manifest')
    dt = number(config['dt'], 'dt')
    if dt <= 0:
        raise ValueError('Nonpositive control timestep')
    if summary.get('episodes_per_condition') != episodes_per_condition:
        raise ValueError('Incorrect episodes_per_condition')
    rows = summary['curve' if kind == 'oracle' else 'rows']
    if len(rows) != conditions:
        raise ValueError(f'Expected {conditions} conditions, found {len(rows)}')
    summaries = {key_for(row, kind):row for row in rows}
    if len(summaries) != conditions:
        raise ValueError('Duplicate summary conditions')
    files = [path/'episodes.csv'] if kind != 'depth' else sorted((path/'episodes').glob('*.csv'))
    if not files or any(not p.is_file() for p in files):
        raise ValueError('Missing primary episode evidence')
    groups = {}
    file_counts = {}
    for p in files:
        with p.open(newline='') as stream:
            records = list(csv.DictReader(stream))
        file_counts[str(p.relative_to(path))] = len(records)
        for record in records:
            check_episode(record, config)
            groups.setdefault(key_for(record, kind), []).append(record)
    if set(groups) != set(summaries):
        raise ValueError('Episode conditions do not match summary conditions')
    common_ids = None
    tables = {}
    for key, records in groups.items():
        ids = [r['scenario_id'] for r in records]
        if len(records) != episodes_per_condition or len(set(ids)) != len(ids):
            raise ValueError(f'{key}: missing/duplicate episode IDs')
        if common_ids is not None and set(ids) != common_ids:
            raise ValueError('Conditions have different paired scenario IDs')
        common_ids = set(ids)
        tab = episode_table(records)
        tables[key] = tab
        actual = ST.summarize_episodes(tab, dt)
        row = summaries[key]
        for name, value in actual.items():
            if value is None:
                if row.get(name) is not None:
                    raise ValueError(f'{key}: {name} should be null for absent subgroup')
            elif not math.isclose(number(row.get(name), name), value, rel_tol=1e-8, abs_tol=1e-9):
                raise ValueError(f'{key}: stored {name} disagrees with pooled episode evidence')
    # Pair labels must match across conditions, not merely their ID sets.
    first = next(iter(tables.values()))
    labels = dict(zip(first['scenario_id'], first['is_miss']))
    for tab in tables.values():
        if any(labels[s] != m for s,m in zip(tab['scenario_id'],tab['is_miss'])):
            raise ValueError('Paired hit/miss labels disagree')
    if kind == 'oracle':
        reactive = [r for r in rows if r['lead_s'] < 0 and int(r['launch']) == 1]
    else:
        reactive = [r for r in rows if r['method'] == 'reactive']
    if not reactive or not any(r['n_contact_hits'] > 0 for r in reactive):
        raise ValueError('No reactive intended-hit contact observed; disturbance gate unverified')
    reference = summary.get('reference_quality', [])
    coverage = None
    needs_reference = config.get('timing_reference', config.get('oracle_timing')) == 'reference_entry'
    if needs_reference and not reference:
        raise ValueError('Reference-entry coverage evidence missing')
    if reference:
        denominator = sum(number(r['n'] if kind == 'oracle' else r['n_hits'], 'reference count', integer=True) for r in reference)
        numerator = sum(number(r['n_reference_events'] if kind == 'oracle' else r['n_reference_events_hits'], 'reference events', integer=True) for r in reference)
        coverage = numerator / denominator if denominator else 0.
        if not 0 <= numerator <= denominator or coverage < REFERENCE_COVERAGE_MIN:
            raise ValueError(f'Reference-entry coverage {coverage:.3f} below {REFERENCE_COVERAGE_MIN:.2f}; inspect exclusions by mass/speed')
    paired = load_json(path/'paired.json')
    if not isinstance(paired, list) or (conditions > 1 and not paired):
        raise ValueError('Paired comparisons missing')
    for result in paired:
        a, b = (comparison_key(result, side, kind) for side in ('a', 'b'))
        actual = ST.paired_comparison(tables[a], tables[b], dt)
        for metric in ('failure_hits','false_preparation_misses','contact_hits',
                       'failure_common_contact','progress_m','braced_s'):
            compare_evidence(result.get(metric), actual[metric], metric)
    return {'schema_version': ST.SCHEMA_VERSION, 'status':'passed', 'passed':True,
            'kind':kind,'source_commit':source_commit,'conditions':conditions,
            'episodes_per_condition':episodes_per_condition,'episode_rows':sum(file_counts.values()),
            'episode_files':file_counts,'paired_comparisons':len(paired),
            'reference_entry_coverage':coverage,'reference_coverage_minimum':REFERENCE_COVERAGE_MIN,
            'interpretation':'Operational/data-validity checks only; not a performance or superiority gate.'}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir',type=Path,required=True)
    parser.add_argument('--kind',choices=('oracle','frontier','depth'),required=True)
    parser.add_argument('--episodes-per-condition',type=int,required=True)
    parser.add_argument('--conditions',type=int,required=True)
    parser.add_argument('--source-commit',required=True)
    args=parser.parse_args(argv)
    try:
        audit=validate_stage(args.stage_dir,args.kind,args.episodes_per_condition,args.conditions,args.source_commit)
        code=0
    except (OSError,ValueError,KeyError,TypeError) as error:
        audit={'schema_version':ST.SCHEMA_VERSION,'status':'failed','passed':False,'error':str(error),
               'kind':args.kind,'source_commit':args.source_commit,
               'reference_coverage_minimum':REFERENCE_COVERAGE_MIN,
               'interpretation':'Operational/data-validity gate failed; no performance conclusion.'}
        code=1
    ST.write_json(args.stage_dir/'audit.json',audit)
    print(json.dumps(audit,allow_nan=False))
    return code


if __name__=='__main__':
    raise SystemExit(main())
