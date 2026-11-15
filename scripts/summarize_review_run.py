#!/usr/bin/env python3
"""Summarize frozen reviewed evidence, including failed and partial runs.

This script performs no simulation or controller selection. Figures compare the
fixed operating points requested by the run plan, not an optimized frontier.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from foresight.eval import statistics as ST

SUCCESS = {'complete', 'completed', 'passed'}
OUTPUT_FILES = ['report.md', 'analysis.json', 'fixed_controller_comparison.png',
                'depth_comparison.png', 'readiness.png']


def read_json(path):
    def invalid(value):
        raise ValueError(f'{path.name}: nonstandard JSON value {value}')
    return json.loads(path.read_text(), parse_constant=invalid)


def collect_run(run_dir):
    root = Path(run_dir)
    run = read_json(root / 'run.json')
    source = run.get('source_commit')
    planned = run.get('planned_stages') or []
    if not planned and (root / 'plan.json').is_file():
        planned = read_json(root / 'plan.json').get('stages', [])
    actual = {s['name']: s for s in run.get('stages', [])}
    specs = {s['name']: s for s in planned}
    specs.update({name: {**specs.get(name, {}), **stage} for name, stage in actual.items()})
    stages = []
    for name, spec in specs.items():
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Stage name must be a simple directory name')
        entry = {'name': name, 'kind': spec.get('kind'), 'split': spec.get('split'),
                 'runner_status': actual.get(name, {}).get('status', 'not_started'),
                 'status': 'not_started' if name not in actual else 'unvalidated',
                 'elapsed_seconds': spec.get('elapsed_seconds'),
                 'expected_conditions': spec.get('conditions'),
                 'expected_episodes_per_condition': spec.get('episodes_per_condition')}
        try:
            if entry['kind'] == 'fit':
                if not (root / 'response_model.json').exists():
                    entry['status'] = entry['runner_status']
                else:
                    model = read_json(root / 'response_model.json')
                    entry['model'] = model
                    matched = model.get('source_provenance', {}).get('source_commit') == source
                    if (model.get('schema_version') == 'foresight.review_response.v1'
                            and model.get('contact_conditioned') is False and matched
                            and spec.get('checks', {}).get('passed') is True
                            and entry['runner_status'] in SUCCESS):
                        entry['status'] = 'validated'
                    else:
                        entry['error'] = 'Response fit missing completed checks or matching provenance'
            elif (root / name / 'summary.json').exists():
                summary = read_json(root / name / 'summary.json')
                audit_path = root / name / 'audit.json'
                audit = read_json(audit_path) if audit_path.exists() else {}
                cfg_file = read_json(root / name / 'config.json')
                config = cfg_file.get('config') or {k: v for k, v in cfg_file.items() if k != 'provenance'}
                entry.update(summary=summary, audit=audit, config=config)
                provenance = summary.get('provenance', {})
                schema = summary.get('schema_version', summary.get('schema'))
                if (summary.get('status') == 'complete' and schema == ST.SCHEMA_VERSION
                        and audit.get('passed') is True and audit.get('status') == 'passed'
                        and audit.get('source_commit') == source == provenance.get('source_commit')
                        and provenance.get('source_dirty') is False
                        and cfg_file.get('provenance') == provenance
                        and entry['runner_status'] in SUCCESS):
                    entry['status'] = 'validated'
                    entry['paired'] = read_json(root / name / 'paired.json')
                else:
                    entry['error'] = audit.get('error', 'Missing completed checks or matching source provenance')
            else:
                entry['status'] = entry['runner_status'] if entry['runner_status'] not in SUCCESS else 'missing_evidence'
        except (OSError, ValueError, KeyError, TypeError) as error:
            entry['status'] = 'invalid_evidence'
            entry['error'] = str(error)
        stages.append(entry)
    complete = bool(stages) and run.get('status') in SUCCESS and all(s['status'] == 'validated' for s in stages)
    return {'schema_version': ST.SCHEMA_VERSION, 'run_id': run.get('run_id', root.name),
            'source_commit': source, 'mode': run.get('mode'),
            'status': 'complete' if complete else 'partial',
            'experiment_status': run.get('status', 'unknown'),
            'experiment_error': run.get('error'), 'elapsed_seconds': run.get('elapsed_seconds'),
            'stages': stages, 'plots': {}, 'output_files': ['report.md', 'analysis.json'],
            'interpretation': 'Fixed operating points; scenario-paired evidence conditional on one frozen skill pair. Operational checks do not establish superiority.'}


def preferred_stages(analysis, kind):
    stages = [s for s in analysis['stages'] if s['kind'] == kind and s['status'] == 'validated']
    tests = [s for s in stages if s.get('split') == 'test' or '_test_' in s['name']]
    if tests:
        return tests
    if kind == 'oracle':
        development = [s for s in stages if s.get('split') == 'development' or 'development' in s['name']]
        if development:
            return development
    return stages


def finite(value):
    return isinstance(value, (float, int)) and math.isfinite(value)


def fmt(value, percent=False):
    if not finite(value):
        return '—'
    return f'{100*value:.1f}%' if percent else f'{value:.3g}'


def cell(value):
    if isinstance(value, list):
        value = ' / '.join(str(x) for x in value)
    return str(value).replace('|', '/').replace('\n', ' ')


def paired_cell(part):
    if not part or not part.get('n'):
        return '— (n=0)'
    effect = 100 * part['difference']
    ci = part.get('normal95')
    interval = f' [{100*ci[0]:+.1f}, {100*ci[1]:+.1f}]' if ci is not None else ' [interval unavailable]'
    return f'{effect:+.1f}{interval}; n={part["n"]}'


def make_plots(analysis, run_dir):
    root = Path(run_dir)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as error:
        analysis['plots']['availability'] = {'status': 'unavailable', 'reason': str(error)}
        return
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
                         'savefig.dpi': 180})

    def fixed_plot(kind, filename):
        stages = preferred_stages(analysis, kind)
        if not stages:
            analysis['plots'][filename] = {'status': 'not_created', 'reason': 'No validated stage evidence'}
            return
        max_rows = max(len(s['summary']['rows']) for s in stages)
        fig, axes = plt.subplots(len(stages), 2, squeeze=False,
                                 figsize=(12, max(3.2, .28*max_rows+1.6)*len(stages)))
        for i, stage in enumerate(stages):
            rows = stage['summary']['rows']
            cfg = stage['config']
            labels = []
            for r in rows:
                label = f"{r['method']} ({r['param']:g})"
                if kind == 'frontier':
                    label += f" | delay {r['latency_steps']} ticks"
                else:
                    label = f"{r['source']} | " + label
                labels.append(label)
            for j, (metric, title) in enumerate([('fall_rate_hits', 'Failures / intended hits'),
                                                ('false_prep_rate', 'Prepared / intended misses')]):
                ax = axes[i, j]
                values = [100*r[metric] if finite(r.get(metric)) else np.nan for r in rows]
                ax.barh(np.arange(len(rows)), values, color='#36648b' if j == 0 else '#c27d38')
                ax.set_yticks(np.arange(len(rows)), labels=labels)
                ax.invert_yaxis()
                ax.set_xlim(0, 100)
                ax.set_xlabel('Percent')
                ax.set_title(f"{stage['name']}: {title}")
                ax.grid(axis='x', alpha=.2)
            axes[i, 0].text(0, -0.10, f"n={stage['summary']['episodes_per_condition']} per condition; "
                           f"capture period {cfg.get('period', '—')} ticks; split {stage.get('split') or 'unspecified'}",
                           transform=axes[i, 0].transAxes, fontsize=8)
        fig.suptitle('Fixed controller operating points — paired uncertainty in report and analysis', fontsize=11)
        fig.tight_layout(rect=(0, .02, 1, .98))
        fig.savefig(root/filename, bbox_inches='tight')
        plt.close(fig)
        analysis['plots'][filename] = {'status': 'created', 'stages': [s['name'] for s in stages]}
        analysis['output_files'].append(filename)

    for kind, filename in [('frontier', 'fixed_controller_comparison.png'), ('depth', 'depth_comparison.png')]:
        fixed_plot(kind, filename)
    stages = preferred_stages(analysis, 'oracle')
    if not stages:
        analysis['plots']['readiness.png'] = {'status': 'not_created', 'reason': 'No validated oracle evidence'}
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for stage in stages:
        rows = sorted([r for r in stage['summary']['curve'] if r['mode'] == 'brace'
                       and r['launch'] == 1 and r['lead_s'] >= 0], key=lambda r:r['lead_s'])
        if not rows:
            continue
        x = [r['lead_s'] for r in rows]
        axes[0].plot(x, [100*r['fall_rate_hits'] if finite(r.get('fall_rate_hits')) else np.nan for r in rows],
                     'o-', label=stage['name'])
        for key, label, style in [('ready_at_contact_given_contact', 'Ready at contact / contacts', 'o-'),
                                  ('readiness_ever_after_switch_rate', 'Ready at any time / all episodes', 's--')]:
            axes[1].plot(x, [100*r[key] if finite(r.get(key)) else np.nan for r in rows], style,
                         label=label + ('; ' + stage['name'] if len(stages) > 1 else ''))
    axes[0].set_title('Launched BRACE: observed failure')
    axes[1].set_title('Readiness definitions have different denominators')
    for ax in axes:
        ax.set_xlabel('Requested lead to reference entry (s)')
        ax.set_ylabel('Percent')
        ax.set_ylim(0, 100)
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(root/'readiness.png', bbox_inches='tight')
    plt.close(fig)
    analysis['plots']['readiness.png'] = {'status': 'created', 'stages': [s['name'] for s in stages]}
    analysis['output_files'].append('readiness.png')


def make_report(analysis):
    lines = ['# Foresight reviewed experiment report', '',
             f"**Evidence status: {analysis['status'].upper()}.** Runner status: {cell(analysis['experiment_status'])}.", '',
             f"Run: `{cell(analysis['run_id'])}`  ",
             f"Source revision: `{cell(analysis['source_commit'])}`  ",
             f"Mode: {cell(analysis['mode'])}; elapsed: {fmt(analysis['elapsed_seconds'])} seconds.", '']
    if analysis['status'] != 'complete':
        lines += ['The declared plan has not completed with all evidence checks passed. '
                  'Only validated stages below support numerical summaries; missing or failed stages are not zero-valued results.', '']
    if analysis.get('experiment_error'):
        lines += ['Runner error: ' + cell(analysis['experiment_error']), '']
    lines += ['## Execution and evidence', '',
              '| Stage | Split | Runner | Evidence | Episodes / condition | Conditions |',
              '|---|---|---|---|---:|---:|']
    for s in analysis['stages']:
        lines.append('| ' + ' | '.join(cell(x) for x in (s['name'], s.get('split'), s['runner_status'],
                    s['status'], s.get('expected_episodes_per_condition'), s.get('expected_conditions'))) + ' |')
    if not analysis['stages']:
        lines.append('| No experiment stages recorded | — | — | unavailable | — | — |')
    for s in analysis['stages']:
        if s.get('error'):
            lines += ['', f"- **{s['name']}**: {cell(s['error'])}"]
        elif s.get('audit', {}).get('reference_entry_coverage') is not None:
            lines += ['', f"- **{s['name']}**: reference-entry coverage {fmt(s['audit']['reference_entry_coverage'], True)} "
                      f"(required minimum {fmt(s['audit']['reference_coverage_minimum'], True)})."]
    lines += ['', '## Fixed controller comparisons', '',
              'Numerators and denominators are pooled from retained episode rows. Failure is measured on intended hits; '
              'contact is a separate outcome. These are predeclared operating points, not a selected Pareto frontier.']
    for kind in ('frontier', 'depth'):
        stages = preferred_stages(analysis, kind)
        if not stages:
            lines += ['', f'No validated {kind} stage is available.']
        for s in stages:
            lines += ['', f"### {s['name']} ({s.get('split') or 'unspecified split'})", '',
                      '| Condition | Failures / intended hits | Contacts / intended hits | Preparation / misses | Mean brace (s) | Progress (m) | Censored |',
                      '|---|---:|---:|---:|---:|---:|---:|']
            for r in s['summary']['rows']:
                label = f"{r['method']} p={r['param']:g}"
                label += f" latency={r['latency_steps']}" if kind == 'frontier' else f" source={r['source']}"
                def ratio(num, den, rate):
                    return f"{r[num]}/{r[den]} ({fmt(r[rate], True)})"
                lines.append('| ' + ' | '.join([label, ratio('n_falls_hits','n_hits','fall_rate_hits'),
                    ratio('n_contact_hits','n_hits','contact_rate_hits'),
                    ratio('n_false_prep_miss','n_misses','false_prep_rate'), fmt(r['braced_s']),
                    fmt(r['progress_m']), str(r['n_censored'])]) + ' |')
            lines += ['', '**Paired effects (percentage points).** Each entry is condition A minus condition B with a descriptive '
                      'approximate 95% normal interval. Negative failure differences favor A. These intervals condition on one '
                      'frozen skill pair and are not adjusted for multiple comparisons.', '',
                      '| A | B | Failure difference [95% interval]; n | False-preparation difference [95% interval]; n |',
                      '|---|---|---:|---:|']
            pairs = s.get('paired', [])
            for pair in pairs:
                # Show baseline and direct timing-method contrasts; every saved pair remains in analysis.json.
                a, b = cell(pair['condition_a']), cell(pair['condition_b'])
                if not ('reactive' in a+b or ('belief_voi' in a+b and ('belief_wait' in a+b or 'belief_lead' in a+b))
                        or kind == 'depth' and 'depth' in a and 'parametric' in b):
                    continue
                lines.append('| ' + ' | '.join([a, b, paired_cell(pair['failure_hits']),
                                                paired_cell(pair['false_preparation_misses'])]) + ' |')
    lines += ['', '## Oracle timing and readiness', '',
              'The reference event is the locally interpolated 0.45 m cylinder entry in the shared CONTINUE trajectory. '
              'It is distinct from policy-dependent physical contact. Readiness requires active BRACE throughout a finite '
              'pre-action posture window; ready-at-contact and ever-ready are reported separately. No physical settling '
              'time is assumed from the controller tuning constant.']
    for s in preferred_stages(analysis, 'oracle'):
        lines += ['', f"### {s['name']}", '',
                  '| Mode | Launch | Requested lead (s) | Failures / episodes | Ready at contact / contacts | Ever ready / episodes | Stable-window completion since onset (s) |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for r in s['summary']['curve']:
            lines.append('| ' + ' | '.join([r['mode'], str(r['launch']), fmt(r['lead_s']),
                f"{r['n_falls_hits']}/{r['n_total']}", fmt(r.get('ready_at_contact_given_contact'), True),
                fmt(r.get('readiness_ever_after_switch_rate'), True), fmt(r.get('readiness_window_complete_mean_s'))]) + ' |')
    lines += ['', '## Interpretation and limits', '',
              '- All completion gates concern execution, pairing, timestamps, and evidence integrity. Passing is not a test of controller superiority.',
              '- Development response data and final test scenarios use separate seeds. A completed smoke run establishes execution only.',
              '- Scenario-level intervals do not quantify variation across independently trained skill seeds. Small subgroup denominators remain visible.',
              '- Contact-conditioned comparisons are selected after intervention and remain descriptive. Primary failure comparisons use all intended-hit scenarios.',
              '- Censoring counts are retained; finite-horizon survival is not proof of eventual recovery.',
              '- Rendered-depth runs are simulation experiments. Parametric and visibility-matched sensor arms are diagnostic comparators; test-frame calibration is not used to tune the fixed likelihood.',
              '- Figures show fixed parameter settings. All recorded paired comparisons, including progress and preparation duration, are in analysis.json.', '',
              '## Artifacts', '']
    for name in analysis['output_files']:
        if name.endswith('.png'):
            lines += [f'![{name}]({name})', '']
        else:
            lines.append(f'- `{name}`')
    for name, result in analysis['plots'].items():
        if result.get('status') != 'created':
            lines += ['', f"{name}: {cell(result.get('reason', result['status']))}."]
    return '\n'.join(lines) + '\n'


def summarize_run(run_dir, *, plots=True):
    root = Path(run_dir)
    analysis = collect_run(root)
    if plots:
        make_plots(analysis, root)
    else:
        analysis['plots']['availability'] = {'status': 'disabled', 'reason': 'Plots explicitly disabled by caller'}
    ST.write_json(root/'analysis.json', analysis)
    (root/'report.md').write_text(make_report(analysis))
    return analysis


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    analysis = summarize_run(args.run_dir)
    print(json.dumps({'status': analysis['status'], 'output_files': analysis['output_files']}, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
