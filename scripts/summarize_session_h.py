#!/usr/bin/env python3
"""Report completed and partial Session H evidence, including paired hold effects."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from foresight.eval import session_h_evidence as SE


def groups(stage):
    records={}
    for path in sorted((stage/'episodes').glob('*.csv')):
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                key=(row['method'],round(float(row['param']),6),int(float(row['latency_steps'])))
                records.setdefault(key,[]).append(row)
    out={}
    for key,rows in records.items():
        tab={}
        for name in ('scenario_id','fell','contact','is_miss','brace_steps_total','progress_m','mass','speed','t_hit_nominal'):
            if not all(name in r for r in rows): continue
            if name=='scenario_id': v=np.array([r[name] for r in rows])
            elif name in ('fell','contact','is_miss'):
                v=np.array([r[name].lower() in ('true','1','1.0') for r in rows])
            else: v=np.array([float(r[name]) for r in rows])
            tab[name]=v
        out[key]=tab
    return out


def fmt(x):
    return '—' if x is None else f'{float(x):.4f}'


def summarize(run_dir):
    run_dir=Path(run_dir)
    run=json.loads((run_dir/'run.json').read_text())
    results=[]; ready={}; execution={s['name']:s for s in run.get('stages',[])}
    for spec in run.get('planned_stages',[]):
        name=spec['name']; p=run_dir/name
        status=execution.get(name,{}).get('status','not_started')
        record={'name':name,'status':status,'kind':spec['kind'],'split':spec['split']}
        if (p/'summary.json').is_file() and (p/'audit.json').is_file():
            summary=json.loads((p/'summary.json').read_text()); audit=json.loads((p/'audit.json').read_text())
            if audit.get('passed') is True and summary.get('status')=='complete' and summary.get('provenance',{}).get('source_commit')==run.get('source_commit'):
                record.update(rows=summary['rows'],audit=audit)
                if status=='passed': ready[name]=summary
        results.append(record)
    complete=(run.get('status')=='completed' and bool(results) and all(s['status']=='passed' and 'rows' in s for s in results))
    comparisons=[]
    base='E18b_hold_0p3'
    if base in ready:
        a=groups(run_dir/base)
        for name in ('E18b_hold_0p6','E18b_hold_1p0'):
            if name not in ready: continue
            b=groups(run_dir/name)
            if set(a)!=set(b): raise ValueError('Hold conditions do not match across stages')
            for key in sorted(a):
                paired=SE.paired_comparison(b[key],a[key],.02)
                comparisons.append({'stage_a':name,'stage_b':base,'method':key[0],'param':key[1],
                                    'latency_steps':key[2],**paired})
    analysis={'schema_version':SE.SCHEMA_VERSION,'status':'complete' if complete else 'partial',
              'source_commit':run.get('source_commit'),'run_id':run.get('run_id'),'stages':results,
              'paired_hold_comparisons':comparisons,
              'interpretation':'Session G controller retained; fixed response model transferred across execution holds; repeated G scenarios; one frozen policy pair.',
              'hold_definition':'Release may occur after estimated closest approach OR staleness, subject to minimum commitment; re-entry remains possible.',
              'uncertainty':'Scenario-paired normal95 intervals conditional on this skill pair, no multiple-selection adjustment.'}
    SE.write_json(run_dir/'analysis.json',analysis)
    lines=['# Session H experiment report','',f"Status: **{analysis['status']}**. Run `{run.get('run_id')}`; source `{run.get('source_commit')}`.",'',
           '## Stage completion','', '| Stage | Status | Evidence |','| --- | --- | --- |']
    for s in results: lines.append(f"| {s['name']} | {s['status']} | {'audited' if 'rows' in s else 'incomplete or unaudited'} |")
    lines+=['','## Primary outcomes','', 'Preparation time below excludes intervals after termination. The raw legacy duration is retained in the stage files.','',
            '| Stage | Source | Method | Parameter | Delay (steps) | Hit failures | Contact on hits | Braced time (s) | Progress (m) |',
            '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for s in results:
        if s['split']!='followup': continue
        for row in s.get('rows',[]):
            latency=row.get('latency_steps',ready.get(s['name'],{}).get('config',{}).get('extra_latency',0))
            lines.append(f"| {s['name']} | {row.get('source','parametric')} | {row['method']} | {fmt(row.get('param'))} | {latency} | {fmt(row.get('fall_rate_hits'))} | {fmt(row.get('contact_rate_hits'))} | {fmt(row.get('braced_s'))} | {fmt(row.get('progress_m'))} |")
    lines+=['','## Paired hold effects','', 'Effects are longer hold minus 0.3 s on the same scenario IDs. Negative failure difference favors the longer hold.','',
            '| Hold stage | Method | Parameter | Delay | Failure difference | Approximate 95% interval | Added brace time (s) |',
            '| --- | --- | ---: | ---: | ---: | --- | ---: |']
    for c in comparisons:
        d=c['failure_hits']; ci=d['normal95']; interval='unavailable' if ci is None else f"[{fmt(ci[0])}, {fmt(ci[1])}]"
        lines.append(f"| {c['stage_a']} | {c['method']} | {c['param']} | {c['latency_steps']} | {fmt(d['difference'])} | {interval} | {fmt(c['braced_s']['difference'])} |")
    lines+=['','## Interpretation limits','',
            '- These runs test execution hold and the parked-object correction while keeping the Session G controller and fitted response fixed.',
            '- A longer execution hold is a response-model transfer condition; its cost/response fit is not recalibrated. The study does not identify an optimal controller.',
            '- Prelaunch false detections are counted per capture frame. False preparation is counted per near-miss episode. They are different quantities.',
            '- The scene is simulated; radius and robot/camera pose assumptions remain. Render wall time is separate from the configured simulated delay.',
            '- Gate data and partial runs are operational diagnostics. A failed gate does not mean a method failed scientifically.',
            '- Scenario uncertainty for one skill pair does not measure population variation across training seeds.']
    (run_dir/'report.md').write_text('\n'.join(lines)+'\n')
    if comparisons:
        with (run_dir/'hold_comparisons.csv').open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['stage_a','stage_b','method','param','latency_steps','failure_difference','ci_low','ci_high','brace_difference_s'])
            for c in comparisons:
                ci=c['failure_hits']['normal95'] or [None,None]
                w.writerow([c['stage_a'],c['stage_b'],c['method'],c['param'],c['latency_steps'],c['failure_hits']['difference'],*ci,c['braced_s']['difference']])
    print(f"Session H report: {analysis['status']} ({len(ready)}/{len(results)} audited stages).")
    return analysis


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',required=True,type=Path)
    summarize(p.parse_args().run_dir)


if __name__=='__main__': main()
