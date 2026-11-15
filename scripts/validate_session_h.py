#!/usr/bin/env python3
"""Check Session H evidence integrity; no controller-superiority requirement."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from foresight.eval import session_h_evidence as ST


def load(path):
    def bad(x): raise ValueError('Nonstandard JSON number '+x)
    return json.loads(Path(path).read_text(),parse_constant=bad)


def num(value, label, integer=False):
    try: x=float(value)
    except (TypeError,ValueError): raise ValueError(label+': missing numeric value') from None
    if not math.isfinite(x) or (integer and int(x)!=x): raise ValueError(label+': invalid number')
    return int(x) if integer else x


def boolean(value):
    if value in (True,1,'1','true','True'): return True
    if value in (False,0,'0','false','False'): return False
    raise ValueError('Invalid boolean '+str(value))


def key(row,kind):
    common=(str(row['method']),round(num(row['param'],'param'),8))
    return (str(row['source']),*common) if kind=='depth' else (*common,num(row['latency_steps'],'latency',True))


def equal(actual,stored,label):
    if actual is None:
        if stored is not None: raise ValueError(label+': expected null')
    elif not math.isclose(num(stored,label),actual,abs_tol=1e-9,rel_tol=1e-8):
        raise ValueError(label+': disagrees with raw episodes')


def validate(stage_dir,kind,episodes_per_condition,conditions,expected_release,source_commit):
    p=Path(stage_dir); summary=load(p/'summary.json'); config_file=load(p/'config.json')
    config={k:v for k,v in config_file.items() if k!='provenance'}
    prov=summary.get('provenance',{})
    if summary.get('status')!='complete': raise ValueError('Incomplete stage')
    for record in (summary,config):
        if record.get('schema_version',record.get('schema'))!=ST.SCHEMA_VERSION: raise ValueError('Unexpected schema')
    if not re.fullmatch(r'[a-f0-9]{40}|[a-f0-9]{64}',source_commit): raise ValueError('Invalid requested source')
    if prov.get('source_commit')!=source_commit or prov.get('source_dirty') is not False: raise ValueError('Source revision is not the clean requested commit')
    if prov!=config_file.get('provenance') or summary.get('config')!=config: raise ValueError('Configuration/provenance changed')
    digest=hashlib.sha256(json.dumps(ST.json_safe(config),sort_keys=True).encode()).hexdigest()
    if digest!=prov.get('config_sha256'): raise ValueError('Configuration hash mismatch')
    if not prov.get('artifacts'): raise ValueError('Missing checkpoint/model provenance')
    for artifact in prov['artifacts'].values():
        if not artifact.get('files'): raise ValueError('Empty artifact manifest')
        for item in artifact['files']:
            if not re.fullmatch('[a-f0-9]{64}',item.get('sha256','')): raise ValueError('Invalid artifact hash')
            if num(item.get('bytes'),'artifact bytes',True)<0: raise ValueError('Invalid artifact size')
    release=config.get('release_after',config.get('rule',{}).get('release_after'))
    equal(num(expected_release,'expected release'),release,'release_after')
    if num(release,'release_after')<0: raise ValueError('Negative release_after')
    dt=num(config['dt'],'dt'); steps=num(config['steps'],'steps',True)
    if dt<=0 or steps<=0 or episodes_per_condition<=0 or conditions<=0: raise ValueError('Nonpositive stage contract')
    if summary.get('episodes_per_condition')!=episodes_per_condition: raise ValueError('Wrong episodes_per_condition')
    rows={key(r,kind):r for r in summary['rows']}
    if len(rows)!=conditions or len(summary['rows'])!=conditions: raise ValueError('Wrong/duplicate conditions')
    groups={}
    paths=sorted((p/'episodes').glob('*.csv'))
    if not paths: raise ValueError('Missing episode CSVs')
    brace_name='brace_steps' if kind=='depth' else 'brace_steps_total'
    for path in paths:
        with path.open(newline='') as stream:
            for row in csv.DictReader(stream):
                n=num(row[brace_name],brace_name,True)
                if not 0<=n<=steps: raise ValueError('Preparation duration outside horizon')
                num(row['progress_m'],'progress_m')
                for flag in ('fell','contact','is_miss','censored'): boolean(row[flag])
                cs=num(row['contact_step'],'contact_step',True)
                if boolean(row['contact'])!=(0<=cs<steps): raise ValueError('Contact flag/time mismatch')
                ts=num(row['term_step'],'term_step',True)
                if not -1<=ts<steps or (boolean(row['fell']) and ts<0): raise ValueError('Failure/time mismatch')
                if kind=='depth':
                    for suffix in ('','_raw'):
                        frames=num(row['prelaunch_frames'+suffix],'prelaunch frames',True)
                        detects=num(row['prelaunch_detections'+suffix],'prelaunch detections',True)
                        if not 0<=detects<=frames: raise ValueError('Invalid prelaunch counters')
                    if num(row['prelaunch_frames'],'active frames')>num(row['prelaunch_frames_raw'],'raw frames'): raise ValueError('Active frames exceed raw frames')
                groups.setdefault(key(row,kind),[]).append(row)
    if set(groups)!=set(rows): raise ValueError('Episode/summary conditions differ')
    ids=None; tables={}
    fields=('n_total','n_hits','n_misses','n_contact_hits','n_falls_hits','n_falls_contact_hits','n_false_prep_miss',
            'fall_rate_hits','fall_rate_given_contact','contact_rate_hits','false_prep_rate','braced_on_hits','braced_s','progress_m')
    for condition,records in groups.items():
        this=[r['scenario_id'] for r in records]
        if len(this)!=episodes_per_condition or len(set(this))!=len(this): raise ValueError('Missing or duplicate paired scenario IDs')
        if ids is not None and set(this)!=ids: raise ValueError('Conditions have different scenario IDs')
        ids=set(this)
        tab={'scenario_id':np.asarray(this),'brace_steps_total':np.asarray([num(r[brace_name],brace_name) for r in records])}
        for field in ('fell','contact','is_miss','censored'):
            tab[field]=np.asarray([boolean(r[field]) for r in records])
        for field in ('mass','speed','t_hit_nominal','progress_m'):
            tab[field]=np.asarray([num(r[field],field) for r in records])
        tables[condition]=tab
        pooled=ST.summarize_episodes(tab,dt)
        for field in fields: equal(pooled[field],rows[condition].get(field),str(condition)+':'+field)
        if kind=='depth':
            for suffix in ('','_raw'):
                f=sum(num(r['prelaunch_frames'+suffix],'frames',True) for r in records)
                d=sum(num(r['prelaunch_detections'+suffix],'detections',True) for r in records)
                equal(f,rows[condition].get('prelaunch_frames'+suffix),'pooled frames')
                equal(d,rows[condition].get('prelaunch_detections'+suffix),'pooled detections')
                equal(d/f if f else None,rows[condition].get('false_alarm_rate'+suffix),'pooled false alarms')
    reactive=next((k for k in tables if ('reactive'==k[1] if kind=='depth' else 'reactive'==k[0])),None)
    if reactive is None: raise ValueError('Missing reactive reference')
    ref=tables[reactive]
    if not np.any(ref['contact'] & ~ref['is_miss']): raise ValueError('Reactive reference has no intended-hit contacts')
    for tab in tables.values(): ST.paired_comparison(tab,ref,dt) # validates paired labels, not superiority
    if not (p/'episodes.npz').is_file() or not (p/'paired.json').is_file(): raise ValueError('Missing paired/archive evidence')
    return {'schema_version':ST.SCHEMA_VERSION,'status':'passed','passed':True,'kind':kind,
            'source_commit':source_commit,'conditions':conditions,'episodes_per_condition':episodes_per_condition,
            'release_after':expected_release,'episode_files':len(paths),
            'interpretation':'Execution/evidence gate only; no performance or zero-false-alarm requirement.'}


def main(argv=None):
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--stage-dir',required=True,type=Path)
    a.add_argument('--kind',required=True,choices=('depth','frontier'))
    a.add_argument('--episodes-per-condition',required=True,type=int)
    a.add_argument('--conditions',required=True,type=int)
    a.add_argument('--expected-release',required=True,type=float)
    a.add_argument('--source-commit',required=True)
    x=a.parse_args(argv)
    try:
        record=validate(x.stage_dir,x.kind,x.episodes_per_condition,x.conditions,x.expected_release,x.source_commit)
    except Exception as e:
        record={'schema_version':ST.SCHEMA_VERSION,'status':'failed','passed':False,'kind':x.kind,
                'source_commit':x.source_commit,'error':str(e),'interpretation':'Operational evidence gate failed; no performance conclusion.'}
    x.stage_dir.mkdir(parents=True,exist_ok=True)
    ST.write_json(x.stage_dir/'audit.json',record)
    print(json.dumps(record,indent=2))
    return 0 if record['passed'] else 1


if __name__=='__main__': raise SystemExit(main())
