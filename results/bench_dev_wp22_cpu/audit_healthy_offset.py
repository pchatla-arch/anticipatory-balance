"""Recompute AE-011 per-episode metrics from raw effective-offset traces on the dev split."""
import csv
from pathlib import Path
import numpy as np

def build_traces(raw, methods, dest):
    """Collect alive flags and the maximum absolute effective joint correction each tick."""
    series = {}; cache = {}
    for method, rows in raw.items():
        for row in rows:
            name = row['family'], row['telemetry'], method
            if name not in cache:
                filename = f"trace_{name[0]}_{name[1]}_{method}.npz"
                with np.load(methods[method]/filename, allow_pickle=False) as t:
                    eff = t['effective']
                    assert eff.shape == (300,8,29)
                    cache[name] = t['alive'].T.copy(), np.max(np.abs(eff),axis=-1).T, t['onset'].copy(), t['ret'].copy()
            alive, amplitude, onset, ret = cache[name]
            lane = int(row['scenario_id'].split('_env')[-1])
            truth_return = int(ret[lane])
            assert int(onset[lane])==row['onset_step'] and (truth_return if truth_return<2**29 else -1)==row['return_step']
            ident = '|'.join([*name,row['scenario_id']])
            assert ident not in series
            series[ident] = alive[lane], amplitude[lane]
    ordered=sorted(series)
    np.savez_compressed(dest, episodes=np.asarray(ordered), alive=np.asarray([series[k][0] for k in ordered]),
                        max_abs_effective_rad=np.asarray([series[k][1] for k in ordered]))


def audit(rows, trace):
    with np.load(trace, allow_pickle=False) as data:
        keys = {str(k):i for i,k in enumerate(data['episodes'])}
        alive=data['alive']; magnitude=data['max_abs_effective_rad']
    assert len(keys)==len(rows)==128
    for row in rows:
        k = '|'.join(str(row[f]) for f in ('family','telemetry','method','scenario_id'))
        lane=keys[k]
        obs=alive[lane].astype(bool); q=magnitude[lane].astype(np.float64)
        assert len(q)==len(obs)==300 and np.isfinite(q[obs]).all()
        on,ret=int(row['onset_step']),int(row['return_step'])
        t=np.arange(len(q))
        schedule_inactive=(t<on) | ((ret>=0)&(t>=ret))
        original=obs & schedule_inactive
        healthy=obs & (schedule_inactive | (float(row['gain'])==1.0))
        assert original.any() and healthy.any()
        assert abs(q[original].mean()-float(row['harmful_offset_rad']))<2e-6
        yield {'family':row['family'], 'telemetry':row['telemetry'], 'method':row['method'],
               'scenario_id':row['scenario_id'], 'onset_step':on, 'return_step':ret,
               'plant_gain':float(row['gain']), 'observed_ticks':int(obs.sum()),
               'harness_scored_ticks':int(original.sum()), 'actual_healthy_ticks':int(healthy.sum()),
               'harness_healthy_offset_rad':float(row['harmful_offset_rad']),
               'actual_healthy_offset_rad':float(q[healthy].mean())}

if __name__=='__main__':
    root=Path(__file__).resolve().parent
    with (root/'episodes.csv').open(newline='') as inp:
        audited=list(audit(list(csv.DictReader(inp)),root/'effective_offset_ticks.npz'))
    with (root/'physical_health_episodes.csv').open(newline='') as inp:
        expected=list(csv.DictReader(inp))
    assert len(expected)==len(audited)
    for actual,stored in zip(audited,expected,strict=True):
        for k,v in actual.items():
            assert (float(v)==float(stored[k]) if isinstance(v,(float,int)) else v==stored[k]),(k,stored[k],v)
    print(f"AE-011 physical-health masks and per-episode offsets verified for {len(audited)} rows.")
