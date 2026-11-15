"""Same-code Session H follow-up; no training or response-model refitting."""
from pathlib import Path

BASE_SOURCE = '84c161a6d3782a93385d4a258ecc48487e5bfe57'
PROTOCOL = 'foresight.session_h.v1'


def make_plan(output, mode='full'):
    if mode not in ('full','smoke'):
        raise ValueError('Mode must be full or smoke')
    output=Path(output)
    common=['--nominal','checkpoints/nominal_s1','--brace','checkpoints/brace2_s1',
            '--response_model','results/response_model.json','--impl','warp','--steps','300']
    plan=[]
    def depth(name,n,batches,latency,seed,split):
        plan.append(dict(name=name,kind='depth',episodes_per_condition=n*batches,conditions=9,
                         seed=seed,batches=batches,split=split,release_after=.3,
                         args=['scripts/eval_depth_closed_loop.py',*common,'--episodes',str(n),
                               '--batches',str(batches),'--period','5','--extra_latency',str(latency),
                               '--seed',str(seed),'--release_after','0.3',
                               '--sources','parametric,depth',
                               '--methods','ttc_threshold:0.3,belief_lead:0.05,belief_wait:0.05,belief_voi:0.05',
                               '--out',str(output/name)]))
    def hold(name,n,batches,release,seed,split):
        plan.append(dict(name=name,kind='frontier',episodes_per_condition=n*batches,conditions=9,
                         seed=seed,batches=batches,split=split,release_after=release,
                         args=['scripts/eval_decision_frontier.py',*common,'--episodes',str(n),
                               '--batches',str(batches),'--period','5','--latencies','0,5',
                               '--seed',str(seed),'--release_after',str(release),'--hold','0.5',
                               '--methods','reactive,ttc_threshold,belief_lead','--thetas','0.3,0.4',
                               '--c_preps','0.05,0.1','--calib_params','',
                               '--out',str(output/name)]))
    depth('depth_gate_l0',16,1,0,9300,'gate')
    depth('depth_gate_l5',16,1,5,9300,'gate')
    hold('hold_gate',32,1,.3,9400,'gate')
    if mode=='full':
        # Repeat G scenarios deliberately: paired follow-up, not untouched test.
        for release in (.3,.6,1.):
            hold('E18b_hold_'+str(release).replace('.','p'),1024,2,release,200,'followup')
        depth('E17c_depth_lat0',256,2,0,300,'followup')
        depth('E17c_depth_lat100',256,2,5,300,'followup')
    return plan


def assert_disjoint_splits(plan):
    seen={}
    for s in plan:
        for seed in range(s['seed'],s['seed']+s['batches']):
            old=seen.setdefault(seed,s['split'])
            if old!=s['split']:
                raise ValueError('Gate and follow-up scenario seeds overlap')
    return True
