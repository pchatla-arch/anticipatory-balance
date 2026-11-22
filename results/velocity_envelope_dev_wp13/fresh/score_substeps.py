"""Score development substep positions against causal velocity envelopes."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


GRID=(25,50,100,200,400,800,1600)


def packet_map(data):
    time,env,slot=np.nonzero(data["authority_release_mask"])
    return {(int(e),int(s)): (int(t),int(k)) for t,e,k,s in
            zip(time,env,slot,data["authority_packet_source_step"][time,env,slot])}


def verify_transport(first,second):
    early=packet_map(second)
    delayed=packet_map(first)
    overlap=delayed.keys() & early.keys()
    if not overlap or delayed.keys()-early.keys():
        raise ValueError("Missing packets in zero-delay stream")
    for e,s in overlap:
        t,j=delayed[e,s]
        u,k=early[e,s]
        if s!=u or t<=s:
            raise ValueError("Transport time mismatch")
        for name in ("position_rad","velocity_rad_s","torque_nm","valid","target_rad"):
            left=first["authority_packet_"+name][t,e,j]
            right=second["authority_packet_"+name][u,e,k]
            if not np.array_equal(left,right):
                raise ValueError(f"Transport changed measured packet {name}")
    extras=sorted(early.keys()-delayed.keys())
    if any(s<len(first["alive"])-5 for _,s in extras):
        raise ValueError("Non-tail source released under only one transport")
    return {"common_source_packets":len(overlap),"additional_horizon_tail_released_immediately":extras}


def score(data,info,noise,channel):
    q0=data["authority_decision_q_truth"]
    q=np.concatenate((q0[:,:,None,:],data["audit_substep_q"]),axis=2).astype(float)
    ticks,envs,samples,joints=q.shape
    dt=info["control_dt"]
    if not np.isclose((samples-1)*info["physics_dt"],dt):
        raise ValueError("Physics grid length mismatch")
    phase=np.arange(samples)[None,None,:,None]*info["physics_dt"]
    time=np.arange(ticks)[:,None]
    since_onset=time-data["onset"][None,:]
    s=data["authority_rls_last_source_step"]
    age=(time[:,:,None]-s-1)*dt
    if np.any(age[s>=0]<0):
        raise ValueError("Non-causal cached source")
    tau=age[:,:,None,:]+phase
    position=data["authority_rls_last_position"][:,:,None,:].astype(float)
    velocity=data["authority_rls_last_velocity"][:,:,None,:].astype(float)
    reference=data["authority_reference"]
    pos_error=info["sensor_config"]["bound_position_rad"]
    # Use the same bounded velocity errors at each original source and lane
    # independently of transport. Entries with no source are excluded.
    unit_noise=noise[np.maximum(s,0),np.arange(envs)[None,:,None],np.arange(joints)[None,None,:]][:,:,None,:]
    methods=[("speed_10", 0, 0.)]
    methods.extend((kind,a,bv) for kind in ("velocity_magnitude","velocity_signed")
                   for bv in (0.,.5) for a in GRID)
    rows=[]
    for kind,A,bv in methods:
        vhat=velocity+bv*unit_noise
        if kind=="speed_10":
            center=position
            radius=pos_error+10*tau
        elif kind=="velocity_magnitude":
            center=position
            radius=pos_error+(np.abs(vhat)+bv)*tau+0.5*A*tau**2
        elif kind=="velocity_signed":
            center=position+vhat*tau
            radius=pos_error+bv*tau+0.5*A*tau**2
        lower=center-radius
        upper=center+radius
        # Concavity of the lower bound and convexity of the upper permit this exact hull.
        hold_lo=np.minimum(lower[:,:,0,:],lower[:,:,-1,:])
        hold_hi=np.maximum(upper[:,:,0,:],upper[:,:,-1,:])
        point_excess=np.maximum(0,np.maximum(lower-q,q-upper))
        hull_excess=np.maximum(0,np.maximum(hold_lo[:,:,None,:]-q,q-hold_hi[:,:,None,:]))
        fail=np.any(point_excess>0,axis=2)
        endpoint_fail=(point_excess[:,:,0,:]>0)|(point_excess[:,:,-1,:]>0)
        hull_fail=np.any(hull_excess>0,axis=2)
        end_hull_fail=(hull_excess[:,:,0,:]>0)|(hull_excess[:,:,-1,:]>0)
        sign_known=(reference<hold_lo)|(reference>hold_hi)
        for win,window in (("first_second",since_onset<=50),("all_post_onset",np.ones_like(since_onset,bool))):
            for group,part in (("hips","hip"),("knees","knee")):
                select=np.array([part in n for n in info["joint_names"]])
                samples_live=data["alive"] & (since_onset>=0) & window
                mask=samples_live[:,:,None] & select & (s>=0)
                n=int(mask.sum())
                if n==0 or not np.isfinite(point_excess.max(axis=2)[mask]).all():
                    raise ValueError("Unscorable trajectory")
                rows.append({"channel":channel,"window":win,"joint_group":group,"method":kind,
                             "acceleration_rad_s2":A,"velocity_error_rad_s":bv,"n_joint_decisions":n,
                             "mean_source_age_ms":float(age[mask].mean()*1000),
                             "source_age_max_ticks":round(float(age[mask].max()/dt)),
                             "substeps_per_hold":samples-1,
                             "all_sampled_points_coverage":float(1-fail[mask].mean()),
                             "joint_decisions_failed":int(fail[mask].sum()),
                             "interior_only_failures":int((fail & ~endpoint_fail)[mask].sum()),
                             "decisions_outside_hold_hull":int(hull_fail[mask].sum()),
                             "interior_only_hull_failures":int((hull_fail & ~end_hull_fail)[mask].sum()),
                             "max_excess_rad":float(point_excess.max(axis=2)[mask].max()),
                             "mean_hold_width_rad":float((hold_hi-hold_lo)[mask].mean()),
                             "fraction_with_known_error_sign":float(sign_known[mask].mean())})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.directory/"capture_manifest.json").read_text())
    streams={}
    results=[]
    for channel in ("declared","immediate"):
        p=args.directory/(channel+".npz")
        if hashlib.sha256(p.read_bytes()).hexdigest()!=manifest[channel]["archive_sha256"]:
            raise ValueError("Capture checksum mismatch")
        with np.load(p) as z:
            streams[channel]=dict(z)
    audit=verify_transport(streams["declared"],streams["immediate"])
    noise=np.random.default_rng(47623).uniform(-1,1,size=streams["declared"]["authority_reference"].shape)
    for channel,data in streams.items():
        results.extend(score(data,manifest[channel],noise,channel))
    with (args.directory/"coverage.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=results[0]);w.writeheader();w.writerows(results)
    (args.directory/"packet_pairing.json").write_text(json.dumps(audit,indent=2)+"\n")
    print(f"{len(results)} strata; {audit['common_source_packets']} independently aligned source packets")


if __name__=="__main__":
    main()
