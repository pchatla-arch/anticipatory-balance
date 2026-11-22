"""Fixed 20 ms transport follow-up, reusing the original frozen substep observer and benchmark."""
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from capture_substeps import B, N, R, S, SEED, STEPS, SubstepObserver, default_config, with_substep_trace
from score_substeps import packet_map, score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("repo",type=Path)
    p.add_argument("prior_capture",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    start=time.monotonic()
    if any(d.platform!="cpu" for d in jax.devices()):raise ValueError("CPU protocol")
    cfg=default_config()
    cfg.impact.resample_on_autoreset=False
    cfg.impact.miss_fraction=0.
    cfg.impact.reset_perturb=1.
    cfg.push_config.enable=False
    cfg.episode_length=STEPS
    env=SubstepObserver(config=cfg,config_overrides={"impl":"jax"},telemetry=True)
    policy=R.load_policy(R.latest_checkpoint(args.repo/"checkpoints/nominal_s1"),deterministic=True)
    keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(SEED),i))(jp.arange(N,dtype=jp.int32))
    initial=jax.jit(jax.vmap(env.reset))(keys)
    initial=R.set_scenario(env,initial,launch=False,command=jp.zeros((N,3),jp.float32))
    sc=B.draw_scenarios(B.FAMILIES["gain_070"],SEED,N,env)
    scenario={k:jp.asarray(v) for k,v in sc.items() if k!="ids"}
    sensor=S.DECLARED._replace(latency_min=1,latency_max=1)
    entry=with_substep_trace()(env,policy,B.METHODS["none"],sensor,B.WALK_TURN_STOP_RESUME,STEPS,trace=True,trace_authority_packets=True)
    metrics=jax.device_get(entry(initial,scenario))
    new=metrics.pop("trace")
    new.update(onset=sc["onset"],ret=sc["return"])
    if np.any(new["effective"]!=0) or np.any(metrics["first_nonfinite_step"]>=0):
        raise ValueError("Unexpected active control or nonfinite episode")
    if not np.array_equal(new["audit_substep_q"][:,:,-1][new["alive"]],new["authority_source_q_truth"][new["alive"]]):
        raise ValueError("End of held control differs")
    manifest=json.loads((args.prior_capture/"capture_manifest.json").read_text())
    shared_fields=("audit_substep_q","authority_source_q_truth","authority_source_torque_truth","authority_decision_q_truth","authority_previous_request","authority_reference","command","velocity","alive","effective","loss","onset","ret")
    pairing={}
    new_packets=packet_map(new)
    times,lanes,slots=np.where(new["authority_release_mask"])
    if np.any(new["authority_packet_arrival_step"][times,lanes,slots]-new["authority_packet_source_step"][times,lanes,slots]!=1):raise ValueError("Unexpected transport latency")
    for name in ("declared","immediate"):
        f=args.prior_capture/(name+".npz")
        if hashlib.sha256(f.read_bytes()).hexdigest()!=manifest[name]["archive_sha256"]:raise ValueError("Stored source archive changed")
        with np.load(f) as z:old=dict(z)
        if any(not np.array_equal(new[k],old[k]) for k in shared_fields):raise ValueError("Changed trajectory")
        old_packets=packet_map(old)
        overlap=new_packets.keys() & old_packets.keys()
        diff=set(new_packets)^set(old_packets)
        if any(src<STEPS-5 for _,src in diff):raise ValueError("Non-tail dropped-packet mismatch")
        for e,s in overlap:
            t,i=new_packets[e,s];u,j=old_packets[e,s]
            for k in ("target_rad","valid","position_rad","torque_nm","velocity_rad_s"):
                k="authority_packet_"+k
                if not np.array_equal(new[k][t,e,i],old[k][u,e,j]):raise ValueError("Unpaired source packet")
        pairing[name]={"shared_source_packets":len(overlap),"tail_sources_available_in_only_one_arm":sorted(diff)}
    info={**manifest["immediate"],"sensor_config":sensor._asdict(),"shared_physical_fields":shared_fields,
          "pairing":pairing,"falls":int(np.sum(np.asarray(metrics["term_step"])>=0))}
    for key in ("common_reference_fields","archive_sha256","wall_seconds"):
        info.pop(key,None)
    out=args.out/"one_tick.npz"
    np.savez_compressed(out,**new)
    info["archive_sha256"]=hashlib.sha256(out.read_bytes()).hexdigest()
    info["wall_seconds"]=round(time.monotonic()-start,2)
    (args.out/"capture_manifest.json").write_text(json.dumps(info,indent=2)+"\n")
    noise=np.random.default_rng(47623).uniform(-1,1,size=new["authority_reference"].shape)
    rows=score(new,info,noise,"one_tick")
    with (args.out/"coverage.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    print("Successful fixed 20 ms capture",info["wall_seconds"],"seconds",len(rows),"strata",flush=True)


if __name__=="__main__":
    main()
