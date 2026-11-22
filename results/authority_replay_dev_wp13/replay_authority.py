"""Replay development packets from a zero-offset trajectory using its frozen candidate."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jp
import numpy as np

from foresight.adaptive import candidate as C
from foresight.adaptive import estimation as E
from foresight.adaptive import projection as P
from foresight.adaptive import sensing as S


def bind(meta, channel):
    model = meta["candidate_model"]
    sensor = meta["telemetry"][channel]
    force = [np.inf if x is None else x for x in model["nominal_force_ceiling_nm"]]
    return C.Candidate(C.Config(**meta["candidate_config"]), P.Interval(*model["prior_gain"]),
                       jp.array(model["kp"]), jp.array(model["ctrl_range"]),
                       jp.array(model["ctrl_limited"]), jp.array(force),
                       model["max_offset_rad"], model["max_rate_rad_s"], model["dt_s"],
                       sensor["bound_position_rad"], sensor["bound_torque_nm"],
                       meta["rls"]["min_excitation_rad"])


def replay(data, method, gain, is_fault_group):
    signals = {k[len("authority_"):]: jp.asarray(v) for k, v in data.items() if k.startswith("authority_")}
    reference = signals["reference"]
    ticks, episodes, joints = reference.shape
    previous = signals["previous_request"]
    if np.any(np.asarray(previous) != 0) or np.any(data["effective"] != 0):
        raise ValueError("Common-state replay requires an uncorrected walker.")
    initial = method.init(episodes, joints)
    template = jax.vmap(lambda _: E.rls_init(joints, E.RLSConfig()))(jp.arange(episodes))
    indicators = np.arange(ticks)[:, None] >= data["onset"][None, :]
    truth_gain = jp.where(jp.asarray(indicators[:, :, None]) & jp.asarray(is_fault_group)[None, None, :], gain, 1.0)

    def scan(state, inputs):
        t, observed_gain, x = inputs
        ref, prev = x["reference"], x["previous_request"]
        rls = template._replace(**{f: x["rls_"+f] for f in ("g_hat", "last_position", "last_velocity", "last_source_step")})
        issued, reason, detail = method.evaluate(state, rls, ref, t, prev)
        def actuate(offset):
            z = ref + offset
            return jp.where(method.ctrl_limited, jp.clip(z, method.ctrl_range[:,0], method.ctrl_range[:,1]), z) - ref
        demand = (1.0/observed_gain-1.0) * (ref-x["decision_q_truth"])
        capped = jp.clip(demand,-method.max_offset,method.max_offset)
        oracle = actuate(jp.clip(capped,prev-method.max_rate*method.dt,prev+method.max_rate*method.dt))
        allowed = (reason == C.Reason.APPLY) | (reason == C.Reason.NO_EFFECTIVE_CHANGE)
        upper = actuate(detail["requested_high"])
        lower = actuate(detail["requested_low"])
        capacity = jp.where(allowed, jp.maximum(0, jp.where(oracle>0,upper,jp.where(oracle<0,-lower,0))),0)
        out = dict(low=detail["current_gain_low"],high=detail["current_gain_high"],oracle=jp.abs(oracle),
                   capacity=capacity,attainable=jp.minimum(capacity,jp.abs(oracle)),issued=issued,reason=reason)
        resets = jp.zeros_like(issued, dtype=jp.int32)
        invalids = jp.zeros_like(resets)
        for slot in range(x["release_mask"].shape[1]):
            packet = S.MeasuredPacket(**{f:x["packet_"+f][:,slot] for f in S.MeasuredPacket._fields})
            released = x["release_mask"][:,slot]
            state, status = method.update_with_status(state,packet,released)
            resets += released[:,None] & (status == P.GainStatus.INCONSISTENT)
            invalids += released[:,None] & (status == P.GainStatus.INVALID_INPUT)
        out.update(resets=resets,invalids=invalids)
        return state,out

    _, output = jax.jit(lambda state, xs: jax.lax.scan(scan, state, xs))(
        initial, (jp.arange(ticks), truth_gain, signals))
    return {k:np.asarray(v) for k,v in output.items()}


def fraction(a,b):
    return float(a/b) if b>0 else None


def mean(x):
    return float(np.mean(x)) if len(x) else None


def measurement_counts(data, select, sensor):
    mask=data["authority_release_mask"]
    source=data["authority_packet_source_step"]
    good=mask & (source>=0)
    time,ep,slot=np.where(good)
    safe_source=source[time,ep,slot]
    good=data["alive"][time,ep] & data["alive"][safe_source,ep]
    time,ep,slot,safe_source=(v[good] for v in (time,ep,slot,safe_source))
    valid=data["authority_packet_valid"][time,ep,slot][:,select]
    out={"released_joint_measurements":int(valid.size),"regression_valid_measurements":int(valid.sum())}
    for key,actual,bound in [("position_rad","q","bound_position_rad"),("torque_nm","torque","bound_torque_nm")]:
        measured=data["authority_packet_"+key][time,ep,slot][:,select]
        truth=data["authority_source_"+actual+"_truth"][safe_source,ep][:,select]
        exceeds=np.abs(measured-truth)>sensor[bound]
        out[key+"_violations"]=int(np.count_nonzero(valid & exceeds))
    return out


def summarize(data, out, select, dt, family, channel, group, sensor):
    alive=data["alive"]
    onset=data["onset"]
    age=np.arange(len(alive))[:,None]-onset[None,:]
    exposure=alive[:,:,None] & np.broadcast_to(select,out["low"].shape)
    post=exposure & (age[:,:,None]>=0)
    exclude=(out["low"]>1)|(out["high"]<1)
    width=out["high"]-out["low"]
    def stats(mask):
        oracle=out["oracle"][mask]
        return dict(joint_steps=int(np.sum(mask)), mean_width=mean(width[mask]),
                    fraction_excluding_one=mean(exclude[mask]),
                    mean_directional_authority_rad=mean(out["capacity"][mask]),
                    mean_oracle_demand_rad=mean(oracle),
                    oracle_fraction=fraction(out["attainable"][mask].sum(dtype=float),oracle.sum(dtype=float)))
    base=dict(family=family,channel=channel,group=group)
    curves=[]
    for tick in range(int(age.max())+1):
        mask=post & (age[:,:,None]==tick)
        curves.append(dict(**base,offset_steps=tick,seconds=tick*dt,
                           observed_episodes=int(np.count_nonzero(alive & (age==tick))),**stats(mask)))
    first=[]
    for n,t in enumerate(onset):
        for j in np.flatnonzero(select):
            when=np.flatnonzero(post[:,n,j]&exclude[:,n,j])
            first.append((when[0]-t)*dt if len(when) else np.nan)
    first=np.asarray(first)
    crossings=[x["seconds"] for x in curves if x["seconds"]<=1+1e-8 and x["oracle_fraction"] is not None and x["oracle_fraction"]>=.5]
    good=post & (age[:,:,None]*dt<=1+1e-8)
    return dict(**base,first_second=stats(good),all_post_onset=stats(post),
                empirical_measurements=measurement_counts(data,select,sensor),
                observed_first_exclusions=int(np.isfinite(first).sum()),
                no_observed_exclusion=int(np.isnan(first).sum()),
                never_exposed_joint_episodes=int(np.sum(~np.any(post[:,:,select],axis=0))),
                median_first_exclusion_s_conditional=float(np.nanmedian(first)) if np.isfinite(first).any() else None,
                first_50pct_crossing_s=crossings[0] if crossings else None,
                ticks_reaching_50pct_first_second=len(crossings),
                inconsistent_all_observed=int(out["resets"][exposure].sum()),
                inconsistent_post_onset=int(out["resets"][post].sum()),
                invalid_all_observed=int(out["invalids"][exposure].sum())),curves


def write_json(path,data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    meta=json.loads((args.capture/"config.json").read_text())
    model=meta["candidate_model"]
    names=model["joint_names"]
    hips=np.array(["hip" in x for x in names])
    knees=np.array(["knee" in x for x in names])
    legs=hips | knees | np.array(["ankle" in x for x in names])
    outputs,curves,checksums=[],[],{}
    for family in ("gain_085","gain_070","gain_055"):
        for channel in ("exact","declared"):
            source=args.capture/f"trace_{family}_{channel}_none.npz"
            with np.load(source) as trace:
                data=dict(trace)
            checksums[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
            if np.any(np.diff(data["alive"].astype(int),axis=0)>0):
                raise ValueError("A terminated episode re-entered the denominator")
            method=bind(meta,channel)
            vals=replay(data,method,meta["families"][family]["gain"],legs)
            for group,select in [("hips",hips),("knees",knees)]:
                summary,rows=summarize(data,vals,select,method.dt,family,channel,group,meta["telemetry"][channel])
                outputs.append(summary)
                curves.extend(rows)
            print(f"Replayed {family} / {channel}",flush=True)
    write_json(args.out/"summary.json",{"capture_sha256":checksums,"results":outputs})
    with (args.out/"curves.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(curves[0]))
        writer.writeheader()
        writer.writerows(curves)


if __name__=="__main__":
    main()
