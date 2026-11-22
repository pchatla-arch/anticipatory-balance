"""Score fixed envelopes on two fresh matched development trajectories."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path

import jax
import numpy as np

from replay_velocity import ACCELERATIONS, ReplayVariant, summarize


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path)
    parser.add_argument("capture",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--transport",nargs="+",choices=("declared","immediate","one_tick"),default=("declared","immediate"))
    args=parser.parse_args()
    if jax.default_backend()!="cpu":
        raise SystemExit("Use JAX_PLATFORMS=cpu.")
    spec=importlib.util.spec_from_file_location("original_replay",args.study/"replay_authority.py")
    archive=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    config=json.loads((args.study/"dev/config.json").read_text())
    bounds=archive.bind(config,"declared")
    names=config["candidate_model"]["joint_names"]
    groups={g:np.array([s in n for n in names]) for g,s in [("hips","hip"),("knees","knee")]}
    leg=groups["hips"]|groups["knees"]|np.array(["ankle" in n for n in names])
    rows=[]
    for transport in args.transport:
        with np.load(args.capture/(transport+".npz")) as z:
            frozen={k:z[k] for k in z.files if not k.startswith("audit_")}
        base_velocity=frozen["authority_rls_last_velocity"]
        age_source=frozen["authority_rls_last_source_step"]
        ticks,lanes,joints=base_velocity.shape
        errors=np.random.default_rng(47623).uniform(-1,1,size=(ticks,lanes,joints))
        perturbation=np.where(age_source>=0,errors[np.maximum(age_source,0),np.arange(lanes)[None,:,None],
                                                    np.arange(joints)[None,None,:]],0.0)
        for velocity_error in (0.0,0.5):
            observed={**frozen,"authority_rls_last_velocity":(base_velocity+velocity_error*perturbation).astype(base_velocity.dtype)}
            for a in ACCELERATIONS:
                for name,signed in (("speed_magnitude",False),("signed_velocity",True)):
                    variant=ReplayVariant(bounds,a,signed,velocity_error)
                    out=archive.replay(observed,variant,.70,leg)
                    for group,ids in groups.items():
                        for window in ("first_second","all_post_onset"):
                            row=summarize(observed,out,bounds,group,window,ids)
                            rows.append(dict(transport=transport,velocity_error_rad_s=velocity_error,envelope=name,
                                             acceleration_rad_s2=a,**row))
            print(f"Scored {transport}, velocity error {velocity_error} rad/s",flush=True)
    args.out.mkdir(parents=True,exist_ok=True)
    with (args.out/"fresh_authority.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
