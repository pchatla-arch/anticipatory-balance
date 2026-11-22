"""Run a fresh uncorrected development trajectory with optional transparent physics observations."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import jax
import jax.numpy as jp
import numpy as np
from mujoco import mjx
from mujoco_playground._src import mjx_env

from foresight.adaptive import bench as B
from foresight.adaptive import sensing as S
from foresight.adaptive.actuator_env import AdaptiveJoystick
from foresight.envs.g1_impact.impact_joystick import default_config
from foresight.eval import rollout as R


STEPS = 300
N = 2
SEED = 17623


class SubstepObserver(AdaptiveJoystick):
    """Keep the original step, control, faulted model and solver; record substep q after each advance."""
    def reset(self,key):
        result=super().reset(key)
        info=dict(result.info)
        info["audit_substep_q"]=jp.zeros((self.n_substeps,self.action_size),dtype=jp.float32)
        return result.replace(info=info)

    def step(self,state,action):
        observations=[]
        def instrumented(model,data,action,n_substeps=1):
            def one_step(carry,_):
                after=mjx.step(model, carry.replace(ctrl=action))
                return after,self.measured_joint_positions(after)
            final, q = jax.lax.scan(one_step,data,(),n_substeps)
            observations.append(q)
            return final
        with patch.object(mjx_env, "step", instrumented):
            result=super().step(state,action)
        if len(observations)!=1:
            raise RuntimeError("Expected exactly one physics scan per transition")
        info=dict(result.info)
        info["audit_substep_q"]=observations[0]
        return result.replace(info=info)


def with_substep_trace():
    """Append one trace field to a local copy of the unmodified rollout builder."""
    src=inspect.getsource(B.build_bench_rollout)
    anchor='ys.update(authority_reference=reference,'
    if src.count(anchor)!=1:
        raise RuntimeError("Unexpected benchmark trace structure")
    body=src.replace(anchor,'ys.update(audit_substep_q=next_state.info["audit_substep_q"], authority_reference=reference,')
    namespace=dict(vars(B))
    exec(compile(body,"<trace-only copy of build_bench_rollout>","exec"),namespace)
    return namespace["build_bench_rollout"]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    config=default_config()
    config.impact.resample_on_autoreset=False
    config.impact.miss_fraction=0.
    config.impact.reset_perturb=1.
    config.push_config.enable=False
    config.episode_length=STEPS
    checkpoint=R.latest_checkpoint(args.root/"checkpoints/nominal_s1")
    policy=R.load_policy(checkpoint,deterministic=True)
    print("CPU devices:",jax.devices(),flush=True)
    if any(device.platform!="cpu" for device in jax.devices()):
        raise ValueError("CPU-only protocol")
    results={}
    for name,cls,sensor in [
        ("reference",AdaptiveJoystick,S.DECLARED),
        ("declared",SubstepObserver,S.DECLARED),
        ("immediate",SubstepObserver,S.DECLARED._replace(latency_min=0,latency_max=0))]:
        print("Starting",name,flush=True)
        start=time.monotonic()
        env=cls(config=config,config_overrides={"impl":"jax"},telemetry=True)
        keys=jax.vmap(lambda i:jax.random.fold_in(jax.random.PRNGKey(SEED),i))(jp.arange(N,dtype=jp.int32))
        initial=jax.jit(jax.vmap(env.reset))(keys)
        initial=R.set_scenario(env,initial,launch=False,command=jp.zeros((N,3),jp.float32))
        scenario=B.draw_scenarios(B.FAMILIES["gain_070"],SEED,N,env)
        values={k:jp.asarray(v) for k,v in scenario.items() if k!="ids"}
        builder=B.build_bench_rollout if name=="reference" else with_substep_trace()
        entry=builder(env,policy,B.METHODS["none"],sensor,B.WALK_TURN_STOP_RESUME,STEPS,trace=True,trace_authority_packets=True)
        metrics=jax.device_get(entry(initial,values))
        trace=metrics.pop("trace")
        trace.update(onset=scenario["onset"],ret=scenario["return"])
        if np.any(metrics["first_nonfinite_step"]>=0) or np.any(trace["effective"]!=0):
            raise ValueError("Invalid trajectory or nonzero intervention")
        if name!="reference":
            q=trace["audit_substep_q"]
            live=trace["alive"]
            if not np.array_equal(q[:,:,-1,:][live],trace["authority_source_q_truth"][live]):
                raise ValueError("Observed final substep differs from the original control boundary")
        if name=="declared":
            for k,x in reference.items():
                if not np.array_equal(x,trace[k],equal_nan=True):
                    raise ValueError("Observer altered reference field "+k)
        if name=="reference":
            reference=trace
        if name=="immediate":
            for k in ("audit_substep_q","authority_source_q_truth","authority_decision_q_truth","effective","alive","onset","ret","command","velocity"):
                if not np.array_equal(np.asarray(trace[k]),np.asarray(declared[k]),equal_nan=True):
                    raise ValueError("Transport altered common trajectory "+k)
        if name=="declared":
            declared=trace
        if name!="reference":
            np.savez_compressed(args.out/(name+".npz"),**trace)
        results[name]={"wall_seconds":round(time.monotonic()-start,2),
                       "falls":int(np.sum(np.asarray(metrics["term_step"])>=0)),
                       "onset":scenario["onset"].tolist(),
                       "seed":SEED,"episodes":N,"steps":STEPS,"physics_steps_per_hold":int(env.n_substeps),
                       "physics_dt":float(env.sim_dt),"control_dt":float(env.dt),
                       "joint_names":list(env.actuator_joint_names),
                       "sensor_config":sensor._asdict(),
                       "sensor_keys":scenario["sensor_key"].tolist(),
                       "common_reference_fields":sorted(reference),
                       "source_commit":"b814702456011974e59d9afe2568096208794b1e"}
        print("Finished",name,results[name]["wall_seconds"],"s",flush=True)
    for name in ('declared','immediate'):
        results[name]["archive_sha256"]=hashlib.sha256((args.out/(name+".npz")).read_bytes()).hexdigest()
    (args.out/"capture_manifest.json").write_text(json.dumps(results,indent=2)+"\n")


if __name__=="__main__":
    main()
