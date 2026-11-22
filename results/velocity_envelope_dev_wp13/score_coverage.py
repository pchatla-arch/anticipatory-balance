"""Score the frozen velocity-informed envelope experiment (NumPy; no physics executable)."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ACCEL_LIMITS = (25, 50, 100, 200, 400, 800, 1600)
SOURCE = "b814702456011974e59d9afe2568096208794b1e"
FAMILIES = ("gain_085", "gain_070", "gain_055")
CHANNELS = ("exact", "declared")


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def bounds(position, velocity, age, dt, pos_error, acceleration, envelope):
    """Return pointwise bounds at the decision and hold end, plus their enclosing hull."""
    def at(tau):
        if envelope == "speed_10":
            center = position
            radius = pos_error + 10 * tau
        elif envelope == "velocity_magnitude":
            center = position
            radius = pos_error + np.abs(velocity) * tau + 0.5 * acceleration * tau ** 2
        else:
            center = position + velocity * tau
            radius = pos_error + 0.5 * acceleration * tau ** 2
        return center - radius, center + radius

    lo0, hi0 = at(age)
    lo1, hi1 = at(age + dt)
    return lo0, hi0, lo1, hi1, np.minimum(lo0, lo1), np.maximum(hi0, hi1)


def excess(q, low, high):
    return np.maximum(0, np.maximum(low - q, q - high))


def checked_sources(data):
    """Reconstruct capture-time inputs from delivered packets, independent of the stored RLS cache."""
    source = data["authority_rls_last_source_step"]
    ticks, neps, njoints = source.shape
    if np.any(source > np.arange(ticks)[:, None, None] - 1):
        raise ValueError("The RLS cache contains a future observation")
    for name in ("position_rad", "velocity_rad_s"):
        by_source = np.full(source.shape, np.nan, dtype=np.float64)
        packet_key = "authority_packet_" + name
        released_t, ep, slot = np.nonzero(data["authority_release_mask"])
        index = data["authority_packet_source_step"][released_t, ep, slot]
        if np.any(index > released_t) or np.any(index < 0):
            raise ValueError("Invalid release order")
        by_source[index, ep] = data[packet_key][released_t, ep, slot]
        env = np.arange(neps)[None, :, None]
        joint = np.arange(njoints)[None, None, :]
        present = source >= 0
        key = "authority_rls_last_" + ("position" if name == "position_rad" else "velocity")
        recorded = data[key]
        if not np.array_equal(recorded[present], by_source[source, env, joint][present]):
            raise ValueError(f"Source/measurement mismatch for {key}")
    return source


def statistics(data, names, dt, pos_error):
    source = checked_sources(data)
    ticks, neps, njoints = source.shape
    time = np.arange(ticks)[:, None]
    age_ticks = time[:, :, None] - 1 - source
    age = np.maximum(age_ticks, 0) * dt
    windows = {"first_second": (time - data["onset"] <= 50),
               "all_post_onset": np.ones((ticks, neps), dtype=bool)}
    alive = data["alive"] & (time >= data["onset"])
    q_start = data["authority_decision_q_truth"].astype(np.float64)
    q_end = data["authority_source_q_truth"].astype(np.float64)
    position = data["authority_rls_last_position"].astype(np.float64)
    velocity = data["authority_rls_last_velocity"].astype(np.float64)
    reference = data["authority_reference"]
    # Consecutive observed transitions share the same physical boundary.
    shared = data["alive"][:-1] & data["alive"][1:]
    if not np.array_equal(q_end[:-1][shared], q_start[1:][shared]):
        raise ValueError("Discontinuous position labels between live transitions")
    rows = []
    methods = [("speed_10", None)]
    methods.extend((name, value) for name in ("velocity_magnitude", "velocity_signed") for value in ACCEL_LIMITS)
    for envelope, acceleration in methods:
        lo0, hi0, lo1, hi1, lower, upper = bounds(position, velocity, age, dt, pos_error, acceleration, envelope)
        start_miss = excess(q_start, lo0, hi0)
        end_miss = excess(q_end, lo1, hi1)
        worst = np.maximum(start_miss, end_miss)
        hull_miss = np.maximum(excess(q_start, lower, upper), excess(q_end, lower, upper))
        width = upper - lower
        sign_known = (reference < lower) | (reference > upper)
        for group, select in names.items():
            for window, limit in windows.items():
                exposure = (alive & limit)[:, :, None] & select
                with_source = exposure & (source >= 0)
                for label, age_mask in (("all", True), ("0", age_ticks == 0),
                                        ("1", age_ticks == 1), ("2", age_ticks == 2),
                                        ("3", age_ticks == 3), ("4+", age_ticks >= 4)):
                    mask = with_source & age_mask
                    n = int(mask.sum())
                    if n == 0 and label != "all":
                        continue
                    if n == 0 or not np.all(np.isfinite(worst[mask]) & np.isfinite(width[mask])):
                        raise ValueError("Empty or invalid scoring observations")
                    start_failed = start_miss[mask] > 0
                    end_failed = end_miss[mask] > 0
                    failed = worst[mask] > 0
                    rows.append({
                        "joint_group": group, "window": window, "age_ticks": label,
                        "envelope": envelope,
                        "accel_rad_s2": "" if acceleration is None else acceleration,
                        "scored_joint_steps": n,
                        "missing_source_all_ages": int((exposure & (source < 0)).sum()) if label == "all" else "",
                        "decision_failures": int(start_failed.sum()), "end_failures": int(end_failed.sum()),
                        "either_pointwise_failed": int(failed.sum()),
                        "both_pointwise_coverage": float(1-failed.mean()),
                        "joint_episodes_observed": int(np.any(mask,axis=0).sum()),
                        "joint_episodes_with_any_failure": int(np.any((worst>0)&mask,axis=0).sum()),
                        "outside_hold_hull": int((hull_miss[mask] > 0).sum()),
                        "max_pointwise_excess_rad": float(worst[mask].max()),
                        "max_hold_hull_excess_rad": float(hull_miss[mask].max()),
                        "mean_hold_width_rad": float(width[mask].mean()),
                        "sign_known_joint_steps": int(sign_known[mask].sum()),
                        "sign_known_fraction": float(sign_known[mask].mean())})
    return rows


def acceleration_evidence(data, joint_names, groups, dt, family):
    """Exact-sensor velocities at successive source times; finite differences average acceleration."""
    t, ep, slot = np.nonzero(data["authority_release_mask"])
    source = data["authority_packet_source_step"][t, ep, slot]
    if not np.array_equal(t, source):
        raise ValueError("Expected immediate exact delivery")
    velocity = np.full(data["authority_decision_q_truth"].shape, np.nan, dtype=np.float64)
    velocity[source,ep] = data["authority_packet_velocity_rad_s"][t,ep,slot]
    accel = np.abs(np.diff(velocity,axis=0))/dt
    steps = np.arange(1, len(velocity))[:,None]
    # Both endpoint source packets are generated under the post-onset dynamics.
    age = steps-data["onset"][None,:]
    observed = data["alive"][:-1] & data["alive"][1:] & (age >= 1)
    axes = dict(groups)
    axes.update({name: np.array([i == j for i in range(len(joint_names))])
                 for j,name in enumerate(joint_names) if any(g[j] for g in groups.values())})
    rows=[]
    for scope,interval in (("first_second", observed & (age<=50)), ("all_post_onset", observed)):
        for name, select in axes.items():
            values=accel[interval][:,select].ravel()
            if len(values)==0 or not np.isfinite(values).all():
                raise ValueError("Unobserved derivative samples")
            q=np.quantile(values,[0.5,0.95,0.99,0.999,1.0])
            rows.append({"family":family,"window":scope,"axis":name,"successive_joint_intervals":len(values),
                         **{k:float(v) for k,v in zip(("p50","p95","p99","p999","max"),q)},
                         **{f"above_{a}":int((values>a).sum()) for a in ACCEL_LIMITS}})
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study",type=Path,help="results/authority_replay_dev_wp13 in the immutable source")
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    source=args.study/"dev"
    summary=json.loads((args.study/"summary.json").read_text())
    config=json.loads((source/"config.json").read_text())
    model=config["candidate_model"]
    dt=model["dt_s"]
    if not np.isclose(dt,.02):
        raise ValueError("Protocol requires 20 ms control")
    joint_names=model["joint_names"]
    groups={"hips":np.array(["hip" in j for j in joint_names]),"knees":np.array(["knee" in j for j in joint_names])}
    all_rows=[]
    acc_rows=[]
    for family in FAMILIES:
        for channel in CHANNELS:
            name=f"trace_{family}_{channel}_none.npz"
            path=source/name
            if hashlib.sha256(path.read_bytes()).hexdigest()!=summary["capture_sha256"][name]:
                raise ValueError(f"Capture digest mismatch for {name}")
            with np.load(path) as trace:
                data=dict(trace)
            if np.any(data["effective"]!=0):
                raise ValueError("Expected uncorrected physics")
            rows=statistics(data,groups,dt,config["telemetry"][channel]["bound_position_rad"])
            all_rows.extend({"family":family,"telemetry":channel,**row} for row in rows)
            if channel=="exact":
                acc_rows.extend(acceleration_evidence(data,joint_names,groups,dt,family))
    write_csv(args.out/"coverage.csv", all_rows)
    write_csv(args.out/"acceleration.csv",acc_rows)
    print(f"Verified six source archives; scored {len(all_rows)} stratified rows and {len(acc_rows)} derivative rows.")


if __name__ == "__main__":
    main()
