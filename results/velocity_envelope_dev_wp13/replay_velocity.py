"""CPU-only velocity-envelope screen on the immutable WP1.3 uncorrected traces."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from foresight.adaptive import candidate as C
from foresight.adaptive import estimation as E
from foresight.adaptive import projection as P

ACCELERATIONS = (25, 50, 100, 200, 400, 800, 1600)


def uncertainty_bounds(method, reference, rls, step, acceleration, signed, velocity_error=0.0):
    age = jnp.maximum(step - 1 - rls.last_source_step, 0) * method.dt
    end = age + method.dt
    position = rls.last_position
    velocity = rls.last_velocity
    if signed:
        def edges(t):
            center = position + velocity * t
            radius = method.encoder_error_rad + velocity_error*t + 0.5 * acceleration * t**2
            return center - radius, center + radius

        low_start, high_start = edges(age)
        low_end, high_end = edges(end)
        low = jnp.minimum(low_start, low_end)
        high = jnp.maximum(high_start, high_end)
    else:
        radius = method.encoder_error_rad + (jnp.abs(velocity) + velocity_error) * end + 0.5 * acceleration * end**2
        low, high = position - radius, position + radius
    return P.Interval(reference - high, reference - low)


class ReplayVariant:
    """Use the frozen candidate filter; replace only the hold-error input in the replay."""
    def __init__(self, method, acceleration, signed, velocity_error=0.0):
        self.method = method
        self.acceleration = acceleration
        self.signed = signed
        self.velocity_error = velocity_error

    def __getattr__(self, name):
        return getattr(self.method, name)

    def evaluate(self, state, rls, reference, step, previous):
        m = self.method
        _, _, d = m.evaluate(state, rls, reference, step, previous)
        targets = P.Interval(jnp.where(m.ctrl_limited, m.ctrl_range[:, 0], reference - m.max_offset),
                             jnp.where(m.ctrl_limited, m.ctrl_range[:, 1], reference + m.max_offset))
        q_hat, _ = E.causal_position(rls, step, m.dt)
        projected = P.project_offset(
            P.Interval(d["hold_gain_low"], d["hold_gain_high"]),
            uncertainty_bounds(m, reference, rls, step, self.acceleration, self.signed, self.velocity_error),
            current_gains=P.Interval(d["current_gain_low"], d["current_gain_high"]),
            estimate=rls.g_hat, current_error=reference-q_hat, reference=reference, target_range=targets,
            kp=m.kp, available_force=m.force_ceiling-m.config.force_margin_nm, max_offset=m.max_offset,
            slew_rate=m.max_rate, hold=m.dt, previous=previous, regularizer=m.config.regularizer)
        eligible = d["estimate_supported"] & (projected.status == P.ProjectionStatus.FEASIBLE)
        # The recorded previous command is zero on every trial, so withdrawal issues zero.
        issued = jnp.where(eligible, projected.requested, 0.0)
        reason = jnp.where(eligible, C.Reason.APPLY, C.Reason.NO_TELEMETRY)
        return issued, reason, {**d, "requested_low": projected.requested_interval.low,
                                "requested_high": projected.requested_interval.high}


def ratio(mass, demand):
    return float(mass.sum(dtype=np.float64) / demand.sum(dtype=np.float64)) if demand.any() else None


def summarize(data, out, method, group, window, select):
    relative = np.arange(len(data["alive"]))[:, None] - data["onset"][None, :]
    samples = data["alive"] & (relative >= 0)
    if window == "first_second":
        samples &= relative * method.dt <= 1.0000001
    mask = samples[:, :, None] & select[None, None, :]
    reference = data["authority_reference"]
    issued = out["issued"]
    target = reference + issued
    limits = np.asarray(method.ctrl_range)
    clipped = np.clip(target, limits[:, 0], limits[:, 1])
    effective = np.where(np.asarray(method.ctrl_limited), clipped, target) - reference
    sign = np.sign(reference - data["authority_decision_q_truth"])
    # All replay families are gain losses. The sign of the known-gain inverse matches the tracking error.
    directed = np.clip(effective * sign, 0.0, out["oracle"])
    demand = out["oracle"][mask]
    return dict(group=group, window=window, observed_joint_ticks=int(mask.sum()),
                oracle_sum_rad=float(demand.sum(dtype=np.float64)),
                applied_over_oracle=ratio(np.abs(effective[mask]), demand),
                directional_applied_fraction=ratio(directed[mask], demand),
                directional_authority_fraction=ratio(out["attainable"][mask], demand))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("study", type=Path, help="The archived results/authority_replay_dev_wp13 directory")
    ap.add_argument("out", type=Path)
    args=ap.parse_args()
    if jax.default_backend() != "cpu":
        raise SystemExit("Run with JAX_PLATFORMS=cpu.")
    spec=importlib.util.spec_from_file_location("archived_replay", args.study/"replay_authority.py")
    archive=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    original=json.loads((args.study/"summary.json").read_text())
    capture=args.study/"dev"
    config=json.loads((capture/"config.json").read_text())
    names=config["candidate_model"]["joint_names"]
    groups={key:np.array([s in name for name in names]) for key,s in [("hips","hip"),("knees","knee")]}
    legs=groups["hips"] | groups["knees"] | np.array(["ankle" in s for s in names])
    ref={(r["family"],r["channel"],r["group"]):r for r in original["results"]}
    args.out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for family in ("gain_085","gain_070","gain_055"):
        for channel in ("exact","declared"):
            file=capture/f"trace_{family}_{channel}_none.npz"
            if hashlib.sha256(file.read_bytes()).hexdigest()!=original["capture_sha256"][file.name]:
                raise ValueError("Archived trace checksum differs")
            with np.load(file) as z:
                data=dict(z)
            method=archive.bind(config,channel)
            variants=[("speed10",0,method)]
            for name,signed in [("speed_magnitude",False),("signed_velocity",True)]:
                variants.extend((name,a,ReplayVariant(method,a,signed)) for a in ACCELERATIONS)
            for name,a,candidate in variants:
                out=archive.replay(data,candidate,config["families"][family]["gain"],legs)
                for group,ids in groups.items():
                    for window in ("first_second","all_post_onset"):
                        entry=summarize(data,out,method,group,window,ids)
                        if name=="speed10":
                            expected=ref[family,channel,group][window]["oracle_fraction"]
                            np.testing.assert_allclose(entry["directional_authority_fraction"],expected,rtol=1e-6,atol=1e-8)
                        rows.append(dict(family=family,channel=channel,envelope=name,
                                         acceleration_rad_s2=a,**entry))
            print(f"Replayed {family}, {channel}",flush=True)
    with (args.out/"authority.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    main()
