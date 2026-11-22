#!/usr/bin/env python3
"""WP 1.3 authority study: replay recorded telemetry through the gain set and projection.

Input: bench traces of the `none` method (no correction applied, so the held target equals
the policy reference) written with `eval_adaptive_bench.py --trace`. For every leg actuator
of every episode the recorded measured packets are processed in source order through
ChatGPT's set-membership update (`projection.update_gain_set`), expanded over packet age and
the 20 ms hold, combined with the hold error envelope and projected (`project_offset`) with
the interval midpoint as the inverse candidate. Nothing here touches the robot; the question
is how much correction the declared information permits, and when.

Outputs per (family, telemetry): time after onset until the set excludes g = 1 (median over
actuators and episodes), interval width over time, permitted |x| relative to the oracle's
|(1/g - 1)(r - q)|, bound-violation and INCONSISTENT counts, and the fraction of post-onset
steps with zero admissible correction.

  python scripts/authority_replay.py --traces results/bench_dev_packets_v0 --out results/authority_dev_v0
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DT = 0.02
HOLD = 0.02


def replay_actuator(z, q, tau, valid, arrival, onset, g_true, kp, cfg, alive=None):
    """Replay one actuator's packet stream; returns per-step diagnostics (arrays over steps).

    Steps after the episode terminated (alive False) are excluded: the bench freezes a
    finished lane on its reset state, so its later packets describe nothing physical.
    """
    import jax.numpy as jnp
    from foresight.adaptive import projection as P
    steps = len(z)
    if alive is not None:
        steps = int(np.argmin(alive)) if not np.all(alive) else len(z)
    z, q, tau, valid, arrival, g_true = (a[:steps] for a in (z, q, tau, valid, arrival, g_true))
    prior = P.Interval(jnp.float32(cfg["prior_low"]), jnp.float32(cfg["prior_high"]))
    current = prior
    last_src = None
    width = np.full(steps, np.nan)
    excludes_healthy = np.zeros(steps, bool)
    permitted = np.full(steps, np.nan)
    oracle = np.zeros(steps)
    status_counts = {"updated": 0, "skipped": 0, "inconsistent": 0, "violations": 0}
    zero_auth = np.zeros(steps, bool)
    requested = np.zeros(steps)
    prev = 0.0   # the candidate's own request ramps under the slew limit; carry it across steps
    eps = cfg["bound_torque_nm"] / kp + cfg["prior_high"] * cfg["bound_position_rad"]
    # In-order release: packet s is processed at the first step t >= arrival(s) once all
    # earlier packets were processed or declared lost (arrival beyond t - latency_max - 1).
    cursor = 0
    for t in range(steps):
        # release due packets
        while cursor <= t:
            arr = arrival[cursor]
            lost = arr > t and cursor <= t - cfg["latency_max"] - 1
            if arr <= t:
                elapsed = 0.0 if last_src is None else (cursor - last_src) * DT
                upd = P.update_gain_set(
                    current, prior, drift_rate=cfg["drift_rate"], elapsed=elapsed,
                    held_target=jnp.float32(z[cursor]), position=jnp.float32(q[cursor]),
                    torque=jnp.float32(tau[cursor]), kp=jnp.float32(kp),
                    position_error_bound=cfg["bound_position_rad"], torque_error_bound=cfg["bound_torque_nm"],
                    sample_valid=bool(valid[cursor]))
                st = int(upd.status)
                if st == P.GainStatus.UPDATED:
                    status_counts["updated"] += 1
                    current = upd.interval
                elif st == P.GainStatus.SKIPPED:
                    status_counts["skipped"] += 1
                    current = upd.interval
                elif st == P.GainStatus.INCONSISTENT:
                    status_counts["inconsistent"] += 1
                    current = prior  # reacquire from the full prior (proposal rule)
                else:
                    status_counts["skipped"] += 1
                if valid[cursor]:
                    y = tau[cursor] / kp
                    phi = z[cursor] - q[cursor]
                    if abs(y - g_true[cursor] * phi) > eps:
                        status_counts["violations"] += 1
                last_src = cursor
                cursor += 1
            elif lost:
                cursor += 1
            else:
                break
        # decision at step t: propagate by age + hold, envelope, project
        age = 0.0 if last_src is None else (t - last_src) * DT
        hold_set = P.propagate_gain(current, prior, cfg["drift_rate"], age + HOLD)
        now_set = P.propagate_gain(current, prior, cfg["drift_rate"], age)
        lo, hi = float(hold_set.low), float(hold_set.high)
        width[t] = hi - lo
        excludes_healthy[t] = np.isfinite(lo) and np.isfinite(hi) and (hi < 1.0 or lo > 1.0)
        ref = z[t]                      # none method: held target == reference
        q_meas = q[last_src] if last_src is not None else q[t]
        errors = P.hold_error(jnp.float32(ref), jnp.float32(q_meas), position_error_bound=cfg["bound_position_rad"],
                              max_speed=cfg["max_speed"], age=age, hold=HOLD)
        mid = 0.5 * (float(now_set.low) + float(now_set.high)) if np.isfinite(lo) else 1.0
        proj = P.project_offset(hold_set, errors, current_gains=now_set, estimate=jnp.float32(mid),
                                current_error=jnp.float32(ref - q_meas), reference=jnp.float32(ref),
                                target_range=P.Interval(jnp.float32(-10.0), jnp.float32(10.0)), kp=jnp.float32(kp),
                                available_force=jnp.float32(cfg["available_force"]), max_offset=0.2, slew_rate=2.0,
                                hold=HOLD, previous=jnp.float32(prev), regularizer=0.0)
        if int(proj.status) == P.ProjectionStatus.FEASIBLE:
            lo_x, hi_x = float(proj.requested_interval.low), float(proj.requested_interval.high)
            permitted[t] = max(abs(lo_x), abs(hi_x)) if np.isfinite(lo_x) and np.isfinite(hi_x) else np.nan
            zero_auth[t] = (hi_x - lo_x) < 1e-6
            prev = float(proj.requested)
        else:
            # CONSTRAINT_CONFLICT or no effective offset: decay the request at the slew limit.
            permitted[t] = 0.0
            zero_auth[t] = True
            prev = float(np.sign(prev) * max(abs(prev) - 2.0 * HOLD, 0.0))
        requested[t] = prev
        oracle[t] = abs((1.0 / g_true[t] - 1.0) * (ref - q[t]))
    return dict(width=width, excludes_healthy=excludes_healthy, permitted=permitted, oracle=oracle,
                zero_auth=zero_auth, requested=requested, counts=status_counts)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=4, help="episodes per trace to replay")
    parser.add_argument("--drift-rate", type=float, default=0.05, help="declared |dg/dt| bound, 1/s")
    parser.add_argument("--max-speed", type=float, default=3.0, help="declared joint speed bound, rad/s")
    parser.add_argument("--available-force", type=float, default=80.0, help="supported force ceiling minus margin, N.m")
    parser.add_argument("--prior", default="0.3,1.2")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    cfgfile = json.loads((args.traces / "config.json").read_text())
    actuators = json.loads((args.traces / "actuators.json").read_text())
    kp_all = np.asarray(actuators["kp"], float)
    leg = [i for i, m in enumerate(actuators["leg_mask"]) if m]
    results = {}
    for path in sorted(glob.glob(str(args.traces / "trace_*_none.npz"))):
        m = re.match(r"trace_(.+)_(exact|declared)_none\.npz", Path(path).name)
        family, tele = m.group(1), m.group(2)
        d = np.load(path)
        tcfg = cfgfile["telemetry"][tele]
        fam_tel = cfgfile["families"][family]["telemetry"]
        cfg = dict(bound_position_rad=tcfg["bound_position_rad"], bound_torque_nm=tcfg["bound_torque_nm"],
                   latency_max=max(int(tcfg["latency_max"]), int(fam_tel["latency_max"])),
                   drift_rate=args.drift_rate, max_speed=args.max_speed, available_force=args.available_force,
                   prior_low=float(args.prior.split(",")[0]), prior_high=float(args.prior.split(",")[1]))
        onset = d["onset"]
        gain_active = d["gain_true_active"]            # (steps, n)
        g_fam = float(cfgfile["families"][family]["gain"])
        n = min(args.episodes, d["pkt_target"].shape[1])
        per = []
        counts = {"updated": 0, "skipped": 0, "inconsistent": 0, "violations": 0}
        for i in range(n):
            for a in leg:
                kp = float(kp_all[a])
                g_true = np.where(gain_active[:, i], g_fam, 1.0)
                r = replay_actuator(d["pkt_target"][:, i, a], d["pkt_position"][:, i, a], d["pkt_torque"][:, i, a],
                                    d["pkt_valid"][:, i, a], d["pkt_arrival"][:, i], int(onset[i]), g_true, kp, cfg,
                                    alive=d["alive"][:, i])
                for k in counts:
                    counts[k] += r["counts"][k]
                post = np.arange(len(r["width"])) >= onset[i]
                if not post.any():
                    continue
                first_excl = np.argmax(r["excludes_healthy"] & post) if np.any(r["excludes_healthy"] & post) else -1
                per.append(dict(episode=i, actuator=a,
                                steps_to_exclude_healthy=int(first_excl - onset[i]) if first_excl >= 0 else -1,
                                mean_width_post=float(np.nanmean(r["width"][post])),
                                permitted_over_oracle=float(np.nansum(r["permitted"][post]) / max(np.sum(r["oracle"][post]), 1e-9)),
                                requested_over_oracle=float(np.sum(np.abs(r["requested"][post])) / max(np.sum(r["oracle"][post]), 1e-9)),
                                zero_authority_fraction_post=float(np.mean(r["zero_auth"][post]))))
        excl = [p["steps_to_exclude_healthy"] for p in per if p["steps_to_exclude_healthy"] >= 0]
        results[f"{family}|{tele}"] = dict(
            family=family, telemetry=tele, gain=g_fam, episodes=n, actuators=len(leg), config=cfg,
            steps_to_exclude_healthy_median=float(np.median(excl)) if excl else None,
            never_excluded_fraction=float(np.mean([p["steps_to_exclude_healthy"] < 0 for p in per])),
            mean_width_post=float(np.mean([p["mean_width_post"] for p in per])),
            permitted_over_oracle_mean=float(np.mean([p["permitted_over_oracle"] for p in per])),
            requested_over_oracle_mean=float(np.mean([p["requested_over_oracle"] for p in per])),
            zero_authority_fraction_post=float(np.mean([p["zero_authority_fraction_post"] for p in per])),
            packet_status_counts=counts, per_actuator=per)
        r0 = results[f"{family}|{tele}"]
        print(f"{family} {tele}: exclude-healthy median {r0['steps_to_exclude_healthy_median']} steps, "
              f"never {r0['never_excluded_fraction']:.2f}, width {r0['mean_width_post']:.3f}, "
              f"permitted/oracle {r0['permitted_over_oracle_mean']:.2f}, requested/oracle {r0['requested_over_oracle_mean']:.2f}, "
              f"zero-authority {r0['zero_authority_fraction_post']:.2f}, "
              f"inconsistent {counts['inconsistent']}, violations {counts['violations']}", flush=True)
    (args.out / "authority.json").write_text(json.dumps(results, indent=2) + "\n")
    lines = ["# Authority study (offline replay of recorded packets)", "",
             f"Traces: `{args.traces}`; drift bound {args.drift_rate}/s, speed bound {args.max_speed} rad/s, prior "
             f"[{args.prior}], available force {args.available_force} N.m, hold 20 ms. Inverse candidate = set midpoint.", "",
             "| Family | Telemetry | Steps to exclude g=1 (median) | Never excluded | Mean set width | Permitted/oracle | Requested/oracle | Zero-authority steps | Inconsistent | Bound violations |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, r in results.items():
        lines.append(f"| {r['family']} | {r['telemetry']} | {r['steps_to_exclude_healthy_median']} | {r['never_excluded_fraction']:.2f} | "
                     f"{r['mean_width_post']:.3f} | {r['permitted_over_oracle_mean']:.2f} | {r['requested_over_oracle_mean']:.2f} | {r['zero_authority_fraction_post']:.2f} | "
                     f"{r['packet_status_counts']['inconsistent']} | {r['packet_status_counts']['violations']} |")
    (args.out / "report.md").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
