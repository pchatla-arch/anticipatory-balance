#!/usr/bin/env python3
"""Evaluation harness for actuator-adaptation methods (WP 2.3).

Runs scenario families x sensor models x methods on paired episodes drawn from a split
seed, writes per-episode rows, paired summaries against a reference method, and a
report. The test split needs a registration file written before the run.

  # development: tune and look at everything
  python scripts/eval_adaptive_bench.py --split dev --families gain_070,healthy \
      --methods none,oracle,estimated_inverse,heuristic_gate --telemetry exact,declared \
      --episodes 32 --impl jax --allow-cpu --out results/bench_dev

  # registered test (once): the registration JSON must predate the run
  python scripts/eval_adaptive_bench.py --split test --registration results/registration.json --out results/bench_test

Splits use disjoint seed ranges (dev 11000+, val 12000+, test 13000+) so no test
scenario is ever used for tuning. All methods of one family see identical scenarios
and identical sensor noise draws, so differences are paired by scenario.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SPLIT_SEED = {"dev": 11000, "val": 12000, "test": 13000}
TELEMETRY_PRESETS = {"exact": "EXACT", "declared": "DECLARED"}
REFERENCE_METHOD = "heuristic_gate"
PAIRED_METRICS = ("postfault_loss", "harmful_offset_rad", "progress_m")


def family_seed(split, family_name):
    """Scenario seed depends on the split and the family name only, never on list position,
    so the same family draws the same scenarios in every run of that split."""
    return SPLIT_SEED[split] + 37 * (zlib.crc32(family_name.encode()) % 997)


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    tmp.replace(path)


def write_rows(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def paired_ci(a, b, draws=2000, seed=228):
    """Bootstrap 95% interval of mean(a - b) over paired scenarios."""
    delta = np.asarray(a, np.float64) - np.asarray(b, np.float64)
    if len(delta) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(delta), (draws, len(delta)))
    means = delta[idx].mean(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def summarize(rows, reference=REFERENCE_METHOD, margins=None):
    """Per (family, telemetry, method) summary plus paired differences against `reference`."""
    margins = margins or {}
    out = {"cells": {}, "paired_vs_reference": {}, "reference": reference}
    groups = {}
    for r in rows:
        groups.setdefault((r["family"], r["telemetry"], r["method"]), []).append(r)
    for (family, tele, method), rr in sorted(groups.items()):
        rr = sorted(rr, key=lambda r: r["scenario_id"])
        valid = [r for r in rr if not r["nonfinite"]]
        mean = lambda key: float(np.mean([r[key] for r in valid])) if valid else None  # noqa: E731
        withdrawals = [r["withdrawal_steps"] for r in valid if r["withdrawal_steps"] >= 0]
        cell = {
            "n": len(rr), "invalid_episodes": len(rr) - len(valid),
            "fall_rate": float(np.mean([r["fell"] for r in rr])),
            "pre_fault_failures": int(sum(r["pre_fault_failed"] for r in rr)),
            "mean_postfault_loss": mean("postfault_loss"),
            "mean_progress_m": mean("progress_m"),
            "mean_harmful_offset_rad": mean("harmful_offset_rad"),
            "mean_applied_offset_rad": mean("mean_applied_offset_rad"),
            "mean_gain_abs_error": mean("gain_abs_error"),
            "withdrawal_steps_median": float(np.median(withdrawals)) if withdrawals else None,
            "never_withdrew": int(sum(1 for r in valid if r["return_step"] >= 0 and r["withdrawal_steps"] < 0)),
            "control_bound_violations": int(sum(r["max_correction_rad"] > 0.2 + 1e-5 or r["max_slew_rad_s"] > 2.0 + 1e-5 for r in rr)),
            "lost_packets_mean": float(np.mean([r["lost_packets"] for r in rr])),
        }
        out["cells"][f"{family}|{tele}|{method}"] = cell
    for (family, tele, method), rr in sorted(groups.items()):
        ref = groups.get((family, tele, reference))
        if ref is None or method == reference:
            continue
        a = {r["scenario_id"]: r for r in rr}
        b = {r["scenario_id"]: r for r in ref}
        ids = sorted(set(a) & set(b))
        if len(ids) < 2:
            continue
        paired = {}
        for metric in PAIRED_METRICS:
            x = [a[i][metric] for i in ids]
            y = [b[i][metric] for i in ids]
            paired[metric] = {"mean_difference": float(np.mean(np.subtract(x, y))), "ci95": paired_ci(x, y), "n": len(ids)}
        fx = np.array([a[i]["fell"] for i in ids], float)
        fy = np.array([b[i]["fell"] for i in ids], float)
        paired["fall_rate"] = {"mean_difference": float((fx - fy).mean()), "ci95": paired_ci(fx, fy), "n": len(ids)}
        margin = margins.get(family)
        if margin is not None:
            # Noninferiority on healthy/nuisance families: the method's loss may exceed the
            # reference's by at most `margin` (upper CI bound), rather than "no significant difference".
            paired["noninferior_loss"] = bool(paired["postfault_loss"]["ci95"][1] <= margin)
        out["paired_vs_reference"][f"{family}|{tele}|{method}"] = paired
    return out


def render_report(summary, meta):
    lines = [f"# Adaptive execution bench: {meta['split']} split", "",
             f"Source {meta['source_commit']}, physics {meta['impl']}, {meta['steps']} control steps, "
             f"schedule {meta['schedule']}, {meta['episodes']} episodes per cell.", "",
             "| Family | Telemetry | Method | n | Falls | Post-fault loss | Harmful offset (rad) | Gain error | Withdrawal (steps) |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda v, p=4: "—" if v is None else f"{v:.{p}f}"  # noqa: E731
    for key, c in summary["cells"].items():
        fam, tele, method = key.split("|")
        lines.append(f"| {fam} | {tele} | {method} | {c['n']} | {c['fall_rate']:.3f} | {fmt(c['mean_postfault_loss'])} | "
                     f"{fmt(c['mean_harmful_offset_rad'])} | {fmt(c['mean_gain_abs_error'], 3)} | "
                     f"{'—' if c['withdrawal_steps_median'] is None else int(c['withdrawal_steps_median'])} |")
    lines += ["", f"Paired differences are method minus `{summary['reference']}` on identical scenarios and sensor draws "
              "(95% bootstrap intervals).", "",
              "| Family | Telemetry | Method | Δ loss [CI] | Δ harmful offset [CI] | Δ fall rate [CI] |", "|---|---|---|---|---|---|"]
    for key, p in summary["paired_vs_reference"].items():
        fam, tele, method = key.split("|")

        def cell(m):
            d = p[m]
            return f"{d['mean_difference']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
        lines.append(f"| {fam} | {tele} | {method} | {cell('postfault_loss')} | {cell('harmful_offset_rad')} | {cell('fall_rate')} |")
    lines += ["", "Loss: capped planar-velocity error over the post-onset horizon, failure charged 4 (m/s)^2 for the "
              "remaining steps. Harmful offset: mean |effective correction| while the plant is healthy. "
              "Methods never receive the injected gain, onset or return step; the oracle does.", ""]
    return "\n".join(lines)


def registration_record(args, families, methods, telemetry_names, margins):
    return {
        "split": args.split, "families": families, "methods": methods, "telemetry": telemetry_names,
        "episodes": args.episodes, "steps": args.steps, "schedule": args.schedule,
        "reference_method": REFERENCE_METHOD, "paired_metrics": list(PAIRED_METRICS),
        "noninferiority_margins_loss": margins,
        "primary": "paired mean difference in postfault_loss, candidate minus heuristic_gate, on gain families",
        "secondary": "harmful_offset_rad on healthy and nuisance families; fall_rate; withdrawal_steps on return families",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", choices=sorted(SPLIT_SEED), default="dev")
    parser.add_argument("--families", default="gain_070,healthy")
    parser.add_argument("--methods", default="none,oracle,estimated_inverse,heuristic_gate")
    parser.add_argument("--telemetry", default="exact", help="comma-separated presets: exact,declared, or a JSON file path")
    parser.add_argument("--episodes", type=int, default=32)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--schedule", choices=("walk", "walk_turn_stop_resume"), default="walk_turn_stop_resume")
    parser.add_argument("--impl", choices=("warp", "jax"), default="warp")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--registration", type=Path, help="required for --split test; must predate the run")
    parser.add_argument("--heuristic", type=Path, help="JSON with tuned HeuristicConfig/RLSConfig fields (frozen on val)")
    parser.add_argument("--margin", type=float, default=0.005, help="noninferiority loss margin on healthy/nuisance families")
    parser.add_argument("--candidate-config", type=Path, help="explicit WP 2.2 assumption/authority JSON")
    parser.add_argument("--trace-authority-packets", action="store_true",
                        help="dev-only observer: record decision state, released packets and evaluation truth")
    parser.add_argument("--trace", action="store_true", help="save per-step traces of the first batch of every cell (npz)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    families = [f for f in args.families.split(",") if f]
    methods = [m for m in args.methods.split(",") if m]

    import jax
    import jax.numpy as jp
    from foresight.adaptive import bench as B
    from foresight.adaptive import estimation as E
    from foresight.adaptive import sensing as S
    from foresight.adaptive.actuator_env import AdaptiveJoystick
    from foresight.envs.g1_impact import default_config
    from foresight.eval import rollout as R

    devices = jax.devices()
    if not args.allow_cpu and not any(d.platform == "gpu" for d in devices):
        raise RuntimeError("GPU required; --allow-cpu is for local checks only.")
    for f in families:
        if f not in B.FAMILIES:
            parser.error(f"unknown family {f}; known: {sorted(B.FAMILIES)}")
    telemetry = {}
    for name in args.telemetry.split(","):
        if name in TELEMETRY_PRESETS:
            telemetry[name] = getattr(S, TELEMETRY_PRESETS[name])
        elif Path(name).exists():
            telemetry[Path(name).stem] = S.config_from_dict(json.loads(Path(name).read_text()))
        else:
            parser.error(f"unknown telemetry preset {name}")
    tuned = json.loads(args.heuristic.read_text()) if args.heuristic else {}
    rls_cfg = E.RLSConfig(**tuned.get("rls", {}))
    heur_cfg = E.HeuristicConfig(**tuned.get("heuristic", {}))
    candidate = B.load_candidate()
    candidate_cfg = None
    needs_candidate = "candidate" in methods or args.trace_authority_packets
    if args.trace_authority_packets and (not args.trace or args.split != "dev"):
        parser.error("--trace-authority-packets requires --trace and --split dev")
    if needs_candidate:
        if candidate is None or not args.candidate_config:
            parser.error("candidate/authority runs require the candidate module and --candidate-config")
        candidate_cfg = candidate.Config(**json.loads(args.candidate_config.read_text()))
    specs = {}
    for m in methods:
        if m in B.METHODS:
            specs[m] = B.METHODS[m]._replace(rls=rls_cfg, heuristic=heur_cfg)
        elif m == "candidate":
            if candidate is None:
                parser.error("method 'candidate' requested but foresight.adaptive.candidate is not available")
            specs[m] = B.MethodSpec("candidate", external=True, oracle=False, rls=rls_cfg)
        else:
            parser.error(f"unknown method {m}")
    # Healthy and nuisance families carry a noninferiority margin; gain families are superiority tests.
    margins = {f: args.margin for f in families if B.FAMILIES[f].gain == 1.0}
    registration = registration_record(args, families, methods, sorted(telemetry), margins)
    if needs_candidate:
        registration["candidate_config_sha256"] = hashlib.sha256(args.candidate_config.read_bytes()).hexdigest()
    if args.split == "test":
        if not args.registration or not args.registration.exists():
            parser.error("--split test requires an existing --registration file written before this run")
        recorded = json.loads(args.registration.read_text())
        checked_keys = ("families", "methods", "telemetry", "episodes", "steps", "schedule")
        if needs_candidate:
            checked_keys += ("candidate_config_sha256",)
        for key in checked_keys:
            if recorded.get(key) != registration[key]:
                parser.error(f"registration mismatch on {key}: {recorded.get(key)!r} vs {registration[key]!r}")
        registration["registered_at"] = recorded.get("registered_at")
        registration["registration_sha256"] = hashlib.sha256(args.registration.read_bytes()).hexdigest()
    else:
        registration["registered_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    write_json(args.out / "registration.json", registration)

    cfg = default_config()
    cfg.impact.resample_on_autoreset = False
    cfg.impact.miss_fraction = 0.0
    cfg.impact.reset_perturb = 1.0
    cfg.push_config.enable = False
    cfg.episode_length = args.steps
    env = AdaptiveJoystick(config=cfg, config_overrides={"impl": args.impl}, telemetry=True)
    checkpoint = R.latest_checkpoint(ROOT / "checkpoints/nominal_s1")
    policy = R.load_policy(checkpoint, deterministic=True)
    schedule = B.WALK_ONLY if args.schedule == "walk" else B.WALK_TURN_STOP_RESUME
    versions = {}
    for name in ("jax", "mujoco", "mujoco-mjx", "warp-lang", "brax"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    meta = {
        "split": args.split, "seed": SPLIT_SEED[args.split], "impl": args.impl, "steps": args.steps,
        "schedule": args.schedule, "episodes": args.episodes, "devices": [str(d) for d in devices],
        "versions": versions, "checkpoint": str(checkpoint),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "rls": dict(rls_cfg._asdict()), "heuristic": dict(heur_cfg._asdict()),
        "telemetry": {k: S.summarize(v) for k, v in telemetry.items()},
        "families": {f: {**B.FAMILIES[f]._asdict(), "telemetry": S.summarize(B.FAMILIES[f].telemetry)} for f in families},
        "candidate_available": candidate is not None,
    }
    if needs_candidate:
        from dataclasses import asdict
        meta["candidate_config"] = asdict(candidate_cfg)
        meta["candidate_model"] = candidate.model_record(env, next(iter(telemetry.values())), rls_cfg, candidate_cfg)
        meta["trace_authority_packets"] = args.trace_authority_packets
    write_json(args.out / "config.json", meta)
    write_json(args.out / "actuators.json", {
        "names": list(env.actuator_joint_names),
        "kp": [float(v) for v in env.mj_model.actuator_gainprm[:, 0]],
        "leg_mask": [bool(v) for v in env.leg_actuator_mask],
        "ctrl_range": [[float(a), float(b)] for a, b in env.mj_model.actuator_ctrlrange],
        "ctrl_limited": [bool(v) for v in env.mj_model.actuator_ctrllimited],
    })
    reset = jax.jit(jax.vmap(env.reset))
    rows = []
    for fname in families:
        family = B.FAMILIES[fname]
        seed = family_seed(args.split, fname)
        scenario = B.draw_scenarios(family, seed, args.episodes, env)
        for tele_name, tele_cfg in sorted(telemetry.items()):
            # A family's own sensor confound composes with the channel model: the confound
            # defines scale/bias; the channel preset defines noise, latency and dropout.
            combined = tele_cfg._replace(
                torque_scale=family.telemetry.torque_scale if family.telemetry.torque_scale != 1.0 else tele_cfg.torque_scale,
                torque_bias_nm=family.telemetry.torque_bias_nm or tele_cfg.torque_bias_nm,
                latency_min=max(family.telemetry.latency_min, tele_cfg.latency_min),
                latency_max=max(family.telemetry.latency_max, tele_cfg.latency_max),
            ).validate()
            for mname, spec in specs.items():
                if mname == "candidate":
                    # Only public error bounds are consumed from the measurement channel.
                    spec = spec._replace(candidate=candidate.Candidate.bind(env, combined, rls_cfg, candidate_cfg))
                execute = B.build_bench_rollout(env, policy, spec, combined, schedule, args.steps,
                                                trace=args.trace, trace_authority_packets=args.trace_authority_packets)
                arm_started = time.monotonic()
                for start in range(0, args.episodes, args.batch):
                    stop = min(start + args.batch, args.episodes)
                    keys = jax.vmap(lambda i: jax.random.fold_in(jax.random.PRNGKey(seed), i))(jp.arange(start, stop, dtype=jp.int32))
                    initial = reset(keys)
                    initial = R.set_scenario(env, initial, launch=False, command=jp.zeros((stop - start, 3), jp.float32))
                    sub = {k: jp.asarray(v[start:stop]) for k, v in scenario.items() if k != "ids"}
                    result = jax.device_get(execute(initial, sub))
                    if args.trace and start == 0:
                        np.savez_compressed(args.out / f"trace_{fname}_{tele_name}_{mname}.npz",
                                            onset=scenario["onset"][start:stop], ret=scenario["return"][start:stop],
                                            **{k: np.asarray(v) for k, v in result["trace"].items()})
                    result.pop("trace", None)
                    part = {k: v[start:stop] for k, v in scenario.items()}
                    rows.extend(B.episode_rows(result, part, family, mname, tele_name, args.split))
                    write_rows(args.out / "episodes.csv", rows)
                cell = [r for r in rows if r["family"] == fname and r["telemetry"] == tele_name and r["method"] == mname]
                print(f"{fname} {tele_name} {mname}: falls={np.mean([r['fell'] for r in cell]):.3f} "
                      f"loss={np.mean([r['postfault_loss'] for r in cell]):.4f} "
                      f"harmful={np.mean([r['harmful_offset_rad'] for r in cell]):.4f} "
                      f"gain_err={np.mean([r['gain_abs_error'] for r in cell]):.3f} "
                      f"time={time.monotonic() - arm_started:.0f}s", flush=True)
    summary = summarize(rows, margins=margins)
    summary["meta"] = meta
    summary["elapsed_seconds"] = round(time.monotonic() - started, 1)
    write_json(args.out / "summary.json", summary)
    (args.out / "report.md").write_text(render_report(summary, meta))
    print(f"Bench completed: {len(rows)} episode rows; see {args.out}/report.md", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
