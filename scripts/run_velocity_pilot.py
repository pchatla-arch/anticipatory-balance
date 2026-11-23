#!/usr/bin/env python3
"""WP 2.2 CPU development pilot. Use plan, wiring or run with a fresh output directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CANDIDATE = ROOT / "docs/adaptive_execution/research_candidate/wp22_dev_config.json"
CHECKPOINT = ROOT / "checkpoints/nominal_s1/000202342400"
CHECKPOINT_HASH = "14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de"
FAMILIES = ("gain_070", "gain_070_return", "healthy", "sensor_scale_080")
METHODS = ("none", "oracle", "heuristic_gate", "candidate", "candidate_velocity")


FROZEN_PATH = "results/tune_heuristic_dev_cpu_v0/heuristic.json"


def frozen_heuristic(revision, *, repo=ROOT):
    """Read once from an immutable Git object, then bind the resolved settings and exact bytes."""
    from foresight.adaptive import estimation as E
    commit = subprocess.check_output(["git", "rev-parse", "--verify", "--end-of-options",
                                      revision + "^{commit}"], cwd=repo, text=True).strip()
    read = subprocess.run(["git", "show", f"{commit}:{FROZEN_PATH}"], cwd=repo,
                          capture_output=True, check=False)
    if read.returncode:
        raise ValueError(f"WP 2.1 frozen heuristic is unavailable at {commit}:{FROZEN_PATH}")
    artifact = json.loads(read.stdout)
    tuned = artifact.get("tuned_on", {})
    if (tuned.get("split") != "dev" or not isinstance(tuned.get("evaluations"), int)
            or tuned["evaluations"] < 1 or not artifact.get("rls") or not artifact.get("heuristic")):
        raise ValueError("expected the completed WP 2.1 artifact with dev tuning provenance")
    record = dict(status="frozen_wp21", source_commit=commit, source_path=FROZEN_PATH,
                  sha256=hashlib.sha256(read.stdout).hexdigest(), tuned_on=tuned,
                  rls=E.RLSConfig(**artifact["rls"])._asdict(),
                  heuristic=E.HeuristicConfig(**artifact["heuristic"])._asdict())
    return record, read.stdout


def manifest(mode, baseline):
    from foresight.adaptive import sensing
    wiring = mode == "wiring"
    # Keep the wiring seed family separate from the development measurements and historical split ranges.
    base = 911000 if wiring else 811000
    families = ("healthy",) if wiring else FAMILIES
    methods = ("candidate_velocity",) if wiring else METHODS
    channels = {"declared": sensing.DECLARED._asdict()}
    if not wiring:
        channels["declared_fast"] = sensing.DECLARED_FAST._asdict()
    episodes = 1 if wiring else 8
    tree = hashlib.sha256()
    for file in sorted(CHECKPOINT.rglob("*")):
        if file.is_file():
            tree.update(file.relative_to(CHECKPOINT).as_posix().encode() + b"\0"
                        + hashlib.sha256(file.read_bytes()).hexdigest().encode() + b"\n")
    if tree.hexdigest() != CHECKPOINT_HASH:
        raise ValueError("nominal_s1 checkpoint missing or checksum mismatch")
    git = lambda *args: subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("commit tracked source changes before creating a pilot manifest")
    return dict(mode="wiring" if wiring else "development", split="dev", impl="jax", platform="cpu",
                source_commit=git("rev-parse", "HEAD"), source_tree=git("rev-parse", "HEAD^{tree}"),
                checkpoint=str(CHECKPOINT), checkpoint_tree_sha256=CHECKPOINT_HASH,
                seed_base=base, family_seeds={f: base + 37 * (zlib.crc32(f.encode()) % 997) for f in families},
                families=list(families), methods=list(methods), channels=channels, episodes=episodes,
                batch=episodes, steps=8 if wiring else 300, schedule="walk_turn_stop_resume",
                sensor_velocity_error="uniform source-time independent error, shared key by scenario and tick",
                speed_config=json.loads(CANDIDATE.read_text()), acceleration_rad_s2=400.,
                baseline=baseline, attempt_budget=1,
                expected_episode_rows=len(families) * len(methods) * len(channels) * episodes)


def inspect_hold(trace, names, dt, cfg, sensor):
    """Score each live tick/joint only when the decision has a released valid encoder."""
    q = np.asarray(trace["pilot_substep_q"])  # time, episode, physics substep, actuator
    live = np.asarray(trace["alive"], dtype=bool)
    if not np.array_equal(q[:, :, -1][live], trace["authority_source_q_truth"][live]):
        raise ValueError("observer final substep differs from source position on a live transition")
    source = np.asarray(trace["authority_rls_last_source_step"])
    tick = np.arange(len(q))[:, None, None]
    keep = live[:, :, None] & (source >= 0)
    age = tick - 1 - source
    if np.any(age[keep] < 0):
        raise ValueError("decision consumed a future packet")
    result = {}
    for group in ("hip", "knee"):
        joint = np.array([group in name for name in names])
        mask = keep & joint
        ages = age[mask]
        age_bins, counts = np.unique(ages, return_counts=True)
        result[group] = {
            "live_joint_ticks": int(live.sum() * joint.sum()),
            "source_available_joint_ticks": int(mask.sum()),
            "mean_source_age_ms": float(ages.mean() * dt * 1000.) if ages.size else None,
            "source_age_ticks_histogram": dict(zip(map(str, age_bins), map(int, counts))),
        }
        if cfg is None:
            continue
        # Score a source-time envelope at each observed physical time, as in
        # the G1 replay. The controller conservatively uses the end-of-hold radius.
        elapsed = age[:, :, None, :] * dt + np.linspace(0., dt, q.shape[2] + 1)[None, None, :, None]
        speed = cfg.hold_speed_rad_s
        if cfg.hold_accel_rad_s2 is not None:
            velocity = trace["authority_rls_last_velocity"][:, :, None, :]
            speed = np.abs(velocity) + sensor.bound_velocity_rad_s + 0.5 * cfg.hold_accel_rad_s2 * elapsed
        radius = sensor.bound_position_rad + speed * elapsed
        positions = np.concatenate((trace["authority_decision_q_truth"][:, :, None, :], q), axis=2)
        dist = np.abs(positions - trace["authority_rls_last_position"][:, :, None, :])
        covered = np.all(dist <= radius, axis=2) & mask
        result[group]["whole_hold_covered_joint_ticks"] = int(covered.sum())
        result[group]["whole_hold_coverage"] = float(covered.sum() / mask.sum()) if mask.any() else None
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("plan", "wiring", "run"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--heuristic-rev", help="Git revision containing Claude's frozen WP 2.1 heuristic.json")
    args = parser.parse_args(argv)
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
    raw = None
    if args.heuristic_rev:
        try:
            baseline, raw = frozen_heuristic(args.heuristic_rev)
        except (ValueError, TypeError, subprocess.CalledProcessError) as err:
            parser.error(str(err))
    elif args.mode == "run":
        parser.error("run requires --heuristic-rev with the final WP 2.1 heuristic.json")
    elif args.mode == "wiring":
        from foresight.adaptive import estimation as E
        baseline = dict(status="wiring_only", rls=E.RLSConfig()._asdict(), heuristic=E.HeuristicConfig()._asdict())
    else:
        baseline = dict(status="awaiting_frozen_wp21")
    plan = manifest(args.mode, baseline)
    args.out.mkdir(parents=True, exist_ok=False)
    import eval_adaptive_bench as bench
    if raw is not None:
        (args.out / "frozen_heuristic.json").write_bytes(raw)
    bench.write_json(args.out / "pilot_manifest.json", plan)
    if args.mode == "plan":
        print(args.out / "pilot_manifest.json")
        return 0
    return bench.main(["--out", str(args.out), "--split", "dev", "--impl", "jax", "--allow-cpu",
                       "--families", ",".join(plan["families"]), "--methods", ",".join(plan["methods"]),
                       "--telemetry", ",".join(plan["channels"]), "--episodes", str(plan["episodes"]),
                       "--batch", str(plan["batch"]), "--steps", str(plan["steps"]), "--schedule", plan["schedule"],
                       "--candidate-config", str(CANDIDATE), "--trace", "--trace-authority-packets"],
                      velocity_pilot=plan)


if __name__ == "__main__":
    raise SystemExit(main())
