#!/usr/bin/env python3
"""Tune the heuristic gate and the shared RLS on the development split (WP 2.1).

Coordinate descent over a declared grid, one coordinate at a time, each candidate
configuration evaluated with the bench harness on the same development scenarios.
Every evaluation is logged (configuration, objective, cells), so the tuning budget is
on record and the same budget can be granted to the candidate's lambda.

Objective (declared before tuning, minimized):
    mean post-fault loss over the gain families
  + harm_weight * mean applied offset on healthy-plant families (false adaptation)
  + fall_weight * fall rate over all families

  python scripts/tune_heuristic.py --out results/tune_heuristic_dev --impl jax --allow-cpu --episodes 8
The frozen result (heuristic.json) is passed to the bench with --heuristic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

GRID = {
    "rls.forgetting": [0.90, 0.95, 0.98],
    "heuristic.dead_band": [0.03, 0.05, 0.10],
    "heuristic.max_residual_rms": [0.02, 0.05, 0.10],
    "heuristic.max_age_steps": [3, 5, 10],
    "heuristic.min_samples": [3, 5, 10],
    "heuristic.min_excitation_rad": [0.005, 0.01, 0.02],
}
START = {"rls": {"forgetting": 0.98}, "heuristic": {"dead_band": 0.05, "max_residual_rms": 0.05,
                                                   "max_age_steps": 5, "min_samples": 5, "min_excitation_rad": 0.01}}
GAIN_FAMILIES = ("gain_070", "gain_070_return")
HEALTHY_FAMILIES = ("healthy", "sensor_scale_080")


def set_path(cfg, dotted, value):
    section, key = dotted.split(".")
    out = {k: dict(v) for k, v in cfg.items()}
    out[section][key] = value
    return out


def objective(summary, harm_weight, fall_weight, method="heuristic_gate"):
    cells = summary["cells"]
    gain = [c for k, c in cells.items() if k.split("|")[0] in GAIN_FAMILIES and k.endswith("|" + method)]
    healthy = [c for k, c in cells.items() if k.split("|")[0] in HEALTHY_FAMILIES and k.endswith("|" + method)]
    loss = sum(c["mean_postfault_loss"] for c in gain) / max(len(gain), 1)
    harm = sum(c["mean_applied_offset_rad"] for c in healthy) / max(len(healthy), 1)
    falls = sum(c["fall_rate"] for c in gain + healthy) / max(len(gain + healthy), 1)
    return loss + harm_weight * harm + fall_weight * falls, {"loss": loss, "harm": harm, "falls": falls}


def evaluate(cfg, args, tag):
    out = args.out / f"eval_{tag}"
    cfg_path = args.out / f"cfg_{tag}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    cmd = [sys.executable, str(ROOT / "scripts/eval_adaptive_bench.py"), "--split", "dev",
           "--families", ",".join(GAIN_FAMILIES + HEALTHY_FAMILIES), "--methods", "heuristic_gate",
           "--telemetry", args.telemetry, "--episodes", str(args.episodes), "--batch", str(args.episodes),
           "--steps", str(args.steps), "--schedule", args.schedule, "--impl", args.impl,
           "--heuristic", str(cfg_path), "--out", str(out)]
    if args.allow_cpu:
        cmd.append("--allow-cpu")
    started = time.monotonic()
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    summary = json.loads((out / "summary.json").read_text())
    value, parts = objective(summary, args.harm_weight, args.fall_weight)
    return {"tag": tag, "config": cfg, "objective": value, **parts, "seconds": round(time.monotonic() - started, 1)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--schedule", default="walk_turn_stop_resume")
    parser.add_argument("--telemetry", default="declared")
    parser.add_argument("--impl", default="warp")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--harm-weight", type=float, default=0.5, help="loss units per rad of mean applied offset on a healthy plant")
    parser.add_argument("--fall-weight", type=float, default=0.1)
    parser.add_argument("--passes", type=int, default=1, help="coordinate-descent sweeps over the grid")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log = []
    current = {k: dict(v) for k, v in START.items()}
    best = evaluate(current, args, "start")
    log.append(best)
    budget = 1
    for sweep in range(args.passes):
        for coord, values in GRID.items():
            for value in values:
                section, key = coord.split(".")
                if current[section][key] == value:
                    continue
                trial = set_path(current, coord, value)
                budget += 1
                result = evaluate(trial, args, f"s{sweep}_{coord.replace('.', '_')}_{value}")
                log.append(result)
                print(f"{result['tag']}: objective={result['objective']:.4f} loss={result['loss']:.4f} "
                      f"harm={result['harm']:.4f} falls={result['falls']:.3f}", flush=True)
                if result["objective"] < best["objective"]:
                    best, current = result, trial
            (args.out / "tuning_log.json").write_text(json.dumps({"evaluations": budget, "log": log, "best": best}, indent=2) + "\n")
    frozen = {"rls": best["config"]["rls"], "heuristic": best["config"]["heuristic"],
              "tuned_on": {"split": "dev", "families": GAIN_FAMILIES + HEALTHY_FAMILIES, "telemetry": args.telemetry,
                           "episodes": args.episodes, "evaluations": budget,
                           "objective": "loss + %.2f*harm + %.2f*falls" % (args.harm_weight, args.fall_weight)}}
    (args.out / "heuristic.json").write_text(json.dumps(frozen, indent=2) + "\n")
    print(f"frozen heuristic: {json.dumps(frozen)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
