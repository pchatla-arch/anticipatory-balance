"""Fresh small CPU companion to E20. Use the pinned Foresight environment and repo."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

PROTOCOL = Path(__file__).resolve().parents[1] / "analysis/deterministic_cpu_pilot/protocol.json"
PROTOCOL_HASH = "4d9b52b2c89ead5569f8ee738322dc1f242329088eab74a192e6ae7393b818eb"


def save(path, data):
    with path.open("x") as out:
        json.dump(data, out, indent=2, allow_nan=False)
        out.write("\n")


def verify_inputs(repo):
    manifest = json.loads(PROTOCOL.with_name("input_manifest.json").read_text())
    for name, expected in manifest["paths"].items():
        payload = (repo / name).read_bytes()
        prefix = b"blob " + str(len(payload)).encode("ascii") + b"\0"
        if hashlib.sha1(prefix + payload).hexdigest() != expected:
            raise ValueError(f"Input differs from reviewed evaluator revision: {name}")


def read_runner(repo):
    spec = importlib.util.spec_from_file_location("e20", repo / "scripts/run_foresight_matched_duration.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    return runner


def phase(repo, plan, runner, out, stage, thetas):
    seed = plan[stage + "_seed"]
    count = plan[stage + "_n"]
    output = out / stage
    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)
    env.update(JAX_PLATFORMS="cpu", JAX_DEFAULT_MATMUL_PRECISION="highest",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", PYTHONPATH=str(repo),
               PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    flags = ["--impl=jax" if x == "--impl=warp" else x for x in runner.COMMON]
    cmd = [sys.executable, "scripts/eval_decision_frontier.py", *flags, f"--seed={seed}",
           f"--episodes={count}", "--batches=1", "--thetas=" + ",".join(map(str, thetas)),
           f"--out={output}"]
    print(f"Starting {stage}, n={count}, {len(thetas)} TTC thresholds.", flush=True)
    with (out / (stage + ".log")).open("x") as log:
        subprocess.run(cmd, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    config = json.loads((output / "config.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    if (summary["status"] != "complete" or config["seed"] != seed or config["episodes"] != count
            or config["impl"] != "jax" or config["batches"] != 1 or config["period"] != 5
            or config["latencies_steps"] != [5] or config["steps"] != runner.STEPS
            or abs(config["dt"] - runner.DT) > 1e-8):
        raise ValueError("Recorded evaluator settings differ from the pilot protocol")
    return runner.read_arms(output, seed, count, 1, thetas)


def evidence(out, stage):
    fields = ("method", "param", "scenario_id", "mass", "speed", "t_hit_nominal", "is_miss",
              "fell", "contact_before_termination", "term_step", "progress_m", "brace_steps_total")
    data = []
    for source in sorted((out / stage / "episodes").glob("*.csv")):
        with source.open() as f:
            data.extend({k: row[k] for k in fields} for row in csv.DictReader(f))
    with (out / (stage + "_episodes.csv")).open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    raw = PROTOCOL.read_bytes()
    if hashlib.sha256(raw).hexdigest() != PROTOCOL_HASH:
        raise ValueError("Pilot protocol differs from its frozen version.")
    plan = json.loads(raw)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    repo = args.repo.resolve()
    verify_inputs(repo)
    runner = read_runner(repo)
    grid = list(runner.GRID)
    if grid != plan["ttc_threshold_grid_seconds"] or runner.MARGIN != 0.02:
        raise ValueError("Reviewed E20 grid or matching margin changed.")
    save(out / "protocol.json", plan)
    arms = phase(repo, plan, runner, out, "calibration", grid)
    frozen = runner.choose_theta(arms)
    save(out / "freeze.json", frozen)
    evidence(out, "calibration")
    if frozen["matched_on_calibration"]:
        theta = frozen["theta_s"]
        arms = phase(repo, plan, runner, out, "confirmation", [theta])
        verdict = runner.score(arms, theta)
        evidence(out, "confirmation")
    else:
        verdict = {"status": "no_calibration_match", "confirmation_started": False}
    save(out / "verdict.json", verdict)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
