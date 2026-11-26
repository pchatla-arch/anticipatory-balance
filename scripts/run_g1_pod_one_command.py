#!/usr/bin/env python3
"""One-entry G1 RunPod boot/setup/development-bench/publish/stop supervisor.

This runner executes the already implemented actuator benchmark. The separate
72-episode head-depth obstacle experiment is NOT runnable until a verified
MuJoCo integration runner exists. It never fabricates those results.

RUNPOD_API_KEY, RUNPOD_POD_ID and GH_TOKEN may be supplied in the pod environment;
if missing, values are requested securely in an interactive terminal.
Never paste credentials into chat.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
import uuid

REPO_URL = "https://github.com/pchatla-arch/anticipatory-balance.git"
RESULT_BRANCH = "foresight/experiment-results"
FROZEN_BASELINE = "d756c37d0564837a74132c30f8e366320a6bb1a5"
COST_MAX_USD = 4.44
PRICE_MAX = 0.74
# Stop early to reserve publication time and leave margin below six hours.
LIFETIME_LIMIT_S = 5 * 3600 + 20 * 60
RESERVE_S = 25 * 60
CHECKPOINT_SHA = "14bd6650f06c4e3171b6e2d2403ef0da5d9dd3604709f0c22b488329dd0223de"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = Path("/workspace/g1_run_outputs")
SECRETS = Path("/workspace/.g1-one-command-secrets")
RUNPOD_API = "https://api.runpod.io/v2/pods/"


def log(*args):
    print(datetime.now(timezone.utc).isoformat(), *args, flush=True)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".writing")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def ask_secret(name, path):
    value = os.environ.pop(name, "")
    if not value:
        if not sys.stdin.isatty():
            raise RuntimeError(name + " is required; rerun in an interactive pod terminal")
        value = getpass.getpass(name + " (hidden): ").strip()
    if not value or "\n" in value:
        raise RuntimeError("Invalid " + name)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(value)
    path.chmod(0o600)
    return value


def runpod_request(pod_id, token, *, stop=False):
    req = urllib.request.Request(
        RUNPOD_API + pod_id + ("/action" if stop else ""),
        headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
        data=b'{"action":"stop"}' if stop else None,
        method="POST" if stop else "GET",
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read())


def validate_pod(pod_id, token):
    pod = runpod_request(pod_id, token)
    gpu = pod.get("gpu") or {}
    if (pod.get("id") != pod_id or pod.get("status") != "RUNNING"
            or pod.get("cloud") != "COMMUNITY"
            or pod.get("cudaVersion") != "13.2"
            or gpu.get("count") != 1
            or gpu.get("id") != "NVIDIA RTX 6000 Ada Generation"
            or float(pod.get("cost", 1000)) > PRICE_MAX):
        raise RuntimeError(
            "Pod does not match the approved one Community RTX 6000 Ada, CUDA 13.2, "
            "at <= $0.74/hr; scientific work withheld"
        )
    return pod


def started_epoch(pod):
    s = (pod.get("startedAt") or pod.get("createdAt"))
    if not s:
        raise RuntimeError("Pod has no provider start timestamp")
    return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def stop_pod(pod_id, token):
    for i in range(3):
        try:
            info = runpod_request(pod_id, token)
            if info.get("status") in {"EXITED", "TERMINATED"}:
                log("Provider confirms", info.get("status"))
                return True
            runpod_request(pod_id, token, stop=True)
            log("Provider stop requested, attempt", i + 1)
            time.sleep(2)
        except Exception as exc:
            log("Provider stop retry", type(exc).__name__)
    # The provider can stop the current process before its own response arrives.
    return False


def watchdog(pod_id, deadline, secret_file):
    # Separate process inside this pod, NOT a provider-independent guarantee.
    token = secret_file.read_text().strip()
    while time.time() < deadline:
        time.sleep(min(30, max(1, deadline - time.time())))
    log("WATCHDOG DEADLINE reached; requesting provider stop")
    stop_pod(pod_id, token)


def command(argv, *, cwd=ROOT, deadline, logfile=None, env=None):
    remaining = int(deadline - time.time())
    if remaining <= RESERVE_S:
        raise RuntimeError("GPU budget/deadline reserve reached; no new task starts")
    max_seconds = remaining - RESERVE_S
    log("RUN", " ".join(argv[:5]), "limit_seconds", max_seconds)
    with (open(logfile, "a") if logfile else open(os.devnull, "w")) as stream:
        p = subprocess.run(argv, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT,
                           timeout=max_seconds, check=False)
    if p.returncode:
        raise RuntimeError("Command failed with exit " + str(p.returncode)
                           + "; see " + str(logfile))
    return p.returncode


def checkpoint_digest(directory):
    h = hashlib.sha256()
    if not directory.is_dir():
        raise RuntimeError("Frozen nominal G1 checkpoint is missing")
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(directory).as_posix().encode() + b"\0"
                     + hashlib.sha256(p.read_bytes()).hexdigest().encode() + b"\n")
    return h.hexdigest()


def setup_askpass():
    script = SECRETS / "askpass.sh"
    script.write_text('#!/bin/sh\ncase "$1" in *Username*) echo x-access-token;; '
                      '*Password*) cat /workspace/.g1-one-command-secrets/github;; '
                      '*) exit 1;; esac\n')
    script.chmod(0o700)
    env = os.environ.copy()
    env["GIT_ASKPASS"] = str(script)
    env["GIT_TERMINAL_PROMPT"] = "0"
    return env


def verify_git_write(env, deadline):
    # Non-mutating dry run of a fresh branch name, never force push.
    branch = "refs/heads/g1-publisher-probe-" + uuid.uuid4().hex[:10]
    command(["git", "push", "--dry-run", "origin", "HEAD:" + branch],
            deadline=deadline, env=env, logfile=None)


def publish(run_dir, run_id, env, deadline):
    if deadline - time.time() < 120:
        raise RuntimeError("Too little time left for publication; provider stop still required")
    files = [p for p in run_dir.rglob("*") if p.is_file()]
    # GitHub repository blob limit and quota guard.
    if any(p.stat().st_size > 85 * 1024 * 1024 for p in files):
        raise RuntimeError("A result artifact exceeds safe GitHub blob size; raw files retained locally")
    if sum(p.stat().st_size for p in files) > 180 * 1024 * 1024:
        raise RuntimeError("Results exceed this publisher's size limit; raw files retained locally")
    publication_root = OUTPUT_ROOT / ("publisher_" + run_id)
    command(["git", "clone", "--depth", "1", "--branch", RESULT_BRANCH,
             REPO_URL, str(publication_root)], deadline=deadline,
            env=env, logfile=run_dir / "publish.log")
    dest = publication_root / "results" / ("g1_course_dev_" + run_id)
    shutil.copytree(run_dir, dest, ignore=shutil.ignore_patterns("publish.log"))
    git = lambda *args: subprocess.run(["git", "-C", str(publication_root), *args],
                                        env=env, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, timeout=45)
    git("config", "user.name", "G1 Experiment Publisher")
    git("config", "user.email", "g1-results@users.noreply.github.com")
    git("add", "--", str(dest.relative_to(publication_root)))
    commit = git("commit", "-m", "data: publish G1 development run " + run_id)
    if commit.returncode:
        raise RuntimeError("Git result commit failed")
    push = git("push", "origin", "HEAD:refs/heads/" + RESULT_BRANCH)
    if push.returncode:
        raise RuntimeError("Git result push failed; retry transport only, no physics replay")
    head = git("rev-parse", "HEAD")
    remote = git("ls-remote", "origin", "refs/heads/" + RESULT_BRANCH)
    if head.returncode or remote.returncode or head.stdout.strip() != remote.stdout.split()[0]:
        raise RuntimeError("Published commit read-back mismatch")
    log("PUBLISHED VERIFIED:", RESULT_BRANCH, head.stdout.decode().strip())
    return head.stdout.decode().strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watchdog", action="store_true")
    parser.add_argument("--pod-id")
    parser.add_argument("--deadline", type=float)
    args = parser.parse_args()
    if args.watchdog:
        watchdog(args.pod_id, args.deadline, SECRETS / "runpod")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SECRETS.mkdir(mode=0o700, exist_ok=True)
    lock = open(OUTPUT_ROOT / "one_attempt.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("Another G1 experiment launcher already owns this pod")

    pod_id = os.environ.get("RUNPOD_POD_ID", "").strip()
    if not pod_id:
        if not sys.stdin.isatty():
            raise SystemExit("RUNPOD_POD_ID required")
        pod_id = input("RunPod pod ID: ").strip()
    token = ask_secret("RUNPOD_API_KEY", SECRETS / "runpod")
    try:
        pod = validate_pod(pod_id, token)
    except Exception:
        # Incompatible manually deployed pods must not accrue more billing
        # while the operator reads the failure message.
        log("UNAUTHORIZED CONFIGURATION OR PROVIDER READ FAILURE; attempting stop")
        stop_pod(pod_id, token)
        raise
    price = float(pod["cost"])
    pod_started = started_epoch(pod)
    # 5h20m maximum from pod START, never 6h from starting this script.
    max_budget_s = int(COST_MAX_USD / price * 3600) - 40 * 60
    deadline = pod_started + min(LIFETIME_LIMIT_S, max_budget_s)
    if deadline - time.time() < 45 * 60:
        stop_pod(pod_id, token)
        raise SystemExit("Pod already too old: no safe setup+experiment budget remains")
    log("Verified approved Community pod", pod_id, "price", price,
        "seconds remaining", int(deadline - time.time()))

    # Start detached pod-side deadline guard immediately after provider validation.
    guard_log = OUTPUT_ROOT / "watchdog.log"
    with open(guard_log, "a") as out:
        subprocess.Popen([sys.executable, str(Path(__file__).resolve()),
                          "--watchdog", "--pod-id", pod_id,
                          "--deadline", str(deadline)],
                         stdout=out, stderr=subprocess.STDOUT,
                         start_new_session=True, stdin=subprocess.DEVNULL)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:6]
    outdir = OUTPUT_ROOT / run_id
    outdir.mkdir()
    status = {"kind": "g1_course_development", "run_id": run_id, "split": "dev",
              "pod_id": pod_id, "gpu_rate_per_hr": price, "source_commit": "",
              "frozen_heuristic_commit": FROZEN_BASELINE,
              "obstacle_72_episode_trial": "NOT_EXECUTED", "stages": {},
              "scientific_claims": "Only recorded episode data from completed stages"}
    state_path = outdir / "run_status.json"
    published = False
    try:
        env = setup_askpass()
        ask_secret("GH_TOKEN", SECRETS / "github")
        verify_git_write(env, deadline)
        status["stages"]["github_write_preflight"] = "passed"
        githead = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain",
                                    "--untracked-files=no"], text=True).strip():
            raise RuntimeError("Tracked source is dirty")
        status["source_commit"] = githead
        write_json(state_path, status)

        command(["bash", "scripts/bootstrap_fresh_pod.sh"],
                deadline=deadline, logfile=outdir / "install.log")
        status["stages"]["pinned_bootstrap"] = "passed"
        py = ROOT.parent / "mujoco_playground/rss2025/.venv/bin/python"
        if not py.is_file():
            raise RuntimeError("Pinned venv Python was not installed")
        env["PATH"] = str(py.parent) + os.pathsep + env.get("PATH", "")
        env["PYTHONPATH"] = str(ROOT)
        command(["bash", "scripts/g1_demo_preflight.sh"],
                deadline=deadline, logfile=outdir / "preflight.log", env=env)
        status["stages"]["cpu_preflight_checkpoint"] = "passed"
        status["checkpoint_sha"] = checkpoint_digest(ROOT / "checkpoints/nominal_s1/000202342400")
        if status["checkpoint_sha"] != CHECKPOINT_SHA:
            raise RuntimeError("Checkpoint changed after preflight")

        # A real, already implemented MuJoCo/JAX closed-loop dev benchmark.
        # Distinct from the protected 320-row WP 2.2 science and the proposed
        # 72-row obstacle study; do not merge claims across these experiments.
        subprocess.run(["git", "-C", str(ROOT), "fetch", "origin", "oaa/claude-bench"],
                       check=True, stdout=subprocess.DEVNULL)
        frozen = subprocess.check_output(["git", "-C", str(ROOT), "show",
             FROZEN_BASELINE + ":results/tune_heuristic_dev_cpu_v0/heuristic.json"])
        (outdir / "frozen_heuristic.json").write_bytes(frozen)
        bench = outdir / "bench_dev"
        command([str(py), "scripts/eval_adaptive_bench.py",
                 "--split", "dev", "--impl", "jax", "--allow-cpu",
                 "--out", str(bench), "--families",
                 "gain_070,gain_070_return,healthy,sensor_scale_080",
                 "--methods", "none,oracle,estimated_inverse,heuristic_gate",
                 "--telemetry", "declared", "--episodes", "8", "--batch", "8",
                 "--steps", "300", "--schedule", "walk_turn_stop_resume",
                 "--heuristic", str(outdir / "frozen_heuristic.json"),
                 "--trace"], deadline=deadline, logfile=outdir / "bench.log", env=env)
        import csv
        with (bench / "episodes.csv").open() as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 128:
            raise RuntimeError("Benchmark incomplete: " + str(len(rows)) + " rows, expected 128")
        status["stages"]["frozen_heuristic_g1_dev_bench"] = "completed_128_rows"
        status["bench_rows"] = len(rows)
        # No physical obstacle runner exists in the reviewed branch.
        if not (ROOT / "scripts/eval_g1_obstacle_demo.py").is_file():
            status["stages"]["depth_obstacle_g1_demo"] = "BLOCKED_missing_integrated_runner"
            status["obstacle_72_episode_trial"] = "NOT_EXECUTED"
            status["overall"] = "PARTIAL_REAL_G1_BENCH_NO_OBSTACLE_VIDEO"
        else:
            # No unreviewed interface guessed; new runner requires its own CLI,
            # independent tests, registration and staging.
            status["stages"]["depth_obstacle_g1_demo"] = "PENDING_separate_reviewed_protocol"
            status["overall"] = "PARTIAL_REAL_G1_BENCH"
    except Exception as exc:
        status["overall"] = "FAILED_OR_INCOMPLETE"
        status["error_type"] = type(exc).__name__
        status["error"] = str(exc)[:500]
        status["stages"].setdefault("failed_stage", "see_logs")
        log("FAIL CLOSED:", type(exc).__name__, str(exc)[:180])
    finally:
        write_json(state_path, status)
        try:
            if (SECRETS / "github").exists():
                env = setup_askpass()
                status["publication_commit"] = publish(outdir, run_id, env, deadline)
                published = True
                write_json(state_path, status)
        except Exception as exc:
            log("PUBLISH FAILED:", type(exc).__name__, str(exc)[:220])
            status["publication_error"] = type(exc).__name__
            write_json(state_path, status)
        log("Stopping provider pod; scientific rows are",
            status.get("bench_rows", 0), "published", published)
        stopped = stop_pod(pod_id, token)
        if not stopped:
            log("URGENT: provider EXITED not independently confirmed. Check RunPod console.")
        # Do not claim a successful integrated demonstration from a partial run.
    if status.get("overall") == "FAILED_OR_INCOMPLETE" or not published:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
