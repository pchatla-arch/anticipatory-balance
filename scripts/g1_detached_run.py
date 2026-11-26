#!/usr/bin/env python3
"""Durable foreground enrollment and detached G1 development-run supervisor.

Launch from an interactive pod terminal:
    python3 scripts/g1_detached_run.py

The short foreground enrollment captures secrets with hidden prompts, validates
the authenticated pod and exclusive project ownership, then starts a detached
worker and an independent in-pod monitor. The monitor snapshots completed
experiment cells to GitHub, attempts a final rescue if the worker disappears,
and requests provider stop. This is NOT a provider-hosted guaranteed timer:
if the whole container or network goes down, the monitor cannot execute.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "scripts" / "run_g1_pod_one_command.py"
spec = importlib.util.spec_from_file_location("g1_worker_module", SUPERVISOR)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

WORKSPACE = Path("/workspace")
SESSION_DIR = WORKSPACE / "g1_run_outputs"
SECURE_DIR = WORKSPACE / ".g1-one-command-secrets"
SNAPSHOT_MIN_INTERVAL = 120
INTERIM_MIN_ROWS = 8
DEATH_GRACE_SECONDS = 20


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def log(*args):
    print(now(), *args, flush=True)


def process_start_ticks(pid):
    """Linux proc starttime prevents PID reuse and treats zombie as exited."""
    try:
        stat = Path("/proc") / str(pid) / "stat"
        value = stat.read_text()
        fields = value[value.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def finished_episode_rows(run):
    f = run / "bench_dev" / "episodes.csv"
    if not f.exists():
        return 0
    try:
        with f.open(newline="") as fp:
            lines = list(csv.DictReader(fp))
        keys = {(v.get("scenario_id"), v.get("family"),
                 v.get("method"), v.get("telemetry")) for v in lines}
        if len(keys) != len(lines):
            return 0
        return len(lines)
    except (OSError, csv.Error, UnicodeError):
        return 0


def latest_run():
    if not SESSION_DIR.is_dir():
        return None
    folders = [p for p in SESSION_DIR.iterdir()
               if p.is_dir() and p.name.startswith("20")
               and (p / "run_status.json").is_file()]
    return max(folders, key=lambda p: p.stat().st_mtime) if folders else None


def git_checked(argv, *, cwd=ROOT, env=None, timeout=50):
    try:
        result = subprocess.run(["git", *argv], cwd=cwd, env=env, timeout=timeout,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.TimeoutExpired, OSError) as exc:
        raise RuntimeError("Git stage unavailable: " + type(exc).__name__) from None
    if result.returncode:
        # Never emit Git stderr: token-bearing URL rewrites could appear there.
        raise RuntimeError("Git stage failed at " + argv[0]
                           + " (exit " + str(result.returncode) + ")")
    return result.stdout.decode("utf-8", errors="replace").strip()


def publish_snapshot(run, *, final=False):
    """Publish an immutable snapshot with per-cell causal raw data.

    Only completed NPZ files older than 15 seconds are included; a half-written
    trace is not trusted. An unexpected exit can therefore leave unavailable
    traces, which must be explicitly recorded in the snapshot metadata.
    """
    rows = finished_episode_rows(run)
    if rows < INTERIM_MIN_ROWS and not final:
        return None
    auth_file = SECURE_DIR / "github"
    if not auth_file.is_file() or auth_file.is_symlink() or auth_file.stat().st_mode & 0o077:
        raise RuntimeError("No verified private GitHub credential for interim backup")
    env = R.setup_askpass()
    name = run.name
    nonce = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:5]
    with tempfile.TemporaryDirectory(prefix="g1_publish_") as tmp:
        repo = Path(tmp) / "checkout"
        git_checked(["clone", "--depth", "1", "--branch", R.RESULT_BRANCH,
                     R.REPO_URL, str(repo)], env=env, timeout=80)
        dest = repo / "results" / "g1_interim" / name / nonce
        dest.mkdir(parents=True)
        selected = ["run_status.json", "bench.log", "supervisor_tests.log",
                    "preflight.log", "frozen_heuristic.json"]
        if (run / "bench_dev" / "episodes.csv").is_file():
            selected.append("bench_dev/episodes.csv")
        if (run / "bench_dev" / "registration.json").is_file():
            selected.append("bench_dev/registration.json")
        if (run / "bench_dev" / "summary.json").is_file():
            selected.append("bench_dev/summary.json")
        traces = sorted((run / "bench_dev").glob("trace_*.npz"))
        copied = []
        for f in traces:
            if f.stat().st_size <= 24 * 1024 * 1024 and time.time() - f.stat().st_mtime >= 15:
                selected.append(str(f.relative_to(run)))
        for relative in selected:
            src = run / relative
            if not src.is_file():
                continue
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            sz = src.stat().st_size
            if sz > 24 * 1024 * 1024:
                continue
            shutil.copy2(src, target)
            if src.stat().st_size == sz:
                copied.append(relative)
            else:
                target.unlink(missing_ok=True)
        atomic_json(dest / "checkpoint_metadata.json", {
            "kind": "G1_INTERIM_DEVELOPMENT_SNAPSHOT",
            "run_id": name, "captured_utc": now(),
            "episode_rows_observed": rows, "expected_full_dev_rows": 128,
            "science_complete": False,
            "final_rescue": bool(final),
            "files_copied": copied,
            "warning": "Partial snapshot. Not a completed 128-row run; "
                       "no obstacle/VLA experiment or video claim.",
        })
        git_checked(["config", "user.name", "G1 Interim Rescue"], cwd=repo, env=env)
        git_checked(["config", "user.email", "g1-results@users.noreply.github.com"],
                    cwd=repo, env=env)
        git_checked(["add", "--", str(dest.relative_to(repo))], cwd=repo, env=env)
        git_checked(["commit", "-m", "data: interim G1 experiment snapshot " + nonce],
                    cwd=repo, env=env)
        for attempt in range(3):
            try:
                git_checked(["push", "origin", "HEAD:refs/heads/" + R.RESULT_BRANCH],
                            cwd=repo, env=env, timeout=65)
                break
            except RuntimeError:
                if attempt == 2:
                    raise
                git_checked(["fetch", "origin", R.RESULT_BRANCH], cwd=repo, env=env)
                git_checked(["rebase", "FETCH_HEAD"], cwd=repo, env=env)
        sha = git_checked(["rev-parse", "HEAD"], cwd=repo, env=env)
        remote = git_checked(["ls-remote", "origin", "refs/heads/" + R.RESULT_BRANCH],
                             cwd=repo, env=env)
        if not remote or remote.split()[0] != sha:
            raise RuntimeError("Remote GitHub SHA readback mismatch for interim data")
        log("PUBLISHED INTERIM", rows, "rows; verified", sha)
        return sha


def monitor(pid, start_ticks, pod_id, deadline):
    token_path = SECURE_DIR / "runpod"
    last_rows = -1
    last_push = 0.0
    seen_run = None
    while True:
        current = time.time()
        alive = process_start_ticks(pid) == start_ticks
        run = latest_run()
        if run and run != seen_run:
            seen_run = run
            last_rows = -1
        rows = finished_episode_rows(run) if run else 0
        if (run is not None and rows >= INTERIM_MIN_ROWS
                and rows > last_rows and current - last_push >= SNAPSHOT_MIN_INTERVAL
                and current + 120 < deadline):
            try:
                publish_snapshot(run)
                last_push = time.time()
                last_rows = rows
            except Exception as exc:
                log("INTERIM PUBLISH WARNING:", type(exc).__name__, str(exc)[:120])
        if not alive or current >= deadline:
            reason = "worker_exited" if not alive else "deadline_reached"
            log("MONITOR", reason, "observed_episode_rows", rows)
            # Allow the normal worker's finally block to finish publication/stop.
            for _ in range(DEATH_GRACE_SECONDS // 2):
                try:
                    token = token_path.read_text().strip()
                    info = R.runpod_request(pod_id, token)
                    if info.get("status") in ("EXITED", "TERMINATED"):
                        log("MONITOR verified provider", info["status"])
                        return
                except Exception as exc:
                    log("MONITOR provider check warning:", type(exc).__name__)
                time.sleep(2)
            if run:
                try:
                    publish_snapshot(run, final=True)
                except Exception as exc:
                    log("FINAL RESCUE WARNING:", type(exc).__name__, str(exc)[:120])
            try:
                token = token_path.read_text().strip()
                if not R.stop_pod(pod_id, token):
                    log("URGENT: provider stop not independently verified")
            except Exception as exc:
                log("URGENT: provider stop failed:", type(exc).__name__)
            return
        time.sleep(10)


def launch():
    SESSION_DIR.mkdir(exist_ok=True, parents=True)
    SECURE_DIR.mkdir(exist_ok=True, mode=0o700, parents=True)
    with (SESSION_DIR / "interactive_launch.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        pidfile = SESSION_DIR / "active_worker.json"
        if pidfile.exists():
            old = json.loads(pidfile.read_text())
            if process_start_ticks(old.get("pid")) == old.get("start_ticks"):
                raise RuntimeError("A G1 scientific worker is already active on this pod")
        pod_id = os.environ.get("RUNPOD_POD_ID") or input("RunPod pod ID: ").strip()
        key = R.ask_secret("RUNPOD_API_KEY", SECURE_DIR / "runpod")
        try:
            pod = R.validate_pod(pod_id, key)
            mode = R.verify_no_competing_pod(pod_id, key)
            price = float(pod["cost"])
            start = R.started_epoch(pod)
            deadline = start + min(R.LIFETIME_LIMIT_S,
                                   int(R.COST_MAX_USD / price * 3600) - 40 * 60)
            if deadline - time.time() < 45 * 60:
                raise RuntimeError("Insufficient authorized time for new G1 worker")
            R.ask_secret("GH_TOKEN", SECURE_DIR / "github")
        except BaseException:
            log("ENROLLMENT FAILED: requesting provider stop immediately")
            R.stop_pod(pod_id, key)
            raise
        atomic_json(SECURE_DIR / "ownership_receipt.json", {
            "pod_id": pod_id, "started_at": pod.get("startedAt"),
            "timestamp": time.time(), "mode": mode,
        })
        (SECURE_DIR / "ownership_receipt.json").chmod(0o600)
        log_file = SESSION_DIR / "detached_session.log"
        env = os.environ.copy()
        for name in ("GH_TOKEN", "RUNPOD_API_KEY"):
            env.pop(name, None)
        env["RUNPOD_POD_ID"] = pod_id
        env["G1_DETACHED_WORKER"] = "1"
        with log_file.open("a") as stream:
            worker = subprocess.Popen([sys.executable, str(SUPERVISOR)],
                                      cwd=ROOT, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT,
                                      stdin=subprocess.DEVNULL,
                                      start_new_session=True, close_fds=True)
        ticks = process_start_ticks(worker.pid)
        if ticks is None:
            log("Detached worker failed to start; stopping provider pod")
            R.stop_pod(pod_id, key)
            raise RuntimeError("Detached G1 worker exited before monitor was installed")
        atomic_json(pidfile, {"pid": worker.pid, "start_ticks": ticks,
                              "pod_id": pod_id, "started_utc": now()})
        try:
            with (SESSION_DIR / "detached_monitor.log").open("a") as stream:
                watcher = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--monitor", "--pid", str(worker.pid),
                     "--start-ticks", str(ticks),
                     "--pod-id", pod_id, "--deadline", str(deadline)],
                    cwd=ROOT, env=env, stdout=stream,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    start_new_session=True, close_fds=True)
            if process_start_ticks(watcher.pid) is None:
                raise RuntimeError("Detached rescue monitor did not start")
        except BaseException:
            log("Cannot establish independent in-pod monitor; stopping pod")
            R.stop_pod(pod_id, key)
            raise
        log("DETACHED G1 worker PID", worker.pid, "monitor active")
        log("Watch log: tail -f", log_file)
        log("Progress: tail -f", SESSION_DIR / "detached_monitor.log")
        log("The original development benchmark may take time to compile; "
            "GitHub interim commits follow completed result cells.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--monitor", action="store_true")
    p.add_argument("--pid", type=int)
    p.add_argument("--start-ticks", type=int)
    p.add_argument("--pod-id")
    p.add_argument("--deadline", type=float)
    args = p.parse_args()
    if args.monitor:
        if not args.pid or not args.start_ticks or not args.pod_id or not args.deadline:
            raise SystemExit("Monitor requires pid, start-ticks, pod-id and deadline")
        monitor(args.pid, args.start_ticks, args.pod_id, args.deadline)
    else:
        launch()


if __name__ == "__main__":
    main()
