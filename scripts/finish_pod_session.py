#!/usr/bin/env python3
"""Request this pod's stop only after the current invocation's verified upload.

FORESIGHT_AUTO_STOP=1 authorizes stopping; NO_STOP=1 always overrides it.
No credentials are installed, no pod is terminated, and no other pod is selected.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
OBJECT_ID = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
FINAL_EXPERIMENT_STATES = frozenset(("completed", "failed", "timed_out"))


class StopBlocked(Exception):
    """A check prevented stopping; the pod remains under the user's control."""


def read_json(path: Path) -> dict:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024:
            raise StopBlocked("A required run record is missing or invalid.")
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise StopBlocked("A required run record could not be read.") from None
    if not isinstance(value, dict):
        raise StopBlocked("A required run record is not a JSON object.")
    return value


def verify_current_upload(root: Path, invocation_id: str) -> tuple[dict, dict]:
    """Use the invocation token, publisher result, and frozen run to reject stale receipts."""
    if not invocation_id:
        raise StopBlocked("The current launcher invocation ID is missing.")
    latest = read_json(root / "logs/last_run.json")
    if latest.get("schema_version") != 1 or latest.get("invocation_id") != invocation_id:
        raise StopBlocked("The last-run record belongs to a different launcher invocation.")
    if latest.get("status") not in ("completed", "failed"):
        raise StopBlocked("The current invocation has not finished finalizing its results.")
    if (type(latest.get("publication_exit_code")) is not int
            or latest["publication_exit_code"] != 0
            or latest.get("publication_status") != "verified"):
        raise StopBlocked("The current invocation did not finish a verified upload.")

    run_id = latest.get("run_id")
    source = latest.get("source_commit")
    exit_code = latest.get("experiment_exit_code")
    state = latest.get("experiment_status")
    if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
        raise StopBlocked("The run ID is invalid.")
    if not isinstance(source, str) or OBJECT_ID.fullmatch(source) is None:
        raise StopBlocked("The run source revision is invalid.")
    if type(exit_code) is not int or not 0 <= exit_code <= 255 or state not in FINAL_EXPERIMENT_STATES:
        raise StopBlocked("The experiment lacks a completed, failed, or timed-out final status.")
    if (state == "completed") != (exit_code == 0):
        raise StopBlocked("The experiment status and exit code disagree.")
    raw_path = latest.get("run_dir")
    expected = root / "results" / run_id
    if (not isinstance(raw_path, str) or not Path(raw_path).is_absolute()
            or Path(raw_path).resolve() != expected.resolve() or expected.is_symlink()):
        raise StopBlocked("The run directory does not match the recorded run ID.")
    run = read_json(expected / "run.json")
    if (run.get("run_id") != run_id or run.get("source_commit") != source
            or run.get("status") != state or type(run.get("exit_code")) is not int
            or run["exit_code"] != exit_code):
        raise StopBlocked("The frozen experiment record does not match the final run status.")

    receipt = read_json(expected / "publication_receipt.json")
    commit = receipt.get("commit")
    if (receipt.get("schema_version") != 1 or receipt.get("status") != "verified"
            or receipt.get("verified") is not True or receipt.get("run_id") != run_id
            or receipt.get("run_source") != source or type(receipt.get("run_exit_code")) is not int
            or receipt["run_exit_code"] != exit_code
            or not isinstance(commit, str) or OBJECT_ID.fullmatch(commit) is None
            or latest.get("publication_receipt") != receipt):
        raise StopBlocked("The publication receipt does not verify this exact run and source.")
    return latest, receipt


def run_command(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(command, capture_output=True, text=True,
                              timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        raise StopBlocked("A pod-control check or request timed out; stop was not confirmed.") from None
    except OSError:
        raise StopBlocked("A required pod-control command could not run.") from None


def stop_arguments(cli: str) -> list[str]:
    """Probe help only, supporting both documented CLI generations."""
    for arguments in (("pod", "stop"), ("stop", "pod")):
        result = run_command([cli, *arguments, "--help"], timeout=10)
        help_text = (result.stdout + "\n" + result.stderr).lower()
        # An unrelated top-level help page is not evidence that this verb exists.
        expected_usage = r"\brunpodctl\s+" + r"\s+".join(arguments) + r"\b"
        if result.returncode == 0 and re.search(expected_usage, help_text):
            return list(arguments)
    raise StopBlocked("Installed runpodctl supports neither recognized stop command. No stop was sent.")


def require_idle_gpu(binary: str):
    result = run_command([binary, "--query-compute-apps=pid", "--format=csv,noheader,nounits"], timeout=15)
    if result.returncode:
        raise StopBlocked("GPU process status could not be checked. No stop was sent.")
    rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(row.isdigit() and int(row) > 0 for row in rows):
        raise StopBlocked("A GPU process is still active. No stop was sent.")
    if rows:
        raise StopBlocked("GPU process status was not recognizable. No stop was sent.")


def write_stop_record(root: Path, record: dict):
    path = root / "logs/pod_stop_receipt.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def finalize(root: Path = ROOT, environ=None) -> int:
    env = os.environ if environ is None else environ
    if env.get("NO_STOP") == "1":
        print("Automatic stop disabled by NO_STOP=1. No stop was requested.", flush=True)
        return 0
    if env.get("FORESIGHT_AUTO_STOP") != "1":
        print("Automatic stop is not enabled. No stop was requested.", flush=True)
        return 0
    stop_record = None
    try:
        latest, receipt = verify_current_upload(root, env.get("FORESIGHT_INVOCATION_ID", ""))
        print(f"Upload verified for {latest['run_id']} at commit {receipt['commit']}.", flush=True)
        pod_id = env.get("RUNPOD_POD_ID", "")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", pod_id) is None:
            raise StopBlocked("RUNPOD_POD_ID is missing or invalid. No stop was sent.")
        cli = shutil.which("runpodctl")
        if cli is None:
            raise StopBlocked("runpodctl is unavailable. Results are uploaded; stop this pod in RunPod.")
        gpu_check = shutil.which("nvidia-smi")
        if gpu_check is None:
            raise StopBlocked("nvidia-smi is unavailable, so active GPU work cannot be ruled out.")
        arguments = stop_arguments(cli)
        require_idle_gpu(gpu_check)
        stop_record = {
            "schema_version": 1, "invocation_id": env["FORESIGHT_INVOCATION_ID"],
            "run_id": latest["run_id"], "run_source": latest["source_commit"],
            "publication_commit": receipt["commit"], "pod_id": pod_id,
            "status": "requesting_stop", "command_style": " ".join(arguments),
            "requested_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        write_stop_record(root, stop_record)
        print("Sending the authorized stop request for this pod. Uploaded results are verified.", flush=True)
        print("Stopping clears container storage; /workspace retention depends on its actual mount.", flush=True)
        result = run_command([cli, *arguments, pod_id], timeout=30)
        stop_record["command_exit_code"] = result.returncode
        if result.returncode:
            stop_record["status"] = "request_failed"
            write_stop_record(root, stop_record)
            raise StopBlocked("runpodctl did not accept the stop request. Results are uploaded; check RunPod.")
        stop_record["status"] = "stop_requested"
        write_stop_record(root, stop_record)
        print("Stop request accepted by runpodctl. Pod shutdown has not been independently observed.", flush=True)
        return 0
    except StopBlocked as error:
        if stop_record is not None and stop_record.get("status") == "requesting_stop":
            stop_record["status"] = "request_unconfirmed"
            try:
                write_stop_record(root, stop_record)
            except OSError:
                pass
        print(f"Automatic stop withheld or unconfirmed: {error}", flush=True)
        print("The pod may still be running and billing. Check it in RunPod.", flush=True)
        return 2
    except (OSError, ValueError, TypeError):
        print("Automatic stop withheld or unconfirmed because local run records could not be validated or saved.", flush=True)
        print("The pod may still be running and billing. Check it in RunPod.", flush=True)
        return 2


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    return finalize()


if __name__ == "__main__":
    raise SystemExit(main())
