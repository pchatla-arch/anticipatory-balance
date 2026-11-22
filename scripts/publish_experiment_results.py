#!/usr/bin/env python3
"""Publish a small pod-run record without changing the current Git worktree.

Only the named JSON/CSV outputs, a redacted console log, and publication records
are sent to the dedicated results branch. Git objects and a temporary index are
used directly: no checkout, add, stash, reset, config write, or force push occurs.
The existing Git credentials are used without interactive authentication.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import zipfile
from urllib.parse import quote, urlsplit


REMOTE = "origin"
BRANCH = "foresight/experiment-results"
REF = "refs/heads/" + BRANCH
PUBLIC_REPOSITORY = "https://github.com/pchatla-arch/anticipatory-balance"
SOURCE_DIRECTORY = Path(__file__).resolve().parents[1]
STRUCTURED_FILES = frozenset({
    "run.json", "preflight.json", "summary.json", "manifest.json", "config.json",
    "batches.json", "release_audit.json", "setup.json", "setup_status.json", "status.json",
    "error.json", "errors.json", "scenarios.csv", "episodes.csv", "frontier.csv",
    "error.log", "errors.log", "setup.log", "bootstrap.log",
    "tests.json", "plan.json", "response_model.json", "report.md", "analysis.json",
    "audit.json", "paired.json", "calibration.json", "table.csv", "lead_curve.csv",
    "hold_comparisons.csv", "batch_summaries.json",
})
BINARY_FILES = frozenset({"episodes.npz", "fixed_controller_comparison.png",
                          "depth_comparison.png", "readiness.png", "closed_loop.png", "frontier.png"})
GENERATED_FILES = frozenset({"console.log", "publication.json"})
RECEIPT_FILE = "publication_receipt.json"
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
OBJECT_ID = re.compile(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")
CONDITION_CSV = re.compile(
    r"batch[0-9]{3,}_condition[0-9]{3,}_"
    r"(?:reactive|oracle|always|ttc_threshold|belief_commit)"
    r"(?:_(?:none|oracle|legacy|stale_only|deadline))?\.csv\Z")
REVIEW_TABLE = re.compile(r"episodes_b[0-9]{3,}_(?:(?:brace|zerocmd)_launch[01]|c[0-9]{3,})\.csv\Z")
DEPTH_TABLE = re.compile(r"(?:depth|parametric|parametric_matched|baseline|none)_(?:reactive|ttc_threshold|belief_lead|belief_wait|belief_voi)_p[0-9]+(?:p[0-9]+)?_batch[0-9]{3,}\.csv\Z")
REVIEW_TRACE = re.compile(r"(?:oracle_b[0-9]{3,}_(?:brace|zerocmd)_launch[01]|frontier_b[0-9]{3,}_c[0-9]{3,})\.npz\Z")
SESSION_H_TABLE = re.compile(r"(?:batch[0-9]{3,}_condition[0-9]{3,}|(?:parametric|depth|none)_(?:reactive|ttc_threshold|belief_lead|belief_wait|belief_voi)_[0-9]+(?:\.[0-9]+)?_b[0-9]{3,})\.csv\Z")


class PublicationError(Exception):
  """A safe-to-display message containing no subprocess output or credentials."""


def sanitize(text: str) -> str:
  """Remove credentials commonly found in captured shell and Git diagnostics."""
  text = re.sub(
      r"(?i)\b((?:https?|ssh|git|ftp)://)[^/\s]*@",
      r"\1[REDACTED]@", text)
  text = re.sub(
      r"\b(?:github_pat_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9_]+)",
      "[REDACTED_GITHUB_TOKEN]", text)
  text = re.sub(
      r"(?i)(\bauthorization[\"']?\s*[:=]\s*[\"']?\s*)[^\r\n\"']+",
      r"\1[REDACTED_AUTHORIZATION]", text)
  text = re.sub(
      r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+",
      r"\1 [REDACTED]", text)
  text = re.sub(
      r"(?i)([?&](?:token|access_token|api_key|secret|password)=)[^&\s\"']+",
      r"\1[REDACTED]", text)
  return re.sub(
      r"(?i)([\"']?(?:api[_-]?key|access[_-]?token|token|password|passwd|client[_-]?secret|"
      r"secret[_-]?key|[A-Z][A-Z0-9_]*_(?:TOKEN|API_KEY|PASSWORD|SECRET|SECRET_ACCESS_KEY))"
      r"[\"']?\s*[:=]\s*)([\"']?)([^\r\n\s,;}\"']+)(\2)",
      lambda match: match[1] + match[2] + "[REDACTED]" + match[4], text)


class Git:
  def __init__(self, directory: Path, index: Path):
    self.directory = directory
    self.env = os.environ.copy()
    self.env.update({
        "GIT_INDEX_FILE": str(index),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
        "SSH_ASKPASS_REQUIRE": "never",
        "GCM_INTERACTIVE": "never",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_SSH_COMMAND": "ssh -o BatchMode=yes -o ConnectTimeout=15 -o ConnectionAttempts=1",
    })

  def run(self, *args, operation: str, data: bytes | None = None,
          network: bool = False, allowed=(0,)):
    # Capture all output: Git's errors can include a credential-bearing remote URL.
    command = ["git", "-C", str(self.directory),
               "-c", "credential.interactive=never",
               "-c", "commit.gpgsign=false",
               "-c", "http.lowSpeedLimit=1024", "-c", "http.lowSpeedTime=20",
               *args]
    try:
      with subprocess.Popen(command, env=self.env, stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True) as proc:
        try:
          out, _ = proc.communicate(data, timeout=45 if network else 20)
        except subprocess.TimeoutExpired:
          os.killpg(proc.pid, signal.SIGKILL)
          proc.communicate()
          raise PublicationError(f"{operation} timed out.") from None
        if proc.returncode not in allowed:
          raise PublicationError(f"{operation} failed.")
        return proc.returncode, out
    except OSError:
      raise PublicationError(f"{operation} could not run.") from None

  def oid(self, *args, operation: str, data: bytes | None = None) -> str:
    _, output = self.run(*args, operation=operation, data=data)
    output = output.strip()
    if not OBJECT_ID.fullmatch(output):
      raise PublicationError(f"{operation} returned an invalid object ID.")
    return output.decode("ascii")

  def identity(self):
    for field, fallback in (("name", "Foresight Results Publisher"),
                            ("email", "foresight-results@users.noreply.github.com")):
      _, value = self.run("config", "--get", "user." + field,
                          operation="Reading Git identity", allowed=(0, 1))
      identity = value.decode("utf-8", errors="replace").strip() or fallback
      self.env.setdefault("GIT_AUTHOR_" + field.upper(), identity)
      self.env.setdefault("GIT_COMMITTER_" + field.upper(), identity)

  def validate_origin(self):
    # Check both configured and effective destinations without printing them.
    # Git can redirect fetch and push separately through URL rewrite rules.
    _, urls = self.run("config", "--get-all", f"remote.{REMOTE}.url",
                       operation="Checking origin configuration", allowed=(0, 1))
    _, push_urls = self.run("config", "--get-all", f"remote.{REMOTE}.pushurl",
                            operation="Checking push configuration", allowed=(0, 1))
    configured = urls.splitlines() + push_urls.splitlines()
    if not urls.strip() or not all(expected_repository(url) for url in configured):
      raise PublicationError("The origin remote must identify the expected Foresight repository.")
    for push in (False, True):
      resolved = self.resolved_urls(push=push)
      if not resolved or not all(expected_repository(url) for url in resolved):
        raise PublicationError("The effective Git destination must identify the expected Foresight repository.")

  def resolved_urls(self, *, push=False):
    options = ("--push", "--all") if push else ("--all",)
    _, output = self.run("remote", "get-url", *options, REMOTE,
                         operation="Resolving Git destination")
    return output.splitlines()


def expected_repository(raw: bytes) -> bool:
  try:
    value = raw.decode("utf-8")
    if "://" in value:
      parsed = urlsplit(value)
      if (parsed.scheme not in ("https", "ssh", "git") or
          parsed.hostname != "github.com" or parsed.query or parsed.fragment):
        return False
      if parsed.port not in (None, {"https": 443, "ssh": 22, "git": 9418}[parsed.scheme]):
        return False
      path = parsed.path.strip("/")
    else:
      match = re.fullmatch(r"(?:[^@/:\s]+@)?github\.com:([^\s?#]+)", value)
      if match is None:
        return False
      path = match.group(1).strip("/")
    if path.endswith(".git"):
      path = path[:-4]
    return path.lower() == "pchatla-arch/anticipatory-balance"
  except (UnicodeError, ValueError):
    return False


def allowed_tree_path(path: str) -> bool:
  if path == "latest.json":
    return True
  parts = path.split("/")
  if (not 3 <= len(parts) <= 5 or parts[0] != "results"
      or not all(RUN_ID.fullmatch(part) is not None for part in parts[1:])):
    return False
  relative = parts[2:]
  if relative[-1] in {"registration.json", "actuators.json"}:
    return len(relative) == 1 or (len(relative) == 2 and relative[0] == "cpu_smoke")
  if len(relative) == 1:
    return relative[0] in STRUCTURED_FILES | GENERATED_FILES | BINARY_FILES
  if len(relative) == 2 and (relative[-1] in STRUCTURED_FILES | BINARY_FILES
                            or REVIEW_TABLE.fullmatch(relative[-1])):
    return relative[0] not in {"episodes", "traces", "per_episode"}
  return (len(relative) in (2, 3)
          and relative[-2] in {"episodes", "traces", "per_episode"}
          and (CONDITION_CSV.fullmatch(relative[-1]) is not None
               or DEPTH_TABLE.fullmatch(relative[-1]) is not None
               or SESSION_H_TABLE.fullmatch(relative[-1]) is not None
               or (relative[-2] == "traces" and REVIEW_TRACE.fullmatch(relative[-1]) is not None)))


def read_result(path: Path) -> bytes:
  if path.is_symlink() or not path.is_file():
    raise PublicationError("An allowed result path is not a regular file.")
  if path.stat().st_size > MAX_RESULT_BYTES:
    raise PublicationError("An allowed result file exceeds the 64 MiB publication limit.")
  if path.suffix == '.png':
    data = path.read_bytes()
    if not data.startswith(b'\x89PNG\r\n\x1a\n'):
      raise PublicationError("A report figure is not a PNG file.")
    return data
  if path.suffix == '.npz':
    # The publisher runs with system Python: validate NPY headers without importing
    # NumPy or unpickling. Only known numeric/bool evidence arrays are publishable.
    try:
      with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        if not entries or len(entries) > 1024 or sum(i.file_size for i in entries) > 512*1024*1024:
          raise ValueError('Invalid archive size.')
        for item in entries:
          if not re.fullmatch(r'[A-Za-z0-9_.-]+\.npy', item.filename):
            raise ValueError('Unexpected archive member.')
          with archive.open(item) as stream:
            prefix = stream.read(8)
            if prefix[:6] != b'\x93NUMPY' or prefix[6] not in (1, 2, 3):
              raise ValueError('Not an NPY array.')
            nbytes = 2 if prefix[6] == 1 else 4
            header_len = int.from_bytes(stream.read(nbytes), 'little')
            if not 1 <= header_len <= 65536:
              raise ValueError('Invalid array header.')
            header = ast.literal_eval(stream.read(header_len).decode('latin1'))
            if not isinstance(header, dict) or not re.fullmatch(r'[<>=|]?[?biufc][0-9]*', str(header.get('descr'))):
              raise ValueError('Evidence arrays must be numeric or bool.')
    except (OSError, ValueError, SyntaxError, KeyError, zipfile.BadZipFile) as error:
      raise PublicationError("An evidence archive failed numeric-array validation.") from None
    return path.read_bytes()
  # Also redact text artifacts in case a manifest contains copied command output.
  return sanitize(path.read_text(encoding="utf-8")).encode("utf-8")


def read_log(path: Path) -> tuple[bytes, bool]:
  if not path.exists():
    return b"Console log was unavailable when publication was attempted.\n", False
  if path.is_symlink() or not path.is_file():
    raise PublicationError("The console log is not a regular file.")
  size = path.stat().st_size
  truncated = size > MAX_LOG_BYTES
  with path.open("rb") as stream:
    if truncated:
      stream.seek(size - MAX_LOG_BYTES)
    raw = stream.read(MAX_LOG_BYTES)
  if truncated:
    # Discard a partial first line, which could contain a cut-off credential.
    raw = raw.partition(b"\n")[2]
  text = sanitize(raw.decode("utf-8", errors="replace"))
  if truncated:
    text = "[Console log truncated to its final complete lines, at most 1 MiB.]\n" + text
  return text.encode("utf-8"), truncated


def json_bytes(value: dict) -> bytes:
  return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def atomic_receipt(run_dir: Path, record: dict) -> None:
  """Replace the local receipt only after a complete write and fsync."""
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=run_dir,
                                     prefix=".publication-receipt-", delete=False) as stream:
      temporary = Path(stream.name)
      json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
      stream.write("\n")
      stream.flush()
      os.fsync(stream.fileno())
    temporary.replace(run_dir / RECEIPT_FILE)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def remote_tip(git: Git) -> str | None:
  status, output = git.run("ls-remote", "--exit-code", "--heads", REMOTE, REF,
                           operation="Inspecting results branch", network=True,
                           allowed=(0, 2))
  if status == 2:
    return None
  lines = output.splitlines()
  if len(lines) != 1:
    raise PublicationError("Results branch lookup was ambiguous.")
  fields = lines[0].split(b"\t")
  if len(fields) != 2 or fields[1] != REF.encode() or not OBJECT_ID.fullmatch(fields[0]):
    raise PublicationError("Results branch lookup returned invalid metadata.")
  return fields[0].decode("ascii")


def fetch_object(git: Git, commit: str) -> None:
  # Do not update FETCH_HEAD, remote-tracking branches, or the active checkout.
  git.run("fetch", "--no-tags", "--no-write-fetch-head", "--no-recurse-submodules",
          "--refmap=", REMOTE, commit, operation="Fetching results history", network=True)


def verify_published_commit(git: Git, commit: str, tip: str | None) -> bool:
  """Accept our exact tip or a later tip retaining our publication in its history."""
  if tip is None:
    return False
  if tip == commit:
    return True
  fetch_object(git, tip)
  status, _ = git.run("merge-base", "--is-ancestor", commit, tip,
                      operation="Verifying published results ancestry", allowed=(0, 1))
  return status == 0


def publish(run_dir: Path, log: Path, exit_code: int) -> str:
  run_dir = run_dir.expanduser().resolve()
  run_id = run_dir.name
  if RUN_ID.fullmatch(run_id) is None or not run_dir.is_dir():
    raise PublicationError("Run directory must exist and have a valid publication name.")
  prefix = "results/" + run_id + "/"
  branch_url = PUBLIC_REPOSITORY + "/tree/" + BRANCH + "/results/" + quote(run_id, safe="")
  receipt = {
      "schema_version": 1, "status": "pending", "verified": False,
      "run_id": run_id, "run_exit_code": exit_code, "run_source": None,
      "publisher_source": None, "branch": BRANCH, "ref": REF,
      "results_url": None, "branch_url": branch_url, "commit": None,
      "remote_tip": None, "attempts": 0, "error": None,
      "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
  }
  try:
    # Invalidate a previous verified receipt before any new attempt or validation.
    atomic_receipt(run_dir, receipt)
    run_path = run_dir / "run.json"
    if run_path.is_symlink() or not run_path.is_file():
      raise PublicationError("A regular run.json with the original source_commit is required.")
    try:
      run_record = json.loads(run_path.read_text(encoding="utf-8"))
      source = run_record["source_commit"]
      valid_source = isinstance(source, str) and OBJECT_ID.fullmatch(source.encode("ascii"))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
      valid_source = False
    if not valid_source:
      raise PublicationError("run.json must contain a full source_commit object ID.")
    if run_record.get("run_id", run_id) != run_id:
      raise PublicationError("run.json run_id does not match the run directory.")
    if "exit_code" in run_record and run_record["exit_code"] != exit_code:
      raise PublicationError("The requested exit code does not match the frozen run.json.")
    receipt["run_source"] = source
    atomic_receipt(run_dir, receipt)

    files = {}
    for path in sorted(run_dir.rglob("*")):
      relative = path.relative_to(run_dir).as_posix()
      if relative in GENERATED_FILES or relative == RECEIPT_FILE:
        continue
      if not allowed_tree_path(prefix + relative):
        continue
      if path.is_symlink():
        raise PublicationError("Symlinks are not published.")
      if path.is_file():
        files[prefix + relative] = read_result(path)
    if sum(len(content) for content in files.values()) > 128 * 1024 * 1024:
      raise PublicationError("Compact results exceed the 128 MiB publication limit.")
    files[prefix + "console.log"], truncated = read_log(log.expanduser())

    with tempfile.TemporaryDirectory(prefix="foresight-publish-") as temporary:
      git = Git(SOURCE_DIRECTORY, Path(temporary) / "index")
      publisher_source = git.oid("rev-parse", "HEAD", operation="Reading publisher revision")
      receipt["publisher_source"] = publisher_source
      git.identity()
      git.validate_origin()

      # One retry is permitted only when another writer changed the remote parent.
      for attempt in range(1, 3):
        receipt["attempts"] = attempt
        parent = remote_tip(git)
        if parent is not None:
          fetch_object(git, parent)
          _, tree_paths = git.run("ls-tree", "-r", "--name-only", "-z", parent,
                                  operation="Checking results tree")
          if any(not allowed_tree_path(path.decode("utf-8"))
                 for path in tree_paths.split(b"\0") if path):
            raise PublicationError("Results branch contains files outside the publication allowlist.")
          git.run("read-tree", parent, operation="Preparing results index")
        else:
          git.run("read-tree", "--empty", operation="Preparing empty results index")

        record = {
            "run_id": run_id, "results_path": prefix.rstrip("/"),
            "source_commit": source, "publisher_source_commit": publisher_source,
            "run_exit_code": exit_code, "timestamp_utc": receipt["timestamp_utc"],
        }
        snapshot = dict(files)
        snapshot["latest.json"] = json_bytes(record)
        snapshot[prefix + "publication.json"] = json_bytes({
            **record, "schema_version": 2, "console_log_truncated": truncated,
            "result_files_present": sorted(path[len(prefix):] for path in files),
        })
        _, previous = git.run("ls-files", "-z", "--", prefix,
                              operation="Listing previous run artifacts")
        previous_paths = [path.decode("utf-8") for path in previous.split(b"\0") if path]
        if previous_paths:
          git.run("update-index", "--force-remove", "--", *previous_paths,
                  operation="Preparing run snapshot")
        entries = []
        for path, content in sorted(snapshot.items()):
          blob = git.oid("hash-object", "-w", "--stdin",
                         operation="Writing result blob", data=content)
          entries.append(f"100644 {blob}\t{path}\0".encode("utf-8"))
        git.run("update-index", "-z", "--index-info",
                operation="Indexing result blobs", data=b"".join(entries))
        tree = git.oid("write-tree", operation="Writing results tree")
        parents = ("-p", parent) if parent is not None else ()
        commit = git.oid("commit-tree", tree, *parents, "-m", f"Publish pod results {run_id}",
                         operation="Creating results commit")
        receipt.update(commit=commit, remote_tip=None,
                       results_url=PUBLIC_REPOSITORY + "/tree/" + commit + "/results/" + quote(run_id, safe=""))
        atomic_receipt(run_dir, receipt)
        push_error = None
        try:
          git.run("push", REMOTE, f"{commit}:{REF}",
                  operation="Publishing results branch", network=True)
        except PublicationError as exc:
          push_error = exc

        tip = remote_tip(git)
        if verify_published_commit(git, commit, tip):
          receipt.update(status="verified", verified=True, remote_tip=tip, error=None)
          atomic_receipt(run_dir, receipt)
          return receipt["results_url"]
        if push_error is not None and attempt == 1 and tip is not None and tip != parent:
          continue
        if push_error is not None:
          raise push_error
        raise PublicationError("The pushed results commit could not be verified on the remote branch.")
    raise PublicationError("The results branch changed during both bounded publication attempts.")
  except Exception as exc:
    error = exc if isinstance(exc, PublicationError) else PublicationError(
        "Publication failed during local preparation.")
    receipt.update(status="failed", verified=False, error=str(error))
    try:
      atomic_receipt(run_dir, receipt)
    except OSError:
      pass
    if error is exc:
      raise
    raise error from None


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--run-dir", required=True, type=Path)
  parser.add_argument("--log", required=True, type=Path)
  parser.add_argument("--exit-code", required=True, type=int)
  args = parser.parse_args()
  try:
    url = publish(args.run_dir, args.log, args.exit_code)
  except PublicationError as exc:
    print(f"Result publication failed: {exc} Local run artifacts are unchanged.", file=sys.stderr)
    return 1
  except Exception:
    # Do not expose filenames, subprocess exceptions, or unexpected secret-bearing text.
    print("Result publication failed during local preparation. Local run artifacts are unchanged.",
          file=sys.stderr)
    return 1
  print("Published pod results: " + url)
  print("Results branch: " + PUBLIC_REPOSITORY + "/tree/" + BRANCH)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
