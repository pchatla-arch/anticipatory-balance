#!/usr/bin/env python3
"""Publish a small pod-run record without changing the current Git worktree.

Only the named JSON/CSV outputs, a redacted console log, and publication records
are sent to the dedicated results branch. Git objects and a temporary index are
used directly: no checkout, add, stash, reset, config write, or force push occurs.
The existing Git credentials are used without interactive authentication.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
from urllib.parse import quote, urlsplit


REMOTE = "origin"
BRANCH = "foresight/pod-results"
REF = "refs/heads/" + BRANCH
PUBLIC_REPOSITORY = "https://github.com/pchatla-arch/anticipatory-balance"
RESULT_FILES = ("summary.json", "manifest.json", "scenarios.csv", "episodes.csv")
RUN_FILES = frozenset((*RESULT_FILES, "console.log", "publication.json"))
MAX_RESULT_BYTES = 64 * 1024 * 1024
MAX_LOG_BYTES = 1024 * 1024
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
OBJECT_ID = re.compile(rb"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


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
  return re.sub(
      r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+",
      r"\1 [REDACTED]", text)


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
  return (len(parts) == 3 and parts[0] == "results"
          and RUN_ID.fullmatch(parts[1]) is not None and parts[2] in RUN_FILES)


def read_result(path: Path) -> bytes:
  if path.is_symlink() or not path.is_file():
    raise PublicationError("An allowed result path is not a regular file.")
  if path.stat().st_size > MAX_RESULT_BYTES:
    raise PublicationError("An allowed result file exceeds the 64 MiB publication limit.")
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


def publish(run_dir: Path, log: Path, exit_code: int) -> str:
  run_dir = run_dir.expanduser().resolve()
  run_id = run_dir.name
  if RUN_ID.fullmatch(run_id) is None:
    raise PublicationError("Run directory basename must be 1–128 letters, digits, dots, underscores, or hyphens, starting with a letter or digit.")
  prefix = "results/" + run_id + "/"
  files = {}
  for name in RESULT_FILES:
    path = run_dir / name
    if path.exists() or path.is_symlink():
      files[prefix + name] = read_result(path)
  files[prefix + "console.log"], truncated = read_log(log.expanduser())
  source_directory = Path(__file__).resolve().parents[1]

  with tempfile.TemporaryDirectory(prefix="foresight-publish-") as temporary:
    git = Git(source_directory, Path(temporary) / "index")
    source = git.oid("rev-parse", "HEAD", operation="Reading source revision")
    git.identity()
    git.validate_origin()
    status, output = git.run("ls-remote", "--exit-code", "--heads", REMOTE, REF,
                             operation="Inspecting results branch", network=True,
                             allowed=(0, 2))
    parent = None
    if status == 0:
      lines = output.splitlines()
      if len(lines) != 1:
        raise PublicationError("Results branch lookup was ambiguous.")
      fields = lines[0].split(b"\t")
      if len(fields) != 2 or fields[1] != REF.encode() or not OBJECT_ID.fullmatch(fields[0]):
        raise PublicationError("Results branch lookup returned invalid metadata.")
      parent = fields[0].decode("ascii")
      # Fetch only the advertised object; do not update FETCH_HEAD or tracking refs.
      git.run("fetch", "--no-tags", "--no-write-fetch-head", "--no-recurse-submodules",
              "--refmap=", REMOTE, parent, operation="Fetching results history", network=True)
      _, tree_paths = git.run("ls-tree", "-r", "--name-only", "-z", parent,
                              operation="Checking results tree")
      if any(not allowed_tree_path(path.decode("utf-8"))
             for path in tree_paths.split(b"\0") if path):
        raise PublicationError("Results branch contains files outside the publication allowlist.")
      git.run("read-tree", parent, operation="Preparing results index")
    else:
      # The first results commit is independent of the code branch's tree/history.
      git.run("read-tree", "--empty", operation="Preparing empty results index")

    record = {
        "run_id": run_id,
        "results_path": prefix.rstrip("/"),
        "source_commit": source,
        "run_exit_code": exit_code,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    files["latest.json"] = json_bytes(record)
    files[prefix + "publication.json"] = json_bytes({
        **record, "schema_version": 1, "console_log_truncated": truncated,
        "result_files_present": [name for name in RESULT_FILES if prefix + name in files],
        "result_files_missing": [name for name in RESULT_FILES if prefix + name not in files],
    })
    # A retry of the same run ID must not retain stale files absent in this snapshot.
    git.run("update-index", "--force-remove", "--",
            *(prefix + name for name in sorted(RUN_FILES)),
            operation="Preparing run snapshot")
    entries = []
    for path, content in sorted(files.items()):
      blob = git.oid("hash-object", "-w", "--stdin", operation="Writing result blob", data=content)
      entries.append(f"100644 {blob}\t{path}\0".encode("utf-8"))
    git.run("update-index", "-z", "--index-info", operation="Indexing result blobs",
            data=b"".join(entries))
    tree = git.oid("write-tree", operation="Writing results tree")
    parents = ("-p", parent) if parent is not None else ()
    commit = git.oid("commit-tree", tree, *parents, "-m", f"Publish pod results {run_id}",
                     operation="Creating results commit")
    # A concurrent publisher causes an ordinary non-fast-forward rejection.
    # Nothing is forced; rerunning will build on the new remote parent.
    git.run("push", REMOTE, f"{commit}:{REF}",
            operation="Publishing results branch", network=True)
  return PUBLIC_REPOSITORY + "/tree/" + BRANCH + "/results/" + quote(run_id, safe="")


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
