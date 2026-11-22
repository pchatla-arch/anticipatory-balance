"""Relay a completed native publisher's allowlisted Stage A snapshot without a pod write credential.

export: inspect the failed receipt and export the native candidate tree's files.
verify: reassemble the entire tree on another checkout and compare its pod-displayed Git ID.
confirm: fetch the public results branch and confirm the retained run and exact original tree.
All three operations preserve receipts and refs. No command stops a machine.
"""

import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile

import publish_experiment_results as publisher

REF = publisher.REF
PUBLIC = publisher.PUBLIC_REPOSITORY + ".git"
# All and only the file types emitted by the calibration worker and native publisher.
FILES = {
    "run.json",
    "manifest.json",
    "batches.json",
    "summary.json",
    "episodes.csv",
    "report.md",
    "console.log",
    "setup/preflight.json",
    "publication.json",
}
MANDATORY = {"run.json", "report.md", "console.log", "publication.json"}
MAX_BYTES = 8 * 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def oid(value):
    require(
        isinstance(value, str) and re.fullmatch(r"[a-f0-9]{40}", value),
        "Invalid Git object ID",
    )
    return value


def git(repo, *args, data=None, env=None):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        input=data,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        timeout=90,
    ).stdout


def rev(repo, expr):
    return oid(git(repo, "rev-parse", "--verify", expr).decode().strip())


def check_source(repo, source):
    require(
        rev(repo, "HEAD") == oid(source),
        "Checkout differs from the approved source SHA",
    )


def check_run(name):
    require(
        isinstance(name, str) and publisher.RUN_ID.fullmatch(name), "Invalid run ID"
    )


def parent_of(repo, commit):
    raw = git(repo, "cat-file", "-p", oid(commit)).split(b"\n\n", 1)[0]
    parents = [
        line[7:].decode() for line in raw.splitlines() if line.startswith(b"parent ")
    ]
    require(
        len(parents) == 1,
        "Native snapshot requires exactly one existing results parent",
    )
    return oid(parents[0])


def unpack_entries(meta):
    check_run(meta["run"])
    prefix = "results/" + meta["run"] + "/"
    blobs = {}
    total = 0
    for entry in meta["entries"]:
        path = entry["path"]
        require(
            isinstance(path, str)
            and (
                path == "latest.json"
                or path.startswith(prefix)
                and path[len(prefix) :] in FILES
            )
            and publisher.allowed_tree_path(path),
            "Unexpected path outside calibration publication scope",
        )
        require(
            path not in blobs and entry["mode"] == "100644",
            "Duplicate path or unexpected mode",
        )
        blob = base64.b64decode(entry["data"], validate=True)
        total += len(blob)
        require(total <= MAX_BYTES, "Result bundle too large")
        sha = hashlib.sha1(
            b"blob " + str(len(blob)).encode() + b"\0" + blob
        ).hexdigest()
        require(oid(entry["sha"]) == sha, "Result blob hash mismatch")
        text = blob.decode("utf-8")
        require(publisher.sanitize(text) == text, "Unsanitized publisher text")
        blobs[path] = blob
    require(
        "latest.json" in blobs and all(prefix + name in blobs for name in MANDATORY),
        "Required native publication files missing",
    )
    run = json.loads(blobs[prefix + "run.json"])
    current = json.loads(blobs["latest.json"])
    publication = json.loads(blobs[prefix + "publication.json"])
    expected = {
        "run_id": meta["run"],
        "results_path": prefix[:-1],
        "source_commit": oid(meta["source"]),
        "publisher_source_commit": meta["source"],
        "run_exit_code": run["exit_code"],
    }
    require(
        type(run["exit_code"]) is int
        and 0 <= run["exit_code"] <= 255
        and run.get("mode") == "calibrate-identity"
        and run.get("status") in {"completed", "failed", "timed_out", "interrupted"}
        and run.get("run_id") == meta["run"]
        and run.get("source_commit") == meta["source"],
        "Native run record inconsistent",
    )
    require(
        all(
            current.get(k) == v and publication.get(k) == v for k, v in expected.items()
        )
        and current.get("timestamp_utc") == publication.get("timestamp_utc"),
        "Native manifests disagree",
    )
    declared = {prefix + name for name in publication["result_files_present"]}
    require(
        declared == set(blobs) - {"latest.json", prefix + "publication.json"},
        "Unexpected native snapshot inventory",
    )
    return blobs


def export(repo, source, run, output):
    check_source(repo, source)
    check_run(run)
    receipt = json.loads((repo / "results" / run / publisher.RECEIPT_FILE).read_text())
    require(
        receipt.get("status") == "failed"
        and receipt.get("verified") is False
        and receipt.get("run_id") == run
        and receipt.get("ref") == REF
        and receipt.get("run_source") == source
        and receipt.get("publisher_source") == source,
        "Receipt is not a failed native publication of this run",
    )
    commit = oid(receipt["commit"])
    parent = parent_of(repo, commit)
    prefix = "results/" + run + "/"
    require(
        not git(repo, "ls-tree", parent, "--", prefix[:-1]),
        "Run already existed in parent",
    )
    paths = git(
        repo,
        "diff-tree",
        "--no-commit-id",
        "-r",
        "--name-only",
        "--no-renames",
        "-z",
        parent,
        commit,
    ).split(b"\0")
    entries = []
    for pathbytes in filter(None, paths):
        path = pathbytes.decode()
        require(
            path == "latest.json"
            or path.startswith(prefix)
            and path[len(prefix) :] in FILES,
            "Native candidate modifies an unrelated path",
        )
        row = (
            git(repo, "ls-tree", commit, "--", path)
            .decode()
            .strip()
            .split("\t")[0]
            .split()
        )
        require(
            len(row) == 3 and row[:2] == ["100644", "blob"],
            "Deletion or unexpected file mode",
        )
        content = git(repo, "cat-file", "blob", oid(row[2]))
        entries.append(
            dict(
                path=path,
                mode=row[0],
                sha=row[2],
                data=base64.b64encode(content).decode(),
            )
        )
    meta = dict(
        schema=1,
        ref=REF,
        run=run,
        source=source,
        native_commit=commit,
        parent=parent,
        base_tree=rev(repo, parent + "^{tree}"),
        tree=rev(repo, commit + "^{tree}"),
        entries=entries,
    )
    blobs = unpack_entries(meta)
    require(
        json.loads(blobs[prefix + "run.json"])["exit_code"]
        == receipt.get("run_exit_code"),
        "Exit code differs from receipt",
    )
    compressed = gzip.compress(
        json.dumps(meta, separators=(",", ":")).encode(), mtime=0
    )
    with output.open("xb") as f:
        f.write(compressed)
    return dict(
        run=run,
        source=source,
        tree=meta["tree"],
        parent=parent,
        bundle_sha256=hashlib.sha256(compressed).hexdigest(),
        bytes=len(compressed),
        files=len(entries),
    )


def verify(repo, source, path, expected_tree, expected_sha):
    check_source(repo, source)
    require(path.stat().st_size <= MAX_BYTES, "Compressed bundle exceeds limit")
    bundle = path.read_bytes()
    require(
        hashlib.sha256(bundle).hexdigest() == expected_sha,
        "Transferred bundle SHA-256 mismatch",
    )
    with gzip.open(path, "rb") as f:
        raw = f.read(2 * MAX_BYTES + 1)
    require(len(raw) <= 2 * MAX_BYTES, "Decompressed bundle exceeds limit")
    meta = json.loads(raw)
    require(
        meta["schema"] == 1
        and meta["ref"] == REF
        and meta["source"] == source
        and oid(meta["tree"]) == oid(expected_tree),
        "Source, tree or results ref differs from operator pins",
    )
    blobs = unpack_entries(meta)
    parent = oid(meta["parent"])
    require(
        rev(repo, parent + "^{tree}") == oid(meta["base_tree"])
        and not git(repo, "ls-tree", parent, "--", "results/" + meta["run"]),
        "Invalid results parent",
    )
    with tempfile.TemporaryDirectory() as temp:
        env = dict(os.environ, GIT_INDEX_FILE=temp + "/index")
        git(repo, "read-tree", parent, env=env)
        for path, data in blobs.items():
            sha = git(repo, "hash-object", "-w", "--stdin", data=data).decode().strip()
            git(
                repo,
                "update-index",
                "--add",
                "--cacheinfo",
                "100644",
                oid(sha),
                path,
                env=env,
            )
        rebuilt = git(repo, "write-tree", env=env).decode().strip()
    require(
        rebuilt == expected_tree, "Reassembled tree differs from native pod snapshot"
    )
    return dict(
        source=source,
        run=meta["run"],
        parent=parent,
        tree=rebuilt,
        files=len(blobs),
        verified=True,
    )


def confirm(repo, source, run, published):
    with tempfile.TemporaryDirectory() as temp:
        own = export(repo, source, run, Path(temp) / "native.gz")
    tip_info = git(repo, "ls-remote", "--exit-code", PUBLIC, REF).decode().split()
    require(len(tip_info) == 2 and tip_info[1] == REF, "Cannot verify public branch")
    tip = oid(tip_info[0])
    git(repo, "fetch", "--no-tags", "--no-write-fetch-head", PUBLIC, tip)
    git(repo, "merge-base", "--is-ancestor", oid(published), tip)
    require(
        parent_of(repo, published) == own["parent"]
        and rev(repo, published + "^{tree}") == own["tree"],
        "Remote commit differs from native parent/tree",
    )
    runpath = ":results/" + run
    require(
        rev(repo, tip + runpath) == rev(repo, published + runpath),
        "Run changed in later public commit",
    )
    return dict(
        source=source,
        run=run,
        public_commit=published,
        remote_tip=tip,
        tree=own["tree"],
        external_verified=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--source", required=True, help="Approved full source SHA")
    actions = parser.add_subparsers(dest="action", required=True)
    e = actions.add_parser("export")
    e.add_argument("run")
    e.add_argument("output", type=Path)
    v = actions.add_parser("verify")
    v.add_argument("bundle", type=Path)
    v.add_argument("native_tree")
    v.add_argument("bundle_sha256")
    c = actions.add_parser("confirm")
    c.add_argument("run")
    c.add_argument("public_commit")
    a = parser.parse_args()
    if a.action == "export":
        result = export(a.repo, a.source, a.run, a.output)
    elif a.action == "verify":
        result = verify(a.repo, a.source, a.bundle, a.native_tree, a.bundle_sha256)
    else:
        result = confirm(a.repo, a.source, a.run, a.public_commit)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
