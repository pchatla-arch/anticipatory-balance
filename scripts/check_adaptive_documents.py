#!/usr/bin/env python3
"""Validate registered adaptive-execution documents without changing any file.

Uses only the Python standard library. Checks declared identities, versions,
local content hashes, source/PDF build bindings and decision/review references.
It does not fetch history, authenticate approvals, prove scientific claims or
replace visual PDF review. Historical hashes are format-checked, not refetched.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import urlsplit


HEX256 = re.compile(r"[0-9a-f]{64}\Z")
HEX160 = re.compile(r"[0-9a-f]{40}\Z")
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?\Z")
REVIEW_STATES = {"not_requested", "pending", "in_review", "reviewed", "changes_requested", "not_required"}
APPROVAL_STATES = {"not_recorded", "pending", "approved", "superseded", "not_required"}


def version_tuple(value):
    match = VERSION.fullmatch(value) if isinstance(value, str) else None
    return tuple(int(part or 0) for part in match.groups()) if match else None


class Validator:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.errors = []
        self.checked_hashes = {}

    def error(self, code, message):
        self.errors.append(f"[{code}] {message}")

    def relative_path(self, value, label, require_exists=True):
        if (not isinstance(value, str) or not value or "\\" in value
                or any(ord(c) < 32 for c in value)):
            self.error("PATH", f"{label}: expected a repository-relative POSIX path")
            return None
        rel = PurePosixPath(value)
        if rel.is_absolute() or ".." in rel.parts or str(rel) != value:
            self.error("PATH", f"{label}: invalid relative path {value!r}")
            return None
        path = (self.root / value).resolve()
        if not path.is_relative_to(self.root):
            self.error("PATH", f"{label}: path escapes repository")
            return None
        if require_exists and not path.is_file():
            self.error("MISSING_FILE", f"{label}: {value}")
            return None
        return path

    def read_text(self, value, label):
        path = self.relative_path(value, label)
        if path is None:
            return ""
        try:
            return path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            self.error("READ", f"{label}: {exc}")
            return ""

    def read_json(self, path, label):
        try:
            value = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            self.error("JSON", f"{label}: {exc}")
            return None
        if not isinstance(value, dict):
            self.error("SCHEMA", f"{label}: expected a JSON object")
            return None
        return value

    def digest_map(self, value, label, verify=True):
        if not isinstance(value, dict) or not value:
            self.error("SCHEMA", f"{label}: expected a nonempty path-to-SHA256 object")
            return {}
        valid = {}
        for name, digest in value.items():
            path = self.relative_path(name, label, require_exists=verify)
            if not isinstance(digest, str) or not HEX256.fullmatch(digest):
                self.error("SHA256", f"{label}: invalid SHA256 for {name!r}")
                continue
            valid[name] = digest
            if verify and path is not None:
                try:
                    actual = self.checked_hashes.setdefault(name, hashlib.sha256(path.read_bytes()).hexdigest())
                except OSError as exc:
                    self.error("READ", f"{label}: {exc}")
                    continue
                if actual != digest:
                    self.error("ARTIFACT_HASH", f"{label}: {name} differs from its recorded SHA256")
        return valid

    def evidence_reference(self, value, label):
        if not isinstance(value, str) or not value.strip():
            self.error("EVIDENCE", f"{label}: a nonempty evidence reference is required")
            return
        parsed = urlsplit(value)
        if parsed.scheme:
            if parsed.scheme != "https" or not parsed.netloc or not parsed.path.strip("/"):
                self.error("EVIDENCE", f"{label}: expected an HTTPS evidence URL or local file")
        else:
            self.relative_path(value.split("#", 1)[0], label)

    def check_build(self, doc, artifacts):
        label = doc["document_id"]
        provenance = doc.get("provenance_status")
        if provenance == "legacy_byte_verified":
            if doc.get("build_manifest_path"):
                self.error("PROVENANCE", f"{label}: legacy byte verification must not imply build provenance")
            return
        if provenance != "build_manifest":
            self.error("PROVENANCE", f"{label}: expected build_manifest or legacy_byte_verified")
            return
        manifest_name = doc.get("build_manifest_path")
        manifest_path = self.relative_path(manifest_name, f"{label} build manifest")
        if not isinstance(manifest_name, str) or manifest_name not in artifacts:
            self.error("BUILD_BINDING", f"{label}: build manifest must be included in artifacts")
        if manifest_path is None:
            return
        manifest = self.read_json(manifest_path, f"{label} build manifest")
        if manifest is None:
            return
        if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
                or manifest.get("document_id") != label
                or manifest.get("version") != doc.get("version")):
            self.error("BUILD_BINDING", f"{label}: build manifest schema/document/version mismatch")
        inputs = self.digest_map(manifest.get("inputs"), f"{label} build inputs")
        outputs = self.digest_map(manifest.get("outputs"), f"{label} build outputs")
        if not isinstance(doc.get("source_path"), str) or doc["source_path"] not in inputs:
            self.error("BUILD_BINDING", f"{label}: source_path is not a build input")
        if doc.get("pdf_path") and (not isinstance(doc["pdf_path"], str) or doc["pdf_path"] not in outputs):
            self.error("BUILD_BINDING", f"{label}: pdf_path is not a build output")
        for name, digest in {**inputs, **outputs}.items():
            if artifacts.get(name) != digest:
                self.error("BUILD_BINDING", f"{label}: {name} is missing or differs in artifacts")
        pages = manifest.get("pages")
        if type(pages) is not int or pages < 1:
            self.error("BUILD_BINDING", f"{label}: pages must be a positive integer")
        if not isinstance(manifest.get("build_environment"), dict) or not manifest["build_environment"]:
            self.error("BUILD_BINDING", f"{label}: build_environment must be a nonempty object")
        visual = manifest.get("visual_review")
        if not isinstance(visual, dict) or visual.get("status") not in {"pending", "passed", "failed", "not_recorded"}:
            self.error("VISUAL_REVIEW", f"{label}: invalid visual_review status")
        elif visual["status"] == "passed":
            self.evidence_reference(visual.get("evidence"), f"{label} visual review")
        elif doc.get("approval_status") == "approved":
            self.error("VISUAL_REVIEW", f"{label}: an approved built document needs a passed visual review")

    def check(self, registry, baseline=None):
        if type(registry.get("schema_version")) is not int or registry["schema_version"] != 1:
            self.error("SCHEMA", "registry schema_version must be 1")
        changelog = self.read_text(registry.get("changelog_path"), "changelog")
        decisions = self.read_text(registry.get("decision_log_path"), "decision log")
        docs = registry.get("documents")
        if not isinstance(docs, list) or not docs:
            self.error("SCHEMA", "documents must be a nonempty list")
            docs = []
        seen = set()
        for doc in docs:
            if not isinstance(doc, dict):
                self.error("SCHEMA", "each document entry must be an object")
                continue
            label = doc.get("document_id")
            if not isinstance(label, str) or not re.fullmatch(r"[A-Z][A-Z0-9_-]*", label):
                self.error("SCHEMA", "invalid document_id")
                continue
            if label in seen:
                self.error("SCHEMA", f"duplicate document_id {label}")
            seen.add(label)
            if version_tuple(doc.get("version")) is None:
                self.error("VERSION", f"{label}: expected a numeric two/three-part version")
            for field in ("owner", "reviewer"):
                if not isinstance(doc.get(field), str) or not doc[field].strip():
                    self.error("SCHEMA", f"{label}: {field} is required")
            source = self.read_text(doc.get("source_path"), f"{label} source")
            marker = doc.get("version_marker")
            if (not isinstance(marker, str) or not marker
                    or str(doc.get("version")) not in marker or marker not in source):
                self.error("VERSION_MARKER", f"{label}: version_marker must contain the version and occur in source")
            heading = doc.get("changelog_heading")
            if not isinstance(heading, str) or not heading or heading not in changelog:
                self.error("CHANGELOG_ENTRY", f"{label}: changelog_heading is missing from changelog")
            artifacts = self.digest_map(doc.get("artifacts"), f"{label} artifacts")
            if not isinstance(doc.get("source_path"), str) or doc["source_path"] not in artifacts:
                self.error("SCHEMA", f"{label}: source_path must be registered in artifacts")
            pdf_name = doc.get("pdf_path")
            if pdf_name:
                pdf = self.relative_path(pdf_name, f"{label} PDF")
                if not isinstance(pdf_name, str) or pdf_name not in artifacts:
                    self.error("SCHEMA", f"{label}: pdf_path must be registered in artifacts")
                if pdf is not None:
                    with pdf.open("rb") as handle:
                        if handle.read(5) != b"%PDF-":
                            self.error("PDF_HEADER", f"{label}: {pdf_name} is not a PDF byte stream")
            review = doc.get("review_status")
            if review not in REVIEW_STATES:
                self.error("REVIEW_STATUS", f"{label}: invalid review_status {review!r}")
            elif review in {"reviewed", "changes_requested"}:
                self.evidence_reference(doc.get("review_evidence"), f"{label} peer review")
            approval = doc.get("approval_status")
            if approval not in APPROVAL_STATES:
                self.error("APPROVAL_STATUS", f"{label}: invalid approval_status {approval!r}")
            elif approval == "approved":
                decision = doc.get("approval_decision_id")
                if (not isinstance(decision, str) or not decision
                        or not re.search(r"(?<![A-Za-z0-9_-])" + re.escape(decision)
                                         + r"(?![A-Za-z0-9_-])", decisions)):
                    self.error("APPROVAL_EVIDENCE", f"{label}: approval_decision_id must occur in decision log")
                if review != "reviewed":
                    self.error("APPROVAL_EVIDENCE", f"{label}: approval requires recorded peer review")
            self.check_build(doc, artifacts)
        history = registry.get("history", [])
        if not isinstance(history, list):
            self.error("SCHEMA", "history must be a list")
            history = []
        for item in history:
            if not isinstance(item, dict):
                self.error("SCHEMA", "each history entry must be an object")
                continue
            label = f"history {item.get('document_id')} {item.get('version')}"
            commit = item.get("release_commit")
            if not isinstance(commit, str) or not HEX160.fullmatch(commit):
                self.error("HISTORY_COMMIT", f"{label}: full lowercase 40-hex release_commit required")
            artifacts = item.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                self.error("HISTORY_HASH", f"{label}: artifacts must contain historical file identities")
                continue
            for name, identity in artifacts.items():
                self.relative_path(name, label, require_exists=False)
                if not isinstance(identity, dict) or not any(k in identity for k in ("git_blob_sha1", "sha256")):
                    self.error("HISTORY_HASH", f"{label}: {name} needs git_blob_sha1 or sha256")
                    continue
                for key, pattern in (("git_blob_sha1", HEX160), ("sha256", HEX256)):
                    if key in identity and (not isinstance(identity[key], str) or not pattern.fullmatch(identity[key])):
                        self.error("HISTORY_HASH", f"{label}: invalid {key} for {name}")
        if baseline is not None:
            baseline_docs = baseline.get("documents")
            if (type(baseline.get("schema_version")) is not int
                    or baseline["schema_version"] != 1 or not isinstance(baseline_docs, list)):
                self.error("BASELINE_SCHEMA", "baseline must have schema_version 1 and a documents list")
                baseline_docs = []
            previous = {d["document_id"]: d for d in baseline_docs
                        if isinstance(d, dict) and isinstance(d.get("document_id"), str)}
            current = {d["document_id"]: d for d in docs
                       if isinstance(d, dict) and isinstance(d.get("document_id"), str)}
            for label, old in previous.items():
                if label not in current:
                    self.error("REMOVED_DOCUMENT", f"{label}: keep a superseded entry rather than silently removing it")
                    continue
                new = current[label]
                before, after = version_tuple(old.get("version")), version_tuple(new.get("version"))
                if before is not None and after is not None and after < before:
                    self.error("VERSION_ROLLBACK", f"{label}: version decreased")
                if old.get("artifacts") != new.get("artifacts"):
                    if before is None or after is None or after <= before:
                        self.error("VERSION_BUMP", f"{label}: changed artifacts require a higher version")
                    if old.get("changelog_heading") == new.get("changelog_heading"):
                        self.error("CHANGELOG_ENTRY", f"{label}: changed artifacts need a new changelog heading")
        return {"ok": not self.errors, "documents_checked": len(docs),
                "history_entries_checked": len(history), "errors": self.errors,
                "limitations": ["Historical bytes and external evidence URLs are not fetched.",
                                "Reference existence does not authenticate human approval or prove scientific claims.",
                                "PDF signature/hash checks do not replace rendering and visual review."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--registry", type=Path, default=Path("docs/adaptive_execution/document_versions.json"))
    parser.add_argument("--baseline", type=Path, help="previous registry JSON; changed artifacts require version and changelog updates")
    parser.add_argument("--json", action="store_true", help="print the machine-readable validation report")
    args = parser.parse_args()
    validator = Validator(args.root)
    registry_path = args.registry if args.registry.is_absolute() else args.root / args.registry
    registry = validator.read_json(registry_path, "registry")
    baseline = validator.read_json(args.baseline, "baseline") if args.baseline else None
    if registry is not None:
        report = validator.check(registry, baseline)
    else:
        report = {"ok": False, "documents_checked": 0, "history_entries_checked": 0,
                  "errors": validator.errors}
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"{'PASS' if report['ok'] else 'FAIL'}: {report['documents_checked']} current documents; "
              f"{report['history_entries_checked']} historical entries")
        for error in report["errors"]:
            print(error, file=sys.stderr)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
