"""Publication receipts verified against isolated local Git repositories only."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "receipt_publisher", ROOT / "scripts/publish_experiment_results.py")
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class PublicationReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="foresight-publication-test-")
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.repo = root / "code"
        self.remote = root / "results.git"
        self.env = os.environ.copy()
        self.env.update(GIT_AUTHOR_NAME="CPU Test", GIT_AUTHOR_EMAIL="cpu@example.invalid",
                        GIT_COMMITTER_NAME="CPU Test", GIT_COMMITTER_EMAIL="cpu@example.invalid",
                        GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull)
        for key in ("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"):
            self.env.pop(key, None)
        subprocess.run(["git", "init", "--quiet", "--initial-branch=code", str(self.repo)],
                       env=self.env, check=True)
        subprocess.run(["git", "init", "--quiet", "--bare", str(self.remote)],
                       env=self.env, check=True)
        (self.repo / "source.py").write_text("version = 1\n")
        self.git("add", "source.py")
        self.git("commit", "--quiet", "-m", "Original experiment source")
        self.source = self.git("rev-parse", "HEAD").strip()
        self.git("remote", "add", "origin", str(self.remote))
        self.run_dir = self.repo / "results" / "cpu_run"
        self.run_dir.mkdir(parents=True)
        (self.run_dir / "run.json").write_text(json.dumps({
            "run_id": "cpu_run", "source_commit": self.source,
            "status": "completed", "exit_code": 0,
        }) + "\n")
        self.log = self.run_dir / "console.log"
        self.log.write_text("Experiment completed. RUNPOD_API_KEY=synthetic_secret\n")
        self.root_patch = mock.patch.object(publisher, "SOURCE_DIRECTORY", self.repo)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)

    def git(self, *args, remote=False, data=None):
        directory = self.remote if remote else self.repo
        return subprocess.run(["git", "-C", str(directory), *args], input=data,
                              text=True, capture_output=True, check=True,
                              env=self.env).stdout

    def publish_local_fixture(self):
        # Production rejects non-GitHub destinations. Only the isolated test
        # fixture bypasses that check so no external network/write is possible.
        with mock.patch.object(publisher.Git, "validate_origin", return_value=None):
            return publisher.publish(self.run_dir, self.log, 0)

    def receipt(self):
        return json.loads((self.run_dir / publisher.RECEIPT_FILE).read_text())

    def advance_remote(self, parent=None):
        if parent:
            tree = self.git("rev-parse", parent + "^{tree}", remote=True).strip()
        else:
            blob = self.git("hash-object", "-w", "--stdin", remote=True,
                            data='{"concurrent_writer":true}\n').strip()
            tree = self.git("mktree", remote=True,
                            data=f"100644 blob {blob}\tlatest.json\n").strip()
        parents = ("-p", parent) if parent else ()
        commit = self.git("commit-tree", tree, *parents, "-m", "Concurrent local fixture",
                          remote=True).strip()
        self.git("update-ref", publisher.REF, commit, remote=True)
        return commit

    def test_verified_receipt_uses_original_run_source_and_preserves_checkout(self):
        (self.repo / "source.py").write_text("version = 2\n")
        self.git("commit", "--quiet", "-am", "Publisher code moved after the experiment")
        current = self.git("rev-parse", "HEAD").strip()
        (self.repo / "source.py").write_text("version = 3  # uncommitted local work\n")
        before_diff = self.git("diff", "--binary")
        before_index = (self.repo / ".git/index").read_bytes()
        (self.run_dir / "credentials.json").write_text('{"secret":"MUST_NOT_PUBLISH"}\n')
        stage = self.run_dir / "release_diagnostic"
        (stage / "episodes").mkdir(parents=True)
        (stage / "traces").mkdir()
        (stage / "config.json").write_text('{"period":1}\n')
        conditions = (
            "batch000_condition000_reactive_none.csv",
            "batch000_condition001_oracle_oracle.csv",
            "batch000_condition002_belief_commit_legacy.csv",
            "batch000_condition003_belief_commit_stale_only.csv",
            "batch000_condition004_belief_commit_deadline.csv",
        )
        for folder in ("episodes", "traces"):
            for name in conditions:
                (stage / folder / name).write_text("fell\n0\n")

        url = self.publish_local_fixture()
        receipt = self.receipt()
        self.assertEqual(receipt["status"], "verified")
        self.assertIs(receipt["verified"], True)
        self.assertEqual(receipt["run_source"], self.source)
        self.assertEqual(receipt["publisher_source"], current)
        self.assertEqual(receipt["run_exit_code"], 0)
        self.assertEqual(receipt["commit"], self.git("rev-parse", publisher.REF, remote=True).strip())
        self.assertIn("/tree/" + receipt["commit"] + "/", url)
        published = json.loads(self.git("show", publisher.REF + ":results/cpu_run/publication.json", remote=True))
        self.assertEqual(published["source_commit"], self.source)
        paths = self.git("ls-tree", "-r", "--name-only", publisher.REF, remote=True).splitlines()
        self.assertNotIn("results/cpu_run/credentials.json", paths)
        self.assertNotIn("results/cpu_run/publication_receipt.json", paths)
        for folder in ("episodes", "traces"):
            for name in conditions:
                self.assertIn(f"results/cpu_run/release_diagnostic/{folder}/{name}", paths)
        self.assertTrue(all(publisher.allowed_tree_path(path) for path in paths))
        self.assertNotIn("synthetic_secret", self.git("show", publisher.REF + ":results/cpu_run/console.log", remote=True))
        self.assertEqual(self.git("rev-parse", "HEAD").strip(), current)
        self.assertEqual(self.git("diff", "--binary"), before_diff)
        self.assertEqual((self.repo / ".git/index").read_bytes(), before_index)

    def test_remote_descendant_retaining_publication_is_verified(self):
        original = publisher.Git.run
        descendant = []

        def push_then_advance(instance, *args, **kwargs):
            result = original(instance, *args, **kwargs)
            if args[0] == "push":
                parent = self.git("rev-parse", publisher.REF, remote=True).strip()
                descendant.append(self.advance_remote(parent))
            return result

        with mock.patch.object(publisher.Git, "run", new=push_then_advance):
            self.publish_local_fixture()
        receipt = self.receipt()
        self.assertTrue(receipt["verified"])
        self.assertEqual(receipt["remote_tip"], descendant[0])
        self.assertNotEqual(receipt["commit"], descendant[0])

    def test_concurrent_parent_change_retries_once_without_force(self):
        original = publisher.Git.run
        pushes = []

        def race_first_push(instance, *args, **kwargs):
            if args[0] == "push":
                pushes.append(args)
                if len(pushes) == 1:
                    self.advance_remote()
            return original(instance, *args, **kwargs)

        with mock.patch.object(publisher.Git, "run", new=race_first_push):
            self.publish_local_fixture()
        self.assertEqual(len(pushes), 2)
        self.assertFalse(any("force" in arg for args in pushes for arg in args))
        self.assertEqual(self.receipt()["attempts"], 2)
        self.assertTrue(self.receipt()["verified"])

    def test_push_success_without_remote_verification_is_failure(self):
        with mock.patch.object(publisher, "verify_published_commit", return_value=False):
            with self.assertRaisesRegex(publisher.PublicationError, "could not be verified"):
                self.publish_local_fixture()
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertIs(self.receipt()["verified"], False)
        self.assertIsNotNone(self.receipt()["commit"])

    def test_invalid_run_source_invalidates_old_verified_receipt_without_push(self):
        publisher.atomic_receipt(self.run_dir, {"status": "verified", "verified": True})
        (self.run_dir / "run.json").write_text('{"source_commit":"not-a-commit"}\n')
        with self.assertRaisesRegex(publisher.PublicationError, "full source_commit"):
            self.publish_local_fixture()
        self.assertEqual(self.receipt()["status"], "failed")
        self.assertFalse(self.receipt()["verified"])
        self.assertEqual(self.git("for-each-ref", "--format=%(refname)", remote=True), "")

    def test_wrong_destination_is_rejected_with_failure_receipt(self):
        with self.assertRaisesRegex(publisher.PublicationError, "expected Foresight repository"):
            publisher.publish(self.run_dir, self.log, 0)
        self.assertFalse(self.receipt()["verified"])
        self.assertEqual(self.git("for-each-ref", "--format=%(refname)", remote=True), "")

    def test_exit_code_must_match_frozen_run(self):
        with self.assertRaisesRegex(publisher.PublicationError, "does not match"):
            publisher.publish(self.run_dir, self.log, 1)
        self.assertFalse(self.receipt()["verified"])


class PublicationSchemaTests(unittest.TestCase):
    def test_named_outputs_and_condition_tables_only(self):
        for path in ("results/run1/run.json", "results/run1/preflight.json",
                     "results/run1/errors.json", "results/run1/release_diagnostic/config.json",
                     "results/run1/release_diagnostic/batches.json",
                     "results/run1/release_diagnostic/traces/batch000_condition003_belief_commit_deadline.csv"):
            self.assertTrue(publisher.allowed_tree_path(path), path)
        for path in ("results/run1/credentials.json", "results/run1/tokens.log",
                     "results/run1/episodes/credentials.csv", "results/run1/secret.env",
                     "results/run1/release_diagnostic/private_config.json",
                     "results/run1/publication_receipt.json", "scripts/start.py"):
            self.assertFalse(publisher.allowed_tree_path(path), path)

    def test_structured_and_query_credentials_are_redacted(self):
        examples = (
            "RUNPOD_API_KEY=synthetic_secret_value",
            "https://example.invalid/path?token=synthetic_secret_value",
            '{"api_key":"synthetic_secret_value"}',
            "Authorization: Bearer synthetic_secret_value",
        )
        for text in examples:
            self.assertNotIn("synthetic_secret_value", publisher.sanitize(text))


if __name__ == "__main__":
    unittest.main()
