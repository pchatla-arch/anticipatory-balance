"""Offline rehearsal with a real publisher, rejected receive-pack, and independent reader."""

import base64
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import relay_adaptive_results as relay


def git(cwd, *args, env=None):
    return (
        subprocess.check_output(
            ["git", "-C", str(cwd), *args], env=env, stderr=subprocess.PIPE
        )
        .decode()
        .strip()
    )


class TransferTests(unittest.TestCase):
    def test_native_rejected_push_export_verify_and_pre_stop_remote_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            project = work / "pod"
            remote = work / "results.git"
            reader = work / "reviewer"
            source = git(ROOT, "rev-parse", "HEAD")
            git(work, "init", "--bare", str(remote))
            git(
                work,
                "clone",
                "-q",
                "--no-checkout",
                "--shared",
                str(ROOT),
                str(project),
            )
            git(project, "checkout", "-q", "--detach", source)
            git(project, "config", "user.name", "Relay fixture")
            git(project, "config", "user.email", "fixture@example.invalid")
            tree = (
                subprocess.check_output(
                    ["git", "-C", str(project), "mktree"], input=b""
                )
                .decode()
                .strip()
            )
            parent = git(
                project, "commit-tree", tree, "-m", "Empty disposable results baseline"
            )
            git(project, "push", str(remote), parent + ":" + relay.REF)
            git(
                work,
                "clone",
                "-q",
                "--shared",
                "--no-checkout",
                str(ROOT),
                str(reader),
            )
            git(reader, "checkout", "-q", "--detach", source)
            ssh = work / "ssh"
            ssh.write_text(
                '#!/bin/sh\ncase "$*" in *git-upload-pack*) exec git-upload-pack "$OAA_FIXTURE_REMOTE";; *git-receive-pack*) exec git-receive-pack "$OAA_FIXTURE_REMOTE";; *) exit 1;; esac\n'
            )
            ssh.chmod(0o700)
            reject = remote / "hooks/pre-receive"
            reject.write_text("#!/bin/sh\nexit 1\n")
            reject.chmod(0o700)
            git(
                project,
                "remote",
                "set-url",
                "origin",
                "git@github.com:pchatla-arch/anticipatory-balance.git",
            )
            env = dict(
                os.environ,
                PATH=str(work) + os.pathsep + os.environ["PATH"],
                GIT_SSH_COMMAND=str(ssh),
                GIT_SSH_VARIANT="ssh",
                OAA_FIXTURE_REMOTE=str(remote),
            )
            name = "adaptive_gate_relay_fixture"
            results = project / "results" / name
            results.mkdir(parents=True)
            run = dict(
                run_id=name,
                source_commit=source,
                mode="calibrate-identity",
                exit_code=124,
                status="timed_out",
            )
            (results / "run.json").write_text(json.dumps(run))
            (results / "report.md").write_text("Disposable offline timeout fixture.\n")
            (results / "ignored.private").write_text("Out-of-scope marker")
            (results / "console.log").write_text("Relay fixture only.\n")
            invocation = [
                sys.executable,
                str(project / "scripts/publish_experiment_results.py"),
                "--run-dir",
                str(results),
                "--log",
                str(results / "console.log"),
                "--exit-code",
                "124",
            ]
            job = subprocess.run(invocation, env=env, capture_output=True)
            self.assertNotEqual(job.returncode, 0)
            receipt = (results / "publication_receipt.json").read_bytes()
            local = json.loads(receipt)
            self.assertEqual(local["status"], "failed")
            self.assertIsNotNone(local["commit"], local["error"])
            self.assertEqual(git(remote, "rev-parse", relay.REF), parent)
            with self.assertRaises(subprocess.CalledProcessError):
                git(reader, "cat-file", "-e", local["commit"] + "^{commit}")
            bundle = work / "snapshot.gz"
            info = relay.export(project, source, name, bundle)
            git(reader, "fetch", str(remote), parent)
            result = relay.verify(
                reader, source, bundle, info["tree"], info["bundle_sha256"]
            )
            self.assertEqual(result["files"], 5)
            self.assertEqual(
                result["tree"], git(project, "rev-parse", local["commit"] + "^{tree}")
            )
            with self.assertRaises(ValueError):
                relay.verify(reader, source, bundle, "0" * 40, info["bundle_sha256"])
            with self.assertRaises(ValueError):
                relay.verify(reader, source, bundle, info["tree"], "0" * 64)
            bad = json.loads(gzip.decompress(bundle.read_bytes()))
            bad["entries"][0]["path"] = "private-file.txt"
            with self.assertRaises(ValueError):
                relay.unpack_entries(bad)
            bad = json.loads(gzip.decompress(bundle.read_bytes()))
            bad["entries"][0]["data"] = base64.b64encode(b"changed").decode()
            with self.assertRaises(ValueError):
                relay.unpack_entries(bad)
            # Emulate only the authenticated connector Git writes, on the disposable remote.
            git(reader, "config", "user.name", "Independent publisher")
            git(reader, "config", "user.email", "fixture@example.invalid")
            proposed = git(
                reader,
                "commit-tree",
                result["tree"],
                "-p",
                parent,
                "-m",
                "Verified alternate transport",
            )
            reject.unlink()
            git(reader, "push", str(remote), proposed + ":" + relay.REF)
            with patch.object(relay, "PUBLIC", str(remote)):
                confirmation = relay.confirm(project, source, name, proposed)
                self.assertTrue(confirmation["external_verified"])
                self.assertEqual(confirmation["public_commit"], proposed)
                # Simulate an unrelated branch movement dropping the authenticated commit.
                git(remote, "update-ref", relay.REF, parent, proposed)
                with self.assertRaises(subprocess.CalledProcessError):
                    relay.confirm(project, source, name, proposed)
            self.assertEqual(
                (results / "publication_receipt.json").read_bytes(), receipt
            )


if __name__ == "__main__":
    unittest.main()
