"""Pure local tests for GitHub preflight isolation; do not use a token or network."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "run_g1_pod_one_command.py"
spec = importlib.util.spec_from_file_location("one_command_github", MODULE)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


class GitHubAuthenticationTests(unittest.TestCase):
    def test_askpass_only_reads_private_token_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            secrets = Path(tmp) / "secrets"
            secrets.mkdir()
            with patch.object(L, "SECRETS", secrets):
                env = L.setup_askpass()
                script = secrets / "askpass.sh"
                self.assertTrue(script.exists())
                self.assertTrue(env["GIT_ASKPASS"].endswith("askpass.sh"))
                self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
                self.assertEqual(env["GIT_CONFIG_KEY_0"], "credential.helper")
                self.assertEqual(env["GIT_CONFIG_VALUE_0"], "")
                self.assertNotIn("ghp_", script.read_text())
                self.assertNotIn("github_pat_", script.read_text())
                self.assertEqual(script.stat().st_mode & 0o777, 0o700)

    def test_ask_secret_file_permissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            dest = Path(tmp) / "private" / "github"
            with patch.dict(os.environ, {"GH_TOKEN": "placeholder-only"}, clear=False):
                result = L.ask_secret("GH_TOKEN", dest)
            self.assertEqual(result, "placeholder-only")
            self.assertEqual(dest.read_text(), "placeholder-only")
            self.assertEqual(dest.stat().st_mode & 0o777, 0o600)

    def test_repo_auth_preflight_never_runs_with_wrong_origin(self):
        from subprocess import CalledProcessError
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(L, "ROOT", Path(tmp)):
                with patch.object(L.subprocess, "check_output",
                                  return_value="https://github.com/other/repository.git\n"):
                    with self.assertRaisesRegex(RuntimeError, "Unexpected GitHub remote"):
                        L.verify_git_write({}, 99999999999)

    def test_preflight_refuses_insufficient_budget(self):
        with self.assertRaisesRegex(RuntimeError, "Deadline too close"):
            L.verify_git_write({}, 0)


if __name__ == "__main__":
    unittest.main()
