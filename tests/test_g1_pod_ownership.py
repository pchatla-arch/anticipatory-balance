"""Unit tests for RunPod ownership gate; no credentials or GPU required."""
import importlib.util
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "scripts/run_g1_pod_one_command.py"
spec = importlib.util.spec_from_file_location("g1_pod_launcher", MODULE)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


class Reply:
    def __init__(self, obj):
        self.body = io.BytesIO(json.dumps(obj).encode())
    def __enter__(self): return self
    def __exit__(self, *args): self.body.close()
    def read(self): return self.body.read()


POD = {"id": "current", "name": "robot-research", "status": "RUNNING"}
PAGED = {"pods": [POD], "pagination": {"hasNextPage": False, "nextCursor": None}}


class OwnershipTests(unittest.TestCase):
    def test_one_pod_is_verified(self):
        with patch.object(L.urllib.request, "urlopen", return_value=Reply(PAGED)):
            self.assertEqual(L.verify_no_competing_pod("current", "fake"), "API_VERIFIED")

    def test_named_competitor_fails(self):
        records = {"pods": [POD, {"id": "other", "name": "g1-study", "status": "RUNNING"}],
                   "pagination": {"hasNextPage": False}}
        with patch.object(L.urllib.request, "urlopen", return_value=Reply(records)):
            with self.assertRaisesRegex(RuntimeError, "Another project pod"):
                L.verify_no_competing_pod("current", "fake")

    def test_competitor_in_next_page_fails(self):
        pages = iter([Reply({"pods": [POD], "pagination":
                            {"hasNextPage": True, "nextCursor": "abc"}}),
                      Reply({"pods": [{"id": "other", "name": "oaa-run", "status": "RUNNING"}],
                             "pagination": {"hasNextPage": False}})])
        with patch.object(L.urllib.request, "urlopen", side_effect=lambda *args, **kwargs: next(pages)):
            with self.assertRaisesRegex(RuntimeError, "Another project pod"):
                L.verify_no_competing_pod("current", "fake")

    def test_scope_denial_requires_explicit_confirmation(self):
        denied = HTTPError("https://api.runpod.io/v2/pods", 403, "Forbidden", None, None)
        with patch.object(L.urllib.request, "urlopen", side_effect=denied):
            with patch.object(L.sys, "stdin", type("Terminal", (), {"isatty": lambda self: True})()):
                with patch("builtins.input", return_value="ONLY ONE G1 POD"):
                    self.assertEqual(L.verify_no_competing_pod("current", "fake"),
                                     "USER_CONFIRMED_API_UNAVAILABLE")

    def test_scope_denial_without_tty_fails(self):
        denied = HTTPError("https://api.runpod.io/v2/pods", 403, "Forbidden", None, None)
        with patch.object(L.urllib.request, "urlopen", side_effect=denied):
            with patch.object(L.sys, "stdin", type("Terminal", (), {"isatty": lambda self: False})()):
                with self.assertRaisesRegex(RuntimeError, "interactive explicit"):
                    L.verify_no_competing_pod("current", "fake")

    def test_scope_denial_without_confirmation_fails(self):
        denied = HTTPError("https://api.runpod.io/v2/pods", 403, "Forbidden", None, None)
        with patch.object(L.urllib.request, "urlopen", side_effect=denied):
            with patch.object(L.sys, "stdin", type("Terminal", (), {"isatty": lambda self: True})()):
                with patch("builtins.input", return_value=""):
                    with self.assertRaisesRegex(RuntimeError, "declined"):
                        L.verify_no_competing_pod("current", "fake")

    def test_other_http_error_does_not_bypass(self):
        denied = HTTPError("https://api.runpod.io/v2/pods", 500, "error", None, None)
        with patch.object(L.urllib.request, "urlopen", side_effect=denied):
            with self.assertRaisesRegex(RuntimeError, "HTTP 500"):
                L.verify_no_competing_pod("current", "fake")

    def test_empty_list_not_verified(self):
        with patch.object(L.urllib.request, "urlopen", return_value=Reply(
            {"pods": [], "pagination": {"hasNextPage": False}})):
            with self.assertRaisesRegex(RuntimeError, "absent"):
                L.verify_no_competing_pod("current", "fake")

    def test_bad_response_fail_closed(self):
        with patch.object(L.urllib.request, "urlopen", return_value=Reply({"unexpected": []})):
            with self.assertRaisesRegex(RuntimeError, "Unexpected"):
                L.verify_no_competing_pod("current", "fake")


if __name__ == "__main__":
    unittest.main()
