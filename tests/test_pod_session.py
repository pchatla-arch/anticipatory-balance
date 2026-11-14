"""Mocked worker orchestration: no GPU, installations, network, or pod stop."""
from contextlib import ExitStack, redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
with mock.patch.object(sys, "path", [str(ROOT / "scripts"), *sys.path]):
    SPEC = importlib.util.spec_from_file_location("pod_session_worker", ROOT / "scripts/pod_session.py")
    worker = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(worker)
SPEC = importlib.util.spec_from_file_location("pod_session_finalizer", ROOT / "scripts/finish_pod_session.py")
finalizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(finalizer)


class PodSessionTests(unittest.TestCase):
    SOURCE = "1" * 40
    PUBLICATION = "2" * 40
    INVOCATION = "f" * 32

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="foresight-worker-test-")
        self.addCleanup(self.temporary.cleanup)
        self.previous_cwd = Path.cwd()
        self.addCleanup(os.chdir, self.previous_cwd)
        self.root = Path(self.temporary.name) / "code"
        self.root.mkdir()
        self.setup_root = Path(self.temporary.name) / "setup"
        self.setup_root.mkdir()
        self.venv = self.setup_root / "mujoco_playground/rss2025/.venv"
        self.logs = self.root / "logs"
        self.logs.mkdir()

    def python_stub(self, venv=None):
        python = (venv or self.venv) / "bin/python"
        python.parent.mkdir(parents=True, exist_ok=True)
        python.write_text("# Never executed by these tests.\n")
        return python

    def marker(self):
        path = self.setup_root / ".foresight-bootstrap/state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"status":"in_progress"}\n')

    def receipt(self, run_id, code=0):
        return {"schema_version": 1, "status": "verified", "verified": True,
                "run_id": run_id, "run_source": self.SOURCE,
                "run_exit_code": code, "commit": self.PUBLICATION}

    def completed_run(self, invocation, run_id="f0_fixture"):
        output = self.root / "results" / run_id
        output.mkdir(parents=True)
        receipt = self.receipt(run_id)
        (output / "run.json").write_text(json.dumps({
            "run_id": run_id, "source_commit": self.SOURCE,
            "status": "completed", "exit_code": 0,
        }))
        (output / "publication_receipt.json").write_text(json.dumps(receipt))
        (self.logs / "last_run.json").write_text(json.dumps({
            "schema_version": 1, "invocation_id": invocation, "status": "completed",
            "run_id": run_id, "run_dir": str(output), "source_commit": self.SOURCE,
            "experiment_status": "completed", "experiment_exit_code": 0,
            "publication_status": "verified", "publication_exit_code": 0,
            "publication_receipt": receipt,
        }))

    def main_context(self, child, process, calls):
        stack = ExitStack()
        stack.enter_context(mock.patch.object(worker, "ROOT", self.root))
        stack.enter_context(mock.patch.dict(os.environ, {
            "FORESIGHT_SETUP_ROOT": str(self.setup_root), "FORESIGHT_VENV": str(self.venv),
            "FORESIGHT_INVOCATION_ID": "old-inherited-invocation",
        }))
        stack.enter_context(mock.patch.object(worker.uuid, "uuid4", return_value=SimpleNamespace(hex=self.INVOCATION)))
        stack.enter_context(mock.patch.object(worker.signal, "signal"))
        stack.enter_context(mock.patch.object(worker, "run_child", side_effect=child))
        stack.enter_context(mock.patch.object(worker.subprocess, "Popen", side_effect=process))

        def command_run(command, **kwargs):
            calls.append((list(command), kwargs))
            if command == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, stdout=self.SOURCE + "\n")
            if command[1:] == ["scripts/finish_pod_session.py"]:
                invocation = kwargs["env"]["FORESIGHT_INVOCATION_ID"]
                try:
                    finalizer.verify_current_upload(self.root, invocation)
                except finalizer.StopBlocked:
                    return subprocess.CompletedProcess(command, 2)
                return subprocess.CompletedProcess(command, 0)
            raise AssertionError(f"Unexpected command: {command}")

        stack.enter_context(mock.patch.object(worker.subprocess, "run", side_effect=command_run))
        stack.enter_context(redirect_stdout(io.StringIO()))
        stack.enter_context(redirect_stderr(io.StringIO()))
        return stack

    def test_unowned_warm_default_venv_is_preflighted_without_bootstrap(self):
        python = self.python_stub()
        command, timeout, _ = worker.setup_command(self.setup_root, self.venv, self.logs)
        self.assertEqual(command[:2], [str(python), "scripts/preflight_experiments_v2.py"])
        self.assertEqual(timeout, 300)
        self.assertNotIn("scripts/bootstrap_fresh_pod.sh", command)

    def test_owned_default_setup_uses_resumable_bootstrap_even_when_python_exists(self):
        self.python_stub()
        self.marker()
        command, timeout, _ = worker.setup_command(self.setup_root, self.venv, self.logs)
        self.assertEqual(command, ["bash", "scripts/bootstrap_fresh_pod.sh"])
        self.assertEqual(timeout, 1500)

    def test_existing_custom_venv_is_reused_despite_default_setup_marker(self):
        self.marker()
        custom = self.setup_root / "custom-venv"
        python = self.python_stub(custom)
        command, _, _ = worker.setup_command(self.setup_root, custom, self.logs)
        self.assertEqual(command[:2], [str(python), "scripts/preflight_experiments_v2.py"])

    def test_missing_explicit_custom_venv_does_not_silently_install_elsewhere(self):
        with self.assertRaisesRegex(RuntimeError, "Custom venv is missing"):
            worker.setup_command(self.setup_root, self.setup_root / "missing-custom", self.logs)
        self.assertFalse(self.venv.exists())

    def test_setup_timeout_saves_correct_source_and_publishes_current_failure(self):
        children = []
        calls = []

        def child(command, env, log, timeout, **kwargs):
            children.append((command, env, timeout, kwargs))
            if command == ["bash", "scripts/bootstrap_fresh_pod.sh"]:
                raise TimeoutError("Synthetic setup timeout")
            self.assertEqual(command[1], "scripts/publish_experiment_results.py")
            output = Path(command[command.index("--run-dir") + 1])
            run = json.loads((output / "run.json").read_text())
            self.assertEqual(run["source_commit"], self.SOURCE)
            self.assertEqual(run["status"], "timed_out")
            self.assertEqual(run["exit_code"], 124)
            self.assertEqual(run["mode"], "setup")
            self.assertIn("Synthetic setup timeout", (output / "console.log").read_text())
            self.assertEqual(env["FORESIGHT_INVOCATION_ID"], self.INVOCATION)
            self.assertIs(kwargs["check"], False)
            (output / "publication_receipt.json").write_text(json.dumps(self.receipt(run["run_id"], 124)))
            return 0

        with self.main_context(child, lambda *args, **kwargs: self.fail("Experiment must not start"), calls):
            code = worker.main([])
        self.assertEqual(code, 124)
        self.assertEqual(len(children), 2)
        latest = json.loads((self.logs / "last_run.json").read_text())
        self.assertEqual(latest["source_commit"], self.SOURCE)
        self.assertEqual(latest["invocation_id"], self.INVOCATION)
        self.assertEqual(latest["stage"], "setup")
        self.assertEqual(latest["experiment_status"], "timed_out")
        self.assertEqual(latest["publication_status"], "verified")
        self.assertEqual(latest["publication_exit_code"], 0)
        finalizers = [call for call in calls if call[0][1:] == ["scripts/finish_pod_session.py"]]
        self.assertEqual(len(finalizers), 1)
        self.assertEqual(finalizers[0][1]["env"]["FORESIGHT_INVOCATION_ID"], self.INVOCATION)

    def test_warm_run_passes_new_invocation_and_environment_to_runner_and_finalizer(self):
        self.python_stub()
        children = []
        processes = []
        calls = []

        def child(command, env, log, timeout, **kwargs):
            children.append((command, timeout))
            self.assertEqual(command[1], "scripts/preflight_experiments_v2.py")
            return 0

        def process(command, **kwargs):
            processes.append((command, kwargs))
            invocation = kwargs["env"]["FORESIGHT_INVOCATION_ID"]
            self.completed_run(invocation)
            result = mock.Mock()
            result.wait.return_value = 0
            return result

        with self.main_context(child, process, calls):
            code = worker.main(["--mode", "smoke", "--max-minutes", "4"])
        self.assertEqual(code, 0)
        self.assertEqual(len(children), 1)
        self.assertEqual(len(processes), 1)
        command, kwargs = processes[0]
        self.assertEqual(command[1:], ["scripts/run_experiments_v2.py", "--mode", "smoke", "--max-minutes", "4.0"])
        self.assertEqual(kwargs["env"]["FORESIGHT_INVOCATION_ID"], self.INVOCATION)
        self.assertEqual(kwargs["env"]["FORESIGHT_PYTHON"], str(self.venv / "bin/python"))
        self.assertEqual(calls[-1][1]["env"]["FORESIGHT_INVOCATION_ID"], self.INVOCATION)

    def test_stale_last_run_cannot_authorize_finalization_of_new_invocation(self):
        self.python_stub()
        self.completed_run("old-inherited-invocation", "old_run")
        calls = []
        process = mock.Mock()
        process.wait.return_value = 0  # Simulate runner exit before writing this invocation's last_run.
        with self.main_context(lambda *args, **kwargs: 0, lambda *args, **kwargs: process, calls):
            code = worker.main([])
        self.assertEqual(code, 2)
        self.assertEqual(calls[-1][0][1:], ["scripts/finish_pod_session.py"])
        self.assertEqual(calls[-1][1]["env"]["FORESIGHT_INVOCATION_ID"], self.INVOCATION)
        self.assertNotEqual(self.INVOCATION, json.loads((self.logs / "last_run.json").read_text())["invocation_id"])


if __name__ == "__main__":
    unittest.main()
