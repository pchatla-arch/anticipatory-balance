"""Offline G1 detached monitor tests; no GPU, GitHub token or RunPod access."""
import csv
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "g1_detached_run.py"
spec = importlib.util.spec_from_file_location("g1_detach", MODULE)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


class DetachedReliabilityTests(unittest.TestCase):
    def test_no_process_returns_none(self):
        self.assertIsNone(M.process_start_ticks(99999999))

    def test_own_process_has_start_ticks(self):
        self.assertIsInstance(M.process_start_ticks(os.getpid()), int)

    def test_empty_run_has_zero_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(M.finished_episode_rows(Path(temp)), 0)

    def test_unique_episode_keys_counted(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "bench_dev"
            folder.mkdir()
            with (folder / "episodes.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "method", "telemetry"])
                w.writeheader()
                w.writerow({"scenario_id": "scenario1", "family": "gain_070",
                            "method": "none", "telemetry": "declared"})
                w.writerow({"scenario_id": "scenario1", "family": "gain_070",
                            "method": "oracle", "telemetry": "declared"})
            self.assertEqual(M.finished_episode_rows(Path(temp)), 2)

    def test_duplicate_episode_keys_marked_unverified(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / "bench_dev"
            folder.mkdir()
            with (folder / "episodes.csv").open("w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["scenario_id", "family", "method", "telemetry"])
                w.writeheader()
                for _ in range(2):
                    w.writerow({"scenario_id": "scenario1", "family": "gain_070",
                                "method": "none", "telemetry": "declared"})
            self.assertEqual(M.finished_episode_rows(Path(temp)), 0)

    def test_latest_run_prefers_newest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            p1 = root / "20261126T100000Z_test"
            p2 = root / "20261126T110000Z_test"
            for p in (p1, p2):
                p.mkdir()
                (p / "run_status.json").write_text("{}")
            os.utime(p1 / "run_status.json", (100, 100))
            os.utime(p2 / "run_status.json", (200, 200))
            os.utime(p1, (100, 100))
            os.utime(p2, (200, 200))
            with patch.object(M, "SESSION_DIR", root):
                self.assertEqual(M.latest_run(), p2)

    def test_monitor_normal_provider_stop_no_rescue(self):
        with patch.object(M, "process_start_ticks", return_value=None):
            with patch.object(M, "latest_run", return_value=None):
                with patch.object(M.R, "runpod_request", return_value={"status": "EXITED"}):
                    with patch.object(M.R, "stop_pod") as stop:
                        with tempfile.TemporaryDirectory() as temp:
                            secrets = Path(temp)
                            (secrets / "runpod").write_text("dummy")
                            with patch.object(M, "SECURE_DIR", secrets):
                                M.monitor(123, 456, "pod-id", 9999999999)
                        stop.assert_not_called()

    def test_monitor_on_parent_exit_rescues_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "runpod").write_text("dummy")
            run = root / "20261126T100000Z_test"
            run.mkdir()
            with patch.object(M, "SECURE_DIR", root):
                with patch.object(M, "process_start_ticks", return_value=None):
                    with patch.object(M, "latest_run", return_value=run):
                        with patch.object(M, "finished_episode_rows", return_value=32):
                            with patch.object(M.R, "runpod_request", return_value={"status": "RUNNING"}):
                                with patch.object(M, "publish_snapshot", return_value="sha") as publish:
                                    with patch.object(M.R, "stop_pod", return_value=True) as stop:
                                        with patch.object(M.time, "sleep", return_value=None):
                                            M.monitor(123, 456, "pod-id", 9999999999)
                                    self.assertTrue(publish.called)
                                    stop.assert_called_once_with("pod-id", "dummy")

    def test_snapshot_missing_git_token_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "20261126T100000Z_test"
            run.mkdir()
            with patch.object(M, "SECURE_DIR", Path(temp) / "missing"):
                with self.assertRaisesRegex(RuntimeError, "credential"):
                    M.publish_snapshot(run, final=True)


if __name__ == "__main__":
    unittest.main()
