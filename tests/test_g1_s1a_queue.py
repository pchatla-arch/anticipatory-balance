"""Offline contract tests: queued S1a is NOT allowed to fall back to smoke/dev."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_g1_pod_one_command.py"
spec = importlib.util.spec_from_file_location("s1a_runner", PATH)
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

VARS = {"G1_BENCH_PROFILE": "s1_screen",
        "G1_QUEUE_JOB_ID": "G1-S1A-20261126",
        "G1_PROJECT_BILLING_REVIEWED": "YES",
        "G1_AUTHORIZED_RUN_BUDGET_USD": "1.05"}


class S1AQueueContract(unittest.TestCase):
    def test_frozen_profile_matches_registered_queue(self):
        with patch.dict(os.environ, VARS, clear=True):
            mode, budget = R.validate_queued_profile()
        self.assertEqual((mode, budget), ("s1_screen", 1.05))
        spec = R.BENCH_PROFILES[mode]
        self.assertEqual(spec["expected_rows"], 32)
        self.assertEqual(spec["steps"], 400)
        self.assertEqual(spec["episodes"], 2)
        self.assertEqual(len(spec["families"]) * len(spec["methods"]) * spec["episodes"], 32)
        self.assertIn("gain_070_return", spec["families"])
        self.assertIn("sensor_scale_080", spec["families"])

    def test_smoke_default_retained(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(R.validate_queued_profile(), ("smoke", None))

    def test_unregistered_dev_and_typo_rejected(self):
        for profile in ("dev", "test", "s1"):
            with self.subTest(profile=profile):
                with patch.dict(os.environ, {"G1_BENCH_PROFILE": profile}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "Only smoke or registered"):
                        R.validate_queued_profile()

    def test_queue_id_is_mandatory(self):
        for supplied in ("", "wrong"):
            env = dict(VARS, G1_QUEUE_JOB_ID=supplied)
            with patch.dict(os.environ, env, clear=True):
                with self.assertRaisesRegex(RuntimeError, "QUEUE_JOB_ID"):
                    R.validate_queued_profile()

    def test_account_budget_acknowledgement_mandatory(self):
        env = dict(VARS, G1_PROJECT_BILLING_REVIEWED="NO")
        with patch.dict(os.environ, env, clear=True):
            with self.assertRaisesRegex(RuntimeError, "billing review"):
                R.validate_queued_profile()

    def test_cost_limit_not_exceeded(self):
        for requested in ("1.06", "2.30", "", "not-a-number", "-0.5"):
            with self.subTest(requested=requested):
                env = dict(VARS, G1_AUTHORIZED_RUN_BUDGET_USD=requested)
                with patch.dict(os.environ, env, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "limit"):
                        R.validate_queued_profile()

    def test_modified_registration_denied(self):
        queued = json.loads(R.S1A_QUEUE_MANIFEST.read_text())
        queued["experiment"]["steps_per_episode"] = 160
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "queue.json"
            path.write_text(json.dumps(queued))
            with patch.object(R, "S1A_QUEUE_MANIFEST", path):
                with patch.dict(os.environ, VARS, clear=True):
                    with self.assertRaisesRegex(RuntimeError, "registration differs"):
                        R.validate_queued_profile()

    def test_32_episode_auditor_accepts_paired_complete_s1a(self):
        import csv
        spec = R.BENCH_PROFILES["s1_screen"]
        with tempfile.TemporaryDirectory() as folder:
            p = Path(folder)
            rows = []
            cells = {}
            for family in spec["families"]:
                for method in spec["methods"]:
                    cells[family + "|declared|" + method] = {"mean_postfault_loss": 0.25}
                    for seed in range(spec["episodes"]):
                        rows.append({"split": "dev", "telemetry": "declared",
                                     "family": family, "method": method,
                                     "scenario_id": family + "-seed-" + str(seed),
                                     "postfault_loss": "0.25"})
            with (p / "episodes.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            (p / "summary.json").write_text(json.dumps({"cells": cells}))
            audit = R.verify_bench_rows(p, profile="s1_screen")
            self.assertEqual(audit["episode_rows"], 32)
            self.assertEqual(audit["cells"], 16)
            rows[0]["postfault_loss"] = "0.75"
            with (p / "episodes.csv").open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(RuntimeError, "disagrees with summary"):
                R.verify_bench_rows(p, profile="s1_screen")

    def test_lifetime_and_publication_reserve(self):
        self.assertEqual(R.PROFILE_LIFETIME_S["s1_screen"], 90 * 60)
        self.assertEqual(R.PROFILE_LIFETIME_S["smoke"], 60 * 60)
        self.assertGreaterEqual(R.RESERVE_S, 15 * 60)
        self.assertLessEqual(R.S1A_MAX_NEW_COMPUTE_USD, 1.05)
        self.assertGreater(R.PROFILE_LIFETIME_S["s1_screen"], R.RESERVE_S)


if __name__ == "__main__":
    unittest.main()
