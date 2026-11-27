"""CPU-only preregistration / failure checks for bounded G1 GPU smoke matrix."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[1] / "scripts" / "run_g1_pod_one_command.py"
spec = importlib.util.spec_from_file_location("g1_smoke", MODULE)
L = importlib.util.module_from_spec(spec)
spec.loader.exec_module(L)


class G1SmokeContract(unittest.TestCase):
    def test_smoke_is_twelve_rows_and_not_the_obstacle_study(self):
        p = L.BENCH_PROFILES["smoke"]
        self.assertEqual(p["expected_rows"], 12)
        self.assertEqual(len(p["families"]) * len(p["methods"]) * p["episodes"], 12)
        self.assertGreaterEqual(p["steps"], 150)
        self.assertLess(p["expected_rows"], L.BENCH_PROFILES["dev"]["expected_rows"])
        self.assertNotIn("candidate_velocity", p["methods"])

    def test_provider_lifetime_is_short_and_reserves_publication_time(self):
        self.assertLessEqual(L.LIFETIME_LIMIT_S, 3600)
        self.assertGreaterEqual(L.RESERVE_S, 15 * 60)
        self.assertLess(L.RESERVE_S, L.LIFETIME_LIMIT_S)

    def test_dev_retains_original_128_rows(self):
        p = L.BENCH_PROFILES["dev"]
        self.assertEqual(p["expected_rows"], 128)
        self.assertEqual(p["episodes"], 8)

    def write_fixture(self, folder, *, n=12, duplicate=False):
        p = L.BENCH_PROFILES["smoke"]
        rows, cells = [], {}
        for family in p["families"]:
            for method in p["methods"]:
                key = family + "|declared|" + method
                cells[key] = {"mean_postfault_loss": 0.125}
                for i in range(p["episodes"]):
                    rows.append({"scenario_id": family + "_seed_" + str(i),
                                 "family": family, "method": method, "telemetry": "declared",
                                 "split": "dev", "postfault_loss": 0.125})
        if duplicate:
            rows[1]["scenario_id"] = rows[0]["scenario_id"]
            rows[1]["method"] = rows[0]["method"]
        path = Path(folder)
        with (path / "episodes.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows[:n])
        (path / "summary.json").write_text(json.dumps({"cells": cells}))

    def test_correct_12_rows_pass_paired_audit(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_fixture(d)
            result = L.verify_bench_rows(Path(d), "smoke")
            self.assertEqual(result["episode_rows"], 12)
            self.assertEqual(result["cells"], 6)

    def test_missing_cell_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_fixture(d, n=10)
            with self.assertRaisesRegex(RuntimeError, "expected 12 rows"):
                L.verify_bench_rows(Path(d), "smoke")

    def test_duplicate_scenario_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_fixture(d, duplicate=True)
            with self.assertRaisesRegex(RuntimeError, "Duplicate benchmark"):
                L.verify_bench_rows(Path(d), "smoke")

    def test_missing_summary_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.write_fixture(d)
            (Path(d) / "summary.json").unlink()
            with self.assertRaises(FileNotFoundError):
                L.verify_bench_rows(Path(d), "smoke")


if __name__ == "__main__":
    unittest.main()
