"""CPU tests for atomic benchmark CSV checkpoints; no simulator import."""
import csv
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_adaptive_bench.py"
spec = importlib.util.spec_from_file_location("g1_atomic_bench", SCRIPT)
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class AtomicCsvTests(unittest.TestCase):
    def test_complete_rows_are_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.csv"
            B.write_rows(path, [{"scenario_id": "0", "loss": 1.0},
                                {"scenario_id": "1", "loss": 0.7}])
            with path.open(newline="") as f:
                data = list(csv.DictReader(f))
            self.assertEqual(len(data), 2)
            self.assertEqual(data[1]["loss"], "0.7")
            self.assertEqual(list(Path(folder).glob("*.tmp")), [])

    def test_failed_writer_preserves_previous_complete_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.csv"
            B.write_rows(path, [{"scenario_id": "0", "loss": 1.0}])
            original = path.read_bytes()
            with patch.object(B.csv.DictWriter, "writerows",
                              side_effect=RuntimeError("simulated disk writer fault")):
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    B.write_rows(path, [{"scenario_id": "0", "loss": 0.2}])
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(folder).glob(".*.tmp")), [])

    def test_empty_rows_do_not_destroy_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "episodes.csv"
            B.write_rows(path, [{"scenario_id": "0", "loss": 1.0}])
            before = path.read_bytes()
            B.write_rows(path, [])
            self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
