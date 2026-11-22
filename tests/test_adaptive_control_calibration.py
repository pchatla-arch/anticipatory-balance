"""The prospective calibration scores one batch in its own physical scales."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import calibrate_gate_identity as calibration


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.arms = [
            {key: np.zeros(calibration.BATCH) for key in (*calibration.RETAIN, *calibration.CALIBRATION_METRICS.values())}
            for _ in range(2)
        ]
        for values in self.arms:
            values["first_nonfinite_step"][:] = -1
            values["term_step"][:] = -1

    def test_score_uses_all_lanes_and_distinct_physical_units(self):
        self.arms[1]["max_force"][23] = 2 * calibration.FROZEN["max_abs_actuator_force_nm"]
        self.arms[0]["postfault_loss"][2] = 3 * calibration.FROZEN["postfault_loss"]
        self.arms[1]["term_step"][6] = 225
        result = calibration.validate_records(self.arms, False)
        self.assertEqual(result["score"], 3.)
        self.assertEqual(result["terminal_disagreements"], 1)

    def test_nonfinite_or_active_controls_cannot_establish_envelope(self):
        for key, value in (("first_nonfinite_step", 10), ("progress", float("nan")), ("max_correction", 0.01)):
            with self.subTest(key=key):
                initial = self.arms[0][key][5]
                self.arms[0][key][5] = value
                with self.assertRaises(ValueError):
                    calibration.validate_records(self.arms, False)
                self.arms[0][key][5] = initial
