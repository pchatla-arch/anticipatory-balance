"""Check table reconstruction from the compact pilot and the uploader's nested layout."""
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


def load_summary():
    path = Path(__file__).resolve().parents[1] / "scripts/summarize_threshold_sweep.py"
    spec = importlib.util.spec_from_file_location("sweep_table", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.summarize


def write_json(path, obj):
    path.write_text(json.dumps(obj))


def make_rows(seed, method, param):
    return [dict(method=method, param=param, scenario_id=f"seed{seed}_env{i:05d}",
                 mass=6.0, speed=5.0, t_hit_nominal=3.0, is_miss=(i == 1), fell=(i == 0),
                 contact_before_termination=(i == 0), progress_m=[1.0, float("nan")][i],
                 brace_steps_total=[5, 0][i]) for i in range(2)]


def write_rows(path, rows):
    path.parent.mkdir(exist_ok=True, parents=True)
    with path.open("w", newline="") as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


class SweepTablesTest(unittest.TestCase):
    def test_published_multibatch_preserves_denominators_and_methods(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = dict(experiment="Foresight deterministic TTC matched to belief preparation, revision 1",
                            calibration=dict(seed=5, n=2, theta_grid_s=[0.1, 0.2]),
                            confirmation=dict(batch_seeds=[7, 8], episodes_per_seed=2,
                                              frozen_belief=["belief_lead", 0.05]))
            write_json(root / "protocol.json", protocol)
            write_json(root / "frozen.json", dict(theta_s=0.1, matched_on_calibration=True))
            write_json(root / "verdict.json", dict(status="risk_difference_unresolved"))
            for stage, seeds, grid in (("calibration", [5], [0.1, 0.2]), ("confirmation", [7, 8], [0.1])):
                for batch, seed in enumerate(seeds):
                    for c, (method, par) in enumerate([("ttc_threshold", t) for t in grid] + [("belief_lead", 0.05)]):
                        path = root / stage / "episodes" / f"batch{batch:03d}_condition{c:03d}.csv"
                        write_rows(path, make_rows(seed, method, par))
            write_json(root / "calibration/config.json", dict(dt=0.02))
            out = load_summary()(root)
            self.assertEqual(len(out), 15)
            record, = [r for r in out if (r["stage"], r["method"], r["stratum"])
                       == ("confirmation", "belief_lead", "all")]
            self.assertEqual((record["n"], record["contacts"], record["failures"],
                              record["failures_among_contacts"], record["finite_progress_n"]), (4, 2, 2, 2, 2))
            self.assertAlmostEqual(record["applied_brace_s"], 0.05)
            self.assertAlmostEqual(record["finite_mean_progress_m"], 1.0)
            path = root / "confirmation/episodes/batch001_condition000.csv"
            write_rows(path, make_rows(7, "ttc_threshold", 0.1))
            with self.assertRaisesRegex(ValueError, "Repeated scenario"):
                load_summary()(root)

    def test_pilot_early_stop_needs_no_confirmation_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protocol = dict(schema="foresight.deterministic-cpu-pilot.v1", control_horizon=[300, .02],
                            calibration_seed=77, calibration_n=2, ttc_threshold_grid_seconds=[0.1],
                            reference_rule=["belief_lead", 0.05])
            write_json(root / "protocol.json", protocol)
            write_json(root / "freeze.json", dict(theta_s=0.1, matched_on_calibration=False))
            write_json(root / "verdict.json", dict(status="no_calibration_match", confirmation_started=False))
            rows = make_rows(77, "belief_lead", 0.05) + make_rows(77, "ttc_threshold", 0.1)
            write_rows(root / "calibration_episodes.csv", rows)
            self.assertEqual(len(load_summary()(root)), 6)
            write_rows(root / "calibration_episodes.csv", rows[:-1])
            with self.assertRaisesRegex(ValueError, "missing scenario ID"):
                load_summary()(root)


if __name__ == "__main__":
    unittest.main()
