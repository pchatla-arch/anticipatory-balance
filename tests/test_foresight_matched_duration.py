"""Protocol regressions; no physics or GPU imports."""
import csv
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import run_foresight_matched_duration as exp


class TestMatchedDuration(unittest.TestCase):
    def rows(self, steps, falls=(), invalid_progress=()):
        return {str(i): exp.Episode((8., 4., 2.), i >= 4, i in falls, steps, i not in invalid_progress)
                for i in range(8)}

    def test_selection_uses_duration_only_and_tie_break(self):
        fitted = exp.choose_theta({exp.BELIEF: self.rows(21),
                                   (exp.TTC, .18): self.rows(20, falls=(0,1,2)),
                                   (exp.TTC, .20): self.rows(22)})
        self.assertEqual(fitted['theta_s'], .18)
        self.assertTrue(fitted['matched_on_calibration'])

    def test_duration_equivalence_and_progress_accounting(self):
        arms = {exp.BELIEF:self.rows(21, falls=(0,1,2,3)), (exp.TTC,.16):self.rows(21, invalid_progress=(2,))}
        score = exp.score(arms, .16)
        self.assertEqual(score['status'], 'ttc_lower_failure')
        self.assertEqual(score['nonfinite_or_missing_progress_by_arm']['ttc_threshold_0.16'], 1)
        self.assertEqual(score['hit_risk_difference']['n'], 4)
        self.assertEqual(score['near_miss_preparation_difference']['mean'], 0)
        self.assertEqual(score['near_miss_preparation_difference']['n'], 4)
        arms[(exp.TTC,.16)] = self.rows(25)
        self.assertFalse(exp.choose_theta(arms)['matched_on_calibration'])
        self.assertFalse(exp.score(arms, .16)['duration_equivalence'])

    def test_reordered_rows_join_and_corrupt_evidence_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'episodes'
            out.mkdir()
            seed = 77
            rows = []
            for arm in ((exp.TTC, .16), exp.BELIEF):
                for i in reversed(range(3)):
                    rows.append(dict(method=arm[0],param=arm[1],batch=0,scenario_index=i,
                                     scenario_seed=seed,scenario_id=f'seed77_env{i:05d}',latency_steps=5,
                                     fell='False',is_miss=str(i==2),term_step=-1,brace_steps_total=i,
                                     progress_m='nan' if i==1 else '1',mass=8,speed=4,t_hit_nominal=2))
            def save(values):
                with (out/'one.csv').open('w',newline='') as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0]))
                    w.writeheader()
                    w.writerows(values)
            save(rows)
            parsed = exp.read_arms(root,seed,3,1,[.16])
            self.assertTrue(parsed[exp.BELIEF]['seed77_env00002'].miss)
            for broken in (rows[:-1],rows+[rows[0]],[dict(rows[0],fell='NaN'),*rows[1:]],
                           [dict(rows[0],brace_steps_total=1444),*rows[1:]],
                           [dict(rows[0],mass=10),*rows[1:]]):
                save(broken)
                with self.assertRaises(ValueError):
                    exp.read_arms(root,seed,3,1,[.16])


if __name__ == '__main__':
    unittest.main()
