"""New response fits retain noncontacts and failures before scheduled bracing."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import fit_review_response as fitter


class ReviewResponseTests(unittest.TestCase):
    def fixture(self, root):
        rng = np.random.default_rng(123)
        rows = []
        for seed in (7001,7002):
            for i in range(64):
                J = float(rng.uniform(24,100))
                for lead in (-1,.1,.5,.75):
                    p = float(fitter.sigmoid(-5+1.4*np.log(J)-(2*lead if lead>0 else 0)))
                    fell = int(rng.random()<p)
                    rows.append(dict(scenario_id=f'seed{seed}_env{i}', mode_brace=1, launch=1,
                                     reference_event_time_s=2, censored=0, lead=lead,
                                     momentum=J, fell=fell, contact=int(i%2==0),
                                     scheduled_switch_time_s=2-max(lead,0),
                                     actual_onset_time_s='' if fell else 2-max(lead,0)))
        target = root/'episodes.csv'
        with target.open('w',newline='') as stream:
            w = csv.DictWriter(stream,fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        summary = root/'summary.json'
        summary.write_text(json.dumps({'config':dict(timing_reference='reference_entry',release_mode='deadline',
                                                       dt=.02,hold_s=.5,split_label='development')}))
        return target, summary, rows

    def test_fit_all_scheduled_rows_and_grouped_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes,summary,rows = self.fixture(Path(tmp))
            model = fitter.fit_response(episodes,summary)
            self.assertEqual(model['n_selected'],len(rows))
            self.assertEqual(model['n_cont'],128)
            self.assertEqual(model['n_brace_lead'],384)
            self.assertFalse(model['contact_conditioned'])
            self.assertGreater(model['unexecuted_scheduled_braces_retained'],0)
            self.assertEqual(model['grouped_prediction_check']['brace']['status'],'complete')
            self.assertEqual(len(model['grouped_prediction_check']['brace']['folds']),2)
            self.assertLess(model['c_lead'],0)

    def test_reject_historical_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            episodes,summary,_ = self.fixture(Path(tmp))
            summary.write_text(json.dumps({'config':dict(timing_reference='nominal', release_mode='legacy')}))
            with self.assertRaisesRegex(ValueError,'reference_entry'):
                fitter.fit_response(episodes,summary)

    def test_insufficient_or_constant_outcomes_fail_loudly(self):
        for n in (8,32):
            with self.assertRaises(ValueError):
                fitter.fit(np.ones((n,1)),np.zeros(n))


if __name__ == '__main__':
    unittest.main()
