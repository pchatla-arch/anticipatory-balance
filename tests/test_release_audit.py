"""Planted trace failures for the small deadline-release diagnostic; no simulator."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from foresight.eval import frontier_output as O
from foresight.eval import release_audit as A


class ReleaseAuditTests(unittest.TestCase):
  def setUp(self):
    self.tmp = tempfile.TemporaryDirectory()
    self.addCleanup(self.tmp.cleanup)
    self.root = Path(self.tmp.name)
    self.steps = 7
    self.conditions = [dict(method="reactive", param=0, latency_steps=0, release_mode="legacy"),
                       dict(method="belief_commit", param=.1, latency_steps=0, release_mode="deadline")]
    O.atomic_json(self.root / "config.json", dict(status="complete", steps=7, dt=.02,
                   batches=1, episodes=2, conditions=self.conditions))
    self.rows = []
    self.last_summary = None
    for i, condition in enumerate(self.conditions):
      table = dict(mass=np.array([4., 8.]), speed=np.array([5., 6.]), is_miss=np.array([False, True]),
                   term_step=np.array([-1, 2]), contact_step=np.array([2, 4]),
                   release_mode=np.full(2, condition["release_mode"]))
      for axis in "xyz":
        table[f"launch_pos_{axis}"] = np.array([1., 2.])
        table[f"launch_vel_{axis}"] = np.array([3., 4.])
      traj = self.trajectory(deadline=4, brace=i == 1)
      ep = f"episodes/{i}.csv"
      tr = f"traces/{i}.csv"
      common = dict(seed=200, batch=0, method=condition["method"], param=condition["param"],
                    latency_steps=0, period=1)
      O.write_episodes(self.root / ep, table, **common)
      self.last_summary = A.write_trace(self.root / tr, traj, table,
                                        release_mode=condition["release_mode"], dt=.02, **common)
      self.rows.append(dict(condition, batch=0, seed=200, period=1, episode_file=ep, trace_file=tr))
    O.atomic_json(self.root / "batches.json", dict(completed_condition_batches=2, rows=self.rows))

  def trajectory(self, deadline, brace=True):
    step = np.arange(self.steps)[:, None]
    started = np.broadcast_to((step >= 1) & brace, (self.steps, 2))
    active = started & (step < deadline)
    start = np.broadcast_to((step == 1) & brace, active.shape)
    release = np.broadcast_to((step == deadline) & brace, active.shape)
    return dict(brace_active=active, ctrl_release_deadline_step=np.where(started, deadline, -1),
                ctrl_has_committed=started, ctrl_brace_start=start, ctrl_brace_release=release,
                ctrl_release_reason=release.astype(int), ctrl_t_hit=np.full(active.shape, .12),
                ctrl_pe_t_hit=np.full(active.shape, .1), ctrl_det_valid=np.ones(active.shape, bool),
                ctrl_position=np.zeros((*active.shape, 3)))

  def read(self, name="traces/1.csv"):
    with (self.root / name).open(newline="") as stream:
      return list(csv.DictReader(stream))

  def mutate(self, fn, name="traces/1.csv"):
    rows = self.read(name)
    fn(rows)
    O.atomic_rows(self.root / name, rows)

  def test_valid_run_and_censored_completion(self):
    report = A.validate_release_run(self.root)
    self.assertEqual(report["status"], "passed")
    deadline = report["conditions"][1]
    self.assertEqual(deadline["n_brace_onsets"], 2)
    self.assertEqual(deadline["n_deadlines_observed"], 1)
    self.assertEqual(deadline["n_deadline_censored_termination"], 1)
    self.assertEqual(deadline["n_contact_observed"], 1)
    self.assertEqual(deadline["n_contact_outside_observed_trace"], 1)
    self.assertEqual(deadline["braced_at_contact_fraction"], 1.)

  def test_trace_masks_posttermination_and_keeps_scalar_fields(self):
    rows = self.read()
    second = [r for r in rows if r["episode_index"] == "1"]
    self.assertEqual([int(r["step"]) for r in second], [0, 1, 2])
    self.assertEqual(self.last_summary["n_trace_rows"], 10)
    self.assertIn("ctrl_t_hit", rows[0])
    self.assertIn("ctrl_pe_t_hit", rows[0])
    self.assertIn("ctrl_det_valid", rows[0])
    self.assertNotIn("ctrl_position", rows[0])
    self.assertEqual(second[0]["scenario_id"], "seed200_episode000001")

  def test_early_release_is_rejected(self):
    def early(rows):
      row = next(r for r in rows if r["episode_index"] == "0" and r["step"] == "3")
      row["brace_active"] = "False"
    self.mutate(early)
    with self.assertRaisesRegex(RuntimeError, "deadline contract broken"):
      A.validate_release_run(self.root)

  def test_deadline_drift_is_rejected(self):
    self.mutate(lambda rows: rows[2].update(ctrl_release_deadline_step="5"))
    with self.assertRaisesRegex(RuntimeError, "deadline drift"):
      A.validate_release_run(self.root)

  def test_rearm_is_rejected(self):
    self.mutate(lambda rows: rows[6].update(brace_active="True"))
    with self.assertRaisesRegex(RuntimeError, "rearmed"):
      A.validate_release_run(self.root)

  def test_posttermination_rows_are_rejected(self):
    def append(rows):
      extra = dict(rows[-1], step="3", time_s="0.06")
      rows.append(extra)
    self.mutate(append)
    with self.assertRaisesRegex(RuntimeError, "post-termination"):
      A.validate_release_run(self.root)

  def test_pairing_mismatch_is_rejected(self):
    self.mutate(lambda rows: rows[0].update(speed="5.00001"), "episodes/1.csv")
    with self.assertRaisesRegex(RuntimeError, "pairing mismatch.*speed"):
      A.validate_release_run(self.root)

  def test_missing_step_is_not_treated_as_censoring(self):
    self.mutate(lambda rows: rows.pop(5))
    with self.assertRaisesRegex(RuntimeError, "trace missing steps"):
      A.validate_release_run(self.root)

  def test_no_deadline_onset_is_rejected(self):
    def no_start(rows):
      for row in rows:
        row.update(brace_active="False", ctrl_release_deadline_step="-1", ctrl_has_committed="False",
                   ctrl_brace_start="False", ctrl_brace_release="False", ctrl_release_reason="0")
    self.mutate(no_start)
    with self.assertRaisesRegex(RuntimeError, "No deadline condition ever braces"):
      A.validate_release_run(self.root)

  def test_horizon_censoring_does_not_require_unobserved_release(self):
    def later(rows):
      for row in rows:
        started = int(row["step"]) >= 1
        row.update(brace_active=str(started), ctrl_release_deadline_step="10" if started else "-1",
                   ctrl_brace_release="False", ctrl_release_reason="0")
    self.mutate(later)
    deadline = A.validate_release_run(self.root)["conditions"][1]
    self.assertEqual(deadline["n_deadline_censored_horizon"], 1)
    self.assertEqual(deadline["n_deadline_censored_termination"], 1)


if __name__ == "__main__":
  unittest.main()
