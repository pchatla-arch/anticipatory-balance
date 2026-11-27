"""CPU contract tests for independent paired-G3 evidence auditor."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_g3_paired_navigation import audit, arm


def fixture(root, method, seed=1, value=1):
    root.mkdir()
    config={"adapter":method,"steps":1200,"seed":seed,"depth_res":96,
            "capture_every":5,"goal_x":5.0,"obstacle_x":2.8,
            "fault_gain":0.7,"fault_onset":200,"fault_return":900,
            "render_video":False}
    status={"kind":"G3_DEPTH_ACTUATOR_DEVELOPMENT_PAIRED",
            "adapter_method":method,"result":"G3_COMPLETED_DEVELOPMENT_NOT_GOAL_REACHED",
            "source_commit":"fixed-source","goal_reached":False,"confirmed_obstacle":True,
            "final_xy":[1.0,value],"frames_checked":2,"detected_frames":1,
            "collision_frames":0,"physical_contact_ticks":0,
            "ended_at_step":9,"adapter_nonzero_control_ticks":0,
            "causal_telemetry_evidence":{"released_packets":10}}
    rows=[{"step":t,"x":float(t)/10,"y":0,"yaw":0,"detected":t==0,
           "collisions":0} for t in (0,5)]
    (root/"config.json").write_text(json.dumps(config))
    (root/"run_status.json").write_text(json.dumps(status))
    (root/"frames.jsonl").write_text("".join(json.dumps(x)+"\n" for x in rows))


class G3PairAuditTests(unittest.TestCase):
    def test_two_matched_failure_outcomes_are_valid_data(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a=root/"none";b=root/"gate"
            fixture(a,"none");fixture(b,"heuristic_gate")
            data=audit(a,b)
            self.assertEqual(data["seed"],1)
            self.assertFalse(data["none"]["goal_reached"])
            self.assertEqual(data["heuristic_gate"]["frames"],2)

    def test_mismatched_seed_cannot_be_compared(self):
        with tempfile.TemporaryDirectory() as d:
            a=Path(d)/"none";b=Path(d)/"gate"
            fixture(a,"none");fixture(b,"heuristic_gate",seed=9)
            with self.assertRaisesRegex(ValueError,"Paired configuration"):
                audit(a,b)

    def test_no_causal_packets_rejects_heuristic_result(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"gate";fixture(p,"heuristic_gate")
            st=json.loads((p/"run_status.json").read_text())
            st["causal_telemetry_evidence"]={"released_packets":0}
            (p/"run_status.json").write_text(json.dumps(st))
            with self.assertRaisesRegex(ValueError,"No causal telemetry"):
                arm(p,"heuristic_gate")

    def test_unconfirmed_obstacle_stopped_run_is_valid_negative_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"none";fixture(p,"none")
            st=json.loads((p/"run_status.json").read_text())
            st["confirmed_obstacle"]=False
            (p/"run_status.json").write_text(json.dumps(st))
            result=arm(p,"none")
            self.assertFalse(result["goal_reached"])
            self.assertFalse(result["confirmed_obstacle"])

    def test_false_goal_claim_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"none";fixture(p,"none")
            st=json.loads((p/"run_status.json").read_text())
            st.update(goal_reached=True,result="G3_GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS")
            (p/"run_status.json").write_text(json.dumps(st))
            with self.assertRaisesRegex(ValueError,"False goal success"):
                arm(p,"none")


if __name__=="__main__":
    unittest.main()
