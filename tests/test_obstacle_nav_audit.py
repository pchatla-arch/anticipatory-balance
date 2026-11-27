"""CPU tests for G2 raw trajectory audit integrity (no GPU/video dependencies)."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.audit_g1_depth_navigation import audit


class G2NavigationAuditTests(unittest.TestCase):
    def make_fixture(self, root):
        p=Path(root)
        rows=[
            {"step":0,"x":0.0,"y":0.0,"yaw":0.0,"detected":True,
             "collisions":0,"navigator_reason":"approach_slow"},
            {"step":5,"x":0.05,"y":0.0,"yaw":0.1,"detected":False,
             "collisions":0,"navigator_reason":"detour"},
        ]
        status={"source_commit":"fixture","result":"COMPLETED_DIAGNOSTIC_NOT_VALIDATED_AVOIDANCE",
                "goal_reached":False,"frames_checked":2,"detected_frames":1,
                "collision_frames":0,"physical_contact_ticks":0,
                "final_xy":[0.05,0.0]}
        config={"goal_x":5.0,"capture_every":5}
        (p/"frames.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        (p/"run_status.json").write_text(json.dumps(status))
        (p/"config.json").write_text(json.dumps(config))
        return rows,status

    def test_valid_failure_result_remains_failure(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_fixture(d)
            result=audit(Path(d))
            self.assertFalse(result["goal_reached"])
            self.assertEqual(result["frame_count"],2)
            self.assertEqual(result["navigator_reasons"]["detour"],1)
            self.assertFalse(result["video"]["present"])

    def test_misreported_detection_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _, status=self.make_fixture(d)
            status["detected_frames"]=2
            (Path(d)/"run_status.json").write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,"detector count"):
                audit(Path(d))

    def test_missing_frame_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows,_=self.make_fixture(d)
            rows[-1]["step"]=15
            (Path(d)/"frames.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
            with self.assertRaisesRegex(ValueError,"cadence"):
                audit(Path(d))

    def test_false_success_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            _,status=self.make_fixture(d)
            status["goal_reached"]=True
            status["result"]="GOAL_REACHED_NO_RECORDED_CYLINDER_CONTACTS"
            (Path(d)/"run_status.json").write_text(json.dumps(status))
            with self.assertRaisesRegex(ValueError,"Goal-success"):
                audit(Path(d))

    def test_missing_raw_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            self.make_fixture(d)
            (Path(d)/"frames.jsonl").unlink()
            with self.assertRaisesRegex(ValueError,"No frame-level"):
                audit(Path(d))


if __name__=="__main__":
    unittest.main()
