import unittest
from foresight.adaptive.language_nav import (
    Landmark, parse_instruction, resolve_goal, validate_vlm_proposal,
    navigation_decision,
)


class LanguageNavigationTests(unittest.TestCase):
    def setUp(self):
        self.landmarks = [Landmark("station", 5.0, 2.0, 0.94, 12)]

    def test_language_goal(self):
        c = parse_instruction("Walk to the station")
        self.assertTrue(c.accepted)
        self.assertEqual(c.target, "station")

    def test_goal_grounded(self):
        x, status = resolve_goal(parse_instruction("go to station"), self.landmarks, decision_step=16)
        self.assertEqual(x, (5.0, 2.0))
        self.assertEqual(status, "grounded")

    def test_unsupported_instruction_stops(self):
        c = parse_instruction("jump over the obstacle")
        self.assertFalse(c.accepted)
        self.assertEqual(c.operation, "stop")

    def test_stop(self):
        self.assertEqual(parse_instruction("Stop").reason, "user_stop")

    def test_unknown_goal(self):
        self.assertIsNone(navigation_decision("navigate to kitchen", self.landmarks, 16)["goal_xy"])

    def test_stale_goal(self):
        goal, reason = resolve_goal(parse_instruction("go to station"), self.landmarks, decision_step=30)
        self.assertIsNone(goal)
        self.assertEqual(reason, "stale_or_future_landmark")

    def test_future_packet(self):
        goal, _ = resolve_goal(parse_instruction("go to station"), self.landmarks, decision_step=10)
        self.assertIsNone(goal)

    def test_ambiguous_goal(self):
        data = self.landmarks + [Landmark("station", 5.1, 2.0, .99, 12)]
        goal, _ = resolve_goal(parse_instruction("go to station"), data, decision_step=16)
        self.assertIsNone(goal)

    def test_low_confidence_goal(self):
        goal, _ = resolve_goal(parse_instruction("go to station"), [Landmark("station", 5, 0, .1, 12)], decision_step=16)
        self.assertIsNone(goal)

    def test_reject_coordinates_and_motor_commands(self):
        c = validate_vlm_proposal({"operation":"navigate", "target_id":"station", "confidence":.95,"velocity":2.0},allowed_landmark_ids={"station"})
        self.assertFalse(c.accepted)

    def test_vlm_known_target(self):
        c = validate_vlm_proposal({"operation":"navigate", "target_id":"station", "confidence":.96},allowed_landmark_ids={"station"})
        self.assertTrue(c.accepted)
        self.assertEqual(c.target, "station")

    def test_vlm_unknown_target(self):
        c = validate_vlm_proposal({"operation":"navigate", "target_id":"exit", "confidence":.99},allowed_landmark_ids={"station"})
        self.assertFalse(c.accepted)

    def test_vlm_nan_confidence(self):
        c = validate_vlm_proposal({"operation":"navigate", "target_id":"station", "confidence":float('nan')},allowed_landmark_ids={"station"})
        self.assertFalse(c.accepted)

    def test_long_instruction(self):
        self.assertFalse(parse_instruction("x"*300).accepted)

    def test_reject_vlm_actuator_action(self):
        c = validate_vlm_proposal({"operation":"actuator_gain", "target_id":"station", "confidence":.99},allowed_landmark_ids={"station"})
        self.assertFalse(c.accepted)


if __name__ == "__main__":
    unittest.main()
