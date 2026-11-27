"""Task-goal-to-navigation integration without privileged actuator labels.

This small adapter accepts a validated language choice and a perceived goal
registry, then delegates safe motion generation to the obstacle navigator.
It does not know actuator gain, true obstacle position, or servo commands.
"""
from foresight.adaptive.language_nav import parse_instruction, resolve_goal, validate_vlm_proposal


class LanguageGuidedNavigator:
    def __init__(self, *, navigator_factory):
        self.navigator_factory = navigator_factory
        self.navigator = None
        self.goal = None
        self.choice = None
        self.task_events = []

    def set_language_goal(self, text, landmarks, step):
        return self._set_choice(parse_instruction(text), landmarks, step)

    def set_model_goal(self, proposal, landmarks, step):
        allowed = {l.label.lower() for l in landmarks}
        return self._set_choice(validate_vlm_proposal(proposal, allowed_landmark_ids=allowed),
                                landmarks, step)

    def _set_choice(self, choice, landmarks, step):
        goal, reason = resolve_goal(choice, landmarks, decision_step=step)
        self.choice = choice
        self.task_events.append((step, choice.reason, reason))
        if goal is None:
            self.navigator = None
            self.goal = None
            return reason
        if self.goal != goal:
            self.goal = goal
            self.navigator = self.navigator_factory(goal)
        return reason

    def update_depth(self, detection, observed_step):
        if self.navigator is not None:
            self.navigator.ingest(detection, observed_step=observed_step)

    def command(self, x, y, yaw, step):
        if self.navigator is None:
            return 0.0, 0.0, "stop_no_grounded_goal"
        return self.navigator.command(x, y, yaw, step)
