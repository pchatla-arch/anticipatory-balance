"""Bounded language-conditioned task selection for the G1 demonstration.

The language/model layer selects a goal from fresh *perceived* landmarks. It
cannot supply actuator corrections, robot commands, or arbitrary world points.
The current strict grammar is a reproducible, non-VLM baseline; an optional VLM
may supply only the same schema after separate perception-grounding tests.
"""
from dataclasses import dataclass
import math
import re


@dataclass(frozen=True)
class Landmark:
    label: str
    x: float
    y: float
    confidence: float
    observed_step: int


@dataclass(frozen=True)
class TaskChoice:
    operation: str  # navigate | stop
    target: str = ""
    accepted: bool = False
    reason: str = ""


_COMMAND = re.compile(
    r"^(?:go|walk|navigate|move)\s+to\s+(?:the\s+)?([a-z][a-z0-9 _-]{0,39})$",
    re.IGNORECASE,
)


def parse_instruction(instruction: str) -> TaskChoice:
    """Deterministic instruction baseline: deliberately no VLM claim."""
    if not isinstance(instruction, str) or len(instruction) > 128:
        return TaskChoice("stop", reason="invalid_instruction")
    clean = instruction.strip().rstrip(".!").strip()
    if clean.lower() in {"stop", "halt", "pause"}:
        return TaskChoice("stop", accepted=True, reason="user_stop")
    # Obstacles are detected by depth and avoided by the navigator; text does
    # not supply their position or motor commands.
    avoid = re.fullmatch(
        r"(?:walk|navigate)\s+around\s+the\s+obstacle\s+and\s+"
        r"(?:reach|go\s+to)\s+(?:the\s+)?([a-z][a-z0-9 _-]{0,39})",
        clean, re.IGNORECASE)
    match = avoid if avoid is not None else _COMMAND.fullmatch(clean)
    if match is None:
        return TaskChoice("stop", reason="unsupported_language")
    label = " ".join(match.group(1).lower().split())
    return TaskChoice("navigate", target=label, accepted=True, reason="grammar")


def validate_vlm_proposal(proposal, *, allowed_landmark_ids):
    """Safety firewall for a high-level external model's structured proposal.

    A model can choose only an ID already in the perceived-landmark registry.
    It cannot specify direct velocity, actuator actions or fabricated coordinates.
    """
    if not isinstance(proposal, dict):
        return TaskChoice("stop", reason="invalid_proposal")
    if set(proposal) != {"operation", "target_id", "confidence"}:
        return TaskChoice("stop", reason="unexpected_model_fields")
    if proposal["operation"] != "navigate":
        return TaskChoice("stop", reason="unsupported_model_action")
    target = proposal["target_id"]
    confidence = proposal["confidence"]
    if (not isinstance(target, str) or target not in allowed_landmark_ids
            or not isinstance(confidence, (float, int))
            or isinstance(confidence, bool) or not math.isfinite(confidence)
            or not 0.8 <= confidence <= 1):
        return TaskChoice("stop", reason="ungrounded_or_uncertain_model_goal")
    return TaskChoice("navigate", target=target, accepted=True, reason="vlm_proposal_validated")


def resolve_goal(choice: TaskChoice, landmarks, *, decision_step: int,
                 max_age_steps=10, min_landmark_confidence=0.85):
    """Return a grounded XY goal or None with a reason, never coordinates from text."""
    if not choice.accepted or choice.operation != "navigate":
        return None, choice.reason or "not_navigating"
    if not isinstance(decision_step, int) or decision_step < 0:
        return None, "invalid_decision_step"
    matching = [l for l in landmarks if l.label.lower() == choice.target]
    if len(matching) != 1:
        return None, "missing_or_ambiguous_landmark"
    l = matching[0]
    if (l.observed_step > decision_step or decision_step - l.observed_step > max_age_steps):
        return None, "stale_or_future_landmark"
    if (not math.isfinite(l.confidence) or l.confidence < min_landmark_confidence
            or not all(math.isfinite(v) for v in (l.x, l.y))):
        return None, "unreliable_landmark"
    return (l.x, l.y), "grounded"


def navigation_decision(instruction: str, landmarks, step: int):
    """Conservative high-level interface; caller handles geometry and motor safety."""
    choice = parse_instruction(instruction)
    goal, status = resolve_goal(choice, landmarks, decision_step=step)
    return {"goal_xy": goal, "status": status, "choice": choice}
