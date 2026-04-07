"""
logic/tasks.py — Task definitions, scenario catalogue, and task registry.

THREE TASK TIERS
----------------
easy   : Intent detection only — can the agent name the issue type?
medium : Resolution correctness — does the agent apply the right fix?
hard   : Full response quality — correct fix + professional tone + empathy

TWELVE SCENARIOS
----------------
4 per intent category (refund / delay / complaint), with two sentiment levels
(angry / neutral). Each has a follow-up query for multi-turn episodes.

DESIGN RULES
------------
• Task.evaluate() is deterministic and pure (same input → same output)
• All scenario fields validated at dataclass construction (no raw strings for intent/resolution)
• Task registry (TASKS) uses the config thresholds as single source of truth
• get_task() raises ValueError for unknown names
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from config.config import TASK_THRESHOLDS
from domain.models import Intent, Resolution
from logic.graders import (
    grade_empathy,
    grade_intent_detection,
    grade_politeness,
    grade_resolution,
)

__all__ = [
    "Scenario",
    "Task",
    "EasyTask",
    "MediumTask",
    "HardTask",
    "SCENARIOS",
    "SCENARIO_INDEX",
    "ALL_SCENARIO_IDS",
    "TASKS",
    "get_task",
]


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    """
    A single customer-support interaction scenario.

    Fields
    ------
    scenario_id       : Unique identifier, e.g. "ref_001"
    query             : Opening customer message
    sentiment         : "happy" | "neutral" | "angry"
    intent            : Intent enum value
    correct_resolution: Resolution enum value
    follow_up         : Optional follow-up query (shown after agent's first response)
    """
    scenario_id:        str
    query:              str
    sentiment:          str
    intent:             str   # Runtime value of Intent enum
    correct_resolution: str   # Runtime value of Resolution enum
    follow_up:          str = ""

    def __post_init__(self) -> None:
        """Validate intent and resolution values at construction time."""
        valid_intents    = {e.value for e in Intent} - {"unknown"}
        valid_resolutions = {e.value for e in Resolution} - {"none"}
        if self.intent not in valid_intents:
            raise ValueError(
                f"Scenario {self.scenario_id!r}: invalid intent {self.intent!r}. "
                f"Valid values: {sorted(valid_intents)}"
            )
        if self.correct_resolution not in valid_resolutions:
            raise ValueError(
                f"Scenario {self.scenario_id!r}: invalid resolution {self.correct_resolution!r}. "
                f"Valid values: {sorted(valid_resolutions)}"
            )


# ---------------------------------------------------------------------------
# Scenario catalogue  (12 scenarios, 4 per intent)
# ---------------------------------------------------------------------------

SCENARIOS: List[Scenario] = [

    # ── REFUND ───────────────────────────────────────────────────────────
    Scenario(
        scenario_id="ref_001",
        query=(
            "Hi, I was charged twice for my order #7842. "
            "This is completely unacceptable — I need a refund immediately!"
        ),
        sentiment="angry",
        intent=Intent.refund,
        correct_resolution=Resolution.refund_approved,
        follow_up="When exactly will I see the money back in my account?",
    ),
    Scenario(
        scenario_id="ref_002",
        query=(
            "I returned the defective product two weeks ago "
            "but haven't received my refund yet. My order number is #1023."
        ),
        sentiment="neutral",
        intent=Intent.refund,
        correct_resolution=Resolution.refund_approved,
        follow_up="Can you please check the current status of my refund?",
    ),
    Scenario(
        scenario_id="ref_003",
        query=(
            "The item I received is completely different from what was listed on "
            "your website. I would like my money back, please."
        ),
        sentiment="neutral",
        intent=Intent.refund,
        correct_resolution=Resolution.refund_approved,
        follow_up="How many business days does the refund process typically take?",
    ),
    Scenario(
        scenario_id="ref_004",
        query=(
            "I cancelled my subscription three days ago and I was still charged this month. "
            "I want a full refund for the unauthorised charge."
        ),
        sentiment="angry",
        intent=Intent.refund,
        correct_resolution=Resolution.refund_approved,
        follow_up="Please confirm when the refund will be processed.",
    ),

    # ── DELAY ────────────────────────────────────────────────────────────
    Scenario(
        scenario_id="del_001",
        query=(
            "My order #5512 was supposed to arrive three days ago. "
            "Where is it? I need it urgently."
        ),
        sentiment="angry",
        intent=Intent.delay,
        correct_resolution=Resolution.tracking_provided,
        follow_up="Can you give me the exact tracking number so I can check myself?",
    ),
    Scenario(
        scenario_id="del_002",
        query=(
            "I am waiting for a package that should have arrived last week. "
            "The tracking page has said 'in transit' since Monday — nothing has changed."
        ),
        sentiment="neutral",
        intent=Intent.delay,
        correct_resolution=Resolution.tracking_provided,
        follow_up="Is there any way to expedite my shipment at this point?",
    ),
    Scenario(
        scenario_id="del_003",
        query=(
            "The tracking shows my order was delivered yesterday but I never received it. "
            "Order number is #9901. I am really worried."
        ),
        sentiment="angry",
        intent=Intent.delay,
        correct_resolution=Resolution.tracking_provided,
        follow_up="Should I file a missing package claim with the carrier or with you?",
    ),
    Scenario(
        scenario_id="del_004",
        query=(
            "I placed an express delivery order five days ago and it still has not arrived. "
            "I paid extra for fast shipping!"
        ),
        sentiment="angry",
        intent=Intent.delay,
        correct_resolution=Resolution.tracking_provided,
        follow_up="Can you track where exactly my package is right now?",
    ),

    # ── COMPLAINT ────────────────────────────────────────────────────────
    Scenario(
        scenario_id="cmp_001",
        query=(
            "This is absolutely unacceptable! Your customer service is terrible. "
            "I have been waiting 45 minutes on hold and nobody has helped me."
        ),
        sentiment="angry",
        intent=Intent.complaint,
        correct_resolution=Resolution.escalated,
        follow_up="I want to speak to a manager right now, not another front-line agent.",
    ),
    Scenario(
        scenario_id="cmp_002",
        query=(
            "I am very disappointed with the quality of your product. "
            "It broke after just two uses. I have been a loyal customer for five years."
        ),
        sentiment="angry",
        intent=Intent.complaint,
        correct_resolution=Resolution.escalated,
        follow_up="What are you going to do to make this right for me?",
    ),
    Scenario(
        scenario_id="cmp_003",
        query=(
            "Nobody at your company seems to care about resolving my issue. "
            "I have emailed your support team three times over two weeks with no response."
        ),
        sentiment="angry",
        intent=Intent.complaint,
        correct_resolution=Resolution.escalated,
        follow_up="Can someone please escalate this to your complaints department immediately?",
    ),
    Scenario(
        scenario_id="cmp_004",
        query=(
            "I had an extremely bad experience at your store yesterday. "
            "The staff was rude and dismissive. "
            "I want this on record as a formal complaint."
        ),
        sentiment="angry",
        intent=Intent.complaint,
        correct_resolution=Resolution.escalated,
        follow_up="Who do I speak to about filing a formal complaint?",
    ),
]

# Fast-lookup indices
SCENARIO_INDEX: Dict[str, Scenario] = {s.scenario_id: s for s in SCENARIOS}
ALL_SCENARIO_IDS: List[str]         = [s.scenario_id for s in SCENARIOS]


# ---------------------------------------------------------------------------
# Task base class
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """Base class for evaluation tasks."""

    name:            str
    difficulty:      str   # "easy" | "medium" | "hard"
    description:     str
    objective:       str
    score_threshold: float
    scenario_ids:    List[str] = field(default_factory=list)

    def evaluate(self, response: str, scenario: Scenario, step: int = 0) -> float:
        """Return a task score in [0.0, 1.0] for the agent's response."""
        raise NotImplementedError(
            f"{type(self).__name__}.evaluate() is not implemented."
        )


# ---------------------------------------------------------------------------
# EASY — intent detection
# ---------------------------------------------------------------------------

@dataclass
class EasyTask(Task):
    """
    Score only the agent's intent detection accuracy.
    Grader: grade_intent_detection only.
    """

    def evaluate(self, response: str, scenario: Scenario, step: int = 0) -> float:
        return grade_intent_detection(response, scenario.intent)


# ---------------------------------------------------------------------------
# MEDIUM — resolution correctness
# ---------------------------------------------------------------------------

@dataclass
class MediumTask(Task):
    """
    Score resolution correctness + intent acknowledgement.
    Weights: resolution 70%, intent 30%.
    """

    def evaluate(self, response: str, scenario: Scenario, step: int = 0) -> float:
        res = grade_resolution(response, scenario.correct_resolution)
        intent = grade_intent_detection(response, scenario.intent)
        return round(max(0.0, min(1.0, 0.70 * res + 0.30 * intent)), 4)


# ---------------------------------------------------------------------------
# HARD — full response quality
# ---------------------------------------------------------------------------

@dataclass
class HardTask(Task):
    """
    Score all four quality dimensions.
    Weights: resolution 40%, politeness 25%, empathy 20%, intent 15%.
    """

    def evaluate(self, response: str, scenario: Scenario, step: int = 0) -> float:
        res  = grade_resolution(response, scenario.correct_resolution)
        pol  = grade_politeness(response)
        emp  = grade_empathy(response)
        intn = grade_intent_detection(response, scenario.intent)
        score = (
            0.40 * res
            + 0.25 * pol
            + 0.20 * emp
            + 0.15 * intn
        )
        return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

TASKS: Dict[str, Task] = {
    "easy": EasyTask(
        name="easy",
        difficulty="easy",
        description=(
            "Detect the customer's intent (refund / delay / complaint) "
            "and name it explicitly in your response."
        ),
        objective=(
            "Your goal is to identify and explicitly name the customer's intent. "
            "Use words like 'refund', 'order delay', 'shipping', or 'complaint'. "
            "You do NOT need to resolve the issue — just show you understand "
            "the type of request."
        ),
        score_threshold=TASK_THRESHOLDS["easy"],
        scenario_ids=ALL_SCENARIO_IDS,
    ),
    "medium": MediumTask(
        name="medium",
        difficulty="medium",
        description=(
            "Identify the customer's issue and apply the correct resolution: "
            "approve a refund, provide tracking information, or escalate to a specialist."
        ),
        objective=(
            "Respond with the appropriate resolution action for the customer's issue. "
            "Use phrases like 'I have initiated a full refund', "
            "'here is your tracking number', or "
            "'I am escalating this to a senior specialist'. "
            "Also briefly acknowledge the nature of the issue."
        ),
        score_threshold=TASK_THRESHOLDS["medium"],
        scenario_ids=ALL_SCENARIO_IDS,
    ),
    "hard": HardTask(
        name="hard",
        difficulty="hard",
        description=(
            "Generate a complete, high-quality customer support response: "
            "correct resolution, polite professional tone, and genuine empathy."
        ),
        objective=(
            "Your response MUST contain ALL of the following: "
            "(1) Empathetic acknowledgement of the customer's situation. "
            "(2) The correct resolution action (refund / tracking / escalation). "
            "(3) Polite, professional language throughout. "
            "(4) A closing offer to assist further. "
            "All four elements are required for a top score."
        ),
        score_threshold=TASK_THRESHOLDS["hard"],
        scenario_ids=ALL_SCENARIO_IDS,
    ),
}


def get_task(name: str) -> Task:
    """
    Look up a task by name.

    Raises ValueError for unknown task names, listing valid options.
    """
    if name not in TASKS:
        raise ValueError(
            f"Unknown task {name!r}. Valid tasks: {sorted(TASKS.keys())}"
        )
    return TASKS[name]
