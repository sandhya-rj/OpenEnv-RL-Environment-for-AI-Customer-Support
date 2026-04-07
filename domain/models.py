"""
domain/models.py — Pydantic v2 domain models: the complete internal type contract.

Design decisions
----------------
• Reward is frozen (immutable) — prevents accidental mutation post-construction
• EpisodeState uses use_enum_values so JSON round-trips work transparently
• Action validator strips leading/trailing whitespace and rejects blank-only strings
• All fields use Field() with documentation strings (serves as an API contract)
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import BaseModel, ConfigDict, Field, field_validator

__all__ = [
    "Sentiment",
    "Intent",
    "Resolution",
    "Observation",
    "Action",
    "Reward",
    "EpisodeState",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class Sentiment(str, Enum):
    happy   = "happy"
    neutral = "neutral"
    angry   = "angry"


class Intent(str, Enum):
    refund    = "refund"
    delay     = "delay"
    complaint = "complaint"
    unknown   = "unknown"


class Resolution(str, Enum):
    refund_approved   = "refund_approved"
    tracking_provided = "tracking_provided"
    escalated         = "escalated"
    none              = "none"


# ---------------------------------------------------------------------------
# Observation  — what the agent sees each step
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """The agent's view of the environment at a given step."""

    model_config = ConfigDict(use_enum_values=True)

    query: str = Field(
        ...,
        description="Current customer message (may change to a follow-up after step 1).",
    )
    sentiment: Sentiment = Field(
        ...,
        description="Detected emotional sentiment of the customer.",
    )
    history: List[str] = Field(
        default_factory=list,
        description="Ordered list of all prior agent responses in this episode.",
    )
    step_number: int = Field(
        0,
        ge=0,
        description="Current step index, 0-based. Is 0 on the first call to reset().",
    )
    intent: Intent = Field(
        Intent.unknown,
        description="Ground-truth customer intent (provided to agent as context).",
    )
    episode_done: bool = Field(
        False,
        description="True when the episode has terminated.",
    )

    def __repr__(self) -> str:
        return (
            f"Observation(step={self.step_number}, intent={self.intent!r}, "
            f"sentiment={self.sentiment!r}, history_len={len(self.history)}, "
            f"done={self.episode_done})"
        )


# ---------------------------------------------------------------------------
# Action  — what the agent does each step
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """A single agent turn: a natural-language response to the customer."""

    model_config = ConfigDict(use_enum_values=True)

    response: str = Field(
        ...,
        min_length=1,
        description="The agent's text response. Must be non-empty and non-blank.",
    )

    @field_validator("response")
    @classmethod
    def response_must_not_be_blank(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError(
                "Action.response must not be empty or whitespace-only. "
                "The agent must produce a non-blank response."
            )
        return stripped

    def __repr__(self) -> str:
        preview = self.response[:80].replace("\n", " ")
        ellipsis = "..." if len(self.response) > 80 else ""
        return f"Action(response={preview!r}{ellipsis})"


# ---------------------------------------------------------------------------
# Reward  — structured reward object (immutable by design)
# ---------------------------------------------------------------------------

class Reward(BaseModel):
    """
    Structured reward payload returned at each env.step().

    Frozen so it cannot be mutated after construction.
    All sub-scores are independently auditable.
    """

    model_config = ConfigDict(use_enum_values=True, frozen=True)

    value: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Final scalar reward in [-1, 1].",
    )
    intent_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Intent detection sub-score.",
    )
    resolution_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Resolution correctness sub-score.",
    )
    politeness_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Politeness sub-score.",
    )
    empathy_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Empathy sub-score.",
    )
    task_score: float = Field(
        0.0, ge=0.0, le=1.0,
        description="Weighted composite task score (drives episode resolution).",
    )
    penalty: float = Field(
        0.0, le=0.0,
        description="Total penalty applied this step (always ≤ 0).",
    )
    rationale: str = Field(
        "",
        description="Human-readable reward decomposition string.",
    )

    def __repr__(self) -> str:
        return (
            f"Reward(value={self.value:.3f}, task={self.task_score:.3f}, "
            f"penalty={self.penalty:.3f})"
        )


# ---------------------------------------------------------------------------
# EpisodeState  — full internal episode snapshot managed by StateManager
# ---------------------------------------------------------------------------

class EpisodeState(BaseModel):
    """Complete, serialisable episode state. Never mutated in place."""

    model_config = ConfigDict(use_enum_values=True)

    scenario_id: str
    query: str
    sentiment: Sentiment
    intent: Intent
    correct_resolution: Resolution
    history: List[str]        = Field(default_factory=list)
    step: int                 = 0
    max_steps: int            = 5
    done: bool                = False
    total_reward: float       = 0.0
    rewards: List[float]      = Field(default_factory=list)
    task_name: str            = ""

    def __repr__(self) -> str:
        return (
            f"EpisodeState(scenario={self.scenario_id!r}, task={self.task_name!r}, "
            f"step={self.step}/{self.max_steps}, done={self.done}, "
            f"total_reward={self.total_reward:.3f})"
        )
