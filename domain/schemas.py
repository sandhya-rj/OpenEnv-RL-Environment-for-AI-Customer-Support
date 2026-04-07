"""
domain/schemas.py — External-facing serialisation schemas.

Separates the domain layer (models.py) from external API contracts, log schemas,
and serialisation concerns. External consumers (HTTP handlers, evaluation harnesses,
log parsers) MUST import from here — NEVER from models.py directly.

This file re-exports all domain models so schemas.py is the single import point
for any code at the boundary of the system.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

# Re-export domain models — external callers use schemas.py as the one-stop import
from domain.models import (  # noqa: F401
    Action,
    EpisodeState,
    Intent,
    Observation,
    Resolution,
    Reward,
    Sentiment,
)

__all__ = [
    # Re-exported domain models
    "Action", "EpisodeState", "Intent", "Observation",
    "Resolution", "Reward", "Sentiment",
    # Schema types defined here
    "StepResult", "EpisodeSummary",
    "StartLog", "StepLog", "EndLog",
    "GraderScores",
]


# ---------------------------------------------------------------------------
# StepResult  — serialisable form of one env.step() call
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """Serialisable representation of a single environment step."""

    model_config = ConfigDict(use_enum_values=True)

    step: int = Field(..., ge=0)
    observation: Observation
    reward: float            = Field(..., ge=-1.0, le=1.0)
    done: bool
    task_score: float        = Field(..., ge=0.0, le=1.0)
    intent_score: float      = Field(..., ge=0.0, le=1.0)
    resolution_score: float  = Field(..., ge=0.0, le=1.0)
    politeness_score: float  = Field(..., ge=0.0, le=1.0)
    empathy_score: float     = Field(..., ge=0.0, le=1.0)
    penalty: float           = Field(..., le=0.0)
    rationale: str           = ""
    error: Optional[str]     = None


# ---------------------------------------------------------------------------
# EpisodeSummary  — emitted at the end of a completed episode
# ---------------------------------------------------------------------------

class EpisodeSummary(BaseModel):
    """Full summary of a completed episode, suitable for logging or API response."""

    model_config = ConfigDict(use_enum_values=True)

    scenario_id: str
    task_name: str
    total_steps: int         = Field(..., ge=0)
    total_reward: float
    rewards: List[float]     = Field(default_factory=list)
    final_task_score: float  = Field(..., ge=0.0, le=1.0)
    success: bool
    intent: str
    correct_resolution: str


# ---------------------------------------------------------------------------
# Inference log schemas  — typed [START] / [STEP] / [END] log lines
# ---------------------------------------------------------------------------

class StartLog(BaseModel):
    task:  str
    env:   str
    model: str

    def render(self) -> str:
        return f"[START] task={self.task} env={self.env} model={self.model}"


class StepLog(BaseModel):
    step:   int
    action: str
    reward: float
    done:   bool
    error:  Optional[str] = None

    def render(self) -> str:
        done_str   = "true" if self.done else "false"
        error_str  = self.error if self.error else "null"
        safe_action = self.action.replace("\n", " ").replace('"', "'")[:140]
        return (
            f'[STEP] step={self.step} '
            f'action="{safe_action}" '
            f'reward={self.reward:.2f} '
            f'done={done_str} '
            f'error={error_str}'
        )


class EndLog(BaseModel):
    success: bool
    steps:   int
    score:   float
    rewards: List[float] = Field(default_factory=list)

    def render(self) -> str:
        success_str  = "true" if self.success else "false"
        rewards_str  = ",".join(f"{r:.2f}" for r in self.rewards) if self.rewards else "none"
        return (
            f"[END] success={success_str} "
            f"steps={self.steps} "
            f"score={self.score:.4f} "
            f"rewards={rewards_str}"
        )


# ---------------------------------------------------------------------------
# GraderScores  — raw sub-scores before task-weight combination
# ---------------------------------------------------------------------------

class GraderScores(BaseModel):
    """All grader sub-scores and raw penalty signals for one response."""

    intent_score: float      = Field(..., ge=0.0, le=1.0)
    resolution_score: float  = Field(..., ge=0.0, le=1.0)
    politeness_score: float  = Field(..., ge=0.0, le=1.0)
    empathy_score: float     = Field(..., ge=0.0, le=1.0)
    irrelevance_penalty: float = Field(..., ge=0.0, le=0.35)
    repetition_penalty: float  = Field(..., ge=0.0, le=0.30)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "intent_score":      self.intent_score,
            "resolution_score":  self.resolution_score,
            "politeness_score":  self.politeness_score,
            "empathy_score":     self.empathy_score,
            "irrelevance_penalty": self.irrelevance_penalty,
            "repetition_penalty":  self.repetition_penalty,
        }
