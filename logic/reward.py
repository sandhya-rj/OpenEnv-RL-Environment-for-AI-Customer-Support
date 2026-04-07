"""
logic/reward.py — Continuous, non-binary reward computation.

DESIGN PRINCIPLES
-----------------
1. SMOOTH: All penalties use smooth curves — no hard cliffs or discrete steps.
2. CONTINUOUS: Reward varies meaningfully at every quality level. Not binary.
3. DECOMPOSABLE: RewardResult stores every component for full auditability.
4. IMMUTABLE: RewardResult is a frozen dataclass — not mutable after construction.
5. BOUNDED: Final value is always clamped to [REWARD_MIN, REWARD_MAX].
6. VERBOSITY PENALTY: Excessively long responses receive a small penalty — prevents
   the "wall of text" exploit where padding increases keyword ratios.

PUBLIC API
----------
RewardResult                                              frozen dataclass
compute_reward(task_score, grader_output, step, resolved) -> RewardResult
combine_grader_scores(weights, intent, res, pol, emp)     -> float [0, 1]
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from config.config import (
    IRRELEVANCE_PENALTY_CAP,
    MAX_RESPONSE_WORDS,
    REPETITION_PENALTY_CAP,
    REWARD_MAX,
    REWARD_MIN,
    STEP_PENALTY_CAP,
    VERBOSITY_PENALTY_CAP,
)
from utils.text_processing import word_count

__all__ = [
    "RewardResult",
    "compute_reward",
    "combine_grader_scores",
]


# ---------------------------------------------------------------------------
# Immutable result record
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardResult:
    """
    Immutable, auditable record of one reward computation step.

    Every component is stored separately so the caller (env.py, tests)
    can inspect the full breakdown.
    """
    value: float                 # Final clamped reward ∈ [REWARD_MIN, REWARD_MAX]
    task_score: float            # Weighted grader composite ∈ [0, 1]
    irrelevance_penalty: float   # ∈ [0, IRRELEVANCE_PENALTY_CAP]
    repetition_penalty: float    # ∈ [0, REPETITION_PENALTY_CAP]
    step_penalty: float          # ∈ [0, STEP_PENALTY_CAP]
    verbosity_penalty: float     # ∈ [0, VERBOSITY_PENALTY_CAP]
    total_penalty: float         # = -(irr + rep + stp + vrb), always ≤ 0
    rationale: str = field(default="")

    def as_dict(self) -> dict:
        return {
            "value":               self.value,
            "task_score":          self.task_score,
            "irrelevance_penalty": self.irrelevance_penalty,
            "repetition_penalty":  self.repetition_penalty,
            "step_penalty":        self.step_penalty,
            "verbosity_penalty":   self.verbosity_penalty,
            "total_penalty":       self.total_penalty,
        }


# ---------------------------------------------------------------------------
# Smooth penalty curves
# ---------------------------------------------------------------------------

def _step_penalty(step: int, resolved: bool) -> float:
    """
    Smooth logistic step penalty in [0, STEP_PENALTY_CAP].

    • No penalty for steps 0–1 (agent deserves time to respond)
    • Penalty rises from step 2 onward via a logistic curve
    • Zero penalty if issue was resolved on this step

    Approximate values:
      step 0,1 → 0.00
      step 2   → ~0.03
      step 3   → ~0.12
      step 4   → ~0.20 (cap)
    """
    if resolved or step < 2:
        return 0.0
    x         = float(step - 2)
    logistic  = 1.0 / (1.0 + math.exp(-2.2 * (x - 1.0)))
    return round(min(STEP_PENALTY_CAP * logistic, STEP_PENALTY_CAP), 4)


def _verbosity_penalty(response: str) -> float:
    """
    Small penalty for excessively verbose responses in (0, VERBOSITY_PENALTY_CAP].

    Motivation: very long responses can inflate keyword ratios by accident.
    A typical good response is 30–80 words. Beyond MAX_RESPONSE_WORDS, a tiny
    linearly-rising penalty discourages padding.

    Returns 0.0 for responses at or under MAX_RESPONSE_WORDS words.
    """
    wc = word_count(response)
    if wc <= MAX_RESPONSE_WORDS:
        return 0.0
    excess    = wc - MAX_RESPONSE_WORDS
    raw       = VERBOSITY_PENALTY_CAP * min(excess / 80.0, 1.0)
    return round(min(raw, VERBOSITY_PENALTY_CAP), 4)


# ---------------------------------------------------------------------------
# Main reward function
# ---------------------------------------------------------------------------

def compute_reward(
    task_score: float,
    irrelevance_penalty_raw: float,
    repetition_penalty_raw: float,
    step: int,
    resolved: bool,
    response: str = "",
) -> RewardResult:
    """
    Combine a task score and penalty signals into a final scalar reward.

    Parameters
    ----------
    task_score : float
        Weighted grader composite ∈ [0, 1].
    irrelevance_penalty_raw : float
        Raw penalty from compute_irrelevance_penalty ∈ [0, IRRELEVANCE_PENALTY_CAP].
    repetition_penalty_raw : float
        Raw penalty from compute_repetition_penalty ∈ [0, REPETITION_PENALTY_CAP].
    step : int
        Current step index (0-based), BEFORE incrementing.
    resolved : bool
        True if task_score >= task threshold on this step.
    response : str
        The raw response text (used for verbosity penalty; optional).

    Returns
    -------
    RewardResult (frozen dataclass)
    """
    irr = float(max(0.0, min(irrelevance_penalty_raw, IRRELEVANCE_PENALTY_CAP)))
    rep = float(max(0.0, min(repetition_penalty_raw,  REPETITION_PENALTY_CAP)))
    stp = _step_penalty(step, resolved)
    vrb = _verbosity_penalty(response) if response else 0.0

    total_penalty = -(irr + rep + stp + vrb)
    raw           = task_score + total_penalty
    value         = round(max(REWARD_MIN, min(REWARD_MAX, raw)), 6)

    rationale = (
        f"task={task_score:.3f} "
        f"irr={irr:.3f} rep={rep:.3f} stp={stp:.3f} vrb={vrb:.3f} "
        f"total_pen={total_penalty:.3f} => reward={value:.3f}"
    )

    return RewardResult(
        value=value,
        task_score=task_score,
        irrelevance_penalty=irr,
        repetition_penalty=rep,
        step_penalty=stp,
        verbosity_penalty=vrb,
        total_penalty=total_penalty,
        rationale=rationale,
    )


# ---------------------------------------------------------------------------
# Weighted grader combinator
# ---------------------------------------------------------------------------

def combine_grader_scores(
    weights: Dict[str, float],
    intent_score: float,
    resolution_score: float,
    politeness_score: float,
    empathy_score: float,
) -> float:
    """
    Produce a weighted composite task score in [0, 1].

    Parameters
    ----------
    weights : dict with keys 'intent', 'resolution', 'politeness', 'empathy'.
              Values must sum to 1.0.

    Returns
    -------
    float ∈ [0.0, 1.0]   (clamped)
    """
    score = (
        weights.get("intent",     0.0) * intent_score
        + weights.get("resolution", 0.0) * resolution_score
        + weights.get("politeness", 0.0) * politeness_score
        + weights.get("empathy",    0.0) * empathy_score
    )
    return round(max(0.0, min(1.0, score)), 4)
