"""
logic/graders.py — Six deterministic graders for the AI Customer Support environment.

DESIGN PRINCIPLES
-----------------
1. ALL graders return float in [0.0, 1.0] — enforced via max/min clamps.
2. NO randomness — purely lexical / rule-based.
3. PARTIAL CREDIT at every level — no purely binary outputs.
4. VARIED SCORES — different response quality levels yield meaningfully different scores.
5. NO CIRCULAR IMPORTS — graders.py has zero imports from logic.reward or logic.tasks.
6. ANTI-EXPLOITATION — keyword density is measured by ratio not raw count, making
   keyword-stuffing attacks ineffective (ratio saturates at 1.0 within existing bands).

PUBLIC API
----------
grade_intent_detection(response, correct_intent)   -> float [0, 1]
grade_resolution(response, correct_resolution)     -> float [0, 1]
grade_politeness(response)                         -> float [0, 1]
grade_empathy(response)                            -> float [0, 1]
compute_irrelevance_penalty(response, query)       -> float [0, IRRELEVANCE_PENALTY_CAP]
compute_repetition_penalty(response, history)      -> float [0, REPETITION_PENALTY_CAP]
score_response(response, query, intent, resolution, history) -> GraderOutput
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from config.config import (
    EMPATHY_PHRASES,
    INTENT_KEYWORDS,
    INTENT_PARTIAL_SYNONYMS,
    IRRELEVANCE_FREE_THRESHOLD,
    IRRELEVANCE_PENALTY_CAP,
    POLITE_NEGATIVE,
    POLITE_POSITIVE,
    REPETITION_PENALTY_CAP,
    REPETITION_SIM_THRESHOLD,
    RESOLUTION_KEYWORDS,
    RESOLUTION_PARTIAL,
    STOP_WORDS,
)
from utils.scoring_utils import (
    char_jaccard,
    diminishing_sum,
    keyword_hits,
    overlap_ratio,
)
from utils.text_processing import content_tokens, normalize

__all__ = [
    "GraderOutput",
    "grade_intent_detection",
    "grade_resolution",
    "grade_politeness",
    "grade_empathy",
    "compute_irrelevance_penalty",
    "compute_repetition_penalty",
    "score_response",
]


# ---------------------------------------------------------------------------
# Structured output  (replaces compute_all_scores to eliminate circular import)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GraderOutput:
    """
    Immutable record of all grader scores for one response.

    Contains the four sub-scores and the two RAW penalty signals.
    reward.py is responsible for combining these with the step penalty.
    """
    intent_score: float        # [0, 1]
    resolution_score: float    # [0, 1]
    politeness_score: float    # [0, 1]
    empathy_score: float       # [0, 1]
    irrelevance_penalty: float # [0, IRRELEVANCE_PENALTY_CAP] — already positive
    repetition_penalty: float  # [0, REPETITION_PENALTY_CAP]  — already positive


# ---------------------------------------------------------------------------
# Intent grader  (used by EasyTask — weight 100%)
# ---------------------------------------------------------------------------

def grade_intent_detection(response: str, correct_intent: str) -> float:
    """
    Score how explicitly the response addresses the customer's intent category.

    Scoring bands
    -------------
    0.00        — no primary keyword or synonym match at all
    0.10–0.25   — synonym/partial match only (no primary keyword found)
    0.35–0.60   — primary keyword found with contamination from wrong-intent keywords
    0.60–1.00   — primary keyword density drives score toward 1.0

    Anti-exploitation: scoring is ratio-based, so keyword-stuffing hits the
    same ceiling as a normal response with good keyword coverage.

    Returns float in [0.0, 1.0].
    """
    norm = normalize(response)
    if not norm:
        return 0.0

    correct_kws = INTENT_KEYWORDS.get(correct_intent, [])
    wrong_intents = [i for i in INTENT_KEYWORDS if i != correct_intent]
    correct_hits  = keyword_hits(norm, correct_kws)

    if correct_hits == 0:
        # Fall back to partial synonyms for partial credit
        syns     = INTENT_PARTIAL_SYNONYMS.get(correct_intent, [])
        syn_hits = keyword_hits(norm, syns)
        if syn_hits == 0:
            return 0.0
        ratio = min(syn_hits / max(len(syns), 1), 1.0)
        return round(0.10 + 0.15 * ratio, 4)

    # Primary keyword found — compute density bonus and contamination penalty
    correct_ratio = correct_hits / max(len(correct_kws), 1)
    density_bonus = 0.60 + 0.40 * min(correct_ratio / 0.30, 1.0)   # [0.60, 1.00]

    wrong_hits          = sum(keyword_hits(norm, INTENT_KEYWORDS[wi]) for wi in wrong_intents)
    total_wrong_possible = sum(len(INTENT_KEYWORDS[wi]) for wi in wrong_intents)
    wrong_ratio          = wrong_hits / max(total_wrong_possible, 1)
    contamination        = 0.40 * min(wrong_ratio * 25, 1.0)

    return round(max(0.0, min(1.0, density_bonus - contamination)), 4)


# ---------------------------------------------------------------------------
# Resolution grader  (used by Medium + Hard tasks)
# ---------------------------------------------------------------------------

def grade_resolution(response: str, correct_resolution: str) -> float:
    """
    Score whether the response provides the required resolution action.

    Scoring bands
    -------------
    0.00        — no keyword or synonym match
    0.05–0.25   — partial synonym only
    0.35–0.44   — very low primary keyword density
    0.45–0.69   — moderate keyword density
    0.70–1.00   — good keyword density (≥ 40% of registered keywords found)

    Returns float in [0.0, 1.0].
    """
    norm = normalize(response)
    if not norm:
        return 0.0

    correct_kws = RESOLUTION_KEYWORDS.get(correct_resolution, [])
    if not correct_kws:
        return 0.0

    hits  = keyword_hits(norm, correct_kws)
    ratio = hits / len(correct_kws)

    if ratio == 0.0:
        partial_kws  = RESOLUTION_PARTIAL.get(correct_resolution, [])
        partial_hits = keyword_hits(norm, partial_kws)
        if partial_hits == 0:
            return 0.0
        return round(0.05 + 0.20 * min(partial_hits / max(len(partial_kws), 1), 1.0), 4)

    if ratio >= 0.40:
        scaled = (ratio - 0.40) / 0.60
        return round(0.70 + 0.30 * min(scaled, 1.0), 4)
    elif ratio >= 0.15:
        scaled = (ratio - 0.15) / 0.25
        return round(0.45 + 0.25 * min(scaled, 1.0), 4)
    else:
        scaled = ratio / 0.15
        return round(0.35 + 0.10 * min(scaled, 1.0), 4)


# ---------------------------------------------------------------------------
# Politeness grader
# ---------------------------------------------------------------------------

def grade_politeness(response: str) -> float:
    """
    Score the professional politeness level of the response.

    Scoring mechanism
    -----------------
    • Positive keyword hits: diminishing-returns series (each hit worth less)
    • Negative keyword hits: 0.30 penalty each
    • Length ceiling: very short responses cannot achieve high politeness scores

    Returns float in [0.0, 1.0].
    """
    norm       = normalize(response)
    word_count = len(norm.split()) if norm else 0

    pos_hits = keyword_hits(norm, POLITE_POSITIVE)
    neg_hits = keyword_hits(norm, POLITE_NEGATIVE)

    score = diminishing_sum(pos_hits, base=0.10, decay=0.82)
    score -= neg_hits * 0.30

    # Length ceiling — prevents trivially short responses from scoring high
    if word_count < 4:
        score = min(score, 0.15)
    elif word_count < 10:
        score = min(score, 0.45)
    elif word_count < 20:
        score = min(score, 0.72)

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Empathy grader
# ---------------------------------------------------------------------------

def grade_empathy(response: str) -> float:
    """
    Score the level of emotional empathy in the response.

    Scoring mechanism
    -----------------
    • Empathy phrase hits: diminishing-returns series starting at 0.20 per hit
    • Length ceiling: very short responses cannot achieve high empathy scores
      (genuine empathy requires sentences, not single words)

    Returns float in [0.0, 1.0].
    """
    norm       = normalize(response)
    word_count = len(norm.split()) if norm else 0

    hits  = keyword_hits(norm, EMPATHY_PHRASES)
    score = diminishing_sum(hits, base=0.20, decay=0.80)

    if word_count < 6:
        score = min(score, 0.20)
    elif word_count < 15:
        score = min(score, 0.55)

    return round(max(0.0, min(1.0, score)), 4)


# ---------------------------------------------------------------------------
# Irrelevance penalty
# ---------------------------------------------------------------------------

def compute_irrelevance_penalty(response: str, query: str) -> float:
    """
    Return a penalty in [0.0, IRRELEVANCE_PENALTY_CAP] when the response
    is unrelated to the query.

    Zero penalty when:
    • The query has no qualifying content words (empty / stop-word-only query)
    • Content-word overlap with the query >= IRRELEVANCE_FREE_THRESHOLD

    Below the threshold, the penalty rises smoothly to IRRELEVANCE_PENALTY_CAP.
    """
    if not content_tokens(query, STOP_WORDS, min_length=3):
        return 0.0

    ratio = overlap_ratio(response, query, STOP_WORDS, min_length=3)
    if ratio >= IRRELEVANCE_FREE_THRESHOLD:
        return 0.0

    raw = (1.0 - ratio / IRRELEVANCE_FREE_THRESHOLD) * IRRELEVANCE_PENALTY_CAP
    return round(min(raw, IRRELEVANCE_PENALTY_CAP), 4)


# ---------------------------------------------------------------------------
# Repetition penalty
# ---------------------------------------------------------------------------

def compute_repetition_penalty(response: str, history: List[str]) -> float:
    """
    Return a penalty in [0.0, REPETITION_PENALTY_CAP] when the response
    is similar to any prior agent response in this episode.

    Uses character 4-gram Jaccard similarity.
    Penalty rises smoothly from the similarity threshold to the cap.
    Returns 0.0 when history is empty.
    """
    if not history:
        return 0.0

    max_sim = max(
        (char_jaccard(response, prev, n=4) for prev in history),
        default=0.0,
    )

    if max_sim < REPETITION_SIM_THRESHOLD:
        return 0.0

    penalty = REPETITION_PENALTY_CAP * (
        (max_sim - REPETITION_SIM_THRESHOLD) / (1.0 - REPETITION_SIM_THRESHOLD)
    )
    return round(min(penalty, REPETITION_PENALTY_CAP), 4)


# ---------------------------------------------------------------------------
# Composite scoring function  (no circular import — never imports reward.py)
# ---------------------------------------------------------------------------

def score_response(
    response: str,
    query: str,
    intent: str,
    resolution: str,
    history: List[str],
) -> GraderOutput:
    """
    Run all six graders and return a GraderOutput record.

    This is the primary interface for env.py — it gets all scores and raw
    penalty signals in one call. reward.py is responsible for combining them
    with the step penalty into the final scalar reward.

    Parameters
    ----------
    response   : str   The agent's response this step
    query      : str   Current customer query
    intent     : str   Intent value from the scenario
    resolution : str   Correct resolution from the scenario
    history    : list  Prior agent responses in this episode

    Returns
    -------
    GraderOutput (frozen dataclass)
    """
    return GraderOutput(
        intent_score       = grade_intent_detection(response, intent),
        resolution_score   = grade_resolution(response, resolution),
        politeness_score   = grade_politeness(response),
        empathy_score      = grade_empathy(response),
        irrelevance_penalty = compute_irrelevance_penalty(response, query),
        repetition_penalty  = compute_repetition_penalty(response, history),
    )
