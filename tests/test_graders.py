"""
tests/test_graders.py — Unit tests for logic/graders.py.

Coverage
--------
• score_response() returns a valid GraderOutput dataclass (frozen)
• All six grader functions return float in [0.0, 1.0]
• Scores vary across different response quality levels (non-constant)
• Partial credit is awarded — no purely binary outputs
• Penalty functions respect their documented ranges
• Edge cases: empty response, stop-word-only query, repeated response
• GraderOutput arithmetic: penalty values are non-negative
"""

from __future__ import annotations

import pytest

from logic.graders import (
    GraderOutput,
    compute_irrelevance_penalty,
    compute_repetition_penalty,
    grade_empathy,
    grade_intent_detection,
    grade_politeness,
    grade_resolution,
    score_response,
)
from config.config import IRRELEVANCE_PENALTY_CAP, REPETITION_PENALTY_CAP


# ---------------------------------------------------------------------------
# Reference responses  (also available from conftest.py as fixtures)
# ---------------------------------------------------------------------------

GOOD_REFUND = (
    "I completely understand your frustration. I sincerely apologize for the "
    "duplicate charge. I have approved a full refund for your order, which will "
    "be processed and refunded within 3–5 business days. Please let me know if "
    "there is anything else I can help you with."
)

GOOD_TRACKING = (
    "I sincerely apologize for the delay. I can see your order is still in transit. "
    "Here is your tracking number: TRK-9901. You can track your order at our "
    "tracking page. The carrier will deliver it by end of day tomorrow."
)

GOOD_ESCALATION = (
    "I completely understand how frustrating this must be and I sincerely apologize. "
    "I am escalating this to a senior specialist and a manager who will prioritize "
    "your case and connect you with the right team within 24 hours. "
    "Please don't hesitate to reach out if there's anything else I can help with."
)

RUDE     = "That's not our problem. Deal with it."
MINIMAL  = "Thank you."
IRRELEVANT = "The weather is nice today in Paris."


# ---------------------------------------------------------------------------
# score_response() — composite grader
# ---------------------------------------------------------------------------

class TestScoreResponse:
    def test_returns_grader_output(self):
        result = score_response(
            response   = GOOD_REFUND,
            query      = "I was charged twice and need a refund.",
            intent     = "refund",
            resolution = "refund_approved",
            history    = [],
        )
        assert isinstance(result, GraderOutput)

    def test_grader_output_is_frozen(self):
        result = score_response(
            response   = GOOD_REFUND,
            query      = "I need a refund.",
            intent     = "refund",
            resolution = "refund_approved",
            history    = [],
        )
        with pytest.raises((AttributeError, TypeError)):
            result.intent_score = 0.5  # type: ignore

    def test_all_scores_in_unit_range(self):
        result = score_response(
            response   = GOOD_REFUND,
            query      = "I was charged twice.",
            intent     = "refund",
            resolution = "refund_approved",
            history    = [],
        )
        assert 0.0 <= result.intent_score       <= 1.0
        assert 0.0 <= result.resolution_score   <= 1.0
        assert 0.0 <= result.politeness_score   <= 1.0
        assert 0.0 <= result.empathy_score      <= 1.0
        assert 0.0 <= result.irrelevance_penalty <= IRRELEVANCE_PENALTY_CAP
        assert 0.0 <= result.repetition_penalty  <= REPETITION_PENALTY_CAP

    def test_penalty_values_are_non_negative(self):
        result = score_response(
            response   = IRRELEVANT,
            query      = "I need a refund for my order.",
            intent     = "refund",
            resolution = "refund_approved",
            history    = [IRRELEVANT],
        )
        assert result.irrelevance_penalty >= 0.0
        assert result.repetition_penalty  >= 0.0


# ---------------------------------------------------------------------------
# grade_intent_detection
# ---------------------------------------------------------------------------

class TestIntentGrader:
    @pytest.mark.parametrize("response, intent", [
        (GOOD_REFUND,      "refund"),
        (GOOD_TRACKING,    "delay"),
        (GOOD_ESCALATION,  "complaint"),
    ])
    def test_correct_intent_scores_high(self, response, intent):
        score = grade_intent_detection(response, intent)
        assert score >= 0.50, f"Expected >= 0.50 for {intent!r}, got {score}"

    def test_scores_in_unit_range(self):
        for intent in ("refund", "delay", "complaint"):
            score = grade_intent_detection(GOOD_REFUND, intent)
            assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        assert isinstance(grade_intent_detection(GOOD_REFUND, "refund"), float)

    def test_empty_response_returns_zero(self):
        assert grade_intent_detection("   ", "refund") == 0.0

    def test_normalised_empty_returns_zero(self):
        assert grade_intent_detection("", "refund") == 0.0

    def test_irrelevant_response_scores_low(self):
        assert grade_intent_detection(IRRELEVANT, "refund") < 0.30

    def test_partial_synonym_gives_partial_credit(self):
        # Only synonym words — no primary keywords
        score = grade_intent_detection(
            "We have your invoice details and can look into your purchase history.",
            "refund",
        )
        assert 0.0 < score < 0.60

    def test_scores_vary_across_intents(self):
        s_refund = grade_intent_detection(GOOD_REFUND, "refund")
        s_delay  = grade_intent_detection(GOOD_REFUND, "delay")
        assert s_refund != s_delay, "Grader must not be constant across different intents"

    def test_correct_intent_beats_wrong_intent(self):
        s_correct = grade_intent_detection(GOOD_REFUND, "refund")
        s_wrong   = grade_intent_detection(GOOD_REFUND, "complaint")
        assert s_correct > s_wrong


# ---------------------------------------------------------------------------
# grade_resolution
# ---------------------------------------------------------------------------

class TestResolutionGrader:
    @pytest.mark.parametrize("response, resolution", [
        (GOOD_REFUND,     "refund_approved"),
        (GOOD_TRACKING,   "tracking_provided"),
        (GOOD_ESCALATION, "escalated"),
    ])
    def test_correct_resolution_scores_high(self, response, resolution):
        score = grade_resolution(response, resolution)
        assert score >= 0.50, f"Expected >= 0.50 for {resolution!r}, got {score}"

    def test_scores_in_unit_range(self):
        for res in ("refund_approved", "tracking_provided", "escalated"):
            score = grade_resolution(GOOD_REFUND, res)
            assert 0.0 <= score <= 1.0

    def test_returns_float(self):
        assert isinstance(grade_resolution(GOOD_REFUND, "refund_approved"), float)

    def test_empty_response_returns_zero(self):
        assert grade_resolution("", "refund_approved") == 0.0

    def test_irrelevant_returns_zero(self):
        assert grade_resolution(IRRELEVANT, "refund_approved") == 0.0

    def test_partial_synonym_gives_partial_credit(self):
        score = grade_resolution(
            "We are happy to process a return for you.", "refund_approved"
        )
        assert 0.0 < score < 0.50

    def test_scores_vary_across_resolutions(self):
        s1 = grade_resolution(GOOD_REFUND, "refund_approved")
        s2 = grade_resolution(GOOD_REFUND, "tracking_provided")
        assert s1 != s2

    def test_unknown_resolution_returns_zero(self):
        assert grade_resolution("I will help you.", "nonexistent_resolution") == 0.0


# ---------------------------------------------------------------------------
# grade_politeness
# ---------------------------------------------------------------------------

class TestPolitenessGrader:
    def test_polite_scores_high(self):
        assert grade_politeness(GOOD_REFUND) >= 0.40

    def test_rude_scores_zero(self):
        assert grade_politeness("That's not our problem. Deal with it.") == 0.0

    def test_scores_in_unit_range(self):
        for r in [GOOD_REFUND, RUDE, MINIMAL]:
            assert 0.0 <= grade_politeness(r) <= 1.0

    def test_returns_float(self):
        assert isinstance(grade_politeness(GOOD_REFUND), float)

    def test_polite_beats_rude(self):
        assert grade_politeness(GOOD_REFUND) > grade_politeness(RUDE)

    def test_very_short_response_capped(self):
        assert grade_politeness("Yes.") <= 0.15

    def test_medium_length_intermediate(self):
        score = grade_politeness("Thank you for reaching out. I will help.")
        assert 0.0 < score < 1.0

    def test_empty_response_not_negative(self):
        score = grade_politeness("")
        assert score >= 0.0


# ---------------------------------------------------------------------------
# grade_empathy
# ---------------------------------------------------------------------------

class TestEmpathyGrader:
    def test_empathetic_scores_high(self):
        assert grade_empathy(GOOD_REFUND) >= 0.30

    def test_no_empathy_scores_low(self):
        assert grade_empathy("Your refund is processed.") <= 0.25

    def test_scores_in_unit_range(self):
        for r in [GOOD_REFUND, RUDE, IRRELEVANT]:
            assert 0.0 <= grade_empathy(r) <= 1.0

    def test_returns_float(self):
        assert isinstance(grade_empathy(GOOD_REFUND), float)

    def test_empathetic_beats_neutral(self):
        assert grade_empathy(GOOD_REFUND) > grade_empathy("Here is your refund.")

    def test_very_short_response_capped(self):
        assert grade_empathy("Sorry.") <= 0.20

    def test_empty_response_not_negative(self):
        assert grade_empathy("") >= 0.0


# ---------------------------------------------------------------------------
# compute_irrelevance_penalty
# ---------------------------------------------------------------------------

class TestIrrelevancePenalty:
    def test_relevant_response_zero_penalty(self):
        query    = "I was charged twice for my order and need a refund."
        response = (
            "I completely understand your concern about the duplicate charge. "
            "I have initiated a full refund for your order right away."
        )
        assert compute_irrelevance_penalty(response, query) == 0.0

    def test_irrelevant_response_high_penalty(self):
        query = "I was charged twice for my order and need a refund."
        assert compute_irrelevance_penalty(IRRELEVANT, query) >= 0.20

    def test_in_range(self):
        for r in [GOOD_REFUND, IRRELEVANT, "OK"]:
            p = compute_irrelevance_penalty(r, "I need a refund for my order.")
            assert 0.0 <= p <= IRRELEVANCE_PENALTY_CAP

    def test_returns_float(self):
        assert isinstance(compute_irrelevance_penalty("hi", "hello"), float)

    def test_empty_query_returns_zero(self):
        assert compute_irrelevance_penalty("hello", "") == 0.0

    def test_stop_words_only_query_returns_zero(self):
        # "a the is" → no content words → zero penalty
        assert compute_irrelevance_penalty("hello there", "a the is") == 0.0

    def test_penalty_rises_as_relevance_falls(self):
        query = "refund order duplicate charge billing"
        p_relevant   = compute_irrelevance_penalty(GOOD_REFUND, query)
        p_irrelevant = compute_irrelevance_penalty(IRRELEVANT, query)
        assert p_irrelevant >= p_relevant


# ---------------------------------------------------------------------------
# compute_repetition_penalty
# ---------------------------------------------------------------------------

class TestRepetitionPenalty:
    def test_no_history_returns_zero(self):
        assert compute_repetition_penalty(GOOD_REFUND, []) == 0.0

    def test_identical_response_high_penalty(self):
        assert compute_repetition_penalty(GOOD_REFUND, [GOOD_REFUND]) >= 0.20

    def test_very_different_response_zero_penalty(self):
        history = ["The tracking number is TRK-001 and your parcel is in transit."]
        assert compute_repetition_penalty(IRRELEVANT, history) == 0.0

    def test_in_range(self):
        for r in [GOOD_REFUND, IRRELEVANT, "A totally different message."]:
            p = compute_repetition_penalty(r, [GOOD_REFUND])
            assert 0.0 <= p <= REPETITION_PENALTY_CAP

    def test_returns_float(self):
        assert isinstance(compute_repetition_penalty("hi", ["hello"]), float)

    def test_multiple_history_entries(self):
        history = [GOOD_REFUND, GOOD_TRACKING, GOOD_ESCALATION]
        penalty = compute_repetition_penalty(GOOD_REFUND, history)
        assert 0.0 <= penalty <= REPETITION_PENALTY_CAP
