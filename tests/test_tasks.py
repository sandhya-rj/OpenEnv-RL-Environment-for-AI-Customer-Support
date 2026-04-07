"""
tests/test_tasks.py — Unit tests for logic/tasks.py.

Coverage
--------
• Task registry (TASKS): all three tasks present, correct types, consistent metadata
• Task thresholds match config values
• Task.evaluate(): in-range, deterministic, varied, correct > wrong
• Scenario.__post_init__: invalid intent/resolution raises ValueError
• Scenario catalogue: 12 unique scenarios, correct intent/resolution alignment
• get_task(): correct instance, raises ValueError for unknown names
• HardTask: all twelve scenarios can be scored without error
"""

from __future__ import annotations

import pytest

from logic.tasks import (
    ALL_SCENARIO_IDS,
    SCENARIO_INDEX,
    SCENARIOS,
    TASKS,
    EasyTask,
    HardTask,
    MediumTask,
    Scenario,
    Task,
    get_task,
)
from config.config import TASK_THRESHOLDS


# ---------------------------------------------------------------------------
# Reference responses  (shared across task classes)
# ---------------------------------------------------------------------------

PERFECT_REFUND = (
    "I completely understand your frustration and I sincerely apologize for the "
    "duplicate charge. I have approved a full refund for your order, which will be "
    "refunded to your account within 3–5 business days. Please let me know if there "
    "is anything else I can help you with."
)

PERFECT_TRACKING = (
    "I sincerely apologize for the delay. I can see your order is still in transit. "
    "Here is your tracking number: TRK-5512 so you can track your order on our "
    "tracking page and check the delivery status with the carrier."
)

PERFECT_ESCALATION = (
    "I completely understand your frustration. I sincerely apologize for the experience. "
    "I am escalating this to a senior specialist and a manager who will prioritize your "
    "case and act immediately. "
    "Please don't hesitate to reach out if there's anything else I can help with."
)

POOR      = "I don't know what to say."
CONTEXTLESS = "The product design is very nice and modern."
RUDE      = "That's not our problem. Deal with it."


# ---------------------------------------------------------------------------
# Task registry
# ---------------------------------------------------------------------------

class TestTaskRegistry:
    def test_all_three_tasks_present(self):
        assert set(TASKS.keys()) == {"easy", "medium", "hard"}

    def test_easy_correct_type(self):
        assert isinstance(TASKS["easy"], EasyTask)

    def test_medium_correct_type(self):
        assert isinstance(TASKS["medium"], MediumTask)

    def test_hard_correct_type(self):
        assert isinstance(TASKS["hard"], HardTask)

    def test_get_task_returns_correct_instance(self):
        for name in ("easy", "medium", "hard"):
            task = get_task(name)
            assert isinstance(task, Task)
            assert task.name == name

    def test_get_task_unknown_raises_value_error(self):
        with pytest.raises(ValueError):
            get_task("impossible_difficulty")

    def test_name_keys_match_task_name_attr(self):
        for name, task in TASKS.items():
            assert task.name == name

    def test_difficulty_attrs_correct(self):
        assert TASKS["easy"].difficulty   == "easy"
        assert TASKS["medium"].difficulty == "medium"
        assert TASKS["hard"].difficulty   == "hard"

    def test_descriptions_non_empty(self):
        for task in TASKS.values():
            assert isinstance(task.description, str) and len(task.description) > 10

    def test_objectives_non_empty(self):
        for task in TASKS.values():
            assert isinstance(task.objective, str) and len(task.objective) > 10

    def test_thresholds_in_valid_range(self):
        for task in TASKS.values():
            assert 0.0 < task.score_threshold < 1.0

    def test_thresholds_match_config(self):
        for name, task in TASKS.items():
            assert task.score_threshold == TASK_THRESHOLDS[name]

    def test_scenario_ids_list_populated(self):
        for task in TASKS.values():
            assert len(task.scenario_ids) == 12


# ---------------------------------------------------------------------------
# EasyTask
# ---------------------------------------------------------------------------

class TestEasyTask:
    task = TASKS["easy"]

    def test_correct_intent_scores_high(self, ref_scenario):
        assert self.task.evaluate(PERFECT_REFUND, ref_scenario) >= 0.50

    def test_scores_in_range(self, ref_scenario):
        for r in [PERFECT_REFUND, POOR, CONTEXTLESS]:
            assert 0.0 <= self.task.evaluate(r, ref_scenario) <= 1.0

    def test_returns_float(self, ref_scenario):
        assert isinstance(self.task.evaluate(PERFECT_REFUND, ref_scenario), float)

    def test_deterministic(self, ref_scenario):
        s1 = self.task.evaluate(PERFECT_REFUND, ref_scenario)
        s2 = self.task.evaluate(PERFECT_REFUND, ref_scenario)
        assert s1 == s2

    def test_scores_vary(self, ref_scenario):
        s_good = self.task.evaluate(PERFECT_REFUND, ref_scenario)
        s_poor = self.task.evaluate(CONTEXTLESS, ref_scenario)
        assert s_good > s_poor

    def test_step_arg_ignored(self, ref_scenario):
        """EasyTask must be step-independent."""
        s0 = self.task.evaluate(PERFECT_REFUND, ref_scenario, step=0)
        s4 = self.task.evaluate(PERFECT_REFUND, ref_scenario, step=4)
        assert s0 == s4


# ---------------------------------------------------------------------------
# MediumTask
# ---------------------------------------------------------------------------

class TestMediumTask:
    task = TASKS["medium"]

    def test_correct_resolution_scores_high(self, ref_scenario):
        assert self.task.evaluate(PERFECT_REFUND, ref_scenario) >= 0.50

    def test_scores_in_range(self, ref_scenario, del_scenario, cmp_scenario):
        combos = [
            (ref_scenario, PERFECT_REFUND),
            (del_scenario, PERFECT_TRACKING),
            (cmp_scenario, PERFECT_ESCALATION),
            (ref_scenario, POOR),
        ]
        for scenario, response in combos:
            assert 0.0 <= self.task.evaluate(response, scenario) <= 1.0

    def test_returns_float(self, ref_scenario):
        assert isinstance(self.task.evaluate(PERFECT_REFUND, ref_scenario), float)

    def test_deterministic(self, del_scenario):
        s1 = self.task.evaluate(PERFECT_TRACKING, del_scenario)
        s2 = self.task.evaluate(PERFECT_TRACKING, del_scenario)
        assert s1 == s2

    def test_correct_resolution_beats_wrong(self, ref_scenario, del_scenario):
        s_correct = self.task.evaluate(PERFECT_REFUND, ref_scenario)
        s_wrong   = self.task.evaluate(PERFECT_TRACKING, ref_scenario)
        assert s_correct > s_wrong

    def test_scores_vary_across_scenarios(self, ref_scenario, del_scenario):
        s_ref = self.task.evaluate(PERFECT_REFUND, ref_scenario)
        s_del = self.task.evaluate(PERFECT_REFUND, del_scenario)
        assert s_ref != s_del


# ---------------------------------------------------------------------------
# HardTask
# ---------------------------------------------------------------------------

class TestHardTask:
    task = TASKS["hard"]

    def test_comprehensive_scores_high(self, ref_scenario):
        assert self.task.evaluate(PERFECT_REFUND, ref_scenario) >= 0.50

    def test_scores_in_range(self, ref_scenario):
        for r in [PERFECT_REFUND, POOR, CONTEXTLESS]:
            assert 0.0 <= self.task.evaluate(r, ref_scenario) <= 1.0

    def test_returns_float(self, ref_scenario):
        assert isinstance(self.task.evaluate(PERFECT_REFUND, ref_scenario), float)

    def test_deterministic(self, cmp_scenario):
        s1 = self.task.evaluate(PERFECT_ESCALATION, cmp_scenario)
        s2 = self.task.evaluate(PERFECT_ESCALATION, cmp_scenario)
        assert s1 == s2

    def test_comprehensive_beats_minimal(self, ref_scenario):
        s_good = self.task.evaluate(PERFECT_REFUND, ref_scenario)
        s_min  = self.task.evaluate("Refund done.", ref_scenario)
        assert s_good > s_min

    def test_rude_scores_low(self, ref_scenario):
        assert self.task.evaluate(RUDE, ref_scenario) < 0.50

    def test_all_twelve_scenarios_scoreable(self):
        for sid, scenario in SCENARIO_INDEX.items():
            score = self.task.evaluate(PERFECT_REFUND, scenario)
            assert 0.0 <= score <= 1.0, f"Bad score {score} for {sid}"


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

class TestScenarioCatalogue:
    def test_exactly_twelve_scenarios(self):
        assert len(SCENARIOS) == 12

    def test_all_expected_ids_present(self):
        expected = {
            "ref_001", "ref_002", "ref_003", "ref_004",
            "del_001", "del_002", "del_003", "del_004",
            "cmp_001", "cmp_002", "cmp_003", "cmp_004",
        }
        assert set(ALL_SCENARIO_IDS) == expected

    def test_scenario_index_matches_list(self):
        assert set(SCENARIO_INDEX.keys()) == set(ALL_SCENARIO_IDS)

    def test_all_queries_non_empty(self):
        for s in SCENARIOS:
            assert isinstance(s.query, str) and len(s.query) > 5

    def test_all_sentiments_valid(self):
        for s in SCENARIOS:
            assert s.sentiment in ("happy", "neutral", "angry")

    def test_all_intents_valid(self):
        for s in SCENARIOS:
            assert s.intent in ("refund", "delay", "complaint")

    def test_all_resolutions_valid(self):
        for s in SCENARIOS:
            assert s.correct_resolution in (
                "refund_approved", "tracking_provided", "escalated"
            )

    def test_refund_intent_maps_to_refund_resolution(self):
        for s in SCENARIOS:
            if s.intent == "refund":
                assert s.correct_resolution == "refund_approved"

    def test_delay_intent_maps_to_tracking_resolution(self):
        for s in SCENARIOS:
            if s.intent == "delay":
                assert s.correct_resolution == "tracking_provided"

    def test_complaint_intent_maps_to_escalated_resolution(self):
        for s in SCENARIOS:
            if s.intent == "complaint":
                assert s.correct_resolution == "escalated"

    def test_scenario_ids_are_unique(self):
        ids = [s.scenario_id for s in SCENARIOS]
        assert len(ids) == len(set(ids))

    def test_all_scenarios_are_scenario_instances(self):
        for s in SCENARIOS:
            assert isinstance(s, Scenario)

    def test_invalid_intent_raises_value_error(self):
        """Scenario.__post_init__ must reject invalid intent values."""
        with pytest.raises(ValueError, match="invalid intent"):
            Scenario(
                scenario_id="bad_001",
                query="Test query.",
                sentiment="angry",
                intent="invalid_intent",
                correct_resolution="refund_approved",
            )

    def test_invalid_resolution_raises_value_error(self):
        """Scenario.__post_init__ must reject invalid resolution values."""
        with pytest.raises(ValueError, match="invalid resolution"):
            Scenario(
                scenario_id="bad_002",
                query="Test query.",
                sentiment="neutral",
                intent="refund",
                correct_resolution="teleport_package",
            )
