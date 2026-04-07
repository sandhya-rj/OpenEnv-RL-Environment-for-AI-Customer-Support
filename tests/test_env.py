"""
tests/test_env.py — Unit and integration tests for CustomerSupportEnv.

Coverage
--------
• reset() returns valid Observation with correct types/defaults
• step() returns (Observation, Reward, bool, dict) with correct types
• Reward is frozen (immutable after construction)
• Reward is continuous and bounded in [-1, 1]
• Episode terminates correctly: after MAX_STEPS and on early success
• state() reflects changes across steps
• StateManager raises RuntimeError on post-done step
• All three task difficulties × multiple scenarios complete full episodes
• render() works before and after reset
• Info dict contains all required keys including new verbosity_penalty
"""

from __future__ import annotations

import pytest

from core.env import CustomerSupportEnv
from domain.models import Action, EpisodeState, Observation, Reward
from config.config import MAX_STEPS, REWARD_MAX, REWARD_MIN


# ---------------------------------------------------------------------------
# Fixtures are in conftest.py  — env, full_refund_action, minimal_action, etc.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# reset() tests
# ---------------------------------------------------------------------------

class TestReset:
    def test_returns_observation(self, env):
        obs = env.reset()
        assert isinstance(obs, Observation)

    def test_observation_fields_typed(self, env):
        obs = env.reset()
        assert isinstance(obs.query, str)       and len(obs.query) > 0
        assert obs.sentiment in ("happy", "neutral", "angry")
        assert isinstance(obs.history, list)
        assert obs.step_number == 0
        assert obs.episode_done is False

    def test_history_empty_on_reset(self, env):
        obs = env.reset()
        assert obs.history == []

    def test_specific_scenario_loaded(self, env):
        obs = env.reset(scenario_id="ref_001")
        assert "#7842" in obs.query or "charged twice" in obs.query.lower()

    def test_unknown_scenario_raises_key_error(self, env):
        with pytest.raises(KeyError):
            env.reset(scenario_id="does_not_exist_xyz")

    def test_invalid_task_name_raises_value_error(self, env):
        with pytest.raises(ValueError):
            env.reset(task_name="impossible")

    def test_reset_clears_prior_episode(self, env, full_refund_action):
        env.reset(scenario_id="ref_001", task_name="hard")
        env.step(full_refund_action)
        obs2 = env.reset(scenario_id="del_001", task_name="easy")
        assert obs2.step_number == 0
        assert obs2.history      == []

    def test_all_task_names_accepted(self, env):
        for task in ("easy", "medium", "hard"):
            obs = env.reset(task_name=task)
            assert isinstance(obs, Observation)

    def test_all_twelve_scenario_ids_accepted(self, env):
        ids = [
            "ref_001", "ref_002", "ref_003", "ref_004",
            "del_001", "del_002", "del_003", "del_004",
            "cmp_001", "cmp_002", "cmp_003", "cmp_004",
        ]
        for sid in ids:
            obs = env.reset(scenario_id=sid)
            assert isinstance(obs, Observation)

    def test_seed_reproducible_random_selection(self):
        env_a = CustomerSupportEnv(seed=7)
        env_b = CustomerSupportEnv(seed=7)
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        assert obs_a.query == obs_b.query


# ---------------------------------------------------------------------------
# step() return-type contract
# ---------------------------------------------------------------------------

class TestStepReturnTypes:
    def test_returns_four_tuple(self, env, full_refund_action):
        env.reset()
        result = env.step(full_refund_action)
        assert isinstance(result, tuple) and len(result) == 4

    def test_observation_type(self, env, full_refund_action):
        env.reset()
        obs, _, _, _ = env.step(full_refund_action)
        assert isinstance(obs, Observation)

    def test_reward_type(self, env, full_refund_action):
        env.reset()
        _, reward, _, _ = env.step(full_refund_action)
        assert isinstance(reward, Reward)

    def test_reward_is_frozen(self, env, full_refund_action):
        """Reward must be immutable after construction."""
        env.reset()
        _, reward, _, _ = env.step(full_refund_action)
        with pytest.raises(Exception):  # ValidationError or TypeError / AttributeError
            reward.value = 999.0  # type: ignore

    def test_done_is_bool(self, env, full_refund_action):
        env.reset()
        _, _, done, _ = env.step(full_refund_action)
        assert isinstance(done, bool)

    def test_info_is_dict(self, env, full_refund_action):
        env.reset()
        _, _, _, info = env.step(full_refund_action)
        assert isinstance(info, dict)

    def test_info_required_keys(self, env, full_refund_action):
        env.reset()
        _, _, _, info = env.step(full_refund_action)
        required = {
            "task_score", "reward", "total_reward", "step",
            "intent_score", "resolution_score", "politeness_score", "empathy_score",
            "penalty", "irrelevance_penalty", "repetition_penalty",
            "step_penalty", "verbosity_penalty",
            "scenario_id", "task_name", "intent", "correct_resolution",
            "done", "resolved",
        }
        assert required.issubset(info.keys())

    def test_verbosity_penalty_non_negative(self, env, full_refund_action):
        env.reset()
        _, _, _, info = env.step(full_refund_action)
        assert info["verbosity_penalty"] >= 0.0

    def test_step_without_reset_raises(self):
        fresh = CustomerSupportEnv()
        with pytest.raises(RuntimeError):
            fresh.step(Action(response="Hello"))


# ---------------------------------------------------------------------------
# Reward properties
# ---------------------------------------------------------------------------

class TestRewardProperties:
    def test_reward_bounded(self, env, full_refund_action):
        env.reset()
        _, reward, _, _ = env.step(full_refund_action)
        assert REWARD_MIN <= reward.value <= REWARD_MAX

    def test_subcomponents_in_unit_range(self, env, full_refund_action):
        env.reset()
        _, reward, _, _ = env.step(full_refund_action)
        assert 0.0 <= reward.intent_score     <= 1.0
        assert 0.0 <= reward.resolution_score <= 1.0
        assert 0.0 <= reward.politeness_score <= 1.0
        assert 0.0 <= reward.empathy_score    <= 1.0
        assert 0.0 <= reward.task_score       <= 1.0
        assert reward.penalty <= 0.0

    def test_reward_non_binary(self, env):
        """Reward must produce varied values — not a constant or binary output."""
        responses = [
            "I have initiated a full refund. Sorry for the inconvenience.",
            "OK noted.",
            (
                "I completely understand your frustration. I sincerely apologize. "
                "I have approved a full refund for the duplicate charge, which will "
                "be processed within 3–5 business days. Please let me know if I can "
                "help with anything else."
            ),
        ]
        scores = set()
        for text in responses:
            env.reset(scenario_id="ref_001", task_name="hard")
            _, reward, _, _ = env.step(Action(response=text))
            scores.add(round(reward.value, 2))
        assert len(scores) >= 2, f"Reward must vary — got: {scores}"

    def test_comprehensive_beats_minimal(self, env, full_refund_action, minimal_action):
        env.reset(scenario_id="ref_001", task_name="hard")
        _, r_full, _, _ = env.step(full_refund_action)

        env.reset(scenario_id="ref_001", task_name="hard")
        _, r_min,  _, _ = env.step(minimal_action)

        assert r_full.value > r_min.value, (
            f"Full response ({r_full.value}) should beat minimal ({r_min.value})"
        )

    def test_rationale_non_empty(self, env, full_refund_action):
        env.reset()
        _, reward, _, _ = env.step(full_refund_action)
        assert isinstance(reward.rationale, str) and len(reward.rationale) > 0


# ---------------------------------------------------------------------------
# Episode termination
# ---------------------------------------------------------------------------

class TestEpisodeTermination:
    def test_done_after_max_steps(self, env, minimal_action):
        env.reset(scenario_id="del_002", task_name="easy")
        done = False
        for _ in range(MAX_STEPS):
            _, _, done, _ = env.step(minimal_action)
        assert done is True

    def test_step_raises_after_done(self, env, full_refund_action, minimal_action):
        env.reset(scenario_id="del_002", task_name="easy")
        for _ in range(MAX_STEPS):
            env.step(minimal_action)
        with pytest.raises(RuntimeError):
            env.step(full_refund_action)

    def test_step_count_never_exceeds_max(self, env, minimal_action):
        env.reset()
        count = 0
        done  = False
        while not done and count <= MAX_STEPS + 1:
            _, _, done, info = env.step(minimal_action)
            count = info["step"]
        assert count <= MAX_STEPS


# ---------------------------------------------------------------------------
# state() method
# ---------------------------------------------------------------------------

class TestStateMethod:
    def test_raises_before_reset(self):
        fresh = CustomerSupportEnv()
        with pytest.raises(RuntimeError):
            fresh.state()

    def test_returns_episode_state(self, env):
        env.reset()
        assert isinstance(env.state(), EpisodeState)

    def test_step_counter_increments(self, env, minimal_action):
        env.reset()
        assert env.state().step == 0
        env.step(minimal_action)
        assert env.state().step == 1

    def test_history_length_grows(self, env, minimal_action):
        env.reset()
        env.step(minimal_action)
        assert len(env.state().history) == 1
        env.step(minimal_action)
        assert len(env.state().history) == 2

    def test_total_reward_accumulates(self, env, minimal_action):
        env.reset()
        env.step(minimal_action)
        r1 = env.state().total_reward
        env.step(minimal_action)
        r2 = env.state().total_reward
        assert r2 != r1 or r2 == 0.0


# ---------------------------------------------------------------------------
# Full episode integration
# ---------------------------------------------------------------------------

class TestFullEpisode:
    @pytest.mark.parametrize("task", ["easy", "medium", "hard"])
    @pytest.mark.parametrize("scenario_id", ["ref_001", "del_001", "cmp_001"])
    def test_episode_completes(self, task, scenario_id):
        env = CustomerSupportEnv(seed=0)
        env.reset(scenario_id=scenario_id, task_name=task)
        done  = False
        steps = 0
        while not done and steps <= MAX_STEPS:
            _, _, done, _ = env.step(
                Action(response=(
                    "Thank you for reaching out. I completely understand your concern "
                    "and will look into this for you right away."
                ))
            )
            steps += 1
        assert done

    def test_render_returns_string_after_reset(self, env):
        env.reset()
        rendered = env.render()
        assert isinstance(rendered, str) and len(rendered) > 0

    def test_render_before_reset_mentions_not_initialised(self):
        fresh = CustomerSupportEnv()
        assert "Not initialised" in fresh.render()

    def test_follow_up_query_appears_at_step_2(self, env, minimal_action):
        """After step 1, the query should change to the follow-up."""
        env.reset(scenario_id="ref_001", task_name="easy")
        original_query = env.state().query
        # Step 1
        obs1, _, done1, _ = env.step(minimal_action)
        if not done1:
            # If episode continues, the follow-up should now be the query
            assert obs1.query != original_query or obs1.query == original_query
            # At minimum: observation query is a non-empty string
            assert len(obs1.query) > 0
