"""
core/state_manager.py — Immutable episode state transitions.

DESIGN DECISIONS
----------------
• build_initial_state() creates the first state from a scenario object
• advance() returns a NEW EpisodeState — the input is NEVER mutated
• RuntimeError is raised when advance() is called on a terminal state
• Follow-up query replacement only happens at step 1 and only when not done

PUBLIC API
----------
StateManager(scenario, task_name, max_steps)
    .build_initial_state()                                   -> EpisodeState
    .advance(state, response, reward, resolved, follow_up)   -> EpisodeState
    .is_terminal(state)                                      -> bool  (static)
"""

from __future__ import annotations

from typing import Optional

from domain.models import EpisodeState
from logic.tasks import Scenario

__all__ = ["StateManager"]


class StateManager:
    """
    Manages episode state transitions for CustomerSupportEnv.

    All advance() calls produce a fresh, immutable EpisodeState.
    The previous state is never mutated.
    """

    def __init__(
        self,
        scenario:  Scenario,
        task_name: str,
        max_steps: int,
    ) -> None:
        self._scenario  = scenario
        self._task_name = task_name
        self._max_steps = max_steps

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def build_initial_state(self) -> EpisodeState:
        """Return a fresh EpisodeState for the start of a new episode."""
        s = self._scenario
        return EpisodeState(
            scenario_id       = s.scenario_id,
            query             = s.query,
            sentiment         = s.sentiment,
            intent            = s.intent,
            correct_resolution = s.correct_resolution,
            history           = [],
            step              = 0,
            max_steps         = self._max_steps,
            done              = False,
            total_reward      = 0.0,
            rewards           = [],
            task_name         = self._task_name,
        )

    # ------------------------------------------------------------------
    # Transition
    # ------------------------------------------------------------------

    def advance(
        self,
        state:        EpisodeState,
        response:     str,
        reward_value: float,
        resolved:     bool,
        follow_up:    Optional[str] = None,
    ) -> EpisodeState:
        """
        Return a new EpisodeState reflecting one completed step.

        Parameters
        ----------
        state        : EpisodeState  — state BEFORE the step
        response     : str           — agent's response this step
        reward_value : float         — clamped scalar reward [-1, 1]
        resolved     : bool          — True when task_score >= task threshold
        follow_up    : str or None   — follow-up query to present (used at step 1)

        Returns
        -------
        EpisodeState  — new state object; input is not mutated

        Raises
        ------
        RuntimeError  — if called on an already-terminal state
        """
        if state.done:
            raise RuntimeError(
                "StateManager.advance() called on a terminal episode state. "
                "Call env.reset() to start a new episode before calling step()."
            )

        new_history = list(state.history) + [response]
        new_step    = state.step + 1
        new_rewards = list(state.rewards) + [reward_value]
        new_total   = state.total_reward + reward_value
        done        = new_step >= state.max_steps or resolved

        # Switch to follow-up query when episode continues past step 1
        new_query = state.query
        if not done and new_step == 1 and follow_up:
            new_query = follow_up

        return EpisodeState(
            scenario_id        = state.scenario_id,
            query              = new_query,
            sentiment          = state.sentiment,
            intent             = state.intent,
            correct_resolution = state.correct_resolution,
            history            = new_history,
            step               = new_step,
            max_steps          = state.max_steps,
            done               = done,
            total_reward       = round(new_total, 6),
            rewards            = new_rewards,
            task_name          = state.task_name,
        )

    # ------------------------------------------------------------------
    # Terminal predicate
    # ------------------------------------------------------------------

    @staticmethod
    def is_terminal(state: EpisodeState) -> bool:
        """Return True when the episode cannot accept further steps."""
        return state.done or state.step >= state.max_steps
