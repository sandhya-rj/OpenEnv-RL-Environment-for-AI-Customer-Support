"""
core/env.py — OpenEnv-compliant AI Customer Support RL Environment.

PUBLIC API (OpenEnv specification)
------------------------------------
  reset(scenario_id=None, task_name="hard") -> Observation
  step(action: Action)                      -> (Observation, Reward, bool, dict)
  state()                                    -> EpisodeState

ADDITIONAL HELPERS
------------------
  render()  -> str   human-readable debug view
  seed      -> int | None

ARCHITECTURE
------------
env.py is the THIN orchestration layer. It delegates to:
  core/state_manager.py  — immutable episode state transitions
  logic/graders.py       — score_response() returns GraderOutput (no circular import)
  logic/reward.py        — compute_reward() + combine_grader_scores()
  logic/tasks.py         — task registry + scenario catalogue
  config/config.py       — all constants
  utils/validation_utils — all boundary guards
  domain/models.py       — typed input/output models

DEPENDENCY ORDER (no circular imports)
---------------------------------------
  config → domain → utils → logic → core
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from config.config import ENV_NAME, MAX_STEPS, TASK_THRESHOLDS, TASK_WEIGHTS
from core.state_manager import StateManager
from domain.models import Action, EpisodeState, Observation, Reward
from logic.graders import score_response
from logic.reward import combine_grader_scores, compute_reward
from logic.tasks import SCENARIO_INDEX, SCENARIOS, Task, get_task
from utils.validation_utils import (
    validate_action,
    validate_env_initialized,
    validate_scenario_id,
    validate_step_not_done,
    validate_task_name,
)

__all__ = ["CustomerSupportEnv"]


class CustomerSupportEnv:
    """
    OpenEnv reinforcement learning environment for AI customer support.

    Episode lifecycle
    -----------------
    1. Call reset()  → Initial Observation
    2. Call step()   → (Observation, Reward, done, info) each turn
    3. Repeat until done=True (max_steps exhausted or task_score >= threshold)

    Reward
    ------
    Continuous float in [-1.0, 1.0]:
      reward = task_score − (irrelevance + repetition + step + verbosity) penalties
    """

    ENV_NAME:  str = ENV_NAME
    MAX_STEPS: int = MAX_STEPS

    def __init__(self, seed: Optional[int] = None) -> None:
        self._seed:          Optional[int]          = seed
        self._rng:           random.Random          = random.Random(seed)
        self._state:         Optional[EpisodeState] = None
        self._task:          Optional[Task]         = None
        self._state_manager: Optional[StateManager] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def seed(self) -> Optional[int]:
        return self._seed

    # ------------------------------------------------------------------
    # OpenEnv core API
    # ------------------------------------------------------------------

    def reset(
        self,
        scenario_id: Optional[str] = None,
        task_name:   str           = "hard",
    ) -> Observation:
        """
        Start a new episode and return the initial Observation.

        Parameters
        ----------
        scenario_id : str or None
            Specific scenario by ID; None = random selection (reproducible via seed).
        task_name : str
            "easy" | "medium" | "hard"

        Raises
        ------
        KeyError   — unknown scenario_id
        ValueError — unknown task_name
        """
        validate_task_name(task_name, tuple(TASK_THRESHOLDS.keys()))
        validate_scenario_id(scenario_id, SCENARIO_INDEX)

        scenario = (
            SCENARIO_INDEX[scenario_id]
            if scenario_id is not None
            else self._rng.choice(SCENARIOS)
        )

        self._task          = get_task(task_name)
        self._state_manager = StateManager(scenario, task_name, self.MAX_STEPS)
        self._state         = self._state_manager.build_initial_state()

        return self._build_observation()

    def step(
        self,
        action: Action,
    ) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the environment by one interaction step.

        Parameters
        ----------
        action : Action  — agent's text response

        Returns
        -------
        observation : Observation
        reward      : Reward     (frozen Pydantic model)
        done        : bool
        info        : dict       complete per-step metrics

        Raises
        ------
        RuntimeError — reset() not called, or episode already done
        """
        validate_env_initialized(self._state)
        validate_step_not_done(self._state.done)

        assert self._task is not None and self._state_manager is not None

        response = validate_action(action.response)
        scenario = SCENARIO_INDEX[self._state.scenario_id]
        weights  = TASK_WEIGHTS[self._state.task_name]

        # ── Run all six graders (no circular import) ─────────────────────
        graded = score_response(
            response   = response,
            query      = self._state.query,
            intent     = self._state.intent,
            resolution = self._state.correct_resolution,
            history    = self._state.history,
        )

        # ── Weighted task score ───────────────────────────────────────────
        task_score = combine_grader_scores(
            weights          = weights,
            intent_score     = graded.intent_score,
            resolution_score = graded.resolution_score,
            politeness_score = graded.politeness_score,
            empathy_score    = graded.empathy_score,
        )
        # Also evaluate via task.evaluate() for consistency with task subclasses
        # (task.evaluate() is the authoritative source for early termination)
        task_score_authoritative = self._task.evaluate(
            response, scenario, step=self._state.step
        )
        resolved = task_score_authoritative >= self._task.score_threshold

        # ── Reward computation ────────────────────────────────────────────
        rr = compute_reward(
            task_score              = task_score_authoritative,
            irrelevance_penalty_raw = graded.irrelevance_penalty,
            repetition_penalty_raw  = graded.repetition_penalty,
            step                    = self._state.step,
            resolved                = resolved,
            response                = response,
        )

        reward = Reward(
            value            = rr.value,
            intent_score     = graded.intent_score,
            resolution_score = graded.resolution_score,
            politeness_score = graded.politeness_score,
            empathy_score    = graded.empathy_score,
            task_score       = task_score_authoritative,
            penalty          = rr.total_penalty,
            rationale        = rr.rationale,
        )

        # ── State transition ──────────────────────────────────────────────
        self._state = self._state_manager.advance(
            state        = self._state,
            response     = response,
            reward_value = rr.value,
            resolved     = resolved,
            follow_up    = scenario.follow_up or None,
        )

        done = self._state.done
        obs  = self._build_observation()

        info: Dict[str, Any] = {
            "task_score":          task_score_authoritative,
            "reward":              rr.value,
            "total_reward":        self._state.total_reward,
            "step":                self._state.step,
            "intent_score":        graded.intent_score,
            "resolution_score":    graded.resolution_score,
            "politeness_score":    graded.politeness_score,
            "empathy_score":       graded.empathy_score,
            "penalty":             rr.total_penalty,
            "irrelevance_penalty": rr.irrelevance_penalty,
            "repetition_penalty":  rr.repetition_penalty,
            "step_penalty":        rr.step_penalty,
            "verbosity_penalty":   rr.verbosity_penalty,
            "scenario_id":         self._state.scenario_id,
            "task_name":           self._state.task_name,
            "intent":              self._state.intent,
            "correct_resolution":  self._state.correct_resolution,
            "done":                done,
            "resolved":            resolved,
        }

        return obs, reward, done, info

    def state(self) -> EpisodeState:
        """
        Return the current internal episode state.

        Raises RuntimeError if reset() has not been called.
        """
        validate_env_initialized(self._state)
        return self._state

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> str:
        """Return a formatted human-readable summary of the current state."""
        if self._state is None:
            return "[CustomerSupportEnv] Not initialised — call reset() first."

        lines: List[str] = [
            "=" * 64,
            f"  {self.ENV_NAME}",
            "=" * 64,
            f"  Scenario  : {self._state.scenario_id}",
            f"  Task      : {self._state.task_name}",
            f"  Step      : {self._state.step} / {self._state.max_steps}",
            f"  Done      : {self._state.done}",
            f"  Sentiment : {self._state.sentiment}",
            f"  Intent    : {self._state.intent}",
            f"  Resolution: {self._state.correct_resolution}",
            "─" * 64,
            "  Current query:",
            f"    {self._state.query}",
        ]
        if self._state.history:
            lines += ["─" * 64, "  Conversation history:"]
            for i, resp in enumerate(self._state.history, 1):
                preview = resp[:120].replace("\n", " ")
                lines.append(
                    f"    [{i}] {preview}{'...' if len(resp) > 120 else ''}"
                )
        lines += [
            "─" * 64,
            f"  Total reward : {self._state.total_reward:.4f}",
            f"  Step rewards : {[round(r, 3) for r in self._state.rewards]}",
            "=" * 64,
        ]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        assert self._state is not None
        return Observation(
            query        = self._state.query,
            sentiment    = self._state.sentiment,
            history      = list(self._state.history),
            step_number  = self._state.step,
            intent       = self._state.intent,
            episode_done = self._state.done,
        )
