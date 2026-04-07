"""
utils/validation_utils.py — Input validation and environment lifecycle guards.

All functions raise typed, human-readable exceptions with full context.
They are pure (no side-effects) and return the validated value when possible
so callers can use them inline.

Public API
----------
is_non_empty_string(value)                              -> bool
validate_action(response)                               -> str
validate_task_name(name, valid_names)                   -> str
validate_scenario_id(scenario_id, scenario_index)       -> None
validate_step_not_done(done)                            -> None
validate_env_initialized(state)                         -> None
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, Optional, Tuple

__all__ = [
    "is_non_empty_string",
    "validate_action",
    "validate_task_name",
    "validate_scenario_id",
    "validate_step_not_done",
    "validate_env_initialized",
]


# ---------------------------------------------------------------------------
# Primitive check
# ---------------------------------------------------------------------------

def is_non_empty_string(value: Any) -> bool:
    """Return True iff value is a non-empty, non-blank string."""
    return isinstance(value, str) and bool(value.strip())


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

def validate_action(response: Any) -> str:
    """
    Validate and normalise an agent response.

    Returns the stripped, non-empty response string.

    Raises
    ------
    TypeError   response is not a string
    ValueError  response is empty or whitespace-only
    """
    if not isinstance(response, str):
        raise TypeError(
            f"Action.response must be str, got {type(response).__name__!r}. "
            "The agent must produce a text response."
        )
    stripped = response.strip()
    if not stripped:
        raise ValueError(
            "Action.response must not be empty or whitespace-only. "
            "The agent must produce a non-blank text response."
        )
    return stripped


# ---------------------------------------------------------------------------
# Task-name validation
# ---------------------------------------------------------------------------

def validate_task_name(
    name: Any,
    valid_names: Tuple[str, ...],
) -> str:
    """
    Validate that name is a recognised task identifier.

    Returns the validated name.

    Raises
    ------
    TypeError   name is not a string
    ValueError  name is not in valid_names
    """
    if not isinstance(name, str):
        raise TypeError(
            f"task_name must be str, got {type(name).__name__!r}."
        )
    if name not in valid_names:
        raise ValueError(
            f"Unknown task_name {name!r}. "
            f"Valid options: {sorted(valid_names)}"
        )
    return name


# ---------------------------------------------------------------------------
# Scenario-ID validation
# ---------------------------------------------------------------------------

def validate_scenario_id(
    scenario_id: Optional[str],
    scenario_index: Dict[str, Any],
) -> None:
    """
    Validate that scenario_id exists in the catalogue (when provided).

    Does nothing when scenario_id is None (caller picks randomly).

    Raises
    ------
    KeyError   scenario_id not found in scenario_index
    """
    if scenario_id is None:
        return
    if scenario_id not in scenario_index:
        raise KeyError(
            f"Unknown scenario_id {scenario_id!r}. "
            f"Valid IDs: {sorted(scenario_index.keys())}"
        )


# ---------------------------------------------------------------------------
# Episode lifecycle guards
# ---------------------------------------------------------------------------

def validate_step_not_done(done: bool) -> None:
    """
    Guard against stepping on a terminated episode.

    Raises
    ------
    RuntimeError   episode is already done
    """
    if done:
        raise RuntimeError(
            "Cannot call step() on a terminated episode. "
            "Call reset() to start a new episode."
        )


def validate_env_initialized(state: Optional[Any]) -> None:
    """
    Guard against calling state() or step() before reset().

    Raises
    ------
    RuntimeError   state is None (env not initialised)
    """
    if state is None:
        raise RuntimeError(
            "Environment has not been initialised. "
            "Call reset() before calling step() or state()."
        )
