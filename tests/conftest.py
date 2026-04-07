"""
tests/conftest.py — Shared pytest fixtures for the full test suite.

All fixtures are module-scoped where safe (stateless domain objects)
or function-scoped (env and mutable objects).
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path for all test modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from core.env import CustomerSupportEnv
from domain.models import Action
from logic.tasks import SCENARIO_INDEX


# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env() -> CustomerSupportEnv:
    """A deterministically seeded environment (function-scoped — fresh each test)."""
    return CustomerSupportEnv(seed=42)


@pytest.fixture
def env_unseeded() -> CustomerSupportEnv:
    """An environment without a seed (for testing non-deterministic selection)."""
    return CustomerSupportEnv()


# ---------------------------------------------------------------------------
# Action fixtures  (module-scoped — stateless, safe to share)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def full_refund_action() -> Action:
    """
    A comprehensive refund response hitting intent, resolution, politeness,
    and empathy keywords — should score highly on all three task difficulties.
    """
    return Action(
        response=(
            "I completely understand how frustrating a duplicate charge must be, "
            "and I sincerely apologize for the inconvenience this has caused you. "
            "I have approved a full refund for the duplicate charge on your order, "
            "which will be processed and returned to your account within 3–5 business days. "
            "Please don't hesitate to reach out if there is anything else I can help you with."
        )
    )


@pytest.fixture(scope="module")
def full_tracking_action() -> Action:
    """A comprehensive delay/tracking response."""
    return Action(
        response=(
            "I sincerely apologize for the delay with your shipment. "
            "I completely understand how frustrating this must be. "
            "I can see your order is still in transit — here is your tracking number "
            "so you can track your order status directly: TRK-9901. "
            "The carrier shows an updated delivery estimate. "
            "Please let me know if there is anything else I can help you with."
        )
    )


@pytest.fixture(scope="module")
def full_escalation_action() -> Action:
    """A comprehensive complaint/escalation response."""
    return Action(
        response=(
            "I completely understand how frustrated you must be, and I sincerely "
            "apologize for the experience you have had. This is absolutely unacceptable "
            "and I am escalating this immediately to a senior specialist and manager who "
            "will prioritize your case and connect you with the right team within 24 hours. "
            "Please don't hesitate to reach out if there is anything else I can help you with."
        )
    )


@pytest.fixture(scope="module")
def minimal_action() -> Action:
    """A minimal, below-threshold action."""
    return Action(response="Thank you for contacting us.")


@pytest.fixture(scope="module")
def rude_action() -> Action:
    """A rude action that should score zero on politeness."""
    return Action(response="That's not our problem. Deal with it.")


# ---------------------------------------------------------------------------
# Scenario fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ref_scenario():
    return SCENARIO_INDEX["ref_001"]


@pytest.fixture(scope="module")
def del_scenario():
    return SCENARIO_INDEX["del_001"]


@pytest.fixture(scope="module")
def cmp_scenario():
    return SCENARIO_INDEX["cmp_001"]
