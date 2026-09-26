from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.orchestration.failure_policy import WorkflowFailurePolicy


def test_default_policy_disables_retry() -> None:
    policy = WorkflowFailurePolicy()

    assert policy.retry_failed_agents is False
    assert policy.max_attempts == 1


def test_retry_policy_can_allow_multiple_attempts() -> None:
    policy = WorkflowFailurePolicy(
        retry_failed_agents=True,
        max_attempts=3,
    )

    assert policy.retry_failed_agents is True
    assert policy.max_attempts == 3


def test_retry_policy_allows_single_attempt() -> None:
    policy = WorkflowFailurePolicy(
        retry_failed_agents=True,
        max_attempts=1,
    )

    assert policy.retry_failed_agents is True
    assert policy.max_attempts == 1


def test_retry_disabled_requires_one_attempt() -> None:
    with pytest.raises(
        ValueError,
        match="max_attempts must be 1",
    ):
        WorkflowFailurePolicy(
            retry_failed_agents=False,
            max_attempts=2,
        )


def test_zero_attempts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=0,
        )


def test_negative_attempts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=-1,
        )


def test_extra_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        WorkflowFailurePolicy(
            retry_failed_agents=True,
            max_attempts=2,
            unexpected="value",
        )


def test_policy_is_deterministic() -> None:
    first = WorkflowFailurePolicy(
        retry_failed_agents=True,
        max_attempts=3,
    )

    second = WorkflowFailurePolicy(
        retry_failed_agents=True,
        max_attempts=3,
    )

    assert first == second
    assert first.model_dump() == second.model_dump()