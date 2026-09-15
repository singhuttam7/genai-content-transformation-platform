from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
)
from app.models_ai.embeddings.retry import (
    EmbeddingRetryPolicy,
    RetryDecision,
)


@pytest.fixture
def policy() -> EmbeddingRetryPolicy:
    return EmbeddingRetryPolicy()


def make_classification(
    category: EmbeddingErrorCategory,
    *,
    retryable: bool = False,
) -> EmbeddingErrorClassification:
    return EmbeddingErrorClassification(
        category=category,
        retryable=retryable,
        confidence=1.0,
        reason="Test classification.",
    )


@pytest.mark.parametrize(
    "category",
    [
        EmbeddingErrorCategory.RATE_LIMIT,
        EmbeddingErrorCategory.TIMEOUT,
        EmbeddingErrorCategory.NETWORK,
        EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
    ],
)
def test_transient_categories_are_retryable(
    policy: EmbeddingRetryPolicy,
    category: EmbeddingErrorCategory,
) -> None:
    classification = make_classification(category)

    decision = policy.evaluate(classification)

    assert isinstance(decision, RetryDecision)
    assert decision.retryable is True
    assert decision.category == category


@pytest.mark.parametrize(
    "category",
    [
        EmbeddingErrorCategory.AUTHENTICATION,
        EmbeddingErrorCategory.AUTHORIZATION,
        EmbeddingErrorCategory.MODEL_NOT_FOUND,
        EmbeddingErrorCategory.INPUT,
        EmbeddingErrorCategory.CONFIGURATION,
        EmbeddingErrorCategory.DIMENSION,
        EmbeddingErrorCategory.BATCH,
        EmbeddingErrorCategory.UNKNOWN,
    ],
)
def test_permanent_or_unknown_categories_are_not_retryable(
    policy: EmbeddingRetryPolicy,
    category: EmbeddingErrorCategory,
) -> None:
    classification = make_classification(category)

    decision = policy.evaluate(classification)

    assert decision.retryable is False
    assert decision.category == category


def test_policy_is_independent_of_classification_retryable_field(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.AUTHENTICATION,
        retryable=True,
    )

    decision = policy.evaluate(classification)

    assert classification.retryable is True
    assert decision.retryable is False


def test_policy_can_override_false_retryable_for_transient_category(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.TIMEOUT,
        retryable=False,
    )

    decision = policy.evaluate(classification)

    assert classification.retryable is False
    assert decision.retryable is True


def test_rate_limit_reason_is_deterministic(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.RATE_LIMIT,
    )

    first = policy.evaluate(classification)
    second = policy.evaluate(classification)

    assert first == second


def test_non_retryable_reason_is_deterministic(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.MODEL_NOT_FOUND,
    )

    first = policy.evaluate(classification)
    second = policy.evaluate(classification)

    assert first == second


def test_retryable_reason_identifies_transient_policy(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.NETWORK,
    )

    decision = policy.evaluate(classification)

    assert "transient" in decision.reason.lower()
    assert "network" in decision.reason.lower()


def test_non_retryable_reason_identifies_policy(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.AUTHORIZATION,
    )

    decision = policy.evaluate(classification)

    assert "non-retryable" in decision.reason.lower()
    assert "authorization" in decision.reason.lower()


def test_policy_metadata_contains_version(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.TIMEOUT,
    )

    decision = policy.evaluate(classification)

    assert decision.metadata["policy"] == "embedding_default"
    assert decision.metadata["policy_version"] == "1.0"


def test_unknown_is_conservative(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.UNKNOWN,
        retryable=True,
    )

    decision = policy.evaluate(classification)

    assert decision.retryable is False


def test_decision_is_immutable(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.NETWORK,
    )

    decision = policy.evaluate(classification)

    with pytest.raises(Exception):
        decision.retryable = False


def test_decision_rejects_invalid_category() -> None:
    with pytest.raises(Exception):
        RetryDecision(
            retryable=True,
            category="not-a-real-category",
            reason="Invalid category.",
        )


def test_decision_requires_reason() -> None:
    with pytest.raises(Exception):
        RetryDecision(
            retryable=True,
            category=EmbeddingErrorCategory.NETWORK,
            reason="",
        )


def test_policy_does_not_mutate_classification(
    policy: EmbeddingRetryPolicy,
) -> None:
    classification = make_classification(
        EmbeddingErrorCategory.TIMEOUT,
    )

    before = classification.model_dump()

    policy.evaluate(classification)

    after = classification.model_dump()

    assert after == before


def test_each_decision_has_independent_metadata() -> None:
    policy = EmbeddingRetryPolicy()

    classification = make_classification(
        EmbeddingErrorCategory.NETWORK,
    )

    first = policy.evaluate(classification)
    second = policy.evaluate(classification)

    assert first.metadata == second.metadata
    assert first.metadata is not second.metadata