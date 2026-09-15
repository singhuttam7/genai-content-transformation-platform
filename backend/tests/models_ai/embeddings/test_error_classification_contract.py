from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
)


def test_classification_requires_category() -> None:
    with pytest.raises(ValidationError):
        EmbeddingErrorClassification(
            retryable=False,
            confidence=1.0,
        )


def test_classification_requires_retryable() -> None:
    with pytest.raises(ValidationError):
        EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            confidence=0.5,
        )


def test_classification_requires_confidence() -> None:
    with pytest.raises(ValidationError):
        EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            retryable=False,
        )


def test_classification_accepts_valid_values() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.RATE_LIMIT,
        retryable=True,
        confidence=1.0,
        reason="Provider reported a rate limit.",
    )

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.retryable is True
    assert result.confidence == 1.0
    assert result.reason == "Provider reported a rate limit."
    assert result.metadata == {}


def test_confidence_accepts_zero() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.UNKNOWN,
        retryable=False,
        confidence=0.0,
    )

    assert result.confidence == 0.0


def test_confidence_accepts_one() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.UNKNOWN,
        retryable=False,
        confidence=1.0,
    )

    assert result.confidence == 1.0


def test_confidence_rejects_negative_value() -> None:
    with pytest.raises(ValidationError):
        EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            retryable=False,
            confidence=-0.01,
        )


def test_confidence_rejects_value_above_one() -> None:
    with pytest.raises(ValidationError):
        EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            retryable=False,
            confidence=1.01,
        )


def test_reason_is_optional() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.NETWORK,
        retryable=True,
        confidence=0.8,
    )

    assert result.reason is None


def test_metadata_defaults_to_empty_dictionary() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.UNKNOWN,
        retryable=False,
        confidence=0.0,
    )

    assert result.metadata == {}


def test_nested_metadata_is_supported() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.RATE_LIMIT,
        retryable=True,
        confidence=1.0,
        metadata={
            "status_code": 429,
            "provider": "example",
            "details": {
                "retry_after": 30,
            },
        },
    )

    assert result.metadata == {
        "status_code": 429,
        "provider": "example",
        "details": {
            "retry_after": 30,
        },
    }


def test_unicode_reason_is_supported() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.INPUT,
        retryable=False,
        confidence=0.9,
        reason="इनपुट अमान्य है",
    )

    assert result.reason == "इनपुट अमान्य है"


def test_unknown_fields_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            retryable=False,
            confidence=0.0,
            unexpected_field="value",
        )


def test_classification_is_immutable() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.NETWORK,
        retryable=True,
        confidence=0.9,
    )

    with pytest.raises(ValidationError):
        result.retryable = False


def test_json_serialization_uses_category_value() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        retryable=True,
        confidence=1.0,
    )

    payload = result.model_dump(mode="json")

    assert payload["category"] == "service_unavailable"
    assert isinstance(payload["category"], str)


def test_json_round_trip_preserves_classification() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.TIMEOUT,
        retryable=True,
        confidence=0.85,
        reason="Provider request timed out.",
        metadata={
            "timeout_seconds": 30,
        },
    )

    serialized = json.dumps(
        result.model_dump(mode="json")
    )

    restored = EmbeddingErrorClassification.model_validate(
        json.loads(serialized)
    )

    assert restored == result


def test_model_dump_json_round_trip() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.MODEL_NOT_FOUND,
        retryable=False,
        confidence=1.0,
        reason="Requested model does not exist.",
    )

    serialized = result.model_dump_json()

    restored = EmbeddingErrorClassification.model_validate_json(
        serialized
    )

    assert restored == result


def test_classification_does_not_require_provider_information() -> None:
    result = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.UNKNOWN,
        retryable=False,
        confidence=0.0,
    )

    assert result.category == EmbeddingErrorCategory.UNKNOWN
    assert result.metadata == {}


def test_boolean_retryable_is_preserved() -> None:
    retryable = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.TIMEOUT,
        retryable=True,
        confidence=1.0,
    )

    non_retryable = EmbeddingErrorClassification(
        category=EmbeddingErrorCategory.INPUT,
        retryable=False,
        confidence=1.0,
    )

    assert retryable.retryable is True
    assert non_retryable.retryable is False