from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingConfigurationError,
    EmbeddingDimensionError,
    EmbeddingError,
    EmbeddingErrorCategory,
    EmbeddingInputError,
    EmbeddingProviderError,
)


def test_error_categories_are_stable() -> None:
    assert EmbeddingErrorCategory.CONFIGURATION.value == "configuration"
    assert EmbeddingErrorCategory.INPUT.value == "input"
    assert EmbeddingErrorCategory.AUTHENTICATION.value == "authentication"
    assert EmbeddingErrorCategory.AUTHORIZATION.value == "authorization"
    assert EmbeddingErrorCategory.RATE_LIMIT.value == "rate_limit"
    assert EmbeddingErrorCategory.TIMEOUT.value == "timeout"
    assert EmbeddingErrorCategory.NETWORK.value == "network"
    assert (
        EmbeddingErrorCategory.SERVICE_UNAVAILABLE.value
        == "service_unavailable"
    )
    assert (
        EmbeddingErrorCategory.MODEL_NOT_FOUND.value
        == "model_not_found"
    )
    assert EmbeddingErrorCategory.DIMENSION.value == "dimension"
    assert EmbeddingErrorCategory.BATCH.value == "batch"
    assert EmbeddingErrorCategory.UNKNOWN.value == "unknown"


def test_configuration_error_has_configuration_category() -> None:
    error = EmbeddingConfigurationError("invalid configuration")

    assert error.category == EmbeddingErrorCategory.CONFIGURATION


def test_input_error_has_input_category() -> None:
    error = EmbeddingInputError("invalid input")

    assert error.category == EmbeddingErrorCategory.INPUT


def test_dimension_error_has_dimension_category() -> None:
    error = EmbeddingDimensionError("dimension mismatch")

    assert error.category == EmbeddingErrorCategory.DIMENSION


def test_batch_error_has_batch_category() -> None:
    error = EmbeddingBatchError(
        "batch failed",
        batch_index=2,
    )

    assert error.category == EmbeddingErrorCategory.BATCH


def test_provider_error_defaults_to_unknown() -> None:
    error = EmbeddingProviderError(
        "provider failed",
    )

    assert error.category == EmbeddingErrorCategory.UNKNOWN


@pytest.mark.parametrize(
    "category",
    [
        EmbeddingErrorCategory.AUTHENTICATION,
        EmbeddingErrorCategory.AUTHORIZATION,
        EmbeddingErrorCategory.RATE_LIMIT,
        EmbeddingErrorCategory.TIMEOUT,
        EmbeddingErrorCategory.NETWORK,
        EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        EmbeddingErrorCategory.MODEL_NOT_FOUND,
        EmbeddingErrorCategory.UNKNOWN,
    ],
)
def test_provider_error_accepts_provider_independent_categories(
    category: EmbeddingErrorCategory,
) -> None:
    error = EmbeddingProviderError(
        "provider failure",
        category=category,
    )

    assert error.category == category


def test_provider_error_preserves_message() -> None:
    error = EmbeddingProviderError(
        "provider connection timed out",
        category=EmbeddingErrorCategory.TIMEOUT,
    )

    assert str(error) == "provider connection timed out"


def test_provider_error_is_embedding_error() -> None:
    error = EmbeddingProviderError("provider failure")

    assert isinstance(error, EmbeddingError)


def test_all_embedding_errors_are_embedding_errors() -> None:
    errors = [
        EmbeddingConfigurationError("configuration"),
        EmbeddingInputError("input"),
        EmbeddingProviderError("provider"),
        EmbeddingDimensionError("dimension"),
        EmbeddingBatchError("batch"),
    ]

    assert all(
        isinstance(error, EmbeddingError)
        for error in errors
    )


def test_provider_error_category_can_be_rate_limit() -> None:
    error = EmbeddingProviderError(
        "rate limit exceeded",
        category=EmbeddingErrorCategory.RATE_LIMIT,
    )

    assert error.category == EmbeddingErrorCategory.RATE_LIMIT


def test_provider_error_category_can_be_timeout() -> None:
    error = EmbeddingProviderError(
        "provider timeout",
        category=EmbeddingErrorCategory.TIMEOUT,
    )

    assert error.category == EmbeddingErrorCategory.TIMEOUT


def test_provider_error_category_can_be_authentication() -> None:
    error = EmbeddingProviderError(
        "invalid credentials",
        category=EmbeddingErrorCategory.AUTHENTICATION,
    )

    assert error.category == EmbeddingErrorCategory.AUTHENTICATION


def test_provider_error_category_can_be_model_not_found() -> None:
    error = EmbeddingProviderError(
        "embedding model not found",
        category=EmbeddingErrorCategory.MODEL_NOT_FOUND,
    )

    assert error.category == EmbeddingErrorCategory.MODEL_NOT_FOUND


def test_provider_error_category_can_be_service_unavailable() -> None:
    error = EmbeddingProviderError(
        "service unavailable",
        category=EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
    )

    assert (
        error.category
        == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    )


def test_provider_error_category_is_provider_independent() -> None:
    error = EmbeddingProviderError(
        "temporary provider failure",
        category=EmbeddingErrorCategory.NETWORK,
    )

    assert error.category == EmbeddingErrorCategory.NETWORK


def test_error_category_is_a_string_enum() -> None:
    category = EmbeddingErrorCategory.TIMEOUT

    assert isinstance(category, str)
    assert category.value == "timeout"