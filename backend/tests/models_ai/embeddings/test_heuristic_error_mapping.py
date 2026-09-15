from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingErrorCategory,
    HeuristicEmbeddingErrorMapper,
)


@pytest.fixture
def mapper() -> HeuristicEmbeddingErrorMapper:
    return HeuristicEmbeddingErrorMapper()


@pytest.mark.parametrize(
    "message",
    [
        "request timeout",
        "request timed out",
        "deadline exceeded",
        "TIMEOUT while embedding",
    ],
)
def test_timeout_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True
    assert result.confidence == 0.90


@pytest.mark.parametrize(
    "message",
    [
        "connection refused",
        "connection reset by peer",
        "network unreachable",
        "network error",
        "connection error",
        "DNS error",
    ],
)
def test_network_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.retryable is True


@pytest.mark.parametrize(
    "message",
    [
        "rate limit exceeded",
        "too many requests",
        "quota exceeded",
        "request rate exceeded",
        "request throttled",
    ],
)
def test_rate_limit_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.retryable is True
    assert result.confidence == 0.85


@pytest.mark.parametrize(
    "message",
    [
        "service unavailable",
        "temporarily unavailable",
        "server unavailable",
        "backend unavailable",
        "service overloaded",
    ],
)
def test_service_unavailable_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert (
        result.category
        == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    )
    assert result.retryable is True


@pytest.mark.parametrize(
    "message",
    [
        "authentication failed",
        "invalid API key",
        "API key invalid",
        "credentials invalid",
        "missing API key",
    ],
)
def test_authentication_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert result.retryable is False


@pytest.mark.parametrize(
    "message",
    [
        "authorization failed",
        "permission denied",
        "access denied",
        "forbidden",
        "not authorized",
    ],
)
def test_authorization_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHORIZATION
    assert result.retryable is False


@pytest.mark.parametrize(
    "message",
    [
        "model not found",
        "model does not exist",
        "unknown model",
        "invalid model",
        "model unavailable",
    ],
)
def test_model_not_found_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.MODEL_NOT_FOUND
    assert result.retryable is False


@pytest.mark.parametrize(
    "message",
    [
        "invalid input",
        "invalid request",
        "invalid text",
        "validation error",
        "malformed request",
    ],
)
def test_input_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.INPUT
    assert result.retryable is False


@pytest.mark.parametrize(
    "message",
    [
        "configuration error",
        "configuration invalid",
        "misconfigured",
        "missing configuration",
        "invalid configuration",
    ],
)
def test_configuration_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.CONFIGURATION
    assert result.retryable is False


@pytest.mark.parametrize(
    "message",
    [
        "dimension mismatch",
        "invalid embedding dimension",
        "embedding dimension is incorrect",
        "vector dimension mismatch",
    ],
)
def test_dimension_messages_are_classified(
    mapper: HeuristicEmbeddingErrorMapper,
    message: str,
) -> None:
    result = mapper.classify(Exception(message))

    assert result is not None
    assert result.category == EmbeddingErrorCategory.DIMENSION
    assert result.retryable is False
    assert result.confidence == 0.95


def test_unknown_message_returns_none(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception("something completely unexpected happened")
    )

    assert result is None


def test_empty_message_returns_none(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(Exception(""))

    assert result is None


def test_whitespace_message_returns_none(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(Exception("   "))

    assert result is None


def test_matching_is_case_insensitive(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception("CONNECTION REFUSED")
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK


def test_extra_whitespace_is_normalized(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception("  request   timed    out  ")
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_exception_type_timeout_is_classified(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        TimeoutError()
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True


def test_exception_type_connection_error_is_classified(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        ConnectionError()
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.retryable is True


def test_message_signal_takes_precedence_over_generic_exception_type(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        TimeoutError("authentication failed")
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHENTICATION


def test_structured_status_code_is_ignored(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    class StructuredError(Exception):
        status_code = 429

    result = mapper.classify(
        StructuredError("unrelated error")
    )

    assert result is None


def test_heuristic_source_is_recorded(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception("connection refused")
    )

    assert result is not None
    assert result.metadata["classification_source"] == (
        "heuristic"
    )


def test_message_signal_is_recorded(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception("connection refused")
    )

    assert result is not None
    assert result.metadata["signal"] == "message"


def test_exception_type_signal_is_recorded(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        TimeoutError()
    )

    assert result is not None
    assert result.metadata["signal"] == "exception_type"


def test_classification_is_deterministic(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    error = Exception("request timed out")

    first = mapper.classify(error)
    second = mapper.classify(error)

    assert first == second


def test_mapper_does_not_mutate_exception(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    error = Exception("connection refused")

    original_message = str(error)

    mapper.classify(error)

    assert str(error) == original_message


def test_unicode_message_without_matching_signal_returns_none(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception("埋め込み処理に失敗しました")
    )

    assert result is None


def test_api_key_value_is_not_copied_to_metadata(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        Exception(
            "invalid API key secret-value-12345"
        )
    )

    assert result is not None
    assert "secret-value-12345" not in str(
        result.metadata
    )


def test_full_message_is_not_copied_to_metadata(
    mapper: HeuristicEmbeddingErrorMapper,
) -> None:
    message = (
        "connection refused with sensitive-value-12345"
    )

    result = mapper.classify(Exception(message))

    assert result is not None
    assert message not in str(result.metadata)