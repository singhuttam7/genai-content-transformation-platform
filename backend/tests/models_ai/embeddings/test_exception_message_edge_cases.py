from __future__ import annotations

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
)
from app.models_ai.embeddings.heuristic_classification import (
    HeuristicEmbeddingErrorMapper,
)


def test_empty_message_returns_none() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception("")

    result = mapper.classify(error)

    assert result is None


def test_whitespace_only_message_returns_none() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception("   \t\n   ")

    result = mapper.classify(error)

    assert result is None


def test_message_with_newlines_is_normalized() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "request\n\t timed \n out"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_message_with_excessive_whitespace_is_normalized() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "connection     refused"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK


def test_timeout_matching_is_case_insensitive() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "TIMEOUT",
        "Timeout",
        "timeout",
        "TiMeOuT",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is not None
        assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_network_matching_is_case_insensitive() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    result = mapper.classify(
        Exception("CONNECTION REFUSED")
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK


def test_rate_limit_matching_handles_punctuation() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    result = mapper.classify(
        Exception("Error: rate-limit exceeded!")
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.RATE_LIMIT


def test_service_unavailable_matching_handles_punctuation() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    result = mapper.classify(
        Exception(
            "[503] SERVICE UNAVAILABLE!"
        )
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE


def test_authentication_matching_handles_api_key_variants() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "invalid api key",
        "invalid api-key",
        "invalid api_key",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is not None
        assert result.category == EmbeddingErrorCategory.AUTHENTICATION


def test_authorization_matching_handles_common_phrasing() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "permission denied",
        "access denied",
        "not authorized",
        "forbidden",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is not None
        assert result.category == EmbeddingErrorCategory.AUTHORIZATION


def test_model_not_found_matching_is_case_insensitive() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    result = mapper.classify(
        Exception("MODEL NOT FOUND")
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.MODEL_NOT_FOUND


def test_input_validation_matching() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "invalid input",
        "invalid request",
        "invalid text",
        "validation error",
        "malformed request",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is not None
        assert result.category == EmbeddingErrorCategory.INPUT


def test_configuration_matching() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "configuration error",
        "configuration invalid",
        "misconfigured",
        "missing configuration",
        "invalid configuration",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is not None
        assert result.category == EmbeddingErrorCategory.CONFIGURATION


def test_dimension_matching() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "dimension mismatch",
        "invalid embedding dimension",
        "embedding dimension is invalid",
        "vector dimension mismatch",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is not None
        assert result.category == EmbeddingErrorCategory.DIMENSION


def test_unicode_message_with_known_signal_is_classified() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "请求超时 — timeout — 请求失败"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_multilingual_message_with_known_signal_is_classified() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "नेटवर्क त्रुटि — connection refused"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK


def test_unicode_without_known_signal_returns_none() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "请求失败 — कोई ज्ञात त्रुटि संकेत नहीं"
    )

    result = mapper.classify(error)

    assert result is None


def test_very_large_message_with_signal_is_classified() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    message = (
        "x" * 1_000_000
        + " timeout "
        + "y" * 1_000_000
    )

    result = mapper.classify(
        Exception(message)
    )

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_repeated_known_signal_is_deterministic() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    message = (
        "timeout timeout timeout "
        "timeout timeout"
    )

    first = mapper.classify(
        Exception(message)
    )
    second = mapper.classify(
        Exception(message)
    )

    assert first is not None
    assert second is not None

    assert first == second


def test_timeout_inside_identifier_does_not_match() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "timeout_value was configured"
    )

    result = mapper.classify(error)

    assert result is None


def test_timeout_without_word_boundary_does_not_match() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "timeoutvalue was configured"
    )

    result = mapper.classify(error)

    assert result is None


def test_partial_network_word_does_not_match() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "networking configuration completed"
    )

    result = mapper.classify(error)

    assert result is None


def test_partial_authorization_word_does_not_match() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "authorizationization"
    )

    result = mapper.classify(error)

    assert result is None


def test_multiple_signals_follow_declared_pattern_precedence() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "timeout and connection refused"
    )

    result = mapper.classify(error)

    assert result is not None

    assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_authentication_and_timeout_follow_pattern_precedence() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "invalid api key after timeout"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT


def test_rate_limit_and_service_unavailable_follow_pattern_precedence() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "service unavailable because rate limit was exceeded"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.RATE_LIMIT


def test_timeout_error_type_is_classified_when_message_has_no_signal() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = TimeoutError(
        "operation failed"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True


def test_connection_error_type_is_classified_when_message_has_no_signal() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = ConnectionError(
        "operation failed"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.retryable is True


def test_known_message_takes_precedence_over_exception_type() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = TimeoutError(
        "invalid api key"
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert result.retryable is False


def test_mapper_does_not_mutate_exception() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "connection refused"
    )

    original_message = str(error)

    result = mapper.classify(error)

    assert result is not None
    assert str(error) == original_message


def test_classifier_handles_exception_with_broken_str() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    class BrokenStringError(Exception):
        def __str__(self) -> str:
            raise RuntimeError(
                "broken string conversion"
            )

    error = BrokenStringError()

    result = mapper.classify(error)

    assert result is None


def test_classifier_handles_repeated_calls_with_same_exception() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    error = Exception(
        "service unavailable"
    )

    first = mapper.classify(error)
    second = mapper.classify(error)

    assert first is not None
    assert second is not None

    assert first == second


def test_no_signal_does_not_produce_false_positive() -> None:
    mapper = HeuristicEmbeddingErrorMapper()

    messages = [
        "operation failed",
        "provider returned an unexpected result",
        "something went wrong",
        "embedding request could not be completed",
        "temporary issue",
    ]

    for message in messages:
        result = mapper.classify(
            Exception(message)
        )

        assert result is None