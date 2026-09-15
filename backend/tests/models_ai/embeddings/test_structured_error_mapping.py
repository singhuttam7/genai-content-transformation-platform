from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingErrorCategory,
    StructuredEmbeddingErrorMapper,
)


class StructuredError(Exception):
    """Fake provider-independent structured exception."""

    def __init__(
        self,
        message: str = "provider error",
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type


@pytest.fixture
def mapper() -> StructuredEmbeddingErrorMapper:
    return StructuredEmbeddingErrorMapper()


@pytest.mark.parametrize(
    ("status_code", "expected_category"),
    [
        (400, EmbeddingErrorCategory.INPUT),
        (401, EmbeddingErrorCategory.AUTHENTICATION),
        (403, EmbeddingErrorCategory.AUTHORIZATION),
        (404, EmbeddingErrorCategory.MODEL_NOT_FOUND),
        (408, EmbeddingErrorCategory.TIMEOUT),
        (429, EmbeddingErrorCategory.RATE_LIMIT),
        (500, EmbeddingErrorCategory.SERVICE_UNAVAILABLE),
        (502, EmbeddingErrorCategory.SERVICE_UNAVAILABLE),
        (503, EmbeddingErrorCategory.SERVICE_UNAVAILABLE),
        (504, EmbeddingErrorCategory.TIMEOUT),
    ],
)
def test_supported_status_codes_are_classified(
    mapper: StructuredEmbeddingErrorMapper,
    status_code: int,
    expected_category: EmbeddingErrorCategory,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=status_code)
    )

    assert result is not None
    assert result.category == expected_category


def test_structured_classification_has_full_confidence(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=429)
    )

    assert result is not None
    assert result.confidence == 1.0


def test_rate_limit_is_retryable(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=429)
    )

    assert result is not None
    assert result.retryable is True


@pytest.mark.parametrize(
    "status_code",
    [400, 401, 403, 404],
)
def test_client_side_errors_are_not_retryable(
    mapper: StructuredEmbeddingErrorMapper,
    status_code: int,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=status_code)
    )

    assert result is not None
    assert result.retryable is False


@pytest.mark.parametrize(
    "status_code",
    [408, 500, 502, 503, 504],
)
def test_transient_statuses_are_retryable(
    mapper: StructuredEmbeddingErrorMapper,
    status_code: int,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=status_code)
    )

    assert result is not None
    assert result.retryable is True


@pytest.mark.parametrize(
    "status_code",
    [100, 199, 200, 201, 204, 300, 301, 399],
)
def test_unsupported_status_codes_return_none(
    mapper: StructuredEmbeddingErrorMapper,
    status_code: int,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=status_code)
    )

    assert result is None


def test_missing_status_code_returns_none(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        RuntimeError("provider failure")
    )

    assert result is None


def test_non_integer_status_code_returns_none(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    error = StructuredError()
    error.status_code = "429"

    result = mapper.classify(error)

    assert result is None


def test_boolean_status_code_returns_none(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    error = StructuredError()
    error.status_code = True

    result = mapper.classify(error)

    assert result is None


@pytest.mark.parametrize(
    "status_code",
    [-1, 0, 99, 600, 999],
)
def test_invalid_status_code_returns_none(
    mapper: StructuredEmbeddingErrorMapper,
    status_code: int,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=status_code)
    )

    assert result is None


def test_error_code_is_preserved_in_metadata(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(
            status_code=429,
            error_code="rate_limit_exceeded",
        )
    )

    assert result is not None
    assert result.metadata["error_code"] == (
        "rate_limit_exceeded"
    )


def test_error_type_is_preserved_in_metadata(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(
            status_code=404,
            error_type="model_not_found",
        )
    )

    assert result is not None
    assert result.metadata["error_type"] == (
        "model_not_found"
    )


def test_empty_error_code_is_ignored(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(
            status_code=429,
            error_code="   ",
        )
    )

    assert result is not None
    assert "error_code" not in result.metadata


def test_non_string_error_code_is_ignored(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    error = StructuredError(status_code=429)
    error.error_code = 123

    result = mapper.classify(error)

    assert result is not None
    assert "error_code" not in result.metadata


def test_empty_error_type_is_ignored(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(
            status_code=404,
            error_type="   ",
        )
    )

    assert result is not None
    assert "error_type" not in result.metadata


def test_structured_source_is_recorded(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=503)
    )

    assert result is not None
    assert result.metadata["classification_source"] == (
        "structured"
    )


def test_status_code_is_recorded_in_metadata(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=429)
    )

    assert result is not None
    assert result.metadata["status_code"] == 429


def test_reason_contains_status_and_category(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(status_code=401)
    )

    assert result is not None
    assert "401" in (result.reason or "")
    assert "authentication" in (result.reason or "")


def test_classification_is_deterministic(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    error = StructuredError(
        status_code=429,
        error_code="rate_limit",
        error_type="quota",
    )

    first = mapper.classify(error)
    second = mapper.classify(error)

    assert first == second


def test_mapper_does_not_mutate_exception(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    error = StructuredError(
        status_code=429,
        error_code="rate_limit",
    )

    original_status = error.status_code
    original_code = error.error_code

    mapper.classify(error)

    assert error.status_code == original_status
    assert error.error_code == original_code


def test_classification_is_json_serializable(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    result = mapper.classify(
        StructuredError(
            status_code=429,
            error_code="rate_limit",
        )
    )

    assert result is not None

    payload = result.model_dump(mode="json")

    assert payload["category"] == "rate_limit"
    assert payload["retryable"] is True
    assert payload["confidence"] == 1.0
    assert payload["metadata"]["status_code"] == 429


def test_mapper_does_not_use_error_message_for_classification(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    error = StructuredError(
        "this message says nothing useful",
        status_code=None,
    )

    result = mapper.classify(error)

    assert result is None


def test_unknown_exception_attributes_are_ignored(
    mapper: StructuredEmbeddingErrorMapper,
) -> None:
    class CustomError(Exception):
        provider_status = 429

    result = mapper.classify(CustomError("rate limited"))

    assert result is None