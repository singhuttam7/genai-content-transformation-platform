from __future__ import annotations

from app.models_ai.embeddings.classification import (
    StructuredEmbeddingErrorMapper,
)
from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
)


class ProviderError(Exception):
    """Simple provider-style exception for structured error tests."""

    def __init__(
        self,
        message: str = "provider error",
        *,
        status_code: object = None,
        error_code: object = None,
        error_type: object = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type


def test_valid_status_code_is_classified() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=401,
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert result.retryable is False
    assert result.confidence == 1.0
    assert result.metadata["classification_source"] == "structured"
    assert result.metadata["status_code"] == 401


def test_boolean_status_code_is_not_treated_as_integer() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=True,
    )

    result = mapper.classify(error)

    assert result is None


def test_false_status_code_is_not_treated_as_integer() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=False,
    )

    result = mapper.classify(error)

    assert result is None


def test_string_status_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code="401",
    )

    result = mapper.classify(error)

    assert result is None


def test_float_status_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=401.0,
    )

    result = mapper.classify(error)

    assert result is None


def test_none_status_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=None,
    )

    result = mapper.classify(error)

    assert result is None


def test_status_code_below_http_range_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=99,
    )

    result = mapper.classify(error)

    assert result is None


def test_status_code_above_http_range_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=600,
    )

    result = mapper.classify(error)

    assert result is None


def test_unsupported_valid_status_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=418,
    )

    result = mapper.classify(error)

    assert result is None


def test_supported_status_codes_map_to_expected_categories() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    expected = {
        400: EmbeddingErrorCategory.INPUT,
        401: EmbeddingErrorCategory.AUTHENTICATION,
        403: EmbeddingErrorCategory.AUTHORIZATION,
        404: EmbeddingErrorCategory.MODEL_NOT_FOUND,
        408: EmbeddingErrorCategory.TIMEOUT,
        429: EmbeddingErrorCategory.RATE_LIMIT,
        500: EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        502: EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        503: EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        504: EmbeddingErrorCategory.TIMEOUT,
    }

    for status_code, category in expected.items():
        error = ProviderError(
            status_code=status_code,
        )

        result = mapper.classify(error)

        assert result is not None
        assert result.category == category


def test_malformed_error_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=429,
        error_code=12345,
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert "error_code" not in result.metadata


def test_malformed_error_type_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=503,
        error_type={"unexpected": "object"},
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert "error_type" not in result.metadata


def test_empty_error_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=429,
        error_code="",
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert "error_code" not in result.metadata


def test_whitespace_error_code_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=429,
        error_code="   ",
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert "error_code" not in result.metadata


def test_empty_error_type_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=503,
        error_type="",
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert "error_type" not in result.metadata


def test_whitespace_error_type_is_ignored() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=503,
        error_type="   ",
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert "error_type" not in result.metadata


def test_valid_structured_strings_are_normalized() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=429,
        error_code="  RATE_LIMITED  ",
        error_type="  quota_error  ",
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.metadata["error_code"] == "RATE_LIMITED"
    assert result.metadata["error_type"] == "quota_error"


def test_structured_classification_does_not_depend_on_message() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        message="completely unrelated message with timeout",
        status_code=401,
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHENTICATION


def test_structured_classification_is_deterministic() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=503,
        error_code="SERVICE_UNAVAILABLE",
        error_type="temporary_failure",
    )

    first = mapper.classify(error)
    second = mapper.classify(error)

    assert first is not None
    assert second is not None

    assert first == second
    assert first.model_dump(mode="json") == second.model_dump(
        mode="json"
    )


def test_mapper_does_not_mutate_exception() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=429,
        error_code="RATE_LIMIT",
        error_type="QUOTA",
    )

    original_status_code = error.status_code
    original_error_code = error.error_code
    original_error_type = error.error_type

    result = mapper.classify(error)

    assert result is not None

    assert error.status_code == original_status_code
    assert error.error_code == original_error_code
    assert error.error_type == original_error_type


def test_structured_metadata_contains_only_supported_fields() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        status_code=429,
        error_code="RATE_LIMIT",
        error_type="QUOTA",
    )

    result = mapper.classify(error)

    assert result is not None

    assert result.metadata == {
        "classification_source": "structured",
        "status_code": 429,
        "error_code": "RATE_LIMIT",
        "error_type": "QUOTA",
    }


def test_retryability_follows_structured_category() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    retryable_error = ProviderError(
        status_code=503,
    )

    permanent_error = ProviderError(
        status_code=401,
    )

    retryable_result = mapper.classify(
        retryable_error
    )
    permanent_result = mapper.classify(
        permanent_error
    )

    assert retryable_result is not None
    assert permanent_result is not None

    assert retryable_result.retryable is True
    assert permanent_result.retryable is False


def test_status_code_property_that_raises_is_handled_safely() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    class BrokenStatusError(Exception):
        @property
        def status_code(self) -> int:
            raise RuntimeError("broken status property")

    error = BrokenStatusError(
        "provider error"
    )

    result = mapper.classify(error)

    assert result is None


def test_structured_mapper_does_not_use_heuristic_fallback() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        message="timeout while connecting",
        status_code=418,
    )

    result = mapper.classify(error)

    assert result is None