from __future__ import annotations

from app.models_ai.embeddings.classification import (
    CompositeEmbeddingErrorClassifier,
    StructuredEmbeddingErrorMapper,
)
from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
)


class ProviderError(Exception):
    """Provider-style exception carrying structured HTTP information."""

    def __init__(
        self,
        message: str,
        *,
        status_code: object = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_401_overrides_timeout_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "timeout while contacting embedding service",
        status_code=401,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert result.retryable is False
    assert result.confidence == 1.0


def test_403_overrides_connection_refused_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "connection refused by provider",
        status_code=403,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.AUTHORIZATION
    assert result.retryable is False
    assert result.confidence == 1.0


def test_404_overrides_service_unavailable_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "service unavailable",
        status_code=404,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.MODEL_NOT_FOUND
    assert result.retryable is False
    assert result.confidence == 1.0


def test_408_overrides_invalid_api_key_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "invalid api key",
        status_code=408,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True
    assert result.confidence == 1.0


def test_429_overrides_authentication_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "authentication failed",
        status_code=429,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.retryable is True
    assert result.confidence == 1.0


def test_500_overrides_invalid_api_key_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "invalid api key",
        status_code=500,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert result.retryable is True
    assert result.confidence == 1.0


def test_503_overrides_permission_denied_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "permission denied",
        status_code=503,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert result.retryable is True
    assert result.confidence == 1.0


def test_504_overrides_model_not_found_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "model not found",
        status_code=504,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True
    assert result.confidence == 1.0


def test_structured_signal_wins_over_multiple_heuristic_signals() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        (
            "timeout, connection refused, invalid api key, "
            "permission denied, model not found"
        ),
        status_code=429,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.retryable is True
    assert result.confidence == 1.0


def test_structured_classification_source_is_preserved() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "timeout",
        status_code=401,
    )

    result = classifier.classify(error)

    assert result.metadata["classification_source"] == "structured"
    assert result.metadata["status_code"] == 401


def test_heuristic_message_cannot_modify_structured_metadata() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "timeout",
        status_code=401,
    )

    result = classifier.classify(error)

    assert result.metadata == {
        "classification_source": "structured",
        "status_code": 401,
    }


def test_structured_retryability_is_used() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    permanent_error = ProviderError(
        "timeout",
        status_code=401,
    )

    transient_error = ProviderError(
        "invalid api key",
        status_code=503,
    )

    permanent_result = classifier.classify(
        permanent_error
    )
    transient_result = classifier.classify(
        transient_error
    )

    assert permanent_result.retryable is False
    assert transient_result.retryable is True


def test_unsupported_status_code_allows_heuristic_fallback() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "timeout",
        status_code=418,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.confidence == 0.90
    assert result.metadata["classification_source"] == "heuristic"


def test_missing_status_code_allows_heuristic_fallback() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "connection refused",
        status_code=None,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.confidence == 0.90
    assert result.metadata["classification_source"] == "heuristic"


def test_malformed_status_code_allows_heuristic_fallback() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "service unavailable",
        status_code="503",
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert result.confidence == 0.80
    assert result.metadata["classification_source"] == "heuristic"


def test_boolean_status_code_cannot_override_heuristic_signal() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "timeout",
        status_code=True,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.metadata["classification_source"] == "heuristic"


def test_malicious_message_cannot_override_authentication_status() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        (
            "timeout timeout timeout "
            "connection refused "
            "service unavailable "
            "model not found "
            "invalid api key"
        ),
        status_code=401,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert result.retryable is False
    assert result.confidence == 1.0


def test_malicious_message_cannot_override_service_status() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        (
            "invalid api key "
            "permission denied "
            "model not found"
        ),
        status_code=503,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert result.retryable is True
    assert result.confidence == 1.0


def test_message_with_structured_status_is_not_used_for_category() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "model not found",
        status_code=500,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    assert result.metadata["classification_source"] == "structured"
    assert result.metadata["status_code"] == 500


def test_structured_mapper_can_be_replaced() -> None:
    class EmptyStructuredMapper:
        def classify(
            self,
            error: Exception,
        ):
            return None

    classifier = CompositeEmbeddingErrorClassifier(
        structured_mapper=EmptyStructuredMapper()
    )

    error = ProviderError(
        "timeout",
        status_code=401,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.metadata["classification_source"] == "heuristic"


def test_heuristic_mapper_can_be_replaced() -> None:
    class EmptyHeuristicMapper:
        def classify(
            self,
            error: Exception,
        ):
            return None

    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=EmptyHeuristicMapper()
    )

    error = ProviderError(
        "timeout",
        status_code=418,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.UNKNOWN
    assert result.retryable is False
    assert result.confidence == 0.0


def test_unknown_error_reaches_unknown_fallback() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = Exception(
        "something completely unrelated happened"
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.UNKNOWN
    assert result.retryable is False
    assert result.confidence == 0.0
    assert result.metadata["classification_source"] == "fallback"


def test_classification_is_deterministic_for_conflicting_signals() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        (
            "timeout and connection refused and "
            "invalid api key and model not found"
        ),
        status_code=403,
    )

    first = classifier.classify(error)
    second = classifier.classify(error)

    assert first == second
    assert first.category == EmbeddingErrorCategory.AUTHORIZATION


def test_structured_classification_does_not_mutate_exception() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    error = ProviderError(
        "timeout",
        status_code=401,
    )

    original_message = str(error)
    original_status = error.status_code

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert str(error) == original_message
    assert error.status_code == original_status


def test_structured_mapper_alone_has_precedence_boundary() -> None:
    mapper = StructuredEmbeddingErrorMapper()

    error = ProviderError(
        "timeout",
        status_code=401,
    )

    result = mapper.classify(error)

    assert result is not None
    assert result.category == EmbeddingErrorCategory.AUTHENTICATION
    assert result.confidence == 1.0