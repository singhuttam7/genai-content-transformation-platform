from __future__ import annotations

from app.models_ai.embeddings import (
    CompositeEmbeddingErrorClassifier,
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
    EmbeddingErrorClassifier,
    HeuristicEmbeddingErrorMapper,
)


class StructuredError(Exception):
    """Fake exception exposing structured provider information."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


def test_structured_classification_has_precedence() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "authentication failed",
        status_code=429,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.metadata["classification_source"] == (
        "structured"
    )


def test_structured_classification_wins_when_messages_conflict() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "model not found",
        status_code=401,
    )

    result = classifier.classify(error)

    assert result.category == (
        EmbeddingErrorCategory.AUTHENTICATION
    )


def test_structured_classification_wins_over_timeout_message() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "request timed out",
        status_code=429,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT


def test_heuristic_classification_is_used_without_structured_signal() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = Exception("connection refused")

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.retryable is True
    assert result.metadata["classification_source"] == (
        "heuristic"
    )


def test_heuristic_classification_is_used_for_timeout() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        TimeoutError("request timed out")
    )

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True


def test_unknown_fallback_is_used_when_no_strategy_matches() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        Exception("completely unexpected failure")
    )

    assert result.category == EmbeddingErrorCategory.UNKNOWN
    assert result.retryable is False
    assert result.confidence == 0.0


def test_unknown_fallback_is_provider_independent() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        RuntimeError("unexpected provider failure")
    )

    assert result.category == EmbeddingErrorCategory.UNKNOWN


def test_structured_mapper_can_be_injected() -> None:
    class CustomStructuredMapper:
        def classify(
            self,
            error: Exception,
        ) -> EmbeddingErrorClassification:
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.AUTHORIZATION,
                retryable=False,
                confidence=1.0,
                reason="Custom structured classification.",
            )

    classifier = CompositeEmbeddingErrorClassifier(
        structured_mapper=CustomStructuredMapper(),
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        Exception("connection refused")
    )

    assert result.category == (
        EmbeddingErrorCategory.AUTHORIZATION
    )


def test_custom_structured_mapper_is_higher_priority() -> None:
    class CustomStructuredMapper:
        def classify(
            self,
            error: Exception,
        ) -> EmbeddingErrorClassification:
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.MODEL_NOT_FOUND,
                retryable=False,
                confidence=1.0,
            )

    classifier = CompositeEmbeddingErrorClassifier(
        structured_mapper=CustomStructuredMapper(),
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        Exception("connection refused")
    )

    assert result.category == (
        EmbeddingErrorCategory.MODEL_NOT_FOUND
    )


def test_default_classifier_includes_heuristic_mapper() -> None:
    classifier = CompositeEmbeddingErrorClassifier()

    result = classifier.classify(
        Exception("connection refused")
    )

    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.metadata["classification_source"] == (
        "heuristic"
    )

def test_structured_result_is_returned_unchanged() -> None:
    class CustomStructuredMapper:
        def classify(
            self,
            error: Exception,
        ) -> EmbeddingErrorClassification:
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.RATE_LIMIT,
                retryable=True,
                confidence=1.0,
                reason="Exact structured result.",
                metadata={
                    "custom": True,
                },
            )

    classifier = CompositeEmbeddingErrorClassifier(
        structured_mapper=CustomStructuredMapper(),
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        Exception("authentication failed")
    )

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.retryable is True
    assert result.confidence == 1.0
    assert result.reason == "Exact structured result."
    assert result.metadata == {
        "custom": True,
    }


def test_heuristic_result_is_returned_when_structured_returns_none() -> None:
    class EmptyStructuredMapper:
        def classify(
            self,
            error: Exception,
        ) -> EmbeddingErrorClassification | None:
            return None

    class CustomHeuristicMapper(
        EmbeddingErrorClassifier
    ):
        def classify(
            self,
            error: Exception,
        ) -> EmbeddingErrorClassification:
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.TIMEOUT,
                retryable=True,
                confidence=0.7,
                reason="Custom heuristic result.",
            )

    classifier = CompositeEmbeddingErrorClassifier(
        structured_mapper=EmptyStructuredMapper(),
        heuristic_mapper=CustomHeuristicMapper(),
    )

    result = classifier.classify(
        Exception("anything")
    )

    assert result.category == EmbeddingErrorCategory.TIMEOUT
    assert result.retryable is True
    assert result.confidence == 0.7
    assert result.reason == "Custom heuristic result."


def test_unknown_fallback_has_fallback_source() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        Exception("no recognizable signal")
    )

    assert result.metadata == {
        "classification_source": "fallback",
    }


def test_precedence_is_deterministic() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "too many requests",
        status_code=401,
    )

    first = classifier.classify(error)
    second = classifier.classify(error)

    assert first == second


def test_structured_status_with_unmapped_value_allows_heuristic_fallback() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "connection refused",
        status_code=418,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.NETWORK
    assert result.metadata["classification_source"] == (
        "heuristic"
    )


def test_structured_status_without_heuristic_signal_uses_unknown() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "unexpected provider behavior",
        status_code=418,
    )

    result = classifier.classify(error)

    assert result.category == EmbeddingErrorCategory.UNKNOWN


def test_structured_evidence_preserves_full_confidence() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    error = StructuredError(
        "invalid api key",
        status_code=429,
    )

    result = classifier.classify(error)

    assert result.confidence == 1.0


def test_heuristic_evidence_preserves_heuristic_confidence() -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper(),
    )

    result = classifier.classify(
        Exception("too many requests")
    )

    assert result.confidence == 0.85