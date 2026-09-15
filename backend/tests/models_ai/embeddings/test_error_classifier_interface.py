from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingErrorCategory,
    EmbeddingErrorClassification,
    EmbeddingErrorClassifier,
)


class FakeEmbeddingErrorClassifier(EmbeddingErrorClassifier):
    """Minimal fake implementation used to verify the classifier port."""

    def classify(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification:
        return EmbeddingErrorClassification(
            category=EmbeddingErrorCategory.UNKNOWN,
            retryable=False,
            confidence=0.0,
            reason=str(error),
        )


class FailingClassifier(EmbeddingErrorClassifier):
    """Implementation used to verify the abstract contract."""

    def classify(
        self,
        error: Exception,
    ) -> EmbeddingErrorClassification:
        raise RuntimeError("classifier failure")


def test_classifier_is_abstract() -> None:
    with pytest.raises(TypeError):
        EmbeddingErrorClassifier()


def test_fake_classifier_can_be_instantiated() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    assert isinstance(
        classifier,
        EmbeddingErrorClassifier,
    )


def test_classifier_returns_classification() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    result = classifier.classify(
        RuntimeError("unknown provider failure")
    )

    assert isinstance(
        result,
        EmbeddingErrorClassification,
    )


def test_classifier_returns_provider_independent_category() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    result = classifier.classify(
        RuntimeError("provider failure")
    )

    assert result.category == EmbeddingErrorCategory.UNKNOWN


def test_classifier_receives_exception() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    error = RuntimeError("connection failed")

    result = classifier.classify(error)

    assert result.reason == "connection failed"


def test_classifier_does_not_modify_exception() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    error = RuntimeError("original error")

    original_message = str(error)

    classifier.classify(error)

    assert str(error) == original_message


def test_classifier_is_deterministic_for_same_error() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    error = RuntimeError("same failure")

    first = classifier.classify(error)
    second = classifier.classify(error)

    assert first == second


def test_classifier_can_handle_different_exception_types() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    exceptions = [
        ValueError("invalid value"),
        RuntimeError("runtime failure"),
        OSError("operating system failure"),
        TimeoutError("timeout"),
    ]

    for error in exceptions:
        result = classifier.classify(error)

        assert isinstance(
            result,
            EmbeddingErrorClassification,
        )


def test_classifier_result_is_serializable() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    result = classifier.classify(
        RuntimeError("failure")
    )

    payload = result.model_dump(mode="json")

    assert payload["category"] == "unknown"
    assert payload["retryable"] is False
    assert payload["confidence"] == 0.0


def test_classifier_can_be_subclassed() -> None:
    classifier = FailingClassifier()

    assert isinstance(
        classifier,
        EmbeddingErrorClassifier,
    )


def test_classifier_implementation_controls_classification_result() -> None:
    class RateLimitClassifier(EmbeddingErrorClassifier):
        def classify(
            self,
            error: Exception,
        ) -> EmbeddingErrorClassification:
            return EmbeddingErrorClassification(
                category=EmbeddingErrorCategory.RATE_LIMIT,
                retryable=True,
                confidence=1.0,
                reason="Rate limit detected.",
            )

    classifier = RateLimitClassifier()

    result = classifier.classify(
        RuntimeError("too many requests")
    )

    assert result.category == EmbeddingErrorCategory.RATE_LIMIT
    assert result.retryable is True
    assert result.confidence == 1.0


def test_classifier_does_not_require_provider_sdk() -> None:
    classifier = FakeEmbeddingErrorClassifier()

    result = classifier.classify(
        Exception("generic error")
    )

    assert result.category == EmbeddingErrorCategory.UNKNOWN