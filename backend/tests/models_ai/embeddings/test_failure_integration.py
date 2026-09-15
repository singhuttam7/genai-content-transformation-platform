from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingErrorCategory,
    EmbeddingFailureService,
    EmbeddingOperation,
)
from app.models_ai.embeddings.adapters import (
    BaseEmbeddingAdapter,
)
from app.models_ai.embeddings.retry import (
    EmbeddingRetryPolicy,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelConfig,
    EmbeddingRequest,
    EmbeddingVector,
)


class FakeEmbeddingAdapter(BaseEmbeddingAdapter):
    """Minimal adapter for integration tests."""

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        raise NotImplementedError

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        raise NotImplementedError


class ProviderError(Exception):
    """Fake structured provider exception."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def adapter() -> FakeEmbeddingAdapter:
    return FakeEmbeddingAdapter(
        EmbeddingModelConfig(
            provider="fake-provider",
            model_name="fake-model",
            expected_dimension=3,
        )
    )


@pytest.fixture
def service() -> EmbeddingFailureService:
    return EmbeddingFailureService()


def test_transient_error_produces_retryable_decision(
    adapter: FakeEmbeddingAdapter,
    service: EmbeddingFailureService,
) -> None:
    provider_error = ProviderError(
        "service unavailable",
        status_code=503,
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    decision = service.evaluate_retry(
        error=error
    )

    assert error.category == (
        EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    )
    assert decision.retryable is True


def test_permanent_error_produces_non_retryable_decision(
    adapter: FakeEmbeddingAdapter,
    service: EmbeddingFailureService,
) -> None:
    provider_error = ProviderError(
        "authentication failed",
        status_code=401,
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    decision = service.evaluate_retry(
        error=error
    )

    assert error.category == (
        EmbeddingErrorCategory.AUTHENTICATION
    )
    assert decision.retryable is False


def test_batch_error_preserves_batch_context(
    adapter: FakeEmbeddingAdapter,
    service: EmbeddingFailureService,
) -> None:
    provider_error = ProviderError(
        "rate limit exceeded",
        status_code=429,
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=32,
        batch_index=4,
    )

    decision = service.evaluate_retry(
        error=error
    )

    assert error.batch_index == 4
    assert error.context is not None
    assert error.context.batch_index == 4
    assert decision.retryable is True


def test_classification_is_first_class_on_error(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = ProviderError(
        "request timed out",
        status_code=408,
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.classification is not None
    assert error.classification.category == (
        EmbeddingErrorCategory.TIMEOUT
    )


def test_context_contains_serializable_classification(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = ProviderError(
        "connection refused",
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.context is not None

    classification = error.context.metadata[
        "classification"
    ]

    assert classification["category"] == "network"
    assert classification["retryable"] is True


def test_runtime_classification_and_serialized_classification_agree(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = ProviderError(
        "connection refused",
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.classification is not None
    assert error.context is not None

    serialized = error.context.metadata[
        "classification"
    ]

    assert serialized["category"] == (
        error.classification.category.value
    )
    assert serialized["retryable"] == (
        error.classification.retryable
    )
    assert serialized["confidence"] == (
        error.classification.confidence
    )


def test_missing_classification_is_rejected(
    service: EmbeddingFailureService,
) -> None:
    from app.models_ai.embeddings.exceptions import (
        EmbeddingProviderError,
    )

    error = EmbeddingProviderError(
        "provider failure",
        category=EmbeddingErrorCategory.UNKNOWN,
    )

    with pytest.raises(ValueError):
        service.evaluate_retry(
            error=error
        )


def test_custom_retry_policy_is_used() -> None:
    class AlwaysNoRetryPolicy(EmbeddingRetryPolicy):
        def evaluate(self, classification):
            from app.models_ai.embeddings.retry import (
                RetryDecision,
            )

            return RetryDecision(
                retryable=False,
                category=classification.category,
                reason="Custom test policy.",
            )

    service = EmbeddingFailureService(
        retry_policy=AlwaysNoRetryPolicy()
    )

    adapter = FakeEmbeddingAdapter(
        EmbeddingModelConfig(
            provider="fake-provider",
            model_name="fake-model",
        )
    )

    error = adapter.translate_and_classify_provider_exception(
        error=ProviderError(
            "service unavailable",
            status_code=503,
        ),
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    decision = service.evaluate_retry(
        error=error
    )

    assert decision.retryable is False
    assert decision.reason == "Custom test policy."


def test_failure_service_does_not_mutate_error(
    adapter: FakeEmbeddingAdapter,
    service: EmbeddingFailureService,
) -> None:
    error = adapter.translate_and_classify_provider_exception(
        error=ProviderError(
            "timeout",
            status_code=408,
        ),
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    before = error.classification

    service.evaluate_retry(
        error=error
    )

    assert error.classification == before