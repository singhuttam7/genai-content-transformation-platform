from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    CompositeEmbeddingErrorClassifier,
    EmbeddingErrorCategory,
    EmbeddingOperation,
)
from app.models_ai.embeddings.adapters import (
    BaseEmbeddingAdapter,
)
from app.models_ai.embeddings.heuristic_classification import (
    HeuristicEmbeddingErrorMapper,
)
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelConfig,
    EmbeddingRequest,
    EmbeddingVector,
)


class FakeEmbeddingAdapter(BaseEmbeddingAdapter):
    """Minimal adapter used only for testing the base contract."""

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


class StructuredProviderError(Exception):
    """Fake provider exception exposing a structured status code."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.fixture
def model_config() -> EmbeddingModelConfig:
    return EmbeddingModelConfig(
        provider="fake-provider",
        model_name="fake-embedding-model",
        expected_dimension=3,
        options={
            "temperature": 0,
        },
    )


@pytest.fixture
def adapter(
    model_config: EmbeddingModelConfig,
) -> FakeEmbeddingAdapter:
    return FakeEmbeddingAdapter(model_config)


def test_adapter_requires_model_config() -> None:
    with pytest.raises(TypeError):
        FakeEmbeddingAdapter()  # type: ignore[call-arg]


def test_adapter_stores_model_config(
    adapter: FakeEmbeddingAdapter,
    model_config: EmbeddingModelConfig,
) -> None:
    assert adapter.model_config == model_config


def test_adapter_creates_default_classifier(
    adapter: FakeEmbeddingAdapter,
) -> None:
    assert isinstance(
        adapter.error_classifier,
        CompositeEmbeddingErrorClassifier,
    )


def test_custom_classifier_can_be_injected(
    model_config: EmbeddingModelConfig,
) -> None:
    classifier = CompositeEmbeddingErrorClassifier(
        heuristic_mapper=HeuristicEmbeddingErrorMapper()
    )

    adapter = FakeEmbeddingAdapter(
        model_config,
        error_classifier=classifier,
    )

    assert adapter.error_classifier is classifier


def test_build_error_context_contains_provider(
    adapter: FakeEmbeddingAdapter,
) -> None:
    context = adapter.build_error_context(
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert context.provider == "fake-provider"


def test_build_error_context_contains_model(
    adapter: FakeEmbeddingAdapter,
) -> None:
    context = adapter.build_error_context(
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert context.model == "fake-embedding-model"


def test_build_error_context_contains_operation(
    adapter: FakeEmbeddingAdapter,
) -> None:
    context = adapter.build_error_context(
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=5,
    )

    assert context.operation == EmbeddingOperation.EMBED_BATCH


def test_build_error_context_contains_input_count(
    adapter: FakeEmbeddingAdapter,
) -> None:
    context = adapter.build_error_context(
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=5,
    )

    assert context.input_count == 5


def test_build_error_context_contains_batch_index(
    adapter: FakeEmbeddingAdapter,
) -> None:
    context = adapter.build_error_context(
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=5,
        batch_index=2,
    )

    assert context.batch_index == 2


def test_build_error_context_contains_extra_metadata(
    adapter: FakeEmbeddingAdapter,
) -> None:
    context = adapter.build_error_context(
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata={
            "provider_request_id": "req-123",
        },
    )

    assert context.metadata["provider_request_id"] == "req-123"


def test_build_provider_error_contains_category(
    adapter: FakeEmbeddingAdapter,
) -> None:
    error = adapter.build_provider_error(
        message="request failed",
        category=EmbeddingErrorCategory.RATE_LIMIT,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=10,
        batch_index=1,
    )

    assert error.category == EmbeddingErrorCategory.RATE_LIMIT


def test_build_provider_error_contains_context(
    adapter: FakeEmbeddingAdapter,
) -> None:
    error = adapter.build_provider_error(
        message="request failed",
        category=EmbeddingErrorCategory.TIMEOUT,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.context is not None
    assert error.context.operation == EmbeddingOperation.EMBED


def test_build_provider_error_contains_message(
    adapter: FakeEmbeddingAdapter,
) -> None:
    error = adapter.build_provider_error(
        message="provider unavailable",
        category=EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert str(error) == "provider unavailable"


def test_build_provider_error_preserves_batch_index(
    adapter: FakeEmbeddingAdapter,
) -> None:
    error = adapter.build_provider_error(
        message="batch failed",
        category=EmbeddingErrorCategory.NETWORK,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=4,
        batch_index=3,
    )

    assert error.batch_index == 3
    assert error.context is not None
    assert error.context.batch_index == 3


def test_translate_provider_exception_returns_common_error(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError("provider exploded")

    error = adapter.translate_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        category=EmbeddingErrorCategory.SERVICE_UNAVAILABLE,
    )

    assert str(error) == "provider exploded"
    assert error.category == (
        EmbeddingErrorCategory.SERVICE_UNAVAILABLE
    )


def test_translate_provider_exception_contains_context(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError("provider exploded")

    error = adapter.translate_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=8,
        category=EmbeddingErrorCategory.TIMEOUT,
        batch_index=2,
    )

    assert error.context is not None
    assert error.context.operation == (
        EmbeddingOperation.EMBED_BATCH
    )
    assert error.context.input_count == 8
    assert error.context.batch_index == 2


def test_translation_preserves_original_error_when_raised(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError("provider exploded")

    with pytest.raises(Exception) as captured:
        translated = adapter.translate_provider_exception(
            error=provider_error,
            operation=EmbeddingOperation.EMBED,
            input_count=1,
            category=EmbeddingErrorCategory.NETWORK,
        )
        raise translated from provider_error

    assert captured.value.__cause__ is provider_error


def test_adapter_boundary_does_not_modify_provider_exception(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError("provider exploded")

    before = str(provider_error)

    adapter.translate_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        category=EmbeddingErrorCategory.NETWORK,
    )

    assert str(provider_error) == before


def test_adapter_is_an_embedding_port(
    adapter: FakeEmbeddingAdapter,
) -> None:
    from app.models_ai.embeddings.port import EmbeddingPort

    assert isinstance(adapter, EmbeddingPort)


def test_classify_provider_exception_uses_classifier(
    adapter: FakeEmbeddingAdapter,
) -> None:
    error = StructuredProviderError(
        "authentication failed",
        status_code=429,
    )

    classification = adapter.classify_provider_exception(
        error
    )

    assert classification.category == (
        EmbeddingErrorCategory.RATE_LIMIT
    )


def test_classify_provider_exception_uses_heuristic_fallback(
    adapter: FakeEmbeddingAdapter,
) -> None:
    error = RuntimeError("connection refused")

    classification = adapter.classify_provider_exception(
        error
    )

    assert classification.category == (
        EmbeddingErrorCategory.NETWORK
    )


def test_translate_and_classify_uses_structured_precedence(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = StructuredProviderError(
        "authentication failed",
        status_code=429,
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=5,
        batch_index=1,
    )

    assert error.category == EmbeddingErrorCategory.RATE_LIMIT
    assert error.batch_index == 1
    assert error.context is not None
    assert error.context.batch_index == 1


def test_translate_and_classify_uses_heuristic_fallback(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "connection refused"
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.category == EmbeddingErrorCategory.NETWORK


def test_translate_and_classify_uses_unknown_fallback(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "unexpected provider failure"
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert error.category == EmbeddingErrorCategory.UNKNOWN


def test_translate_and_classify_stores_classification_metadata(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "connection refused"
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
    assert classification["confidence"] == 0.9


def test_translate_and_classify_preserves_provider_metadata(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "connection refused"
    )

    error = adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
        metadata={
            "provider_request_id": "req-123",
        },
    )

    assert error.context is not None
    assert error.context.metadata[
        "provider_request_id"
    ] == "req-123"


def test_translate_and_classify_does_not_modify_original_error(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "connection refused"
    )

    before = str(provider_error)

    adapter.translate_and_classify_provider_exception(
        error=provider_error,
        operation=EmbeddingOperation.EMBED,
        input_count=1,
    )

    assert str(provider_error) == before


def test_translate_and_classify_preserves_original_cause(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "connection refused"
    )

    with pytest.raises(Exception) as captured:
        translated = (
            adapter.translate_and_classify_provider_exception(
                error=provider_error,
                operation=EmbeddingOperation.EMBED,
                input_count=1,
            )
        )
        raise translated from provider_error

    assert captured.value.__cause__ is provider_error


def test_translate_and_classify_is_deterministic(
    adapter: FakeEmbeddingAdapter,
) -> None:
    provider_error = RuntimeError(
        "connection refused"
    )

    first = (
        adapter.translate_and_classify_provider_exception(
            error=provider_error,
            operation=EmbeddingOperation.EMBED,
            input_count=1,
        )
    )

    second = (
        adapter.translate_and_classify_provider_exception(
            error=provider_error,
            operation=EmbeddingOperation.EMBED,
            input_count=1,
        )
    )

    assert first.category == second.category
    assert first.context == second.context