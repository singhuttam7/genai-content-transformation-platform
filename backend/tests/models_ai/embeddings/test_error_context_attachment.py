from __future__ import annotations

import pytest

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingErrorCategory,
    EmbeddingErrorContext,
    EmbeddingOperation,
    EmbeddingProviderError,
)


def make_batch_context(
    *,
    batch_index: int | None = None,
    input_count: int | None = None,
) -> EmbeddingErrorContext:
    return EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=batch_index,
        input_count=input_count,
        metadata={
            "request_id": "request-123",
        },
    )


def test_provider_error_attaches_context() -> None:
    context = make_batch_context(
        input_count=8,
    )

    error = EmbeddingProviderError(
        "provider failed",
        category=EmbeddingErrorCategory.TIMEOUT,
        context=context,
    )

    assert error.context is context


def test_provider_error_preserves_category_with_context() -> None:
    context = make_batch_context()

    error = EmbeddingProviderError(
        "provider failed",
        category=EmbeddingErrorCategory.NETWORK,
        context=context,
    )

    assert error.category == EmbeddingErrorCategory.NETWORK
    assert error.context is context


def test_provider_error_preserves_message_with_context() -> None:
    context = make_batch_context()

    error = EmbeddingProviderError(
        "connection timed out",
        category=EmbeddingErrorCategory.TIMEOUT,
        context=context,
    )

    assert str(error) == "connection timed out"


def test_batch_error_attaches_context() -> None:
    context = make_batch_context(
        batch_index=2,
        input_count=16,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=2,
        context=context,
    )

    assert error.context is context
    assert error.batch_index == 2


def test_matching_batch_indices_are_allowed() -> None:
    context = make_batch_context(
        batch_index=5,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=5,
        context=context,
    )

    assert error.batch_index == 5
    assert error.context.batch_index == 5


def test_mismatched_batch_indices_are_rejected() -> None:
    context = make_batch_context(
        batch_index=4,
    )

    with pytest.raises(
        ValueError,
        match="batch_index must match context.batch_index",
    ):
        EmbeddingBatchError(
            "batch failed",
            batch_index=3,
            context=context,
        )


def test_batch_error_can_derive_batch_identity_from_context() -> None:
    context = make_batch_context(
        batch_index=7,
    )

    error = EmbeddingBatchError(
        "batch failed",
        context=context,
    )

    assert error.batch_index is None
    assert error.context.batch_index == 7


def test_batch_error_can_have_no_context() -> None:
    error = EmbeddingBatchError(
        "batch failed",
        batch_index=3,
    )

    assert error.context is None
    assert error.batch_index == 3


def test_provider_error_can_have_no_context() -> None:
    error = EmbeddingProviderError(
        "provider failed",
        category=EmbeddingErrorCategory.UNKNOWN,
    )

    assert error.context is None


def test_context_remains_immutable_after_attachment() -> None:
    context = make_batch_context(
        batch_index=2,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=2,
        context=context,
    )

    with pytest.raises(
        Exception,
    ):
        error.context.batch_index = 10


def test_context_identity_is_preserved() -> None:
    context = make_batch_context(
        batch_index=1,
    )

    error = EmbeddingBatchError(
        "batch failed",
        context=context,
    )

    assert error.context is context


def test_provider_context_contains_provider_information() -> None:
    context = make_batch_context()

    error = EmbeddingProviderError(
        "provider failure",
        context=context,
    )

    assert error.context.provider == "fake"
    assert error.context.model == "fake-model"


def test_batch_context_contains_input_count() -> None:
    context = make_batch_context(
        batch_index=1,
        input_count=32,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=1,
        context=context,
    )

    assert error.context.input_count == 32


def test_context_metadata_survives_attachment() -> None:
    context = make_batch_context(
        batch_index=3,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=3,
        context=context,
    )

    assert error.context.metadata == {
        "request_id": "request-123",
    }


def test_provider_error_context_is_serializable() -> None:
    context = make_batch_context(
        batch_index=2,
        input_count=10,
    )

    error = EmbeddingProviderError(
        "provider failed",
        category=EmbeddingErrorCategory.RATE_LIMIT,
        context=context,
    )

    dumped = error.context.model_dump(mode="json")

    assert dumped["operation"] == "embed_batch"
    assert dumped["provider"] == "fake"
    assert dumped["model"] == "fake-model"
    assert dumped["batch_index"] == 2
    assert dumped["input_count"] == 10


def test_batch_error_context_is_serializable() -> None:
    context = make_batch_context(
        batch_index=4,
        input_count=20,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=4,
        context=context,
    )

    dumped = error.context.model_dump(mode="json")

    assert dumped["batch_index"] == 4
    assert dumped["input_count"] == 20


def test_batch_error_rejects_negative_batch_index_before_attachment() -> None:
    context = make_batch_context(
        batch_index=0,
    )

    with pytest.raises(
        ValueError,
        match="batch_index cannot be negative",
    ):
        EmbeddingBatchError(
            "batch failed",
            batch_index=-1,
            context=context,
        )


def test_context_can_have_no_batch_index_for_provider_error() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="fake",
        model="fake-model",
    )

    error = EmbeddingProviderError(
        "single embedding failed",
        category=EmbeddingErrorCategory.TIMEOUT,
        context=context,
    )

    assert error.context.batch_index is None


def test_batch_error_with_batch_context_has_batch_category() -> None:
    context = make_batch_context(
        batch_index=2,
    )

    error = EmbeddingBatchError(
        "batch failed",
        context=context,
    )

    assert error.category == EmbeddingErrorCategory.BATCH