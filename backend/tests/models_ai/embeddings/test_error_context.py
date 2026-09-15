from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings import (
    EmbeddingBatchError,
    EmbeddingErrorCategory,
    EmbeddingErrorContext,
    EmbeddingOperation,
    EmbeddingProviderError,
)


def test_minimal_error_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
    )

    assert context.operation == EmbeddingOperation.EMBED
    assert context.provider is None
    assert context.model is None
    assert context.batch_index is None
    assert context.input_count is None
    assert context.metadata == {}


def test_complete_error_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="local_sentence_transformer",
        model="all-MiniLM-L6-v2",
        batch_index=3,
        input_count=32,
        metadata={
            "request_id": "request-123",
            "attempt": 1,
        },
    )

    assert context.operation == EmbeddingOperation.EMBED_BATCH
    assert context.provider == "local_sentence_transformer"
    assert context.model == "all-MiniLM-L6-v2"
    assert context.batch_index == 3
    assert context.input_count == 32
    assert context.metadata["request_id"] == "request-123"


def test_context_is_immutable() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="fake",
        model="fake-model",
    )

    with pytest.raises(
        ValidationError,
    ):
        context.provider = "another-provider"


def test_context_rejects_unknown_fields() -> None:
    with pytest.raises(
        ValidationError,
    ):
        EmbeddingErrorContext(
            operation=EmbeddingOperation.EMBED,
            unknown_field="invalid",
        )


def test_context_rejects_blank_provider() -> None:
    with pytest.raises(
        ValidationError,
    ):
        EmbeddingErrorContext(
            operation=EmbeddingOperation.EMBED,
            provider="",
        )


def test_context_rejects_blank_model() -> None:
    with pytest.raises(
        ValidationError,
    ):
        EmbeddingErrorContext(
            operation=EmbeddingOperation.EMBED,
            model="",
        )


def test_context_rejects_negative_batch_index() -> None:
    with pytest.raises(
        ValidationError,
    ):
        EmbeddingErrorContext(
            operation=EmbeddingOperation.EMBED_BATCH,
            batch_index=-1,
        )


def test_context_rejects_negative_input_count() -> None:
    with pytest.raises(
        ValidationError,
    ):
        EmbeddingErrorContext(
            operation=EmbeddingOperation.EMBED_BATCH,
            input_count=-1,
        )


def test_context_supports_zero_input_count() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        input_count=0,
    )

    assert context.input_count == 0


def test_context_supports_batch_index_zero() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        batch_index=0,
    )

    assert context.batch_index == 0


def test_context_serializes_to_json_compatible_structure() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=2,
        input_count=10,
        metadata={
            "source_id": str(uuid4()),
            "attempt": 1,
        },
    )

    dumped = context.model_dump(mode="json")

    assert dumped["operation"] == "embed_batch"
    assert dumped["provider"] == "fake"
    assert dumped["model"] == "fake-model"
    assert dumped["batch_index"] == 2
    assert dumped["input_count"] == 10
    assert isinstance(dumped["metadata"], dict)


def test_context_serialization_is_deterministic() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="fake",
        model="fake-model",
        metadata={
            "z": 3,
            "a": 1,
        },
    )

    first = context.model_dump(mode="json")
    second = context.model_dump(mode="json")

    assert first == second


def test_provider_error_can_carry_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="fake",
        model="fake-model",
        input_count=1,
    )

    error = EmbeddingProviderError(
        "provider failed",
        category=EmbeddingErrorCategory.TIMEOUT,
        context=context,
    )

    assert error.context is context
    assert error.category == EmbeddingErrorCategory.TIMEOUT


def test_provider_error_without_context_remains_valid() -> None:
    error = EmbeddingProviderError(
        "provider failed",
        category=EmbeddingErrorCategory.NETWORK,
    )

    assert error.context is None
    assert error.category == EmbeddingErrorCategory.NETWORK


def test_batch_error_can_carry_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=4,
        input_count=32,
    )

    error = EmbeddingBatchError(
        "batch failed",
        batch_index=4,
        context=context,
    )

    assert error.batch_index == 4
    assert error.context is context
    assert error.context.batch_index == 4


def test_batch_error_can_exist_without_context() -> None:
    error = EmbeddingBatchError(
        "batch failed",
        batch_index=2,
    )

    assert error.batch_index == 2
    assert error.context is None


def test_context_metadata_supports_nested_structures() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        metadata={
            "provider_response": {
                "status": 429,
                "headers": {
                    "retry_after": "10",
                },
            },
            "request": {
                "chunk_indices": [1, 2, 3],
            },
        },
    )

    assert context.metadata["provider_response"]["status"] == 429
    assert (
        context.metadata["provider_response"]["headers"]["retry_after"]
        == "10"
    )
    assert context.metadata["request"]["chunk_indices"] == [
        1,
        2,
        3,
    ]


def test_context_accepts_unicode_metadata() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "language": "hi",
            "message": "एम्बेडिंग विफल हुई",
        },
    )

    assert context.metadata["language"] == "hi"
    assert context.metadata["message"] == "एम्बेडिंग विफल हुई"


def test_context_does_not_require_provider_specific_fields() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
    )

    assert context.provider is None
    assert context.model is None


def test_operation_values_are_stable() -> None:
    assert EmbeddingOperation.EMBED.value == "embed"
    assert EmbeddingOperation.EMBED_BATCH.value == "embed_batch"


def test_operation_is_string_compatible() -> None:
    assert isinstance(
        EmbeddingOperation.EMBED,
        str,
    )


def test_context_can_be_reconstructed_from_dump() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=2,
        input_count=16,
        metadata={
            "attempt": 1,
        },
    )

    dumped = context.model_dump()

    reconstructed = EmbeddingErrorContext.model_validate(
        dumped
    )

    assert reconstructed == context