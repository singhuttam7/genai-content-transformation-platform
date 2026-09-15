from __future__ import annotations

import json
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings import (
    EmbeddingErrorContext,
    EmbeddingOperation,
)


def test_embed_context_serializes_to_json() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="local_sentence_transformer",
        model="all-MiniLM-L6-v2",
        input_count=1,
    )

    payload = context.model_dump(mode="json")

    assert payload == {
        "operation": "embed",
        "provider": "local_sentence_transformer",
        "model": "all-MiniLM-L6-v2",
        "batch_index": None,
        "input_count": 1,
        "metadata": {},
    }


def test_batch_context_serializes_to_json() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="local_sentence_transformer",
        model="all-MiniLM-L6-v2",
        batch_index=4,
        input_count=32,
    )

    payload = context.model_dump(mode="json")

    assert payload["operation"] == "embed_batch"
    assert payload["provider"] == "local_sentence_transformer"
    assert payload["model"] == "all-MiniLM-L6-v2"
    assert payload["batch_index"] == 4
    assert payload["input_count"] == 32


def test_json_dump_is_accepted_by_json_module() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=2,
        input_count=10,
    )

    payload = context.model_dump(mode="json")

    serialized = json.dumps(payload)

    assert isinstance(serialized, str)
    assert json.loads(serialized) == payload


def test_json_round_trip_preserves_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=3,
        input_count=24,
        metadata={
            "request_id": "request-123",
            "attempt": 2,
        },
    )

    payload = context.model_dump(mode="json")
    serialized = json.dumps(payload)
    decoded = json.loads(serialized)

    restored = EmbeddingErrorContext.model_validate(
        decoded
    )

    assert restored == context


def test_operation_enum_becomes_string() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
    )

    payload = context.model_dump(mode="json")

    assert payload["operation"] == "embed_batch"
    assert isinstance(payload["operation"], str)


def test_none_fields_are_json_compatible() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
    )

    payload = context.model_dump(mode="json")

    assert payload["provider"] is None
    assert payload["model"] is None
    assert payload["batch_index"] is None
    assert payload["input_count"] is None


def test_nested_metadata_is_preserved() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        metadata={
            "request": {
                "id": "request-1",
                "chunks": [1, 2, 3],
            },
            "provider_response": {
                "status": 429,
                "retry_after": 10,
            },
        },
    )

    payload = context.model_dump(mode="json")

    assert payload["metadata"] == {
        "request": {
            "id": "request-1",
            "chunks": [1, 2, 3],
        },
        "provider_response": {
            "status": 429,
            "retry_after": 10,
        },
    }


def test_unicode_metadata_survives_serialization() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "language": "hi",
            "message": "एम्बेडिंग विफल हुई",
            "japanese": "埋め込みに失敗しました",
            "arabic": "فشل التضمين",
        },
    )

    payload = context.model_dump(mode="json")

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
    )

    decoded = json.loads(serialized)

    assert decoded["metadata"] == context.metadata


def test_uuid_values_inside_metadata_are_json_serialized() -> None:
    request_id = uuid4()

    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "request_id": request_id,
        },
    )

    payload = context.model_dump(mode="json")

    assert payload["metadata"]["request_id"] == str(request_id)


def test_uuid_string_can_be_reconstructed_from_metadata() -> None:
    request_id = uuid4()

    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "request_id": request_id,
        },
    )

    payload = context.model_dump(mode="json")

    reconstructed_id = UUID(
        payload["metadata"]["request_id"]
    )

    assert reconstructed_id == request_id


def test_integer_and_float_metadata_are_preserved() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "attempt": 3,
            "latency_seconds": 1.25,
        },
    )

    payload = context.model_dump(mode="json")

    assert payload["metadata"]["attempt"] == 3
    assert payload["metadata"]["latency_seconds"] == 1.25


def test_boolean_metadata_is_preserved() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "retryable": True,
            "cached": False,
        },
    )

    payload = context.model_dump(mode="json")

    assert payload["metadata"]["retryable"] is True
    assert payload["metadata"]["cached"] is False


def test_null_metadata_value_is_preserved() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "error_detail": None,
        },
    )

    payload = context.model_dump(mode="json")

    assert payload["metadata"]["error_detail"] is None


def test_empty_metadata_serializes_as_empty_object() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
    )

    payload = context.model_dump(mode="json")

    assert payload["metadata"] == {}


def test_serialization_is_deterministic() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=5,
        input_count=20,
        metadata={
            "request_id": "abc",
            "attempt": 1,
        },
    )

    first = context.model_dump(mode="json")
    second = context.model_dump(mode="json")

    assert first == second


def test_serialized_json_is_deterministic_for_same_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=5,
        input_count=20,
        metadata={
            "request_id": "abc",
            "attempt": 1,
        },
    )

    first = json.dumps(
        context.model_dump(mode="json"),
        sort_keys=True,
    )

    second = json.dumps(
        context.model_dump(mode="json"),
        sort_keys=True,
    )

    assert first == second


def test_serialization_does_not_mutate_context() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=2,
        input_count=8,
        metadata={
            "attempt": 1,
        },
    )

    before = context.model_dump(mode="json")

    context.model_dump(mode="json")

    after = context.model_dump(mode="json")

    assert before == after


def test_original_context_remains_immutable_after_serialization() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="fake",
        model="fake-model",
    )

    context.model_dump(mode="json")

    with pytest.raises(ValidationError):
        context.provider = "another-provider"


def test_serialized_payload_is_independent_from_context_dump() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        metadata={
            "attempt": 1,
        },
    )

    payload = context.model_dump(mode="json")

    payload["metadata"]["attempt"] = 99

    assert context.metadata["attempt"] == 1


def test_complex_metadata_round_trip() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=7,
        input_count=50,
        metadata={
            "source": {
                "project_id": "project-1",
                "document_id": "document-1",
            },
            "location": {
                "pages": [1, 2, 3],
                "section_path": [
                    "Introduction",
                    "Background",
                ],
            },
            "timing": {
                "start": 12.5,
                "end": 30.75,
            },
        },
    )

    payload = context.model_dump(mode="json")

    restored = EmbeddingErrorContext.model_validate(
        json.loads(json.dumps(payload))
    )

    assert restored == context


def test_model_dump_json_returns_valid_json() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED,
        provider="fake",
        model="fake-model",
    )

    serialized = context.model_dump_json()

    decoded = json.loads(serialized)

    assert decoded["operation"] == "embed"
    assert decoded["provider"] == "fake"
    assert decoded["model"] == "fake-model"


def test_model_validate_json_round_trip() -> None:
    context = EmbeddingErrorContext(
        operation=EmbeddingOperation.EMBED_BATCH,
        provider="fake",
        model="fake-model",
        batch_index=2,
        input_count=12,
    )

    serialized = context.model_dump_json()

    restored = EmbeddingErrorContext.model_validate_json(
        serialized
    )

    assert restored == context


def test_json_schema_contains_expected_fields() -> None:
    schema = EmbeddingErrorContext.model_json_schema()

    properties = schema["properties"]

    assert "operation" in properties
    assert "provider" in properties
    assert "model" in properties
    assert "batch_index" in properties
    assert "input_count" in properties
    assert "metadata" in properties


def test_json_schema_rejects_unknown_fields() -> None:
    schema = EmbeddingErrorContext.model_json_schema()

    assert schema.get("additionalProperties") is False