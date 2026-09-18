from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.schemas import (
    VectorBatchPersistenceRequest,
    VectorDeleteResult,
    VectorRecord,
)


def create_model() -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        dimension=3,
        normalized=True,
    )


def create_record() -> VectorRecord:
    return VectorRecord(
        chunk_id=uuid4(),
        values=[0.1, 0.2, 0.3],
        model=create_model(),
    )


def test_vector_record_accepts_valid_vector() -> None:
    record = create_record()

    assert len(record.values) == 3
    assert record.model.dimension == 3


def test_vector_record_rejects_empty_vector() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(
            chunk_id=uuid4(),
            values=[],
            model=create_model(),
        )


def test_vector_record_rejects_nan() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(
            chunk_id=uuid4(),
            values=[0.1, float("nan"), 0.3],
            model=create_model(),
        )


def test_vector_record_rejects_positive_infinity() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(
            chunk_id=uuid4(),
            values=[0.1, float("inf"), 0.3],
            model=create_model(),
        )


def test_vector_record_rejects_negative_infinity() -> None:
    with pytest.raises(ValidationError):
        VectorRecord(
            chunk_id=uuid4(),
            values=[0.1, float("-inf"), 0.3],
            model=create_model(),
        )


def test_vector_record_is_immutable() -> None:
    record = create_record()

    with pytest.raises(ValidationError):
        record.chunk_id = uuid4()


def test_batch_request_requires_records() -> None:
    with pytest.raises(ValidationError):
        VectorBatchPersistenceRequest(records=[])


def test_delete_result_allows_zero_deleted() -> None:
    result = VectorDeleteResult(
        chunk_id=uuid4(),
        deleted_count=0,
    )

    assert result.deleted_count == 0