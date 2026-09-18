from __future__ import annotations

from uuid import uuid4

import pytest

from app.vector_store.port import VectorStorePort
from app.vector_store.schemas import (
    VectorBatchPersistenceRequest,
    VectorDeleteRequest,
    VectorDeleteResult,
    VectorPersistenceRequest,
    VectorPersistenceResult,
    VectorRecord,
)
from app.models_ai.embeddings.schemas import EmbeddingModelInfo


class FakeVectorStore(VectorStorePort):
    def __init__(self) -> None:
        self.records: dict = {}

    async def persist(
        self,
        request: VectorPersistenceRequest,
    ) -> VectorRecord:
        self.records[request.embedding.chunk_id] = request.embedding
        return request.embedding

    async def persist_batch(
        self,
        request: VectorBatchPersistenceRequest,
    ) -> VectorPersistenceResult:
        for record in request.records:
            self.records[record.chunk_id] = record

        return VectorPersistenceResult(
            records=request.records,
            count=len(request.records),
        )

    async def get_by_chunk_id(
        self,
        chunk_id,
    ) -> VectorRecord | None:
        return self.records.get(chunk_id)

    async def exists(
        self,
        chunk_id,
    ) -> bool:
        return chunk_id in self.records

    async def delete(
        self,
        request: VectorDeleteRequest,
    ) -> VectorDeleteResult:
        deleted = 1 if request.chunk_id in self.records else 0
        self.records.pop(request.chunk_id, None)

        return VectorDeleteResult(
            chunk_id=request.chunk_id,
            deleted_count=deleted,
        )


def create_record() -> VectorRecord:
    return VectorRecord(
        chunk_id=uuid4(),
        values=[0.1, 0.2, 0.3],
        model=EmbeddingModelInfo(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=3,
            normalized=True,
        ),
    )


@pytest.mark.asyncio
async def test_port_supports_single_persistence() -> None:
    store = FakeVectorStore()
    record = create_record()

    result = await store.persist(
        VectorPersistenceRequest(
            chunk_id=record.chunk_id,
            embedding=record,
        )
    )

    assert result == record


@pytest.mark.asyncio
async def test_port_supports_batch_persistence() -> None:
    store = FakeVectorStore()

    records = [
        create_record(),
        create_record(),
        create_record(),
    ]

    result = await store.persist_batch(
        VectorBatchPersistenceRequest(
            records=records,
        )
    )

    assert result.count == 3
    assert result.records == records


@pytest.mark.asyncio
async def test_port_supports_chunk_lookup() -> None:
    store = FakeVectorStore()
    record = create_record()

    await store.persist(
        VectorPersistenceRequest(
            chunk_id=record.chunk_id,
            embedding=record,
        )
    )

    result = await store.get_by_chunk_id(record.chunk_id)

    assert result == record


@pytest.mark.asyncio
async def test_port_supports_existence_check() -> None:
    store = FakeVectorStore()
    record = create_record()

    assert await store.exists(record.chunk_id) is False

    await store.persist(
        VectorPersistenceRequest(
            chunk_id=record.chunk_id,
            embedding=record,
        )
    )

    assert await store.exists(record.chunk_id) is True


@pytest.mark.asyncio
async def test_port_supports_delete() -> None:
    store = FakeVectorStore()
    record = create_record()

    await store.persist(
        VectorPersistenceRequest(
            chunk_id=record.chunk_id,
            embedding=record,
        )
    )

    result = await store.delete(
        VectorDeleteRequest(
            chunk_id=record.chunk_id,
        )
    )

    assert result.deleted_count == 1
    assert await store.exists(record.chunk_id) is False


def test_vector_store_port_is_abstract() -> None:
    with pytest.raises(TypeError):
        VectorStorePort()