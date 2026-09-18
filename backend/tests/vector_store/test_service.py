from __future__ import annotations

from uuid import uuid4

import pytest

from app.models_ai.embeddings.schemas import (
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingVector,
)
from app.vector_store.port import VectorStorePort
from app.vector_store.schemas import (
    VectorBatchPersistenceRequest,
    VectorDeleteRequest,
    VectorDeleteResult,
    VectorPersistenceRequest,
    VectorPersistenceResult,
    VectorRecord,
)
from app.vector_store.service import VectorPersistenceService


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
        deleted = int(request.chunk_id in self.records)
        self.records.pop(request.chunk_id, None)

        return VectorDeleteResult(
            chunk_id=request.chunk_id,
            deleted_count=deleted,
        )


def create_embedding(
    index: int = 0,
) -> EmbeddingVector:
    return EmbeddingVector(
        values=[0.1, 0.2, 0.3],
        model=EmbeddingModelInfo(
            provider="sentence-transformers",
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            dimension=3,
            normalized=True,
        ),
        input_index=index,
    )


def create_service() -> tuple[
    VectorPersistenceService,
    FakeVectorStore,
]:
    store = FakeVectorStore()
    return VectorPersistenceService(store), store


@pytest.mark.asyncio
async def test_persist_embedding_converts_and_persists_record() -> None:
    service, store = create_service()
    chunk_id = uuid4()

    result = await service.persist_embedding(
        chunk_id=chunk_id,
        embedding=create_embedding(),
    )

    assert result.chunk_id == chunk_id
    assert result.values == [0.1, 0.2, 0.3]
    assert chunk_id in store.records


@pytest.mark.asyncio
async def test_persist_embedding_preserves_model_metadata() -> None:
    service, _ = create_service()

    result = await service.persist_embedding(
        chunk_id=uuid4(),
        embedding=create_embedding(),
    )

    assert result.model.provider == "sentence-transformers"
    assert result.model.dimension == 3
    assert result.model.normalized is True


@pytest.mark.asyncio
async def test_persist_embedding_copies_metadata() -> None:
    service, _ = create_service()

    metadata = {
        "source": {
            "page": 3,
        },
    }

    result = await service.persist_embedding(
        chunk_id=uuid4(),
        embedding=create_embedding(),
        metadata=metadata,
    )

    metadata["source"]["page"] = 99

    assert result.metadata["source"]["page"] == 3


@pytest.mark.asyncio
async def test_persist_embedding_records_input_index() -> None:
    service, _ = create_service()

    result = await service.persist_embedding(
        chunk_id=uuid4(),
        embedding=create_embedding(index=7),
    )

    assert result.metadata["input_index"] == 7


@pytest.mark.asyncio
async def test_persist_batch_preserves_chunk_order() -> None:
    service, _ = create_service()

    chunk_ids = [
        uuid4(),
        uuid4(),
        uuid4(),
    ]

    result = EmbeddingBatchResult(
        embeddings=[
            create_embedding(0),
            create_embedding(1),
            create_embedding(2),
        ],
        model=create_embedding().model,
        input_count=3,
        dimension=3,
    )

    persisted = await service.persist_batch(
        chunk_ids=chunk_ids,
        result=result,
    )

    assert [
        record.chunk_id
        for record in persisted.records
    ] == chunk_ids


@pytest.mark.asyncio
async def test_persist_batch_preserves_vectors() -> None:
    service, _ = create_service()

    embeddings = [
        create_embedding(0),
        create_embedding(1),
    ]

    result = EmbeddingBatchResult(
        embeddings=embeddings,
        model=embeddings[0].model,
        input_count=2,
        dimension=3,
    )

    persisted = await service.persist_batch(
        chunk_ids=[uuid4(), uuid4()],
        result=result,
    )

    assert [
        record.values
        for record in persisted.records
    ] == [
        [0.1, 0.2, 0.3],
        [0.1, 0.2, 0.3],
    ]


@pytest.mark.asyncio
async def test_persist_batch_rejects_count_mismatch() -> None:
    service, _ = create_service()

    result = EmbeddingBatchResult(
        embeddings=[create_embedding()],
        model=create_embedding().model,
        input_count=1,
        dimension=3,
    )

    with pytest.raises(ValueError):
        await service.persist_batch(
            chunk_ids=[uuid4(), uuid4()],
            result=result,
        )


@pytest.mark.asyncio
async def test_get_embedding_delegates_to_store() -> None:
    service, _ = create_service()
    chunk_id = uuid4()

    await service.persist_embedding(
        chunk_id=chunk_id,
        embedding=create_embedding(),
    )

    result = await service.get_embedding(chunk_id)

    assert result is not None
    assert result.chunk_id == chunk_id


@pytest.mark.asyncio
async def test_exists_delegates_to_store() -> None:
    service, _ = create_service()
    chunk_id = uuid4()

    assert await service.exists(chunk_id) is False

    await service.persist_embedding(
        chunk_id=chunk_id,
        embedding=create_embedding(),
    )

    assert await service.exists(chunk_id) is True


@pytest.mark.asyncio
async def test_delete_delegates_to_store() -> None:
    service, _ = create_service()
    chunk_id = uuid4()

    await service.persist_embedding(
        chunk_id=chunk_id,
        embedding=create_embedding(),
    )

    result = await service.delete(chunk_id)

    assert result.deleted_count == 1
    assert await service.exists(chunk_id) is False


def test_service_rejects_invalid_store() -> None:
    with pytest.raises(TypeError):
        VectorPersistenceService(object())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_persist_embedding_rejects_invalid_chunk_id() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError):
        await service.persist_embedding(
            chunk_id="invalid",  # type: ignore[arg-type]
            embedding=create_embedding(),
        )


@pytest.mark.asyncio
async def test_persist_embedding_rejects_invalid_embedding() -> None:
    service, _ = create_service()

    with pytest.raises(TypeError):
        await service.persist_embedding(
            chunk_id=uuid4(),
            embedding=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_persist_batch_rejects_empty_chunk_ids() -> None:
    service, _ = create_service()

    result = EmbeddingBatchResult(
        embeddings=[create_embedding()],
        model=create_embedding().model,
        input_count=1,
        dimension=3,
    )

    with pytest.raises(ValueError):
        await service.persist_batch(
            chunk_ids=[],
            result=result,
        )