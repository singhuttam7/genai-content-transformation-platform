from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.database.session import SessionFactory
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_chunk_embedding import (
    KnowledgeChunkEmbedding,
)
from app.models.knowledge_document import KnowledgeDocument
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.models_ai.embeddings.schemas import (
    EmbeddingModelInfo,
    EmbeddingVector,
)
from app.vector_store.adapters.postgres import PostgresVectorStore
from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)
from app.vector_store.retrieval.schemas import (
    VectorRetrievalRequest,
)
from app.vector_store.retrieval.service import (
    VectorRetrievalService,
)
from app.vector_store.schemas import VectorDeleteRequest
from app.vector_store.service import VectorPersistenceService


VECTOR_DIMENSION = 384


def make_vector(
    first: float = 1.0,
    second: float = 0.0,
) -> list[float]:
    """Create a deterministic vector."""
    return [first, second] + [0.0] * (
        VECTOR_DIMENSION - 2
    )


def make_model(
    provider: str = "persistence-provider",
    model_name: str = "persistence-model",
) -> EmbeddingModelInfo:
    """Create a deterministic embedding model."""
    return EmbeddingModelInfo(
        provider=provider,
        model_name=model_name,
        dimension=VECTOR_DIMENSION,
        normalized=True,
    )


def make_embedding(
    *,
    vector: list[float],
    model: EmbeddingModelInfo,
    input_index: int = 0,
) -> EmbeddingVector:
    """Create an embedding vector."""
    return EmbeddingVector(
        values=vector,
        model=model,
        input_index=input_index,
    )


async def create_user() -> UUID:
    """Create a test user."""
    user_id = uuid4()

    async with SessionFactory() as session:
        session.add(
            User(
                id=user_id,
                email=f"a569-persistence-{user_id}@example.com",
                name="A5.6.9.4 Persistence User",
            )
        )

        await session.commit()

    return user_id


async def create_project(
    user_id: UUID,
) -> UUID:
    """Create a test project."""
    project_id = uuid4()

    async with SessionFactory() as session:
        session.add(
            Project(
                id=project_id,
                owner_id=user_id,
                name=(
                    "A5.6.9.4 Persistence Project "
                    f"{project_id}"
                ),
                description=(
                    "Persistence consistency test project."
                ),
            )
        )

        await session.commit()

    return project_id


async def create_source(
    project_id: UUID,
) -> UUID:
    """Create a test source."""
    source_id = uuid4()

    async with SessionFactory() as session:
        session.add(
            Source(
                id=source_id,
                project_id=project_id,
                source_type="text",
                title="Persistence Test Source",
                original_filename="persistence.txt",
                mime_type="text/plain",
                storage_uri=(
                    f"file:///tmp/a5.6.9.4-{source_id}.txt"
                ),
                content_hash=f"source-{source_id}",
                source_metadata={
                    "test": True,
                    "stage": "A5.6.9.4.3",
                },
                status="COMPLETED",
            )
        )

        await session.commit()

    return source_id


async def create_chunk(
    project_id: UUID,
    source_id: UUID,
    *,
    version: int,
    text: str,
    metadata: dict[str, object] | None = None,
) -> UUID:
    """Create a knowledge document and chunk."""
    document_id = uuid4()
    chunk_id = uuid4()

    async with SessionFactory() as session:
        session.add(
            KnowledgeDocument(
                id=document_id,
                project_id=project_id,
                source_id=source_id,
                version=version,
                title=f"Persistence Document {version}",
                language="en",
                content_hash=f"document-{document_id}",
                status="COMPLETED",
                document_metadata={},
            )
        )

        await session.flush()

        session.add(
            KnowledgeChunk(
                id=chunk_id,
                document_id=document_id,
                chunk_index=0,
                text=text,
                content_hash=f"chunk-{chunk_id}",
                token_count=len(text.split()),
                chunk_metadata=metadata or {},
            )
        )

        await session.commit()

    return chunk_id


async def retrieve(
    request: VectorRetrievalRequest,
):
    """Retrieve through the application service."""
    async with SessionFactory() as session:
        adapter = PostgresVectorRetrieval(session)
        service = VectorRetrievalService(adapter)

        return await service.search(request)


@pytest.mark.asyncio
async def test_persisted_embedding_is_visible_after_commit() -> None:
    """A committed embedding must be visible to retrieval."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_id = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="Committed persistence chunk.",
    )

    model = make_model()

    embedding = make_embedding(
        vector=make_vector(),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        persisted = await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
            metadata={
                "persistence": "committed",
            },
        )

        await session.commit()

    assert persisted.chunk_id == chunk_id

    result = await retrieve(
        VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=model,
            top_k=5,
            project_id=project_id,
        )
    )

    assert result.count == 1
    assert result.matches[0].chunk_id == chunk_id
    assert result.matches[0].similarity > 0.99
    assert (
        result.matches[0].metadata["embedding"]["persistence"]
        == "committed"
    )


@pytest.mark.asyncio
async def test_uncommitted_embedding_is_not_visible_to_new_session() -> None:
    """Uncommitted persistence must not leak across sessions."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_id = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="Uncommitted persistence chunk.",
    )

    model = make_model()

    embedding = make_embedding(
        vector=make_vector(),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
        )

        result = await retrieve(
            VectorRetrievalRequest(
                query_vector=list(embedding.values),
                model=model,
                top_k=5,
                project_id=project_id,
            )
        )

        assert result.count == 0

        await session.rollback()


@pytest.mark.asyncio
async def test_multiple_chunks_remain_independently_retrievable() -> None:
    """Persistence must preserve chunk-to-vector associations."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_one = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="First persistence chunk.",
    )

    chunk_two = await create_chunk(
        project_id,
        source_id,
        version=2,
        text="Second persistence chunk.",
    )

    model = make_model()

    embedding_one = make_embedding(
        vector=make_vector(1.0, 0.0),
        model=model,
    )

    embedding_two = make_embedding(
        vector=make_vector(0.0, 1.0),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        await service.persist_embedding(
            chunk_id=chunk_one,
            embedding=embedding_one,
        )

        await service.persist_embedding(
            chunk_id=chunk_two,
            embedding=embedding_two,
        )

        await session.commit()

    result = await retrieve(
        VectorRetrievalRequest(
            query_vector=list(embedding_one.values),
            model=model,
            top_k=2,
            project_id=project_id,
        )
    )

    assert result.count == 2

    assert result.matches[0].chunk_id == chunk_one
    assert result.matches[0].similarity > 0.99

    assert result.matches[1].chunk_id == chunk_two
    assert result.matches[1].similarity < 0.01


@pytest.mark.asyncio
async def test_deleting_one_embedding_preserves_other_embeddings() -> None:
    """Deleting one vector must not remove another chunk's vector."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_one = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="First deletion consistency chunk.",
    )

    chunk_two = await create_chunk(
        project_id,
        source_id,
        version=2,
        text="Second deletion consistency chunk.",
    )

    model = make_model()

    embedding_one = make_embedding(
        vector=make_vector(1.0, 0.0),
        model=model,
    )

    embedding_two = make_embedding(
        vector=make_vector(0.0, 1.0),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        await service.persist_embedding(
            chunk_id=chunk_one,
            embedding=embedding_one,
        )

        await service.persist_embedding(
            chunk_id=chunk_two,
            embedding=embedding_two,
        )

        await session.commit()

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)

        delete_result = await store.delete(
            VectorDeleteRequest(
                chunk_id=chunk_one,
            )
        )

        await session.commit()

    assert delete_result.deleted_count == 1

    result = await retrieve(
        VectorRetrievalRequest(
            query_vector=list(embedding_two.values),
            model=model,
            top_k=5,
            project_id=project_id,
        )
    )

    assert result.count == 1
    assert result.matches[0].chunk_id == chunk_two


@pytest.mark.asyncio
async def test_persistence_round_trip_preserves_model_identity() -> None:
    """Persistence and retrieval must preserve model identity."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_id = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="Model identity persistence chunk.",
    )

    model = make_model(
        provider="round-trip-provider",
        model_name="round-trip-model",
    )

    embedding = make_embedding(
        vector=make_vector(),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
        )

        await session.commit()

    result = await retrieve(
        VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=model,
            top_k=1,
            project_id=project_id,
        )
    )

    assert result.count == 1

    returned_model = result.matches[0].model

    assert returned_model.provider == model.provider
    assert returned_model.model_name == model.model_name
    assert returned_model.dimension == model.dimension
    assert returned_model.normalized == model.normalized


@pytest.mark.asyncio
async def test_persistence_round_trip_preserves_chunk_metadata() -> None:
    """Persistence must not alter knowledge chunk metadata."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_id = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="Chunk metadata persistence test.",
        metadata={
            "page": 12,
            "section": "Persistence",
            "nested": {
                "source": "test",
            },
        },
    )

    model = make_model()

    embedding = make_embedding(
        vector=make_vector(),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
        )

        await session.commit()

    result = await retrieve(
        VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=model,
            top_k=1,
            project_id=project_id,
        )
    )

    assert result.count == 1

    metadata = result.matches[0].metadata

    assert metadata["page"] == 12
    assert metadata["section"] == "Persistence"
    assert metadata["nested"]["source"] == "test"


@pytest.mark.asyncio
async def test_persistence_round_trip_preserves_embedding_metadata() -> None:
    """Embedding metadata must survive persistence and retrieval."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)

    chunk_id = await create_chunk(
        project_id,
        source_id,
        version=1,
        text="Embedding metadata persistence test.",
    )

    model = make_model()

    embedding = make_embedding(
        vector=make_vector(),
        model=model,
    )

    async with SessionFactory() as session:
        store = PostgresVectorStore(session)
        service = VectorPersistenceService(store)

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
            metadata={
                "runtime": "persistence-test",
                "version": 1,
                "nested": {
                    "enabled": True,
                },
            },
        )

        await session.commit()

    result = await retrieve(
        VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=model,
            top_k=1,
            project_id=project_id,
        )
    )

    assert result.count == 1

    embedding_metadata = result.matches[0].metadata[
        "embedding"
    ]

    assert embedding_metadata["runtime"] == "persistence-test"
    assert embedding_metadata["version"] == 1
    assert embedding_metadata["nested"]["enabled"] is True