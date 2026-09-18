from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import SessionFactory
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding
from app.models.knowledge_document import KnowledgeDocument
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)
from app.vector_store.retrieval.schemas import VectorRetrievalRequest


VECTOR_DIMENSION = 384


def make_vector(first_value: float = 1.0) -> list[float]:
    """Create a deterministic 384-dimensional vector."""
    return [first_value] + [0.0] * (VECTOR_DIMENSION - 1)


def make_model(
    *,
    provider: str = "test-provider",
    model_name: str = "test-model",
    dimension: int = VECTOR_DIMENSION,
) -> EmbeddingModelInfo:
    """Create an embedding model descriptor."""
    return EmbeddingModelInfo(
        provider=provider,
        model_name=model_name,
        dimension=dimension,
        normalized=True,
    )


def make_request(
    *,
    vector: list[float] | None = None,
    provider: str = "test-provider",
    model_name: str = "test-model",
    dimension: int = VECTOR_DIMENSION,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    project_id: UUID | None = None,
    metadata_filter: dict | None = None,
) -> VectorRetrievalRequest:
    """Create a vector retrieval request."""
    return VectorRetrievalRequest(
        query_vector=vector or make_vector(),
        model=make_model(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
        ),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        project_id=project_id,
        metadata_filter=metadata_filter or {},
    )


async def create_test_user() -> UUID:
    """Create and persist a test user."""
    user_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"vector-retrieval-{user_id}@example.com",
            name="Vector Retrieval Test User",
        )

        session.add(user)
        await session.commit()

    return user_id


async def create_test_project(user_id: UUID) -> UUID:
    """Create and persist a test project."""
    project_id = uuid4()

    async with SessionFactory() as session:
        project = Project(
            id=project_id,
            owner_id=user_id,
            name=f"Vector Retrieval Project {project_id}",
            description="Vector retrieval integration test project.",
        )

        session.add(project)
        await session.commit()

    return project_id


async def create_test_source(project_id: UUID) -> UUID:
    """Create and persist a test source."""
    source_id = uuid4()

    async with SessionFactory() as session:
        source = Source(
            id=source_id,
            project_id=project_id,
            source_type="text",
            title="Vector Retrieval Test Source",
            original_filename="vector-retrieval-test.txt",
            mime_type="text/plain",
            storage_uri=f"file:///tmp/vector-retrieval-{source_id}.txt",
            content_hash=f"vector-source-{source_id}",
            source_metadata={},
            status="COMPLETED",
        )

        session.add(source)
        await session.commit()

    return source_id


async def create_test_chunk(
    project_id: UUID,
    source_id: UUID,
    *,
    version: int = 1,
    text: str = "Test knowledge chunk.",
    metadata: dict | None = None,
) -> UUID:
    """Create and persist a knowledge document and chunk."""
    document_id = uuid4()
    chunk_id = uuid4()

    async with SessionFactory() as session:
        document = KnowledgeDocument(
            id=document_id,
            project_id=project_id,
            source_id=source_id,
            version=version,
            title=f"Vector Retrieval Document {version}",
            language="en",
            content_hash=f"document-{document_id}",
            status="COMPLETED",
            document_metadata={},
        )

        session.add(document)

        await session.flush()

        chunk = KnowledgeChunk(
            id=chunk_id,
            document_id=document_id,
            chunk_index=0,
            text=text,
            content_hash=f"chunk-{chunk_id}",
            token_count=len(text.split()),
            chunk_metadata=metadata or {},

        )

        session.add(chunk)

        await session.commit()

    return chunk_id


async def create_test_embedding(
    chunk_id: UUID,
    *,
    provider: str = "test-provider",
    model_name: str = "test-model",
    vector: list[float] | None = None,
    metadata: dict | None = None,
) -> None:
    """Create and persist a test embedding."""
    async with SessionFactory() as session:
        embedding = KnowledgeChunkEmbedding(
            id=uuid4(),
            chunk_id=chunk_id,
            provider=provider,
            model_name=model_name,
            dimension=VECTOR_DIMENSION,
            normalized=True,
            embedding=vector or make_vector(),
            vector_metadata=metadata or {},
        )

        session.add(embedding)

        await session.commit()


async def cleanup_test_project(
    user_id: UUID,
    project_id: UUID,
) -> None:
    """
    Remove the test project and user.

    Related sources, knowledge documents, chunks, and embeddings
    are removed through the database foreign-key cascade.
    """
    async with SessionFactory() as session:
        project = await session.get(
            Project,
            project_id,
        )

        user = await session.get(
            User,
            user_id,
        )

        if project is not None:
            await session.delete(project)

        if user is not None:
            await session.delete(user)

        await session.commit()


async def create_retrieval_session() -> AsyncSession:
    """
    Create a dedicated database session for retrieval.

    The caller owns the returned session and must close it.
    """
    return SessionFactory()


@pytest.mark.asyncio
async def test_search_returns_matches_in_similarity_order() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        chunk_1 = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="Most similar chunk.",
        )

        chunk_2 = await create_test_chunk(
            project_id,
            source_id,
            version=2,
            text="Second most similar chunk.",
        )

        chunk_3 = await create_test_chunk(
            project_id,
            source_id,
            version=3,
            text="Least similar chunk.",
        )

        await create_test_embedding(
            chunk_1,
            vector=make_vector(1.0),
        )

        await create_test_embedding(
            chunk_2,
            vector=make_vector(0.8),
        )

        await create_test_embedding(
            chunk_3,
            vector=make_vector(-1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    project_id=project_id,
                    top_k=5,
                )
            )

        assert result.count == 3
        assert len(result.matches) == 3

        assert result.matches[0].chunk_id == chunk_1
        assert result.matches[1].chunk_id == chunk_2
        assert result.matches[2].chunk_id == chunk_3

        assert result.matches[0].similarity > result.matches[1].similarity
        assert result.matches[1].similarity > result.matches[2].similarity

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_respects_top_k() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        for version in range(1, 6):
            chunk_id = await create_test_chunk(
                project_id,
                source_id,
                version=version,
                text=f"Test chunk {version}.",
            )

            await create_test_embedding(
                chunk_id,
                vector=make_vector(1.0),
            )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    project_id=project_id,
                    top_k=2,
                )
            )

        assert result.count == 2
        assert len(result.matches) == 2

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_isolates_embedding_model() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        chunk_1 = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="Model A chunk.",
        )

        chunk_2 = await create_test_chunk(
            project_id,
            source_id,
            version=2,
            text="Model B chunk.",
        )

        await create_test_embedding(
            chunk_1,
            provider="test-provider",
            model_name="model-a",
            vector=make_vector(1.0),
        )

        await create_test_embedding(
            chunk_2,
            provider="test-provider",
            model_name="model-b",
            vector=make_vector(1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    provider="test-provider",
                    model_name="model-a",
                    project_id=project_id,
                )
            )

        assert result.count == 1
        assert result.matches[0].chunk_id == chunk_1
        assert result.matches[0].model.provider == "test-provider"
        assert result.matches[0].model.model_name == "model-a"

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_respects_project_filter() -> None:
    user_id = await create_test_user()

    project_one = await create_test_project(user_id)
    project_two = await create_test_project(user_id)

    source_one = await create_test_source(project_one)
    source_two = await create_test_source(project_two)

    try:
        chunk_one = await create_test_chunk(
            project_one,
            source_one,
            version=1,
            text="Project one chunk.",
        )

        chunk_two = await create_test_chunk(
            project_two,
            source_two,
            version=1,
            text="Project two chunk.",
        )

        await create_test_embedding(
            chunk_one,
            vector=make_vector(1.0),
        )

        await create_test_embedding(
            chunk_two,
            vector=make_vector(1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    project_id=project_one,
                    top_k=5,
                )
            )

        assert result.count == 1
        assert result.matches[0].chunk_id == chunk_one

    finally:
        await cleanup_test_project(
            user_id,
            project_one,
        )

        await cleanup_test_project(
            user_id,
            project_two,
        )


@pytest.mark.asyncio
async def test_search_respects_similarity_threshold() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        high_similarity_chunk = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="High similarity chunk.",
        )

        low_similarity_chunk = await create_test_chunk(
            project_id,
            source_id,
            version=2,
            text="Low similarity chunk.",
        )

        await create_test_embedding(
            high_similarity_chunk,
            vector=make_vector(1.0),
        )

        await create_test_embedding(
            low_similarity_chunk,
            vector=make_vector(-1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    project_id=project_id,
                    similarity_threshold=0.5,
                )
            )

        assert result.count == 1
        assert result.matches[0].chunk_id == high_similarity_chunk
        assert result.matches[0].similarity >= 0.5

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_returns_empty_result_when_no_vectors_match() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        chunk_id = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="Different embedding model.",
        )

        await create_test_embedding(
            chunk_id,
            provider="other-provider",
            model_name="other-model",
            vector=make_vector(1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    provider="test-provider",
                    model_name="test-model",
                    project_id=project_id,
                )
            )

        assert result.count == 0
        assert result.matches == []

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_returns_chunk_metadata() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        chunk_id = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="Chunk with metadata.",
            metadata={
                "page": 5,
                "section": "Introduction",
                "language": "en",
            },
        )

        await create_test_embedding(
            chunk_id,
            vector=make_vector(1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    project_id=project_id,
                )
            )

        assert result.count == 1

        metadata = result.matches[0].metadata

        assert metadata["page"] == 5
        assert metadata["section"] == "Introduction"
        assert metadata["language"] == "en"

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_rejects_wrong_model_dimension() -> None:
    async with SessionFactory() as session:
        retrieval = PostgresVectorRetrieval(session)

        request = make_request(
            vector=[1.0] * 128,
            dimension=128,
        )

        with pytest.raises(
            ValueError,
            match="Unsupported embedding dimension",
        ):
            await retrieval.search(request)


@pytest.mark.asyncio
async def test_search_rejects_wrong_query_dimension() -> None:
    async with SessionFactory() as session:
        retrieval = PostgresVectorRetrieval(session)

        request = make_request(
            vector=[1.0] * 383,
            dimension=VECTOR_DIMENSION,
        )

        with pytest.raises(
            ValueError,
            match="Query vector dimension",
        ):
            await retrieval.search(request)


@pytest.mark.asyncio
async def test_search_returns_model_information() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        chunk_id = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="Model information test chunk.",
        )

        await create_test_embedding(
            chunk_id,
            provider="sentence-transformers",
            model_name="all-MiniLM-L6-v2",
            vector=make_vector(1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    provider="sentence-transformers",
                    model_name="all-MiniLM-L6-v2",
                    project_id=project_id,
                )
            )

        assert result.count == 1

        model = result.matches[0].model

        assert model.provider == "sentence-transformers"
        assert model.model_name == "all-MiniLM-L6-v2"
        assert model.dimension == VECTOR_DIMENSION
        assert model.normalized is True

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )


@pytest.mark.asyncio
async def test_search_result_contains_retrieval_metadata() -> None:
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    try:
        chunk_id = await create_test_chunk(
            project_id,
            source_id,
            version=1,
            text="Retrieval metadata test chunk.",
        )

        await create_test_embedding(
            chunk_id,
            provider="test-provider",
            model_name="test-model",
            vector=make_vector(1.0),
        )

        async with SessionFactory() as session:
            retrieval = PostgresVectorRetrieval(session)

            result = await retrieval.search(
                make_request(
                    vector=make_vector(1.0),
                    provider="test-provider",
                    model_name="test-model",
                    project_id=project_id,
                    top_k=3,
                )
            )

        assert result.count == 1

        assert result.metadata["provider"] == "test-provider"
        assert result.metadata["model_name"] == "test-model"
        assert result.metadata["top_k"] == 3
        assert result.metadata["similarity_metric"] == "cosine"

    finally:
        await cleanup_test_project(
            user_id,
            project_id,
        )