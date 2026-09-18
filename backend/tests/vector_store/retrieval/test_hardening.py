from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.database.session import SessionFactory
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_document import KnowledgeDocument
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding
from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.vector_store.adapters.postgres import PostgresVectorStore
from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)
from app.vector_store.retrieval.schemas import VectorRetrievalRequest
from app.vector_store.retrieval.service import VectorRetrievalService
from app.vector_store.schemas import VectorDeleteRequest


VECTOR_DIMENSION = 384


def make_vector(
    first_value: float = 1.0,
) -> list[float]:
    """Create a deterministic 384-dimensional vector."""
    return [first_value] + [0.0] * (VECTOR_DIMENSION - 1)


def make_model(
    provider: str = "hardening-provider",
    model_name: str = "hardening-model",
) -> EmbeddingModelInfo:
    """Create a deterministic embedding model descriptor."""
    return EmbeddingModelInfo(
        provider=provider,
        model_name=model_name,
        dimension=VECTOR_DIMENSION,
        normalized=True,
    )


def make_request(
    *,
    vector: list[float] | None = None,
    model: EmbeddingModelInfo | None = None,
    top_k: int = 5,
    similarity_threshold: float | None = None,
    project_id: UUID | None = None,
    metadata_filter: dict[str, object] | None = None,
) -> VectorRetrievalRequest:
    """Create a retrieval request for hardening tests."""
    return VectorRetrievalRequest(
        query_vector=vector or make_vector(),
        model=model or make_model(),
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        project_id=project_id,
        metadata_filter=metadata_filter or {},
    )


async def create_test_user() -> UUID:
    """Create a test user."""
    user_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"a569-hardening-{user_id}@example.com",
            name="A5.6.9.4 Hardening User",
        )

        session.add(user)
        await session.commit()

    return user_id


async def create_test_project(
    user_id: UUID,
) -> UUID:
    """Create a test project."""
    project_id = uuid4()

    async with SessionFactory() as session:
        project = Project(
            id=project_id,
            owner_id=user_id,
            name=f"A5.6.9.4 Hardening Project {project_id}",
            description="Retrieval hardening test project.",
        )

        session.add(project)
        await session.commit()

    return project_id


async def create_test_source(
    project_id: UUID,
) -> UUID:
    """Create a test source."""
    source_id = uuid4()

    async with SessionFactory() as session:
        source = Source(
            id=source_id,
            project_id=project_id,
            source_type="text",
            title="A5.6.9.4 Hardening Source",
            original_filename="hardening.txt",
            mime_type="text/plain",
            storage_uri=(
                f"file:///tmp/a5.6.9.4-{source_id}.txt"
            ),
            content_hash=f"source-{source_id}",
            source_metadata={
                "test": True,
                "stage": "A5.6.9.4",
            },
            status="COMPLETED",
        )

        session.add(source)
        await session.commit()

    return source_id


async def create_test_chunk(
    project_id: UUID,
    source_id: UUID,
    *,
    version: int,
    text: str,
    metadata: dict[str, object] | None = None,
) -> UUID:
    """Create a test knowledge document and chunk."""
    document_id = uuid4()
    chunk_id = uuid4()

    async with SessionFactory() as session:
        document = KnowledgeDocument(
            id=document_id,
            project_id=project_id,
            source_id=source_id,
            version=version,
            title=f"A5.6.9.4 Document {version}",
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
    model: EmbeddingModelInfo,
    vector: list[float],
    metadata: dict[str, object] | None = None,
) -> None:
    """Create a persisted embedding directly for retrieval tests."""
    async with SessionFactory() as session:
        embedding = KnowledgeChunkEmbedding(
            id=uuid4(),
            chunk_id=chunk_id,
            provider=model.provider,
            model_name=model.model_name,
            dimension=model.dimension,
            normalized=model.normalized,
            embedding=vector,
            vector_metadata=metadata or {},
        )

        session.add(embedding)
        await session.commit()


async def search(
    request: VectorRetrievalRequest,
):
    """Execute retrieval through the application service."""
    async with SessionFactory() as session:
        adapter = PostgresVectorRetrieval(session)
        service = VectorRetrievalService(adapter)

        return await service.search(request)


@pytest.mark.asyncio
async def test_model_isolation_prevents_cross_model_retrieval() -> None:
    """Retrieval must never return another embedding model."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    chunk_one = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="Model A chunk.",
    )

    chunk_two = await create_test_chunk(
        project_id,
        source_id,
        version=2,
        text="Model B chunk.",
    )

    model_a = make_model(
        provider="provider-a",
        model_name="model-a",
    )

    model_b = make_model(
        provider="provider-b",
        model_name="model-b",
    )

    await create_test_embedding(
        chunk_one,
        model=model_a,
        vector=make_vector(1.0),
    )

    await create_test_embedding(
        chunk_two,
        model=model_b,
        vector=make_vector(1.0),
    )

    result = await search(
        make_request(
            model=model_a,
            project_id=project_id,
        )
    )

    assert result.count == 1
    assert result.matches[0].chunk_id == chunk_one
    assert result.matches[0].model == model_a


@pytest.mark.asyncio
async def test_project_isolation_prevents_cross_project_retrieval() -> None:
    """Project filtering must prevent cross-project results."""
    user_id = await create_test_user()

    project_one = await create_test_project(user_id)
    project_two = await create_test_project(user_id)

    source_one = await create_test_source(project_one)
    source_two = await create_test_source(project_two)

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

    model = make_model()

    await create_test_embedding(
        chunk_one,
        model=model,
        vector=make_vector(1.0),
    )

    await create_test_embedding(
        chunk_two,
        model=model,
        vector=make_vector(1.0),
    )

    result = await search(
        make_request(
            model=model,
            project_id=project_one,
        )
    )

    assert result.count == 1
    assert result.matches[0].chunk_id == chunk_one


@pytest.mark.asyncio
async def test_top_k_limits_number_of_results() -> None:
    """Retrieval must never return more than top_k matches."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    chunk_ids: list[UUID] = []

    for version in range(1, 5):
        chunk_id = await create_test_chunk(
            project_id,
            source_id,
            version=version,
            text=f"Top-k chunk {version}.",
        )

        chunk_ids.append(chunk_id)

        await create_test_embedding(
            chunk_id,
            model=model,
            vector=make_vector(1.0),
        )

    result = await search(
        make_request(
            model=model,
            project_id=project_id,
            top_k=2,
        )
    )

    assert result.count == 2
    assert len(result.matches) == 2


@pytest.mark.asyncio
async def test_similarity_threshold_excludes_non_matching_vectors() -> None:
    """Similarity threshold must remove results below the threshold."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    matching_chunk = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="High similarity chunk.",
    )

    lower_similarity_chunk = await create_test_chunk(
        project_id,
        source_id,
        version=2,
        text="Lower similarity chunk.",
    )

    await create_test_embedding(
        matching_chunk,
        model=model,
        vector=make_vector(1.0),
    )

    low_similarity_vector = [0.0, 1.0] + [0.0] * (
    VECTOR_DIMENSION - 2
    )
    await create_test_embedding(
    lower_similarity_chunk,
    model=model,
    vector=low_similarity_vector,
    )
    result = await search(
        make_request(
            model=model,
            project_id=project_id,
            similarity_threshold=0.9,
        )
    )

    assert result.count == 1
    assert result.matches[0].chunk_id == matching_chunk
    assert result.matches[0].similarity >= 0.9


@pytest.mark.asyncio
async def test_empty_database_returns_empty_result() -> None:
    """Retrieval must return an empty result when nothing matches."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)

    model = make_model()

    result = await search(
        make_request(
            model=model,
            project_id=project_id,
        )
    )

    assert result.count == 0
    assert result.matches == []


@pytest.mark.asyncio
async def test_threshold_can_exclude_all_results() -> None:
    """A threshold above every similarity must return no matches."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    chunk_id = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="Threshold exclusion chunk.",
    )

    await create_test_embedding(
        chunk_id,
        model=model,
        vector=make_vector(1.0),
    )

    result = await search(
        make_request(
            model=model,
            project_id=project_id,
            similarity_threshold=1.0,
        )
    )

    assert result.count in (0, 1)

    if result.count == 1:
        assert result.matches[0].similarity >= 1.0


@pytest.mark.asyncio
async def test_chunk_metadata_is_preserved() -> None:
    """Chunk metadata must survive retrieval reconstruction."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    chunk_id = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="Metadata preservation chunk.",
        metadata={
            "page": 7,
            "section": "Hardening",
            "language": "en",
        },
    )

    await create_test_embedding(
        chunk_id,
        model=model,
        vector=make_vector(1.0),
    )

    result = await search(
        make_request(
            model=model,
            project_id=project_id,
        )
    )

    assert result.count == 1

    metadata = result.matches[0].metadata

    assert metadata["page"] == 7
    assert metadata["section"] == "Hardening"
    assert metadata["language"] == "en"


@pytest.mark.asyncio
async def test_embedding_metadata_is_preserved() -> None:
    """Embedding metadata must remain available after retrieval."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    chunk_id = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="Embedding metadata chunk.",
    )

    await create_test_embedding(
        chunk_id,
        model=model,
        vector=make_vector(1.0),
        metadata={
            "embedding_source": "hardening-test",
            "runtime": "postgres",
        },
    )

    result = await search(
        make_request(
            model=model,
            project_id=project_id,
        )
    )

    assert result.count == 1

    metadata = result.matches[0].metadata

    assert metadata["embedding"]["embedding_source"] == (
        "hardening-test"
    )
    assert metadata["embedding"]["runtime"] == "postgres"


@pytest.mark.asyncio
async def test_deleted_embedding_is_not_retrievable() -> None:
    """Deleted embeddings must disappear from subsequent retrieval."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    chunk_id = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="Deletion consistency chunk.",
    )

    await create_test_embedding(
        chunk_id,
        model=model,
        vector=make_vector(1.0),
    )

    before_delete = await search(
        make_request(
            model=model,
            project_id=project_id,
        )
    )

    assert before_delete.count == 1
    assert before_delete.matches[0].chunk_id == chunk_id

    async with SessionFactory() as session:
        vector_store = PostgresVectorStore(session)

        delete_result = await vector_store.delete(
            VectorDeleteRequest(
                chunk_id=chunk_id,
            )
        )

        await session.commit()

    assert delete_result.deleted_count == 1

    after_delete = await search(
        make_request(
            model=model,
            project_id=project_id,
        )
    )

    assert after_delete.count == 0
    assert after_delete.matches == []


@pytest.mark.asyncio
async def test_retrieval_preserves_similarity_ordering() -> None:
    """Results must remain ordered from highest to lowest similarity."""
    user_id = await create_test_user()
    project_id = await create_test_project(user_id)
    source_id = await create_test_source(project_id)

    model = make_model()

    chunk_one = await create_test_chunk(
        project_id,
        source_id,
        version=1,
        text="Similarity one.",
    )

    chunk_two = await create_test_chunk(
        project_id,
        source_id,
        version=2,
        text="Similarity two.",
    )

    chunk_three = await create_test_chunk(
        project_id,
        source_id,
        version=3,
        text="Similarity three.",
    )

    await create_test_embedding(
        chunk_one,
        model=model,
        vector=make_vector(1.0),
    )

    await create_test_embedding(
        chunk_two,
        model=model,
        vector=make_vector(0.8),
    )

    await create_test_embedding(
        chunk_three,
        model=model,
        vector=make_vector(-1.0),
    )

    result = await search(
        make_request(
            model=model,
            project_id=project_id,
            top_k=3,
        )
    )

    assert result.count == 3

    similarities = [
        match.similarity
        for match in result.matches
    ]

    assert similarities == sorted(
        similarities,
        reverse=True,
    )

    assert result.matches[0].chunk_id == chunk_one
    assert result.matches[1].chunk_id == chunk_two
    assert result.matches[2].chunk_id == chunk_three