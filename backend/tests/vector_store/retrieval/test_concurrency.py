from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import select

from app.database.session import SessionFactory
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding
from app.models.knowledge_document import KnowledgeDocument
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.models_ai.embeddings.schemas import (
    EmbeddingModelInfo,
    EmbeddingVector,
)
from app.vector_store.adapters.postgres import PostgresVectorStore
from app.vector_store.retrieval.adapters.postgres import PostgresVectorRetrieval
from app.vector_store.retrieval.schemas import VectorRetrievalRequest
from app.vector_store.retrieval.service import VectorRetrievalService
from app.vector_store.schemas import VectorDeleteRequest
from app.vector_store.service import VectorPersistenceService


VECTOR_DIMENSION = 384


# ============================================================
# Deterministic test data
# ============================================================


def make_vector(
    first: float = 1.0,
    second: float = 0.0,
) -> list[float]:
    vector = [0.0] * VECTOR_DIMENSION
    vector[0] = first
    vector[1] = second
    return vector


def make_model(
    *,
    provider: str = "sentence-transformers",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider=provider,
        model_name=model_name,
        dimension=VECTOR_DIMENSION,
        normalized=True,
        metadata={
            "runtime": "sentence-transformers",
        },
    )


def make_embedding(
    *,
    first: float = 1.0,
    second: float = 0.0,
    model: EmbeddingModelInfo | None = None,
    input_index: int = 0,
) -> EmbeddingVector:
    selected_model = model or make_model()

    return EmbeddingVector(
        values=make_vector(first, second),
        model=selected_model,
        input_index=input_index,
    )


# ============================================================
# Database setup helpers
# ============================================================


async def create_test_user(session) -> User:
    user_id = uuid4()

    user = User(
        id=user_id,
        email=f"concurrency-test-{user_id}@example.com",
        name="Concurrency Test User",
        password_hash=None,
        role="operator",
        is_active=True,
    )

    session.add(user)
    await session.flush()

    return user


async def create_test_project(
    session,
    user: User,
    *,
    name_suffix: str,
) -> Project:
    project = Project(
        id=uuid4(),
        name=f"Concurrency Project {name_suffix}-{uuid4()}",
        owner_id=user.id,
        description="Concurrency hardening test project.",
        metadata={
            "test": "concurrency",
        },
    )

    session.add(project)
    await session.flush()

    return project


async def create_test_source(
    session,
    project: Project,
    *,
    name_suffix: str,
) -> Source:
    source = Source(
        id=uuid4(),
        project_id=project.id,
        source_type="text",
        title=f"Concurrency Source {name_suffix}-{uuid4()}",
        original_filename="concurrency-test.txt",
        mime_type="text/plain",
        storage_uri=f"memory://concurrency/{uuid4()}",
        content_hash=uuid4().hex,
        source_metadata={
            "test": "concurrency",
        },
        status="completed",
    )

    session.add(source)
    await session.flush()

    return source


async def create_test_document(
    session,
    project: Project,
    source: Source,
    *,
    version: int = 1,
) -> KnowledgeDocument:
    document = KnowledgeDocument(
        id=uuid4(),
        project_id=project.id,
        source_id=source.id,
        version=version,
        title=f"Concurrency Document {uuid4()}",
        language="en",
        content_hash=uuid4().hex,
        status="completed",
        document_metadata={
            "test": "concurrency",
        },
    )

    session.add(document)
    await session.flush()

    return document


async def create_test_chunk(
    session,
    document: KnowledgeDocument,
    *,
    chunk_index: int,
) -> KnowledgeChunk:
    chunk = KnowledgeChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=chunk_index,
        text=f"Concurrency test chunk {chunk_index} {uuid4()}",
        content_hash=uuid4().hex,
        token_count=5,
        chunk_metadata={
            "test": "concurrency",
            "chunk_index": chunk_index,
        },
    )

    session.add(chunk)
    await session.flush()

    return chunk


async def create_committed_project_with_chunks(
    *,
    chunk_count: int,
    project_suffix: str,
) -> tuple[Project, list[KnowledgeChunk]]:
    async with SessionFactory() as session:
        user = await create_test_user(session)

        project = await create_test_project(
            session,
            user,
            name_suffix=project_suffix,
        )

        source = await create_test_source(
            session,
            project,
            name_suffix=project_suffix,
        )

        document = await create_test_document(
            session,
            project,
            source,
        )

        chunks: list[KnowledgeChunk] = []

        for chunk_index in range(chunk_count):
            chunk = await create_test_chunk(
                session,
                document,
                chunk_index=chunk_index,
            )
            chunks.append(chunk)

        await session.commit()

        return project, chunks


# ============================================================
# Retrieval helper
# ============================================================


async def retrieve(
    request: VectorRetrievalRequest,
):
    """
    Every retrieval operation gets its own independent session.
    """
    async with SessionFactory() as session:
        retrieval_port = PostgresVectorRetrieval(session)
        retrieval_service = VectorRetrievalService(
            retrieval_port
        )

        return await retrieval_service.search(request)


# ============================================================
# Concurrent persistence
# ============================================================


async def persist_embedding(
    chunk_id,
    *,
    model: EmbeddingModelInfo,
    first: float,
    second: float,
) -> None:
    """
    Persist one embedding using an independent transaction.
    """
    async with SessionFactory() as session:
        service = VectorPersistenceService(
            PostgresVectorStore(session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(
                first=first,
                second=second,
                model=model,
            ),
        )

        await session.commit()


@pytest.mark.asyncio
async def test_concurrent_persistence_of_independent_chunks():
    """
    Independent chunks can be persisted concurrently without
    cross-session corruption or lost writes.
    """
    _, chunks = await create_committed_project_with_chunks(
        chunk_count=4,
        project_suffix="independent-persistence",
    )

    model = make_model()

    await asyncio.gather(
        persist_embedding(
            chunks[0].id,
            model=model,
            first=1.0,
            second=0.0,
        ),
        persist_embedding(
            chunks[1].id,
            model=model,
            first=0.0,
            second=1.0,
        ),
        persist_embedding(
            chunks[2].id,
            model=model,
            first=1.0,
            second=1.0,
        ),
        persist_embedding(
            chunks[3].id,
            model=model,
            first=-1.0,
            second=0.0,
        ),
    )

    async with SessionFactory() as verification_session:
        result = await verification_session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id.in_(
                    [chunk.id for chunk in chunks]
                )
            )
        )

        embeddings = result.scalars().all()

    assert len(embeddings) == 4
    assert {
        embedding.chunk_id
        for embedding in embeddings
    } == {
        chunk.id
        for chunk in chunks
    }


# ============================================================
# Concurrent retrieval
# ============================================================


@pytest.mark.asyncio
async def test_concurrent_retrieval_from_independent_sessions():
    """
    Multiple independent readers can retrieve simultaneously without
    leaking state between sessions.
    """
    project, chunks = await create_committed_project_with_chunks(
        chunk_count=2,
        project_suffix="concurrent-retrieval",
    )

    model = make_model()

    await asyncio.gather(
        persist_embedding(
            chunks[0].id,
            model=model,
            first=1.0,
            second=0.0,
        ),
        persist_embedding(
            chunks[1].id,
            model=model,
            first=0.0,
            second=1.0,
        ),
    )

    request = VectorRetrievalRequest(
        query_vector=make_vector(1.0, 0.0),
        model=model,
        top_k=2,
        project_id=project.id,
    )

    results = await asyncio.gather(
        retrieve(request),
        retrieve(request),
        retrieve(request),
        retrieve(request),
    )

    assert len(results) == 4

    for result in results:
        assert result.count == 2
        assert result.matches[0].chunk_id == chunks[0].id
        assert result.matches[1].chunk_id == chunks[1].id


# ============================================================
# Concurrent persistence followed by retrieval
# ============================================================


@pytest.mark.asyncio
async def test_concurrent_persistence_is_fully_visible_after_commit():
    """
    After all independent writer transactions commit, a new reader
    transaction must observe the complete committed state.
    """
    project, chunks = await create_committed_project_with_chunks(
        chunk_count=5,
        project_suffix="visibility-after-commit",
    )

    model = make_model()

    await asyncio.gather(
        *[
            persist_embedding(
                chunk.id,
                model=model,
                first=1.0 if index % 2 == 0 else 0.0,
                second=0.0 if index % 2 == 0 else 1.0,
            )
            for index, chunk in enumerate(chunks)
        ]
    )

    request = VectorRetrievalRequest(
        query_vector=make_vector(1.0, 0.0),
        model=model,
        top_k=5,
        project_id=project.id,
    )

    result = await retrieve(request)

    assert result.count == 5

    returned_chunk_ids = {
        match.chunk_id
        for match in result.matches
    }

    assert returned_chunk_ids == {
        chunk.id
        for chunk in chunks
    }


# ============================================================
# Concurrent delete
# ============================================================


async def delete_embedding(chunk_id) -> int:
    """
    Delete one chunk's embedding in its own transaction.
    """
    async with SessionFactory() as session:
        store = PostgresVectorStore(session)

        result = await store.delete(
            VectorDeleteRequest(
                chunk_id=chunk_id,
            )
        )

        await session.commit()

        return result.deleted_count


@pytest.mark.asyncio
async def test_concurrent_delete_of_independent_embeddings():
    """
    Concurrent deletion of independent chunks must not affect the
    embeddings belonging to other chunks.
    """
    _, chunks = await create_committed_project_with_chunks(
        chunk_count=4,
        project_suffix="concurrent-delete",
    )

    model = make_model()

    await asyncio.gather(
        *[
            persist_embedding(
                chunk.id,
                model=model,
                first=1.0,
                second=0.0,
            )
            for chunk in chunks
        ]
    )

    delete_counts = await asyncio.gather(
        *[
            delete_embedding(chunk.id)
            for chunk in chunks[:2]
        ]
    )

    assert delete_counts == [1, 1]

    async with SessionFactory() as verification_session:
        result = await verification_session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id.in_(
                    [chunk.id for chunk in chunks]
                )
            )
        )

        remaining = result.scalars().all()

    assert len(remaining) == 2
    assert {
        embedding.chunk_id
        for embedding in remaining
    } == {
        chunks[2].id,
        chunks[3].id,
    }


# ============================================================
# Project isolation under concurrency
# ============================================================


@pytest.mark.asyncio
async def test_concurrent_operations_preserve_project_isolation():
    """
    Concurrent operations belonging to different projects must not
    cause retrieval results to cross project boundaries.
    """
    project_a, chunks_a = await create_committed_project_with_chunks(
        chunk_count=2,
        project_suffix="project-a",
    )

    project_b, chunks_b = await create_committed_project_with_chunks(
        chunk_count=2,
        project_suffix="project-b",
    )

    model = make_model()

    await asyncio.gather(
        *[
            persist_embedding(
                chunk.id,
                model=model,
                first=1.0,
                second=0.0,
            )
            for chunk in chunks_a + chunks_b
        ]
    )

    request_a = VectorRetrievalRequest(
        query_vector=make_vector(),
        model=model,
        top_k=10,
        project_id=project_a.id,
    )

    request_b = VectorRetrievalRequest(
        query_vector=make_vector(),
        model=model,
        top_k=10,
        project_id=project_b.id,
    )

    result_a, result_b = await asyncio.gather(
        retrieve(request_a),
        retrieve(request_b),
    )

    assert result_a.count == 2
    assert result_b.count == 2

    assert {
        match.chunk_id
        for match in result_a.matches
    } == {
        chunk.id
        for chunk in chunks_a
    }

    assert {
        match.chunk_id
        for match in result_b.matches
    } == {
        chunk.id
        for chunk in chunks_b
    }


# ============================================================
# Model isolation under concurrency
# ============================================================


@pytest.mark.asyncio
async def test_concurrent_operations_preserve_model_isolation():
    """
    Different embedding models persisted for different chunks must
    remain isolated during concurrent retrieval.
    """
    project, chunks = await create_committed_project_with_chunks(
        chunk_count=2,
        project_suffix="model-isolation",
    )

    model_a = make_model(
        provider="sentence-transformers",
        model_name="model-a",
    )

    model_b = make_model(
        provider="sentence-transformers",
        model_name="model-b",
    )

    await asyncio.gather(
        persist_embedding(
            chunks[0].id,
            model=model_a,
            first=1.0,
            second=0.0,
        ),
        persist_embedding(
            chunks[1].id,
            model=model_b,
            first=1.0,
            second=0.0,
        ),
    )

    request_a = VectorRetrievalRequest(
        query_vector=make_vector(),
        model=model_a,
        top_k=10,
        project_id=project.id,
    )

    request_b = VectorRetrievalRequest(
        query_vector=make_vector(),
        model=model_b,
        top_k=10,
        project_id=project.id,
    )

    result_a, result_b = await asyncio.gather(
        retrieve(request_a),
        retrieve(request_b),
    )

    assert result_a.count == 1
    assert result_b.count == 1

    assert result_a.matches[0].chunk_id == chunks[0].id
    assert result_b.matches[0].chunk_id == chunks[1].id

    assert result_a.matches[0].model.provider == model_a.provider
    assert result_a.matches[0].model.model_name == model_a.model_name

    assert result_b.matches[0].model.provider == model_b.provider
    assert result_b.matches[0].model.model_name == model_b.model_name


# ============================================================
# Session state isolation
# ============================================================


@pytest.mark.asyncio
async def test_concurrent_sessions_do_not_share_uncommitted_state():
    """
    A writer's uncommitted embedding must remain invisible to an
    independent reader even when both operations execute concurrently.
    """
    project, chunks = await create_committed_project_with_chunks(
        chunk_count=1,
        project_suffix="session-state-isolation",
    )

    chunk_id = chunks[0].id
    model = make_model()

    writer_ready = asyncio.Event()
    allow_writer_commit = asyncio.Event()

    async def writer() -> None:
        async with SessionFactory() as writer_session:
            service = VectorPersistenceService(
                PostgresVectorStore(writer_session)
            )

            await service.persist_embedding(
                chunk_id=chunk_id,
                embedding=make_embedding(
                    model=model,
                ),
            )

            writer_ready.set()

            await allow_writer_commit.wait()

            await writer_session.commit()

    async def reader_before_commit():
        await writer_ready.wait()

        request = VectorRetrievalRequest(
            query_vector=make_vector(),
            model=model,
            top_k=5,
            project_id=project.id,
        )

        return await retrieve(request)

    writer_task = asyncio.create_task(writer())
    reader_task = asyncio.create_task(reader_before_commit())

    before_commit_result = await reader_task

    assert before_commit_result.count == 0

    allow_writer_commit.set()

    await writer_task

    after_commit_request = VectorRetrievalRequest(
        query_vector=make_vector(),
        model=model,
        top_k=5,
        project_id=project.id,
    )

    after_commit_result = await retrieve(after_commit_request)

    assert after_commit_result.count == 1
    assert after_commit_result.matches[0].chunk_id == chunk_id