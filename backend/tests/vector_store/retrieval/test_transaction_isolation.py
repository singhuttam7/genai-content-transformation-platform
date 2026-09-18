from __future__ import annotations

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
# Test data helpers
# ============================================================


def make_vector(
    first: float = 1.0,
    second: float = 0.0,
) -> list[float]:
    """
    Build a deterministic 384-dimensional vector.
    """
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
# Database fixture helpers
# ============================================================


async def create_test_user(session) -> User:
    user_id = uuid4()

    user = User(
        id=user_id,
        email=f"transaction-test-{user_id}@example.com",
        name="Transaction Isolation Test User",
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
) -> Project:
    project = Project(
        id=uuid4(),
        name=f"Transaction Isolation Project {uuid4()}",
        owner_id=user.id,
        description="Transaction isolation test project.",
        metadata={},
    )

    session.add(project)
    await session.flush()

    return project


async def create_test_source(
    session,
    project: Project,
) -> Source:
    source = Source(
        id=uuid4(),
        project_id=project.id,
        source_type="text",
        title=f"Transaction Isolation Source {uuid4()}",
        original_filename="transaction-isolation.txt",
        mime_type="text/plain",
        storage_uri=f"memory://transaction-isolation/{uuid4()}",
        content_hash=uuid4().hex,
        source_metadata={
            "test": "transaction-isolation",
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
) -> KnowledgeDocument:
    document = KnowledgeDocument(
        id=uuid4(),
        project_id=project.id,
        source_id=source.id,
        version=1,
        title=f"Transaction Isolation Document {uuid4()}",
        language="en",
        content_hash=uuid4().hex,
        status="completed",
        document_metadata={
            "test": "transaction-isolation",
        },
    )

    session.add(document)
    await session.flush()

    return document


async def create_test_chunk(
    session,
    document: KnowledgeDocument,
) -> KnowledgeChunk:
    chunk = KnowledgeChunk(
        id=uuid4(),
        document_id=document.id,
        chunk_index=0,
        text=f"Transaction isolation test chunk {uuid4()}",
        content_hash=uuid4().hex,
        token_count=5,
        chunk_metadata={
            "test": "transaction-isolation",
        },
    )

    session.add(chunk)
    await session.flush()

    return chunk


async def create_committed_test_data() -> tuple:
    """
    Create all parent entities in a committed transaction.

    Returning UUIDs instead of ORM instances makes the tests independent
    from the lifecycle of the setup session.
    """
    async with SessionFactory() as session:
        user = await create_test_user(session)
        project = await create_test_project(session, user)
        source = await create_test_source(session, project)
        document = await create_test_document(session, project, source)
        chunk = await create_test_chunk(session, document)

        await session.commit()

        return (
            user.id,
            project.id,
            source.id,
            document.id,
            chunk.id,
        )


async def retrieve(
    request: VectorRetrievalRequest,
):
    """
    Execute retrieval through a completely independent database session.
    """
    async with SessionFactory() as session:
        retrieval_port = PostgresVectorRetrieval(session)
        retrieval_service = VectorRetrievalService(retrieval_port)

        return await retrieval_service.search(request)


def make_request(
    *,
    vector: list[float],
    model: EmbeddingModelInfo | None = None,
    project_id=None,
    top_k: int = 5,
) -> VectorRetrievalRequest:
    selected_model = model or make_model()

    return VectorRetrievalRequest(
        query_vector=vector,
        model=selected_model,
        top_k=top_k,
        project_id=project_id,
    )


# ============================================================
# Transaction ownership
# ============================================================


@pytest.mark.asyncio
async def test_persistence_service_does_not_commit_caller_transaction():
    """
    VectorPersistenceService must flush its work but must not commit
    the caller-owned transaction.
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    async with SessionFactory() as session:
        service = VectorPersistenceService(
            PostgresVectorStore(session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(),
        )

        assert session.in_transaction()
        assert session.get_transaction() is not None

        result = await session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
        )

        assert result.scalar_one_or_none() is not None

        # Explicit caller-owned rollback.
        await session.rollback()

    # The persistence service did not commit, therefore the row must
    # disappear after the caller rolls the transaction back.
    request = make_request(
        vector=make_vector(),
        model=make_model(),
        project_id=project_id,
    )

    result = await retrieve(request)

    assert result.count == 0


@pytest.mark.asyncio
async def test_persistence_service_does_not_rollback_caller_transaction():
    """
    VectorPersistenceService must not rollback a caller transaction.
    """
    _, _, _, _, chunk_id = await create_committed_test_data()

    async with SessionFactory() as session:
        service = VectorPersistenceService(
            PostgresVectorStore(session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(),
        )

        result = await session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
        )

        embedding = result.scalar_one_or_none()

        assert embedding is not None
        assert session.in_transaction()

        # The rollback is intentionally performed by the caller.
        await session.rollback()

        result = await session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
        )

        assert result.scalar_one_or_none() is None


# ============================================================
# Insert visibility / isolation
# ============================================================


@pytest.mark.asyncio
async def test_uncommitted_insert_is_invisible_to_independent_session():
    """
    An embedding flushed but not committed in one session must not
    become visible to another independent session.
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    model = make_model()

    async with SessionFactory() as writer_session:
        service = VectorPersistenceService(
            PostgresVectorStore(writer_session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(model=model),
        )

        # Verify the writer session can see its own uncommitted row.
        writer_result = await writer_session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
        )

        assert writer_result.scalar_one_or_none() is not None

        # A completely independent session must not see the row.
        request = make_request(
            vector=make_vector(),
            model=model,
            project_id=project_id,
        )

        reader_result = await retrieve(request)

        assert reader_result.count == 0

        await writer_session.rollback()


@pytest.mark.asyncio
async def test_committed_insert_is_visible_to_independent_session():
    """
    Once the caller commits the transaction, an independent retrieval
    session must be able to observe the persisted embedding.
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    model = make_model()

    async with SessionFactory() as writer_session:
        service = VectorPersistenceService(
            PostgresVectorStore(writer_session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(model=model),
        )

        await writer_session.commit()

    request = make_request(
        vector=make_vector(),
        model=model,
        project_id=project_id,
    )

    result = await retrieve(request)

    assert result.count == 1
    assert result.matches[0].chunk_id == chunk_id


# ============================================================
# Rollback semantics
# ============================================================


@pytest.mark.asyncio
async def test_rollback_removes_uncommitted_embedding():
    """
    Explicit caller rollback must remove an embedding that was only
    flushed by the persistence service.
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    model = make_model()

    async with SessionFactory() as session:
        service = VectorPersistenceService(
            PostgresVectorStore(session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(model=model),
        )

        local_result = await session.execute(
            select(KnowledgeChunkEmbedding).where(
                KnowledgeChunkEmbedding.chunk_id == chunk_id
            )
        )

        assert local_result.scalar_one_or_none() is not None

        await session.rollback()

    request = make_request(
        vector=make_vector(),
        model=model,
        project_id=project_id,
    )

    result = await retrieve(request)

    assert result.count == 0


@pytest.mark.asyncio
async def test_rollback_of_delete_preserves_embedding():
    """
    A delete performed by the vector store must remain reversible by
    the caller's transaction rollback.
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    model = make_model()

    async with SessionFactory() as session:
        service = VectorPersistenceService(
            PostgresVectorStore(session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(model=model),
        )

        await session.commit()

    # Perform the delete in a separate transaction.
    async with SessionFactory() as delete_session:
        store = PostgresVectorStore(delete_session)

        delete_result = await store.delete(
            VectorDeleteRequest(chunk_id=chunk_id)
        )

        assert delete_result.deleted_count == 1

        # Do not commit the deletion.
        await delete_session.rollback()

    request = make_request(
        vector=make_vector(),
        model=model,
        project_id=project_id,
    )

    result = await retrieve(request)

    assert result.count == 1
    assert result.matches[0].chunk_id == chunk_id


@pytest.mark.asyncio
async def test_commit_of_delete_makes_embedding_invisible():
    """
    A committed delete must become visible to independent sessions.
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    model = make_model()

    async with SessionFactory() as session:
        service = VectorPersistenceService(
            PostgresVectorStore(session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(model=model),
        )

        await session.commit()

    async with SessionFactory() as delete_session:
        store = PostgresVectorStore(delete_session)

        delete_result = await store.delete(
            VectorDeleteRequest(chunk_id=chunk_id)
        )

        assert delete_result.deleted_count == 1

        await delete_session.commit()

    request = make_request(
        vector=make_vector(),
        model=model,
        project_id=project_id,
    )

    result = await retrieve(request)

    assert result.count == 0


# ============================================================
# Independent transaction isolation
# ============================================================


@pytest.mark.asyncio
async def test_independent_transactions_do_not_leak_uncommitted_state():
    """
    Verify the complete visibility transition:

        writer flush
            ↓
        reader cannot see row
            ↓
        writer commit
            ↓
        reader can see row
    """
    _, project_id, _, _, chunk_id = await create_committed_test_data()

    model = make_model()

    async with SessionFactory() as writer_session:
        service = VectorPersistenceService(
            PostgresVectorStore(writer_session)
        )

        await service.persist_embedding(
            chunk_id=chunk_id,
            embedding=make_embedding(model=model),
        )

        request = make_request(
            vector=make_vector(),
            model=model,
            project_id=project_id,
        )

        # Independent transaction before commit.
        async with SessionFactory() as reader_session:
            retrieval_port = PostgresVectorRetrieval(reader_session)
            retrieval_service = VectorRetrievalService(
                retrieval_port
            )

            before_commit = await retrieval_service.search(request)

            assert before_commit.count == 0

        # Commit belongs to the caller/application transaction.
        await writer_session.commit()

    # New independent transaction after commit.
    async with SessionFactory() as reader_session:
        retrieval_port = PostgresVectorRetrieval(reader_session)
        retrieval_service = VectorRetrievalService(
            retrieval_port
        )

        after_commit = await retrieval_service.search(request)

        assert after_commit.count == 1
        assert after_commit.matches[0].chunk_id == chunk_id