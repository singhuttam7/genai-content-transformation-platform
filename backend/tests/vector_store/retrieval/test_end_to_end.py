from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from app.database.session import SessionFactory
from app.models.knowledge_chunk import KnowledgeChunk
from app.models.knowledge_document import KnowledgeDocument
from app.models.project import Project
from app.models.source import Source
from app.models.user import User
from app.models_ai.embeddings.port import EmbeddingPort
from app.models_ai.embeddings.schemas import (
    EmbeddingBatchRequest,
    EmbeddingBatchResult,
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)
from app.models_ai.embeddings.service import EmbeddingService
from app.vector_store.adapters.postgres import PostgresVectorStore
from app.vector_store.retrieval.adapters.postgres import (
    PostgresVectorRetrieval,
)
from app.vector_store.retrieval.schemas import VectorRetrievalRequest
from app.vector_store.retrieval.service import VectorRetrievalService
from app.vector_store.service import VectorPersistenceService


VECTOR_DIMENSION = 384


def make_vector(first_value: float = 1.0) -> list[float]:
    """Create a deterministic 384-dimensional vector."""
    return [first_value] + [0.0] * (VECTOR_DIMENSION - 1)


class DeterministicEmbeddingPort(EmbeddingPort):
    """Deterministic embedding provider for integration testing."""

    def __init__(self) -> None:
        self.model = EmbeddingModelInfo(
            provider="integration-provider",
            model_name="integration-model",
            dimension=VECTOR_DIMENSION,
            normalized=True,
        )

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        """Return a deterministic embedding for one input."""
        return EmbeddingVector(
            values=make_vector(1.0),
            model=self.model,
            input_index=0,
        )

    async def embed_batch(
        self,
        request: EmbeddingBatchRequest,
    ) -> EmbeddingBatchResult:
        """Return deterministic embeddings for a batch."""
        embeddings = [
            EmbeddingVector(
                values=make_vector(1.0),
                model=self.model,
                input_index=index,
            )
            for index, _ in enumerate(request.requests)
        ]

        return EmbeddingBatchResult(
            embeddings=embeddings,
            model=self.model,
            input_count=len(embeddings),
            dimension=VECTOR_DIMENSION,
        )


async def create_user() -> UUID:
    """Create and persist a test user."""
    user_id = uuid4()

    async with SessionFactory() as session:
        user = User(
            id=user_id,
            email=f"a56-integration-{user_id}@example.com",
            name="A5.6 Integration User",
        )

        session.add(user)
        await session.commit()

    return user_id


async def create_project(user_id: UUID) -> UUID:
    """Create and persist a test project."""
    project_id = uuid4()

    async with SessionFactory() as session:
        project = Project(
            id=project_id,
            owner_id=user_id,
            name=f"A5.6 Integration Project {project_id}",
            description="A5.6 end-to-end integration test.",
        )

        session.add(project)
        await session.commit()

    return project_id


async def create_source(project_id: UUID) -> UUID:
    """Create and persist a test source."""
    source_id = uuid4()

    async with SessionFactory() as session:
        source = Source(
            id=source_id,
            project_id=project_id,
            source_type="text",
            title="A5.6 Integration Test Source",
            original_filename="a5.6-integration.txt",
            mime_type="text/plain",
            storage_uri=f"file:///tmp/a5.6-integration-{source_id}.txt",
            content_hash=f"source-{source_id}",
            source_metadata={
                "test": True,
                "purpose": "a5.6-end-to-end",
            },
            status="COMPLETED",
        )

        session.add(source)
        await session.commit()

    return source_id


async def create_chunk(
    project_id: UUID,
    source_id: UUID,
) -> UUID:
    """Create and persist a test knowledge document and chunk."""
    document_id = uuid4()
    chunk_id = uuid4()

    async with SessionFactory() as session:
        document = KnowledgeDocument(
            id=document_id,
            project_id=project_id,
            source_id=source_id,
            version=1,
            title="A5.6 Integration Document",
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
            text="End-to-end A5.6 integration chunk.",
            content_hash=f"chunk-{chunk_id}",
            token_count=5,
            chunk_metadata={
                "section": "integration",
                "page": 1,
            },
        )

        session.add(chunk)
        await session.commit()

    return chunk_id


@pytest.mark.asyncio
async def test_embedding_persistence_and_retrieval_end_to_end() -> None:
    """Verify the complete embedding-to-retrieval pipeline."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)
    chunk_id = await create_chunk(project_id, source_id)

    embedding_port = DeterministicEmbeddingPort()
    embedding_service = EmbeddingService(embedding_port)

    embedding = await embedding_service.embed_text(
        "End-to-end A5.6 integration chunk."
    )

    assert embedding.model.provider == "integration-provider"
    assert embedding.model.model_name == "integration-model"
    assert embedding.model.dimension == VECTOR_DIMENSION
    assert embedding.model.normalized is True
    assert len(embedding.values) == VECTOR_DIMENSION

    async with SessionFactory() as session:
        vector_store = PostgresVectorStore(session)
        persistence_service = VectorPersistenceService(
            vector_store,
        )

        persisted = await persistence_service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
            metadata={
                "source": "a5.6-integration",
            },
        )

        assert persisted.chunk_id == chunk_id
        assert persisted.model.provider == "integration-provider"
        assert persisted.model.model_name == "integration-model"
        assert persisted.model.dimension == VECTOR_DIMENSION
        assert len(persisted.values) == VECTOR_DIMENSION

        await session.commit()

    async with SessionFactory() as session:
        retrieval_adapter = PostgresVectorRetrieval(session)

        retrieval_service = VectorRetrievalService(
            retrieval_adapter,
        )

        request = VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=embedding.model,
            top_k=5,
            project_id=project_id,
        )

        result = await retrieval_service.search(request)

        assert result.count == 1
        assert len(result.matches) == 1

        match = result.matches[0]

        assert match.chunk_id == chunk_id
        assert match.similarity > 0.99

        assert match.model.provider == "integration-provider"
        assert match.model.model_name == "integration-model"
        assert match.model.dimension == VECTOR_DIMENSION
        assert match.model.normalized is True

        assert match.metadata["section"] == "integration"
        assert match.metadata["page"] == 1
        assert (
            match.metadata["embedding"]["source"]
            == "a5.6-integration"
        )


@pytest.mark.asyncio
async def test_persisted_embedding_is_retrievable_by_model_identity() -> None:
    """Verify that persisted embeddings are isolated by model identity."""
    user_id = await create_user()
    project_id = await create_project(user_id)
    source_id = await create_source(project_id)
    chunk_id = await create_chunk(project_id, source_id)

    embedding_port = DeterministicEmbeddingPort()
    embedding_service = EmbeddingService(embedding_port)

    embedding = await embedding_service.embed_text(
        "Model identity integration test."
    )

    async with SessionFactory() as session:
        vector_store = PostgresVectorStore(session)
        persistence_service = VectorPersistenceService(
            vector_store,
        )

        await persistence_service.persist_embedding(
            chunk_id=chunk_id,
            embedding=embedding,
        )

        await session.commit()

    async with SessionFactory() as session:
        retrieval_adapter = PostgresVectorRetrieval(session)

        retrieval_service = VectorRetrievalService(
            retrieval_adapter,
        )

        correct_request = VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=embedding.model,
            top_k=5,
            project_id=project_id,
        )

        result = await retrieval_service.search(
            correct_request,
        )

        assert result.count == 1
        assert len(result.matches) == 1
        assert result.matches[0].chunk_id == chunk_id

        wrong_model = EmbeddingModelInfo(
            provider="different-provider",
            model_name="different-model",
            dimension=VECTOR_DIMENSION,
            normalized=True,
        )

        wrong_request = VectorRetrievalRequest(
            query_vector=list(embedding.values),
            model=wrong_model,
            top_k=5,
            project_id=project_id,
        )

        wrong_result = await retrieval_service.search(
            wrong_request,
        )

        assert wrong_result.count == 0
        assert wrong_result.matches == []