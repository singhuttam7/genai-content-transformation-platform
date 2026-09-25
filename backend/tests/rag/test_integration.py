from __future__ import annotations

from uuid import uuid4

import pytest

from app.models_ai.embeddings.port import EmbeddingPort
from app.models_ai.embeddings.schemas import (
    EmbeddingModelInfo,
    EmbeddingRequest,
    EmbeddingVector,
)
from app.models_ai.embeddings.service import EmbeddingService
from app.rag.config import RAGContextConfig
from app.rag.context import RAGContextAssembler
from app.rag.schemas import RAGQuery
from app.rag.service import RAGRetrievalService
from app.vector_store.retrieval.port import VectorRetrievalPort
from app.vector_store.retrieval.schemas import (
    VectorRetrievalMatch,
    VectorRetrievalRequest,
    VectorRetrievalResult,
)
from app.vector_store.retrieval.service import VectorRetrievalService


MODEL = EmbeddingModelInfo(
    provider="sentence-transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    dimension=3,
    normalized=True,
)


class IntegrationEmbeddingPort(EmbeddingPort):
    """Deterministic embedding adapter for RAG integration tests."""

    def __init__(self) -> None:
        self.requests: list[EmbeddingRequest] = []

    async def embed(
        self,
        request: EmbeddingRequest,
    ) -> EmbeddingVector:
        self.requests.append(request)

        return EmbeddingVector(
            values=[0.1, 0.2, 0.3],
            model=MODEL,
            input_index=0,
        )

    async def embed_batch(self, request):
        raise NotImplementedError


class IntegrationRetrievalPort(VectorRetrievalPort):
    """Deterministic vector retrieval adapter for integration tests."""

    def __init__(
        self,
        matches: list[VectorRetrievalMatch],
    ) -> None:
        self.matches = matches
        self.requests: list[VectorRetrievalRequest] = []

    async def search(
        self,
        request: VectorRetrievalRequest,
    ) -> VectorRetrievalResult:
        self.requests.append(request)

        return VectorRetrievalResult(
            matches=self.matches,
            query_model=request.model,
            count=len(self.matches),
            metadata={
                "provider": request.model.provider,
                "model_name": request.model.model_name,
            },
        )


def build_match(
    *,
    text: str,
    similarity: float,
) -> VectorRetrievalMatch:
    return VectorRetrievalMatch(
        chunk_id=uuid4(),
        text=text,
        similarity=similarity,
        model=MODEL,
        metadata={
            "language": "en",
            "source": "integration-test",
        },
    )


def build_pipeline(
    matches: list[VectorRetrievalMatch],
) -> tuple[
    IntegrationEmbeddingPort,
    IntegrationRetrievalPort,
    RAGRetrievalService,
    RAGContextAssembler,
]:
    embedding_port = IntegrationEmbeddingPort()
    retrieval_port = IntegrationRetrievalPort(matches)

    embedding_service = EmbeddingService(
        embedding_port,
    )

    retrieval_service = VectorRetrievalService(
        retrieval_port,
    )

    rag_retrieval_service = RAGRetrievalService(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
    )

    context_assembler = RAGContextAssembler()

    return (
        embedding_port,
        retrieval_port,
        rag_retrieval_service,
        context_assembler,
    )


@pytest.mark.asyncio
async def test_complete_rag_pipeline() -> None:
    matches = [
        build_match(
            text="The platform accepts documents as knowledge sources.",
            similarity=0.95,
        ),
        build_match(
            text="Knowledge is normalized before chunking.",
            similarity=0.89,
        ),
    ]

    (
        embedding_port,
        retrieval_port,
        rag_retrieval_service,
        context_assembler,
    ) = build_pipeline(matches)

    query = RAGQuery(
        text="How does the knowledge pipeline work?",
        top_k=5,
        similarity_threshold=0.75,
    )

    retrieved_chunks = await rag_retrieval_service.retrieve(
        query,
    )

    context = context_assembler.assemble(
        query=query,
        chunks=retrieved_chunks,
    )

    assert len(embedding_port.requests) == 1
    assert (
        embedding_port.requests[0].text
        == "How does the knowledge pipeline work?"
    )

    assert len(retrieval_port.requests) == 1

    retrieval_request = retrieval_port.requests[0]

    assert retrieval_request.query_vector == [
        0.1,
        0.2,
        0.3,
    ]
    assert retrieval_request.model == MODEL
    assert retrieval_request.top_k == 5
    assert retrieval_request.similarity_threshold == 0.75

    assert len(retrieved_chunks) == 2

    assert context.query == query
    assert len(context.chunks) == 2

    assert (
        context.chunks[0].text
        == "The platform accepts documents as knowledge sources."
    )
    assert (
        context.chunks[1].text
        == "Knowledge is normalized before chunking."
    )

    assert "Retrieved Knowledge Context" in context.context_text
    assert "[Chunk 1]" in context.context_text
    assert "[Chunk 2]" in context.context_text


@pytest.mark.asyncio
async def test_rag_pipeline_preserves_query_filters() -> None:
    project_id = uuid4()

    matches = [
        build_match(
            text="Project-specific knowledge.",
            similarity=0.91,
        ),
    ]

    (
        _,
        retrieval_port,
        rag_retrieval_service,
        context_assembler,
    ) = build_pipeline(matches)

    query = RAGQuery(
        text="Find project knowledge.",
        top_k=3,
        similarity_threshold=0.8,
        project_id=project_id,
        metadata_filter={
            "language": "en",
            "source": "integration-test",
        },
    )

    chunks = await rag_retrieval_service.retrieve(
        query,
    )

    context = context_assembler.assemble(
        query=query,
        chunks=chunks,
    )

    request = retrieval_port.requests[0]

    assert request.top_k == 3
    assert request.similarity_threshold == 0.8
    assert request.project_id == project_id
    assert request.metadata_filter == {
        "language": "en",
        "source": "integration-test",
    }

    assert context.query.project_id == project_id
    assert context.query.metadata_filter == {
        "language": "en",
        "source": "integration-test",
    }


@pytest.mark.asyncio
async def test_empty_retrieval_produces_empty_context() -> None:
    (
        _,
        retrieval_port,
        rag_retrieval_service,
        context_assembler,
    ) = build_pipeline([])

    query = RAGQuery(
        text="Question with no matching knowledge.",
    )

    chunks = await rag_retrieval_service.retrieve(
        query,
    )

    context = context_assembler.assemble(
        query=query,
        chunks=chunks,
    )

    assert chunks == []
    assert len(retrieval_port.requests) == 1

    assert context.chunks == []
    assert context.context_text == ""
    assert context.metadata["chunk_count"] == 0
    assert context.metadata["source_chunk_count"] == 0


@pytest.mark.asyncio
async def test_pipeline_preserves_retrieval_order() -> None:
    first = build_match(
        text="First retrieved result.",
        similarity=0.60,
    )
    second = build_match(
        text="Second retrieved result.",
        similarity=0.95,
    )

    (
        _,
        _,
        rag_retrieval_service,
        context_assembler,
    ) = build_pipeline(
        [
            first,
            second,
        ],
    )

    query = RAGQuery(
        text="query",
    )

    chunks = await rag_retrieval_service.retrieve(
        query,
    )

    context = context_assembler.assemble(
        query=query,
        chunks=chunks,
    )

    assert context.chunks[0].chunk_id == first.chunk_id
    assert context.chunks[1].chunk_id == second.chunk_id

    assert context.chunks[0].similarity == 0.60
    assert context.chunks[1].similarity == 0.95


@pytest.mark.asyncio
async def test_pipeline_applies_context_limits() -> None:
    matches = [
        build_match(
            text="First knowledge chunk.",
            similarity=0.95,
        ),
        build_match(
            text="Second knowledge chunk.",
            similarity=0.90,
        ),
        build_match(
            text="Third knowledge chunk.",
            similarity=0.85,
        ),
    ]

    (
        _,
        _,
        rag_retrieval_service,
        _,
    ) = build_pipeline(matches)

    query = RAGQuery(
        text="query",
    )

    chunks = await rag_retrieval_service.retrieve(
        query,
    )

    context_assembler = RAGContextAssembler(
        config=RAGContextConfig(
            max_chunks=2,
            max_context_characters=10_000,
        ),
    )

    context = context_assembler.assemble(
        query=query,
        chunks=chunks,
    )

    assert len(context.chunks) == 2

    assert context.chunks[0].text == (
        "First knowledge chunk."
    )
    assert context.chunks[1].text == (
        "Second knowledge chunk."
    )

    assert context.metadata["source_chunk_count"] == 3
    assert context.metadata["chunk_count"] == 2


@pytest.mark.asyncio
async def test_pipeline_is_deterministic() -> None:
    matches = [
        build_match(
            text="Deterministic first result.",
            similarity=0.93,
        ),
        build_match(
            text="Deterministic second result.",
            similarity=0.88,
        ),
    ]

    (
        _,
        _,
        rag_retrieval_service,
        context_assembler,
    ) = build_pipeline(matches)

    query = RAGQuery(
        text="Explain deterministic retrieval.",
    )

    first_chunks = await rag_retrieval_service.retrieve(
        query,
    )

    first_context = context_assembler.assemble(
        query=query,
        chunks=first_chunks,
    )

    second_chunks = await rag_retrieval_service.retrieve(
        query,
    )

    second_context = context_assembler.assemble(
        query=query,
        chunks=second_chunks,
    )

    assert first_context.context_text == (
        second_context.context_text
    )

    assert first_context.metadata == (
        second_context.metadata
    )

    assert [
        chunk.text
        for chunk in first_context.chunks
    ] == [
        chunk.text
        for chunk in second_context.chunks
    ]


@pytest.mark.asyncio
async def test_pipeline_does_not_mutate_retrieved_chunks() -> None:
    matches = [
        build_match(
            text="Original knowledge.",
            similarity=0.92,
        ),
    ]

    (
        _,
        _,
        rag_retrieval_service,
        context_assembler,
    ) = build_pipeline(matches)

    query = RAGQuery(
        text="query",
    )

    chunks = await rag_retrieval_service.retrieve(
        query,
    )

    original_text = chunks[0].text
    original_metadata = dict(
        chunks[0].metadata,
    )

    context = context_assembler.assemble(
        query=query,
        chunks=chunks,
    )

    context.chunks[0].metadata["mutated"] = True

    assert chunks[0].text == original_text
    assert chunks[0].metadata == original_metadata


@pytest.mark.asyncio
async def test_project_filter_reaches_vector_retrieval() -> None:
    project_id = uuid4()

    (
        _,
        retrieval_port,
        rag_retrieval_service,
        _,
    ) = build_pipeline(
        [
            build_match(
                text="Project knowledge.",
                similarity=0.9,
            ),
        ],
    )

    query = RAGQuery(
        text="project query",
        project_id=project_id,
    )

    await rag_retrieval_service.retrieve(query)

    assert len(retrieval_port.requests) == 1
    assert (
        retrieval_port.requests[0].project_id
        == project_id
    )