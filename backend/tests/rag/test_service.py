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
    model_name="all-MiniLM-L6-v2",
    dimension=3,
    normalized=True,
)


class FakeEmbeddingPort(EmbeddingPort):
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


class FakeRetrievalPort(VectorRetrievalPort):
    def __init__(
        self,
        matches: list[VectorRetrievalMatch] | None = None,
    ) -> None:
        self.matches = matches or []
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


def build_service(
    *,
    matches: list[VectorRetrievalMatch] | None = None,
) -> tuple[
    RAGRetrievalService,
    FakeEmbeddingPort,
    FakeRetrievalPort,
]:
    embedding_port = FakeEmbeddingPort()
    retrieval_port = FakeRetrievalPort(
        matches=matches,
    )

    embedding_service = EmbeddingService(
        embedding_port,
    )

    retrieval_service = VectorRetrievalService(
        retrieval_port,
    )

    rag_service = RAGRetrievalService(
        embedding_service=embedding_service,
        retrieval_service=retrieval_service,
    )

    return (
        rag_service,
        embedding_port,
        retrieval_port,
    )


def build_match(
    *,
    text: str = "Retrieved knowledge content.",
    similarity: float = 0.91,
) -> VectorRetrievalMatch:
    return VectorRetrievalMatch(
        chunk_id=uuid4(),
        text=text,
        similarity=similarity,
        model=MODEL,
        metadata={
            "chunk_index": 2,
            "language": "en",
        },
    )


def test_constructor_rejects_invalid_embedding_service() -> None:
    retrieval_port = FakeRetrievalPort()
    retrieval_service = VectorRetrievalService(
        retrieval_port,
    )

    with pytest.raises(TypeError):
        RAGRetrievalService(
            embedding_service=object(),  # type: ignore[arg-type]
            retrieval_service=retrieval_service,
        )


def test_constructor_rejects_invalid_retrieval_service() -> None:
    embedding_port = FakeEmbeddingPort()
    embedding_service = EmbeddingService(
        embedding_port,
    )

    with pytest.raises(TypeError):
        RAGRetrievalService(
            embedding_service=embedding_service,
            retrieval_service=object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_retrieve_rejects_invalid_query() -> None:
    service, _, _ = build_service()

    with pytest.raises(TypeError):
        await service.retrieve(
            object(),  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_retrieve_embeds_query_text() -> None:
    service, embedding_port, _ = build_service()

    query = RAGQuery(
        text="What does this document describe?",
    )

    await service.retrieve(query)

    assert len(embedding_port.requests) == 1
    assert (
        embedding_port.requests[0].text
        == "What does this document describe?"
    )


@pytest.mark.asyncio
async def test_retrieve_builds_vector_request_from_query() -> None:
    service, _, retrieval_port = build_service()

    project_id = uuid4()

    query = RAGQuery(
        text="What changed?",
        top_k=8,
        similarity_threshold=0.75,
        project_id=project_id,
        metadata_filter={
            "language": "en",
            "section": "Introduction",
        },
    )

    await service.retrieve(query)

    assert len(retrieval_port.requests) == 1

    request = retrieval_port.requests[0]

    assert request.query_vector == [0.1, 0.2, 0.3]
    assert request.model == MODEL
    assert request.top_k == 8
    assert request.similarity_threshold == 0.75
    assert request.project_id == project_id
    assert request.metadata_filter == {
        "language": "en",
        "section": "Introduction",
    }


@pytest.mark.asyncio
async def test_retrieve_maps_matches_to_rag_chunks() -> None:
    match = build_match()

    service, _, _ = build_service(
        matches=[match],
    )

    query = RAGQuery(
        text="Explain the document.",
    )

    chunks = await service.retrieve(query)

    assert len(chunks) == 1

    chunk = chunks[0]

    assert chunk.chunk_id == match.chunk_id
    assert chunk.text == match.text
    assert chunk.similarity == match.similarity
    assert chunk.model == match.model
    assert chunk.metadata == match.metadata
    assert chunk.provenance is None


@pytest.mark.asyncio
async def test_retrieve_preserves_retrieval_order() -> None:
    first = build_match(
        text="First result.",
        similarity=0.95,
    )
    second = build_match(
        text="Second result.",
        similarity=0.87,
    )
    third = build_match(
        text="Third result.",
        similarity=0.81,
    )

    service, _, _ = build_service(
        matches=[
            first,
            second,
            third,
        ],
    )

    chunks = await service.retrieve(
        RAGQuery(text="query"),
    )

    assert [chunk.chunk_id for chunk in chunks] == [
        first.chunk_id,
        second.chunk_id,
        third.chunk_id,
    ]

    assert [chunk.text for chunk in chunks] == [
        "First result.",
        "Second result.",
        "Third result.",
    ]


@pytest.mark.asyncio
async def test_retrieve_allows_empty_results() -> None:
    service, _, retrieval_port = build_service()

    chunks = await service.retrieve(
        RAGQuery(text="query"),
    )

    assert chunks == []
    assert len(retrieval_port.requests) == 1


@pytest.mark.asyncio
async def test_retrieve_does_not_mutate_query_metadata() -> None:
    metadata_filter = {
        "language": "en",
        "nested": {
            "section": "Introduction",
        },
    }

    query = RAGQuery(
        text="query",
        metadata_filter=metadata_filter,
    )

    service, _, retrieval_port = build_service()

    await service.retrieve(query)

    request_metadata = retrieval_port.requests[0].metadata_filter

    assert request_metadata == metadata_filter

    request_metadata["nested"]["section"] = "Changed"

    assert (
        query.metadata_filter["nested"]["section"]
        == "Introduction"
    )


@pytest.mark.asyncio
async def test_retrieve_preserves_valid_provenance() -> None:
    provenance = {
        "source": {
            "project_id": str(uuid4()),
            "source_id": str(uuid4()),
            "source_type": "pdf",
            "source_uri": "file:///document.pdf",
            "title": "Test document",
            "language": "en",
        },
        "document": {
            "document_id": str(uuid4()),
            "document_version": 1,
            "content_hash": "a" * 64,
        },
        "location": {
            "element_orders": [0],
            "block_types": [],
            "page_numbers": [1],
            "section_path": ["Introduction"],
            "start_time": None,
            "end_time": None,
        },
        "chunk_index": 0,
    }

    match = VectorRetrievalMatch(
        chunk_id=uuid4(),
        text="Knowledge with provenance.",
        similarity=0.93,
        model=MODEL,
        metadata={
            "provenance": provenance,
        },
    )

    service, _, _ = build_service(
        matches=[match],
    )

    chunks = await service.retrieve(
        RAGQuery(text="query"),
    )

    assert chunks[0].provenance is not None
    assert (
        chunks[0].provenance.model_dump(mode="json")
        == provenance
    )


@pytest.mark.asyncio
async def test_retrieve_ignores_invalid_provenance() -> None:
    match = VectorRetrievalMatch(
        chunk_id=uuid4(),
        text="Knowledge without valid provenance.",
        similarity=0.8,
        model=MODEL,
        metadata={
            "provenance": {
                "invalid": True,
            },
        },
    )

    service, _, _ = build_service(
        matches=[match],
    )

    chunks = await service.retrieve(
        RAGQuery(text="query"),
    )

    assert chunks[0].provenance is None
    assert chunks[0].metadata["provenance"] == {
        "invalid": True,
    }


def test_map_match_rejects_invalid_match() -> None:
    with pytest.raises(TypeError):
        RAGRetrievalService._map_match(
            object(),  # type: ignore[arg-type]
        )