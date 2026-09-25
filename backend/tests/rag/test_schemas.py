from __future__ import annotations

from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.rag.schemas import RAGContext, RAGQuery, RAGRetrievedChunk


def _model_info() -> EmbeddingModelInfo:
    return EmbeddingModelInfo(
        provider="sentence-transformers",
        model_name="all-MiniLM-L6-v2",
        dimension=384,
        normalized=True,
    )


def _retrieved_chunk() -> RAGRetrievedChunk:
    return RAGRetrievedChunk(
        chunk_id=uuid4(),
        text="Retrieved knowledge content.",
        similarity=0.91,
        model=_model_info(),
        metadata={"source": "test"},
    )


def test_query_accepts_valid_values() -> None:
    project_id = uuid4()

    query = RAGQuery(
        text="What is artificial intelligence?",
        top_k=5,
        similarity_threshold=0.7,
        project_id=project_id,
        metadata_filter={"topic": "ai"},
    )

    assert query.text == "What is artificial intelligence?"
    assert query.top_k == 5
    assert query.similarity_threshold == 0.7
    assert query.project_id == project_id
    assert query.metadata_filter == {"topic": "ai"}


def test_query_uses_expected_defaults() -> None:
    query = RAGQuery(text="Explain machine learning.")

    assert query.top_k == 5
    assert query.similarity_threshold is None
    assert query.project_id is None
    assert query.metadata_filter == {}


@pytest.mark.parametrize("text", ["", "   ", "\t", "\n"])
def test_query_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValidationError):
        RAGQuery(text=text)


@pytest.mark.parametrize("top_k", [0, -1])
def test_query_rejects_invalid_top_k(top_k: int) -> None:
    with pytest.raises(ValidationError):
        RAGQuery(text="query", top_k=top_k)


@pytest.mark.parametrize("threshold", [-1.000001, 1.000001])
def test_query_rejects_invalid_similarity_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        RAGQuery(
            text="query",
            similarity_threshold=threshold,
        )


@pytest.mark.parametrize("threshold", [float("nan"), float("inf"), -float("inf")])
def test_query_rejects_non_finite_similarity_threshold(
    threshold: float,
) -> None:
    with pytest.raises(ValidationError):
        RAGQuery(
            text="query",
            similarity_threshold=threshold,
        )


def test_query_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RAGQuery(
            text="query",
            unknown_field="unexpected",
        )


def test_query_is_immutable() -> None:
    query = RAGQuery(text="query")

    with pytest.raises(ValidationError):
        query.text = "changed"


def test_retrieved_chunk_accepts_valid_values() -> None:
    chunk_id = uuid4()

    chunk = RAGRetrievedChunk(
        chunk_id=chunk_id,
        text="Knowledge content.",
        similarity=0.85,
        model=_model_info(),
        metadata={"page": 3},
    )

    assert chunk.chunk_id == chunk_id
    assert chunk.text == "Knowledge content."
    assert chunk.similarity == 0.85
    assert chunk.model.dimension == 384
    assert chunk.metadata == {"page": 3}


@pytest.mark.parametrize("text", ["", "   ", "\t", "\n"])
def test_retrieved_chunk_rejects_blank_text(text: str) -> None:
    with pytest.raises(ValidationError):
        RAGRetrievedChunk(
            chunk_id=uuid4(),
            text=text,
            similarity=0.9,
            model=_model_info(),
        )


@pytest.mark.parametrize(
    "similarity",
    [float("nan"), float("inf"), -float("inf")],
)
def test_retrieved_chunk_rejects_non_finite_similarity(
    similarity: float,
) -> None:
    with pytest.raises(ValidationError):
        RAGRetrievedChunk(
            chunk_id=uuid4(),
            text="Knowledge content.",
            similarity=similarity,
            model=_model_info(),
        )


def test_retrieved_chunk_allows_missing_provenance() -> None:
    chunk = _retrieved_chunk()

    assert chunk.provenance is None


def test_retrieved_chunk_preserves_metadata() -> None:
    metadata = {
        "source": "document.pdf",
        "page": 4,
        "section": "Introduction",
    }

    chunk = RAGRetrievedChunk(
        chunk_id=uuid4(),
        text="Knowledge content.",
        similarity=0.9,
        model=_model_info(),
        metadata=metadata,
    )

    assert chunk.metadata == metadata


def test_retrieved_chunk_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RAGRetrievedChunk(
            chunk_id=uuid4(),
            text="Knowledge content.",
            similarity=0.9,
            model=_model_info(),
            unknown_field="unexpected",
        )


def test_retrieved_chunk_is_immutable() -> None:
    chunk = _retrieved_chunk()

    with pytest.raises(ValidationError):
        chunk.text = "changed"


def test_context_accepts_empty_retrieval() -> None:
    query = RAGQuery(text="query")

    context = RAGContext(
        query=query,
        chunks=[],
        context_text="",
    )

    assert context.query == query
    assert context.chunks == []
    assert context.context_text == ""


def test_context_preserves_retrieved_chunks() -> None:
    query = RAGQuery(text="query")
    chunk = _retrieved_chunk()

    context = RAGContext(
        query=query,
        chunks=[chunk],
        context_text=chunk.text,
    )

    assert context.chunks == [chunk]
    assert context.context_text == chunk.text


def test_context_preserves_metadata() -> None:
    query = RAGQuery(text="query")
    metadata = {
        "retrieval_count": 2,
        "provider": "sentence-transformers",
    }

    context = RAGContext(
        query=query,
        chunks=[],
        context_text="",
        metadata=metadata,
    )

    assert context.metadata == metadata


def test_context_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RAGContext(
            query=RAGQuery(text="query"),
            chunks=[],
            context_text="",
            unknown_field="unexpected",
        )


def test_context_is_immutable() -> None:
    context = RAGContext(
        query=RAGQuery(text="query"),
        chunks=[],
        context_text="",
    )

    with pytest.raises(ValidationError):
        context.context_text = "changed"


def test_context_supports_multiple_chunks_in_order() -> None:
    query = RAGQuery(text="query")

    first = RAGRetrievedChunk(
        chunk_id=uuid4(),
        text="First chunk.",
        similarity=0.95,
        model=_model_info(),
    )
    second = RAGRetrievedChunk(
        chunk_id=uuid4(),
        text="Second chunk.",
        similarity=0.88,
        model=_model_info(),
    )

    context = RAGContext(
        query=query,
        chunks=[first, second],
        context_text="First chunk.\n\nSecond chunk.",
    )

    assert context.chunks[0] == first
    assert context.chunks[1] == second