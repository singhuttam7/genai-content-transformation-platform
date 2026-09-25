from __future__ import annotations

from uuid import uuid4

import pytest

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.rag.config import RAGContextConfig
from app.rag.context import RAGContextAssembler
from app.rag.schemas import (
    RAGQuery,
    RAGRetrievedChunk,
)


MODEL = EmbeddingModelInfo(
    provider="sentence-transformers",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    dimension=384,
    normalized=True,
)


def build_chunk(
    *,
    text: str = "Knowledge content.",
    similarity: float = 0.9,
) -> RAGRetrievedChunk:
    return RAGRetrievedChunk(
        chunk_id=uuid4(),
        text=text,
        similarity=similarity,
        model=MODEL,
        metadata={
            "language": "en",
        },
    )


def test_default_config() -> None:
    config = RAGContextConfig()

    assert config.max_chunks == 8
    assert config.max_context_characters == 12_000


def test_config_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError):
        RAGContextConfig(max_chunks=0)

    with pytest.raises(ValueError):
        RAGContextConfig(max_context_characters=0)


def test_config_is_immutable() -> None:
    config = RAGContextConfig()

    with pytest.raises(Exception):
        config.max_chunks = 10


def test_assemble_valid_context() -> None:
    query = RAGQuery(
        text="What does the document describe?",
    )

    chunks = [
        build_chunk(
            text="First knowledge chunk.",
            similarity=0.95,
        ),
        build_chunk(
            text="Second knowledge chunk.",
            similarity=0.87,
        ),
    ]

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=chunks,
    )

    assert context.query == query
    assert context.chunks == chunks
    assert context.metadata["chunk_count"] == 2
    assert context.metadata["source_chunk_count"] == 2
    assert context.metadata["similarities"] == [
        0.95,
        0.87,
    ]


def test_context_text_is_deterministic() -> None:
    query = RAGQuery(
        text="Explain the document.",
    )

    chunks = [
        build_chunk(text="First chunk."),
        build_chunk(text="Second chunk."),
    ]

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=chunks,
    )

    assert context.context_text == (
        "Retrieved Knowledge Context\n"
        "\n"
        "[Chunk 1]\n"
        "First chunk.\n"
        "\n"
        "[Chunk 2]\n"
        "Second chunk."
    )


def test_retrieval_order_is_preserved() -> None:
    query = RAGQuery(text="query")

    first = build_chunk(text="First.")
    second = build_chunk(text="Second.")
    third = build_chunk(text="Third.")

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=[
            first,
            second,
            third,
        ],
    )

    assert [
        chunk.chunk_id
        for chunk in context.chunks
    ] == [
        first.chunk_id,
        second.chunk_id,
        third.chunk_id,
    ]


def test_assembler_does_not_sort_by_similarity() -> None:
    query = RAGQuery(text="query")

    low = build_chunk(
        text="Lower similarity.",
        similarity=0.50,
    )
    high = build_chunk(
        text="Higher similarity.",
        similarity=0.95,
    )

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=[low, high],
    )

    assert context.chunks[0].chunk_id == low.chunk_id
    assert context.chunks[1].chunk_id == high.chunk_id


def test_empty_chunks_produce_empty_context() -> None:
    query = RAGQuery(text="query")

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=[],
    )

    assert context.chunks == []
    assert context.context_text == ""
    assert context.metadata == {
        "chunk_count": 0,
        "source_chunk_count": 0,
        "chunk_ids": [],
        "similarities": [],
        "has_provenance": False,
        "max_chunks": 8,
        "max_context_characters": 12_000,
        "truncated_chunk_count": 0,
    }


def test_max_chunks_limit_is_enforced() -> None:
    config = RAGContextConfig(
        max_chunks=2,
        max_context_characters=10_000,
    )

    assembler = RAGContextAssembler(config)

    chunks = [
        build_chunk(text="First."),
        build_chunk(text="Second."),
        build_chunk(text="Third."),
        build_chunk(text="Fourth."),
    ]

    context = assembler.assemble(
        query=RAGQuery(text="query"),
        chunks=chunks,
    )

    assert len(context.chunks) == 2
    assert context.chunks[0].text == "First."
    assert context.chunks[1].text == "Second."
    assert context.metadata["source_chunk_count"] == 4
    assert context.metadata["chunk_count"] == 2


def test_max_character_limit_is_enforced() -> None:
    config = RAGContextConfig(
        max_chunks=10,
        max_context_characters=80,
    )

    assembler = RAGContextAssembler(config)

    chunks = [
        build_chunk(
            text="A" * 100,
        ),
    ]

    context = assembler.assemble(
        query=RAGQuery(text="query"),
        chunks=chunks,
    )

    assert len(context.context_text) <= 80
    assert context.metadata["truncated_chunk_count"] == 1


def test_oversized_first_chunk_is_truncated_deterministically() -> None:
    config = RAGContextConfig(
        max_chunks=10,
        max_context_characters=60,
    )

    assembler = RAGContextAssembler(config)

    chunks = [
        build_chunk(
            text="ABCDEFGHIJKLMNOPQRSTUVWXYZ" * 10,
        ),
    ]

    first = assembler.assemble(
        query=RAGQuery(text="query"),
        chunks=chunks,
    )

    second = assembler.assemble(
        query=RAGQuery(text="query"),
        chunks=chunks,
    )

    assert first.context_text == second.context_text
    assert first.chunks[0].text == second.chunks[0].text
    assert first.metadata == second.metadata


def test_character_limit_preserves_retrieval_order() -> None:
    config = RAGContextConfig(
        max_chunks=10,
        max_context_characters=150,
    )

    assembler = RAGContextAssembler(config)

    first = build_chunk(
        text="First result.",
        similarity=0.5,
    )
    second = build_chunk(
        text="Second result.",
        similarity=0.9,
    )

    context = assembler.assemble(
        query=RAGQuery(text="query"),
        chunks=[first, second],
    )

    assert context.chunks[0].chunk_id == first.chunk_id

    if len(context.chunks) > 1:
        assert context.chunks[1].chunk_id == second.chunk_id


def test_context_contains_chunk_ids() -> None:
    query = RAGQuery(text="query")

    first = build_chunk(text="First.")
    second = build_chunk(text="Second.")

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=[first, second],
    )

    assert context.metadata["chunk_ids"] == [
        str(first.chunk_id),
        str(second.chunk_id),
    ]


def test_context_tracks_provenance_presence() -> None:
    query = RAGQuery(text="query")

    chunk_without_provenance = build_chunk(
        text="No provenance.",
    )

    chunk_with_provenance = RAGRetrievedChunk(
        chunk_id=uuid4(),
        text="Has provenance.",
        similarity=0.8,
        model=MODEL,
        provenance={
            "source": {
                "project_id": str(uuid4()),
                "source_id": str(uuid4()),
                "source_type": "pdf",
                "source_uri": "file:///document.pdf",
                "title": "Document",
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
        },
    )

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=[
            chunk_without_provenance,
            chunk_with_provenance,
        ],
    )

    assert context.metadata["has_provenance"] is True


def test_assembly_does_not_mutate_source_chunks() -> None:
    query = RAGQuery(text="query")

    chunk = build_chunk(
        text="Original content.",
    )

    context = RAGContextAssembler().assemble(
        query=query,
        chunks=[chunk],
    )

    context.chunks[0].metadata["new_key"] = "new_value"

    assert "new_key" not in chunk.metadata


def test_assembly_does_not_mutate_original_oversized_chunk() -> None:
    config = RAGContextConfig(
        max_chunks=2,
        max_context_characters=60,
    )

    assembler = RAGContextAssembler(config)

    original_text = "Original content. " * 20

    chunk = build_chunk(
        text=original_text,
    )

    context = assembler.assemble(
        query=RAGQuery(text="query"),
        chunks=[chunk],
    )

    assert chunk.text == original_text
    assert context.chunks[0].text != original_text


def test_assembly_rejects_invalid_query() -> None:
    with pytest.raises(TypeError):
        RAGContextAssembler().assemble(
            query=object(),  # type: ignore[arg-type]
            chunks=[],
        )


def test_assembly_rejects_non_list_chunks() -> None:
    query = RAGQuery(text="query")

    with pytest.raises(TypeError):
        RAGContextAssembler().assemble(
            query=query,
            chunks=object(),  # type: ignore[arg-type]
        )


def test_assembly_rejects_invalid_chunk_objects() -> None:
    query = RAGQuery(text="query")

    with pytest.raises(TypeError):
        RAGContextAssembler().assemble(
            query=query,
            chunks=[object()],  # type: ignore[list-item]
        )