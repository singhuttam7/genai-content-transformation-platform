from __future__ import annotations

from app.ingestion.schemas import ContentBlockType
from app.knowledge.chunking import (
    ChunkSourceReference,
    ChunkingConfig,
    ChunkingResult,
    ChunkingStatistics,
    KnowledgeChunkDraft,
)


def test_default_chunking_config() -> None:
    config = ChunkingConfig()

    assert config.max_characters == 1200
    assert config.overlap_characters == 150
    assert config.min_characters == 1
    assert config.preserve_headings is True
    assert config.preserve_page_boundaries is True
    assert config.preserve_time_boundaries is True
    assert config.include_non_text_blocks is False


def test_chunking_config_accepts_custom_values() -> None:
    config = ChunkingConfig(
        max_characters=800,
        overlap_characters=100,
        min_characters=20,
        preserve_headings=False,
        preserve_page_boundaries=False,
        preserve_time_boundaries=False,
        include_non_text_blocks=True,
    )

    assert config.max_characters == 800
    assert config.overlap_characters == 100
    assert config.min_characters == 20
    assert config.preserve_headings is False
    assert config.include_non_text_blocks is True


def test_chunking_config_rejects_overlap_equal_to_maximum() -> None:
    try:
        ChunkingConfig(
            max_characters=500,
            overlap_characters=500,
        )
    except ValueError as exc:
        assert "overlap_characters" in str(exc)
    else:
        raise AssertionError(
            "Expected ChunkingConfig validation to fail."
        )


def test_chunking_config_rejects_overlap_greater_than_maximum() -> None:
    try:
        ChunkingConfig(
            max_characters=500,
            overlap_characters=501,
        )
    except ValueError as exc:
        assert "overlap_characters" in str(exc)
    else:
        raise AssertionError(
            "Expected ChunkingConfig validation to fail."
        )


def test_chunking_config_rejects_minimum_greater_than_maximum() -> None:
    try:
        ChunkingConfig(
            max_characters=100,
            min_characters=101,
        )
    except ValueError as exc:
        assert "min_characters" in str(exc)
    else:
        raise AssertionError(
            "Expected ChunkingConfig validation to fail."
        )


def test_chunk_source_reference_preserves_provenance() -> None:
    reference = ChunkSourceReference(
        element_orders=[2, 3, 4],
        block_types=[
            ContentBlockType.HEADING,
            ContentBlockType.PARAGRAPH,
        ],
        page_numbers=[4, 5],
        start_time=12.5,
        end_time=38.75,
        section_path=[
            "Architecture",
            "Knowledge Layer",
        ],
    )

    assert reference.element_orders == [2, 3, 4]
    assert reference.block_types == [
        ContentBlockType.HEADING,
        ContentBlockType.PARAGRAPH,
    ]
    assert reference.page_numbers == [4, 5]
    assert reference.start_time == 12.5
    assert reference.end_time == 38.75
    assert reference.section_path == [
        "Architecture",
        "Knowledge Layer",
    ]


def test_knowledge_chunk_draft_contract() -> None:
    reference = ChunkSourceReference(
        element_orders=[7],
        block_types=[ContentBlockType.PARAGRAPH],
        page_numbers=[3],
        section_path=["Introduction"],
    )

    chunk = KnowledgeChunkDraft(
        chunk_index=0,
        text="This is a knowledge chunk.",
        content_hash="a" * 64,
        source=reference,
        metadata={
            "test": True,
        },
    )

    assert chunk.chunk_index == 0
    assert chunk.text == "This is a knowledge chunk."
    assert chunk.content_hash == "a" * 64
    assert chunk.token_count is None
    assert chunk.source.page_numbers == [3]
    assert chunk.metadata["test"] is True


def test_knowledge_chunk_draft_accepts_token_count() -> None:
    reference = ChunkSourceReference(
        element_orders=[1],
        block_types=[ContentBlockType.PARAGRAPH],
    )

    chunk = KnowledgeChunkDraft(
        chunk_index=1,
        text="Token-aware chunk.",
        content_hash="b" * 64,
        token_count=42,
        source=reference,
    )

    assert chunk.token_count == 42


def test_chunking_statistics_contract() -> None:
    statistics = ChunkingStatistics(
        input_element_count=10,
        output_chunk_count=4,
        input_character_count=5000,
        output_character_count=4700,
        largest_chunk_characters=1200,
        smallest_chunk_characters=900,
    )

    assert statistics.input_element_count == 10
    assert statistics.output_chunk_count == 4
    assert statistics.input_character_count == 5000
    assert statistics.output_character_count == 4700
    assert statistics.largest_chunk_characters == 1200
    assert statistics.smallest_chunk_characters == 900


def test_chunking_result_contract() -> None:
    reference = ChunkSourceReference(
        element_orders=[0],
        block_types=[ContentBlockType.PARAGRAPH],
    )

    chunk = KnowledgeChunkDraft(
        chunk_index=0,
        text="Example chunk.",
        content_hash="c" * 64,
        source=reference,
    )

    statistics = ChunkingStatistics(
        input_element_count=1,
        output_chunk_count=1,
        input_character_count=14,
        output_character_count=14,
        largest_chunk_characters=14,
        smallest_chunk_characters=14,
    )

    result = ChunkingResult(
        chunks=[chunk],
        statistics=statistics,
        strategy="structure_aware",
        strategy_version="1.0",
    )

    assert len(result.chunks) == 1
    assert result.chunks[0].chunk_index == 0
    assert result.statistics.output_chunk_count == 1
    assert result.strategy == "structure_aware"
    assert result.strategy_version == "1.0"


def test_empty_chunking_result_is_valid() -> None:
    statistics = ChunkingStatistics(
        input_element_count=0,
        output_chunk_count=0,
        input_character_count=0,
        output_character_count=0,
        largest_chunk_characters=0,
        smallest_chunk_characters=0,
    )

    result = ChunkingResult(
        chunks=[],
        statistics=statistics,
        strategy="structure_aware",
        strategy_version="1.0",
    )

    assert result.chunks == []
    assert result.statistics.output_chunk_count == 0