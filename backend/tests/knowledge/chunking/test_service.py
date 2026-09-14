from __future__ import annotations

from uuid import uuid4

import pytest

from app.ingestion.schemas import (
    CanonicalContent,
    ContentBlock,
    ContentBlockType,
    SourceReference,
)
from app.knowledge.chunking import (
    ChunkingConfig,
    StructureAwareChunker,
)
from app.knowledge.normalization import KnowledgeContentNormalizer


def make_source() -> SourceReference:
    return SourceReference(
        source_id=uuid4(),
        source_type="txt",
    )


def make_content(
    *,
    text: str,
    segments: list[ContentBlock],
    title: str = "Test Document",
    language: str = "en",
) -> CanonicalContent:
    return CanonicalContent(
        source=make_source(),
        title=title,
        text=text,
        language=language,
        segments=segments,
    )


async def normalize(
    content: CanonicalContent,
):
    normalizer = KnowledgeContentNormalizer()
    return await normalizer.normalize(content)


@pytest.mark.asyncio
async def test_chunker_creates_single_chunk_for_small_document() -> None:
    content = make_content(
        text="This is a small document.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="This is a small document.",
                order=0,
            )
        ],
    )

    document = await normalize(content)

    chunker = StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    )

    result = await chunker.chunk(document)

    assert len(result.chunks) == 1
    assert result.chunks[0].text == "This is a small document."
    assert result.chunks[0].chunk_index == 0


@pytest.mark.asyncio
async def test_chunker_preserves_element_order() -> None:
    content = make_content(
        text="First paragraph.\n\nSecond paragraph.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="First paragraph.",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Second paragraph.",
                order=1,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert len(result.chunks) == 1

    assert result.chunks[0].source.element_orders == [0, 1]


@pytest.mark.asyncio
async def test_chunker_preserves_block_types() -> None:
    content = make_content(
        text="Heading\n\nParagraph content.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Heading",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Paragraph content.",
                order=1,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert result.chunks[0].source.block_types == [
        ContentBlockType.HEADING,
        ContentBlockType.PARAGRAPH,
    ]


@pytest.mark.asyncio
async def test_heading_creates_structural_boundary() -> None:
    content = make_content(
        text=(
            "Introduction\n\n"
            "Introduction paragraph.\n\n"
            "Architecture\n\n"
            "Architecture paragraph."
        ),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Introduction",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Introduction paragraph.",
                order=1,
            ),
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Architecture",
                order=2,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Architecture paragraph.",
                order=3,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert len(result.chunks) == 2

    assert result.chunks[0].text == (
        "Introduction\n\nIntroduction paragraph."
    )
    assert result.chunks[1].text == (
        "Architecture\n\nArchitecture paragraph."
    )


@pytest.mark.asyncio
async def test_heading_context_is_preserved() -> None:
    content = make_content(
        text=(
            "Architecture\n\n"
            "The architecture contains several services."
        ),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Architecture",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=(
                    "The architecture contains several services."
                ),
                order=1,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker().chunk(document)

    assert result.chunks[0].source.section_path == [
        "Architecture"
    ]

    assert result.chunks[0].metadata["section_path"] == [
        "Architecture"
    ]


@pytest.mark.asyncio
async def test_new_heading_replaces_previous_section_context() -> None:
    content = make_content(
        text=(
            "Introduction\n\n"
            "Intro text.\n\n"
            "Architecture\n\n"
            "Architecture text."
        ),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Introduction",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Intro text.",
                order=1,
            ),
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Architecture",
                order=2,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Architecture text.",
                order=3,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert result.chunks[0].source.section_path == [
        "Introduction"
    ]

    assert result.chunks[1].source.section_path == [
        "Architecture"
    ]


@pytest.mark.asyncio
async def test_chunker_preserves_page_provenance() -> None:
    content = make_content(
        text="Page one.\n\nPage two.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Page one.",
                order=0,
                page_number=1,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Page two.",
                order=1,
                page_number=2,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert result.chunks[0].source.page_numbers == [1, 2]


@pytest.mark.asyncio
async def test_chunker_preserves_media_timestamps() -> None:
    content = make_content(
        text="First segment.\n\nSecond segment.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="First segment.",
                order=0,
                start_time=10.0,
                end_time=20.0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Second segment.",
                order=1,
                start_time=20.0,
                end_time=35.0,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert result.chunks[0].source.start_time == 10.0
    assert result.chunks[0].source.end_time == 35.0


@pytest.mark.asyncio
async def test_oversized_paragraph_is_split() -> None:
    text = "word " * 100

    content = make_content(
        text=text.strip(),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text.strip(),
                order=0,
            )
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert len(result.chunks) > 1

    for chunk in result.chunks:
        assert len(chunk.text) <= 100


@pytest.mark.asyncio
async def test_oversized_sentence_uses_word_boundaries() -> None:
    text = (
        "alpha beta gamma delta epsilon zeta eta theta "
        "iota kappa lambda mu"
    )

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=25)
    ).chunk(document)

    assert len(result.chunks) > 1

    for chunk in result.chunks:
        assert len(chunk.text) <= 25
        assert "\n\n" not in chunk.text


@pytest.mark.asyncio
async def test_extremely_long_word_uses_hard_split() -> None:
    text = "x" * 250

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=100)
    ).chunk(document)

    assert len(result.chunks) == 3
    assert [len(chunk.text) for chunk in result.chunks] == [
        100,
        100,
        50,
    ]


@pytest.mark.asyncio
async def test_empty_document_produces_no_chunks() -> None:
    content = make_content(
        text="",
        segments=[],
    )

    document = await normalize(content)

    result = await StructureAwareChunker().chunk(document)

    assert result.chunks == []
    assert result.statistics.output_chunk_count == 0


@pytest.mark.asyncio
async def test_chunk_indices_are_sequential() -> None:
    content = make_content(
        text=(
            "Heading A\n\n"
            "A long paragraph that should be split into "
            "multiple chunks because it contains enough text."
        ),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Heading A",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=(
                    "A long paragraph that should be split into "
                    "multiple chunks because it contains enough text."
                ),
                order=1,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(max_characters=40)
    ).chunk(document)

    assert [chunk.chunk_index for chunk in result.chunks] == list(
        range(len(result.chunks))
    )


@pytest.mark.asyncio
async def test_chunk_hash_is_deterministic() -> None:
    content = make_content(
        text="Deterministic chunk content.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Deterministic chunk content.",
                order=0,
            )
        ],
    )

    document = await normalize(content)

    chunker = StructureAwareChunker()

    first = await chunker.chunk(document)
    second = await chunker.chunk(document)

    assert first.chunks[0].content_hash == (
        second.chunks[0].content_hash
    )


@pytest.mark.asyncio
async def test_same_input_produces_same_complete_result() -> None:
    content = make_content(
        text=(
            "Introduction\n\n"
            "This is deterministic content.\n\n"
            "More deterministic content."
        ),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Introduction",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="This is deterministic content.",
                order=1,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="More deterministic content.",
                order=2,
            ),
        ],
    )

    document = await normalize(content)

    chunker = StructureAwareChunker()

    first = await chunker.chunk(document)
    second = await chunker.chunk(document)

    assert first.model_dump() == second.model_dump()


@pytest.mark.asyncio
async def test_unicode_content_is_preserved() -> None:
    text = (
        "Artificial Intelligence भारत में तेजी से विकसित हो रही है। "
        "生成AI is transforming content creation."
    )

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
        language="en",
    )

    document = await normalize(content)

    result = await StructureAwareChunker().chunk(document)

    assert result.chunks[0].text == text


@pytest.mark.asyncio
async def test_statistics_are_generated() -> None:
    content = make_content(
        text="First.\n\nSecond.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="First.",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Second.",
                order=1,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker().chunk(document)

    assert result.statistics.input_element_count == 2
    assert result.statistics.output_chunk_count == 1
    assert result.statistics.input_character_count == len(
        document.text
    )
    assert result.statistics.output_character_count == len(
        result.chunks[0].text
    )


@pytest.mark.asyncio
async def test_strategy_metadata_is_present() -> None:
    content = make_content(
        text="Strategy test.",
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Strategy test.",
                order=0,
            )
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker().chunk(document)

    assert result.strategy == "structure_aware"
    assert result.strategy_version == "1.0"

    assert result.chunks[0].metadata[
        "chunking_strategy"
    ] == "structure_aware"

    assert result.chunks[0].metadata[
        "chunking_strategy_version"
    ] == "1.0"



@pytest.mark.asyncio
async def test_overlap_is_disabled_when_configured_as_zero() -> None:
    text = " ".join(f"word{i}" for i in range(40))

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(
            max_characters=50,
            overlap_characters=0,
        )
    ).chunk(document)

    assert len(result.chunks) > 1

    for first, second in zip(
        result.chunks,
        result.chunks[1:],
    ):
        first_words = first.text.split()
        second_words = second.text.split()

        assert first_words[-1] != second_words[0]


@pytest.mark.asyncio
async def test_overlap_is_word_boundary_aware() -> None:
    text = " ".join(
        [
            "alpha",
            "bravo",
            "charlie",
            "delta",
            "echo",
            "foxtrot",
            "golf",
            "hotel",
            "india",
            "juliet",
        ]
    )

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(
            max_characters=35,
            overlap_characters=10,
        )
    ).chunk(document)

    assert len(result.chunks) > 1

    for chunk in result.chunks:
        assert len(chunk.text) <= 35


@pytest.mark.asyncio
async def test_overlap_does_not_cross_heading_boundary() -> None:
    content = make_content(
        text=(
            "Introduction\n\n"
            "alpha bravo charlie delta echo foxtrot golf hotel\n\n"
            "Architecture\n\n"
            "india juliet kilo lima mike november oscar"
        ),
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Introduction",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=(
                    "alpha bravo charlie delta echo foxtrot "
                    "golf hotel"
                ),
                order=1,
            ),
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Architecture",
                order=2,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=(
                    "india juliet kilo lima mike november "
                    "oscar"
                ),
                order=3,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(
            max_characters=45,
            overlap_characters=10,
        )
    ).chunk(document)

    assert len(result.chunks) >= 2

    for chunk in result.chunks:
        section = chunk.source.section_path

        if section == ["Introduction"]:
            assert "Architecture" not in chunk.text

        if section == ["Architecture"]:
            assert "Introduction" not in chunk.text


@pytest.mark.asyncio
async def test_overlap_preserves_section_context() -> None:
    text = (
        "Architecture\n\n"
        "alpha bravo charlie delta echo foxtrot golf hotel "
        "india juliet kilo lima mike november"
    )

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.HEADING,
                content="Architecture",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=(
                    "alpha bravo charlie delta echo foxtrot "
                    "golf hotel india juliet kilo lima mike "
                    "november"
                ),
                order=1,
            ),
        ],
    )

    document = await normalize(content)

    result = await StructureAwareChunker(
        ChunkingConfig(
            max_characters=45,
            overlap_characters=10,
        )
    ).chunk(document)

    assert len(result.chunks) > 1

    for chunk in result.chunks:
        assert chunk.source.section_path == [
            "Architecture"
        ]


@pytest.mark.asyncio
async def test_overlap_is_deterministic() -> None:
    text = " ".join(
        f"token{i}"
        for i in range(80)
    )

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
    )

    document = await normalize(content)

    chunker = StructureAwareChunker(
        ChunkingConfig(
            max_characters=100,
            overlap_characters=20,
        )
    )

    first = await chunker.chunk(document)
    second = await chunker.chunk(document)

    assert first.model_dump() == second.model_dump()


@pytest.mark.asyncio
async def test_overlap_does_not_exceed_maximum_chunk_size() -> None:
    text = " ".join(
        f"token{i}"
        for i in range(100)
    )

    content = make_content(
        text=text,
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=text,
                order=0,
            )
        ],
    )

    document = await normalize(content)

    max_characters = 80

    result = await StructureAwareChunker(
        ChunkingConfig(
            max_characters=max_characters,
            overlap_characters=25,
        )
    ).chunk(document)

    assert len(result.chunks) > 1

    for chunk in result.chunks:
        assert len(chunk.text) <= max_characters