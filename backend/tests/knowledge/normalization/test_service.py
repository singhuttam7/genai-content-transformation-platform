from __future__ import annotations

from uuid import uuid4

import pytest

from app.ingestion.schemas import (
    CanonicalContent,
    ContentBlock,
    ContentBlockType,
    SourceReference,
)
from app.knowledge.normalization import (
    KnowledgeContentNormalizer,
)


# ============================================================
# Test Fixtures
# ============================================================


def create_canonical_content(
    *,
    text: str = "Hello world.",
    segments: list[ContentBlock] | None = None,
    title: str | None = "Test Document",
    language: str | None = "EN",
) -> CanonicalContent:
    """Create a valid A4 CanonicalContent test object."""

    return CanonicalContent(
        source=SourceReference(
            source_type="txt",
            source_id=uuid4(),
            uri="test://knowledge/document-001",
        ),
        title=title,
        text=text,
        language=language,
        segments=segments or [],
        entities=[],
        topics=[],
        claims=[],
        keywords=[],
        context={
            "test": True,
        },
        provenance={
            "ingestion": "test",
        },
        metadata={
            "fixture": "knowledge-normalization",
        },
    )


# ============================================================
# Basic Normalization
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_creates_normalized_document() -> None:
    """Verify basic CanonicalContent normalization."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text="Hello world.",
        title=" Test Document ",
        language="EN",
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.text == "Hello world."
    assert result.title == "Test Document"
    assert result.language == "en"

    assert result.source.source_id == canonical.source.source_id

    assert result.provenance["ingestion"] == "test"

    assert (
        result.provenance["knowledge_normalizer"]
        == "default"
    )

    assert (
        result.provenance[
            "knowledge_normalization_version"
        ]
        == "1.0"
    )


# ============================================================
# Text Normalization
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_normalizes_line_endings() -> None:
    """Verify CRLF and CR line endings are normalized."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text=(
            "First line\r\n"
            "Second line\r"
            "Third line"
        ),
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.text == (
        "First line\n"
        "Second line\n"
        "Third line"
    )


@pytest.mark.asyncio
async def test_normalizer_removes_boundary_blank_lines() -> None:
    """Verify unnecessary document boundary whitespace is removed."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text="\n\nHello world.\n\n",
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.text == "Hello world."


@pytest.mark.asyncio
async def test_normalizer_preserves_internal_newlines() -> None:
    """Verify meaningful internal newlines are preserved."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text=(
            "Heading\n"
            "\n"
            "Paragraph one.\n"
            "Paragraph two."
        ),
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.text == (
        "Heading\n"
        "\n"
        "Paragraph one.\n"
        "Paragraph two."
    )


@pytest.mark.asyncio
async def test_normalizer_handles_empty_text() -> None:
    """Verify empty document text is handled safely."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text="",
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.text == ""
    assert result.content_hash
    assert len(result.content_hash) == 64


# ============================================================
# Structural Preservation
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_preserves_segment_structure() -> None:
    """Verify canonical segments become knowledge elements."""

    normalizer = KnowledgeContentNormalizer()

    segments = [
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
        ContentBlock(
            block_type=ContentBlockType.HEADING,
            content="Important section",
            order=2,
        ),
    ]

    canonical = create_canonical_content(
        text=(
            "First paragraph.\n"
            "Second paragraph.\n"
            "Important section"
        ),
        segments=segments,
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert len(result.elements) == 3

    assert [
        element.content
        for element in result.elements
    ] == [
        "First paragraph.",
        "Second paragraph.",
        "Important section",
    ]

    assert [
        element.block_type
        for element in result.elements
    ] == [
        ContentBlockType.PARAGRAPH,
        ContentBlockType.PARAGRAPH,
        ContentBlockType.HEADING,
    ]

    assert [
        element.order
        for element in result.elements
    ] == [0, 1, 2]


@pytest.mark.asyncio
async def test_normalizer_preserves_page_metadata() -> None:
    """Verify PDF-style page metadata survives normalization."""

    normalizer = KnowledgeContentNormalizer()

    segments = [
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="Page one content.",
            order=0,
            page_number=1,
        ),
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="Page two content.",
            order=1,
            page_number=2,
        ),
    ]

    canonical = create_canonical_content(
        text=(
            "Page one content.\n"
            "Page two content."
        ),
        segments=segments,
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert len(result.elements) == 2

    assert result.elements[0].page_number == 1
    assert result.elements[1].page_number == 2


@pytest.mark.asyncio
async def test_normalizer_preserves_timestamp_metadata() -> None:
    """Verify video/audio timestamps survive normalization."""

    normalizer = KnowledgeContentNormalizer()

    segments = [
        ContentBlock(
            block_type=ContentBlockType.TRANSCRIPT,
            content="First spoken segment.",
            order=0,
            start_time=0.0,
            end_time=12.5,
            metadata={
                "provider": "test-asr",
            },
        ),
        ContentBlock(
            block_type=ContentBlockType.TRANSCRIPT,
            content="Second spoken segment.",
            order=1,
            start_time=12.5,
            end_time=28.0,
            metadata={
                "provider": "test-asr",
            },
        ),
    ]

    canonical = create_canonical_content(
        text=(
            "First spoken segment.\n"
            "Second spoken segment."
        ),
        segments=segments,
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert len(result.elements) == 2

    first = result.elements[0]
    second = result.elements[1]

    assert first.start_time == 0.0
    assert first.end_time == 12.5

    assert second.start_time == 12.5
    assert second.end_time == 28.0

    assert first.metadata["provider"] == "test-asr"
    assert second.metadata["provider"] == "test-asr"


# ============================================================
# Empty Structural Elements
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_removes_empty_textual_elements() -> None:
    """Verify empty textual blocks are excluded."""

    normalizer = KnowledgeContentNormalizer()

    segments = [
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="",
            order=0,
        ),
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="Valid paragraph.",
            order=1,
        ),
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="   ",
            order=2,
        ),
    ]

    canonical = create_canonical_content(
        text="Valid paragraph.",
        segments=segments,
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert len(result.elements) == 1
    assert result.elements[0].content == "Valid paragraph."


# ============================================================
# Metadata Preservation
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_preserves_semantic_metadata() -> None:
    """Verify semantic fields survive normalization."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content()

    canonical.entities = [
        "OpenAI",
        "Artificial Intelligence",
    ]

    canonical.topics = [
        "AI",
        "Machine Learning",
    ]

    canonical.claims = [
        "AI can transform content.",
    ]

    canonical.keywords = [
        "AI",
        "content",
        "transformation",
    ]

    result = await normalizer.normalize(
        canonical,
    )

    assert result.entities == [
        "OpenAI",
        "Artificial Intelligence",
    ]

    assert result.topics == [
        "AI",
        "Machine Learning",
    ]

    assert result.claims == [
        "AI can transform content.",
    ]

    assert result.keywords == [
        "AI",
        "content",
        "transformation",
    ]


@pytest.mark.asyncio
async def test_normalizer_preserves_segment_metadata() -> None:
    """Verify arbitrary segment metadata survives."""

    normalizer = KnowledgeContentNormalizer()

    segments = [
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="Important content.",
            order=0,
            metadata={
                "section": "Introduction",
                "confidence": 0.95,
                "custom_field": "custom-value",
            },
        ),
    ]

    canonical = create_canonical_content(
        text="Important content.",
        segments=segments,
    )

    result = await normalizer.normalize(
        canonical,
    )

    metadata = result.elements[0].metadata

    assert metadata["section"] == "Introduction"
    assert metadata["confidence"] == 0.95
    assert metadata["custom_field"] == "custom-value"
    assert metadata["canonical_order"] == 0


# ============================================================
# Determinism and Hashing
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_is_deterministic() -> None:
    """
    Verify identical CanonicalContent produces identical
    normalized results.
    """

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text="Deterministic content.",
    )

    first = await normalizer.normalize(
        canonical,
    )

    second = await normalizer.normalize(
        canonical,
    )

    assert first.model_dump() == second.model_dump()


@pytest.mark.asyncio
async def test_normalizer_generates_sha256_content_hash() -> None:
    """Verify the content hash has SHA-256 length."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text="Hash this content.",
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert isinstance(
        result.content_hash,
        str,
    )

    assert len(result.content_hash) == 64

    assert all(
        character in "0123456789abcdef"
        for character in result.content_hash
    )


@pytest.mark.asyncio
async def test_content_hash_changes_when_text_changes() -> None:
    """Verify different content produces different hashes."""

    normalizer = KnowledgeContentNormalizer()

    first = await normalizer.normalize(
        create_canonical_content(
            text="Original content.",
        )
    )

    second = await normalizer.normalize(
        create_canonical_content(
            text="Changed content.",
        )
    )

    assert first.content_hash != second.content_hash


@pytest.mark.asyncio
async def test_content_hash_changes_when_structure_changes() -> None:
    """Verify structural changes affect the content hash."""

    normalizer = KnowledgeContentNormalizer()

    first = await normalizer.normalize(
        create_canonical_content(
            text="Same content.",
            segments=[
                ContentBlock(
                    block_type=ContentBlockType.PARAGRAPH,
                    content="Same content.",
                    order=0,
                ),
            ],
        )
    )

    second = await normalizer.normalize(
        create_canonical_content(
            text="Same content.",
            segments=[
                ContentBlock(
                    block_type=ContentBlockType.HEADING,
                    content="Same content.",
                    order=0,
                ),
            ],
        )
    )

    assert first.content_hash != second.content_hash


@pytest.mark.asyncio
async def test_content_hash_changes_when_page_changes() -> None:
    """Verify page provenance affects the content hash."""

    normalizer = KnowledgeContentNormalizer()

    first = await normalizer.normalize(
        create_canonical_content(
            text="Page content.",
            segments=[
                ContentBlock(
                    block_type=ContentBlockType.PARAGRAPH,
                    content="Page content.",
                    order=0,
                    page_number=1,
                ),
            ],
        )
    )

    second = await normalizer.normalize(
        create_canonical_content(
            text="Page content.",
            segments=[
                ContentBlock(
                    block_type=ContentBlockType.PARAGRAPH,
                    content="Page content.",
                    order=0,
                    page_number=2,
                ),
            ],
        )
    )

    assert first.content_hash != second.content_hash


@pytest.mark.asyncio
async def test_content_hash_changes_when_timestamp_changes() -> None:
    """Verify timestamp provenance affects the content hash."""

    normalizer = KnowledgeContentNormalizer()

    first = await normalizer.normalize(
        create_canonical_content(
            text="Transcript content.",
            segments=[
                ContentBlock(
                    block_type=ContentBlockType.TRANSCRIPT,
                    content="Transcript content.",
                    order=0,
                    start_time=0.0,
                    end_time=10.0,
                ),
            ],
        )
    )

    second = await normalizer.normalize(
        create_canonical_content(
            text="Transcript content.",
            segments=[
                ContentBlock(
                    block_type=ContentBlockType.TRANSCRIPT,
                    content="Transcript content.",
                    order=0,
                    start_time=5.0,
                    end_time=15.0,
                ),
            ],
        )
    )

    assert first.content_hash != second.content_hash


# ============================================================
# Unicode / Multilingual Content
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_preserves_unicode() -> None:
    """Verify Unicode content is preserved."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        text=(
            "Artificial Intelligence — 人工智能 — "
            "कृत्रिम बुद्धिमत्ता — 🤖"
        ),
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.text == (
        "Artificial Intelligence — 人工智能 — "
        "कृत्रिम बुद्धिमत्ता — 🤖"
    )


@pytest.mark.asyncio
async def test_normalizer_normalizes_language_identifier() -> None:
    """Verify language identifiers are normalized."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        language=" EN ",
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.language == "en"


@pytest.mark.asyncio
async def test_normalizer_handles_missing_language() -> None:
    """Verify missing language remains None."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        language=None,
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert result.language is None


# ============================================================
# Normalization Metadata
# ============================================================


@pytest.mark.asyncio
async def test_normalizer_adds_normalization_metadata() -> None:
    """Verify normalization metadata is recorded."""

    normalizer = KnowledgeContentNormalizer()

    canonical = create_canonical_content(
        segments=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="One.",
                order=0,
            ),
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Two.",
                order=1,
            ),
        ],
    )

    result = await normalizer.normalize(
        canonical,
    )

    assert "knowledge_normalization" in result.metadata

    normalization_metadata = (
        result.metadata[
            "knowledge_normalization"
        ]
    )

    assert normalization_metadata["version"] == "1.0"
    assert normalization_metadata["element_count"] == 2