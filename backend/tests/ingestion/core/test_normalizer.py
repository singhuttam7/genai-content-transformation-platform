import pytest

from app.ingestion.normalization import DefaultContentNormalizer
from app.ingestion.parsers import TextProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)


@pytest.mark.asyncio
async def test_default_content_normalizer() -> None:
    processor = TextProcessor()
    normalizer = DefaultContentNormalizer()

    request = IngestionRequest(
        input_type=InputType.TEXT,
        title="AI Content",
        content=(
            "Artificial intelligence is transforming communication.\n\n\n"
            "Organizations are using AI to generate content."
        ),
        metadata={
            "test": True,
        },
    )

    extracted = await processor.process(request)
    canonical = await normalizer.normalize(extracted)

    # ---------------------------------------------------------
    # Main text normalization
    # ---------------------------------------------------------

    assert canonical.text == (
        "Artificial intelligence is transforming communication.\n\n"
        "Organizations are using AI to generate content."
    )

    # ---------------------------------------------------------
    # Structured canonical segments
    # ---------------------------------------------------------

    assert len(canonical.segments) == 1

    assert (
        canonical.segments[0].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert canonical.segments[0].order == 0

    # ---------------------------------------------------------
    # Semantic fields
    # ---------------------------------------------------------

    assert canonical.entities == []
    assert canonical.topics == []
    assert canonical.claims == []
    assert canonical.keywords == []

    # ---------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------

    assert (
        canonical.provenance["ingestion_normalizer"]
        == "default"
    )

    assert (
        canonical.provenance["normalization_version"]
        == "1.0"
    )

    # ---------------------------------------------------------
    # Metadata preservation
    # ---------------------------------------------------------

    assert canonical.metadata["test"] is True

    # ---------------------------------------------------------
    # Deterministic whitespace normalization
    # ---------------------------------------------------------

    assert (
        DefaultContentNormalizer._normalize_text(
            "  Hello   world  \n\n\n\n  AI  "
        )
        == "Hello   world\n\n  AI"
    )

    # ---------------------------------------------------------
    # Block ordering
    # ---------------------------------------------------------

    extracted.blocks = [
        ContentBlock(
            block_type=ContentBlockType.PARAGRAPH,
            content="Second block",
            order=5,
        ),
        ContentBlock(
            block_type=ContentBlockType.HEADING,
            content="First block",
            order=1,
        ),
    ]

    canonical = await normalizer.normalize(extracted)

    assert canonical.segments[0].content == "First block"
    assert canonical.segments[0].order == 0

    assert canonical.segments[1].content == "Second block"
    assert canonical.segments[1].order == 1


@pytest.mark.asyncio
async def test_default_content_normalizer_preserves_video_blocks() -> None:
    """
    Video blocks represent the original video/media object and therefore
    must not be treated as normal text blocks during canonicalization.

    The normalizer should:
    - preserve the VIDEO block
    - preserve empty video content
    - preserve video metadata
    - preserve transcript blocks
    - preserve transcript timestamps
    - normalize ordering deterministically
    """

    normalizer = DefaultContentNormalizer()

    source = SourceReference(
        source_id="00000000-0000-0000-0000-000000000001",
        source_type=InputType.VIDEO,
        title="AI Threat Intelligence Video",
        filename="threat-intelligence.mp4",
        mime_type="video/mp4",
        content_hash="abc123",
        storage_uri=None,
    )

    extracted = ExtractedContent(
        source=source,
        title="AI Threat Intelligence Video",
        language="en",
        text="AI is transforming cybersecurity.",
        blocks=[
            ContentBlock(
                block_type=ContentBlockType.VIDEO,
                content="",
                order=5,
                metadata={
                    "media_type": "video",
                    "format_name": "mp4",
                    "duration_seconds": 8.153,
                    "size_bytes": 123456,
                },
            ),
            ContentBlock(
                block_type=ContentBlockType.TRANSCRIPT,
                content="AI is transforming cybersecurity.",
                order=10,
                start_time=0.52,
                end_time=2.87,
                metadata={
                    "provider": "faster-whisper",
                    "language": "en",
                },
            ),
        ],
        metadata={
            "video": {
                "format_name": "mp4",
                "duration_seconds": 8.153,
            }
        },
    )

    canonical = await normalizer.normalize(extracted)

    # ---------------------------------------------------------
    # Segment count
    # ---------------------------------------------------------

    assert len(canonical.segments) == 2

    # ---------------------------------------------------------
    # VIDEO block preservation
    # ---------------------------------------------------------

    video_segment = canonical.segments[0]

    assert (
        video_segment.block_type
        == ContentBlockType.VIDEO
    )

    # Video blocks intentionally contain no textual content.
    assert video_segment.content == ""

    assert (
        video_segment.metadata["media_type"]
        == "video"
    )

    assert (
        video_segment.metadata["format_name"]
        == "mp4"
    )

    assert (
        video_segment.metadata["duration_seconds"]
        == 8.153
    )

    assert (
        video_segment.metadata["size_bytes"]
        == 123456
    )

    # ---------------------------------------------------------
    # Transcript preservation
    # ---------------------------------------------------------

    transcript_segment = canonical.segments[1]

    assert (
        transcript_segment.block_type
        == ContentBlockType.TRANSCRIPT
    )

    assert (
        transcript_segment.content
        == "AI is transforming cybersecurity."
    )

    # ---------------------------------------------------------
    # Timestamp preservation
    # ---------------------------------------------------------

    assert transcript_segment.start_time == 0.52
    assert transcript_segment.end_time == 2.87

    # ---------------------------------------------------------
    # Deterministic ordering
    # ---------------------------------------------------------

    assert video_segment.order == 0
    assert transcript_segment.order == 1

    # ---------------------------------------------------------
    # Canonical text
    # ---------------------------------------------------------

    assert (
        canonical.text
        == "AI is transforming cybersecurity."
    )

    # ---------------------------------------------------------
    # Source metadata preservation
    # ---------------------------------------------------------

    assert (
        canonical.metadata["video"]["format_name"]
        == "mp4"
    )

    assert (
        canonical.metadata["video"]["duration_seconds"]
        == 8.153
    )