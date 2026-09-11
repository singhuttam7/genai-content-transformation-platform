import pytest

from app.ingestion.normalization import DefaultContentNormalizer
from app.ingestion.parsers import TextProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    IngestionRequest,
    InputType,
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