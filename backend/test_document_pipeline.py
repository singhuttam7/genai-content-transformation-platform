from __future__ import annotations

import pytest

from app.ingestion.factory import create_ingestion_pipeline
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


@pytest.mark.asyncio
async def test_txt_pipeline() -> None:
    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.TXT,
        filename="pipeline-test.txt",
        mime_type="text/plain",
        content=(
            b"Artificial intelligence is transforming communication.\n\n"
            b"This content passed through the complete ingestion pipeline."
        ),
        metadata={
            "test": True,
            "format": "txt",
        },
    )

    result = await pipeline.run(request)

    assert result.text == (
        "Artificial intelligence is transforming communication.\n\n"
        "This content passed through the complete ingestion pipeline."
    )

    assert len(result.segments) == 2

    # Canonical segments preserve structural information.
    assert (
        result.segments[0].content
        == "Artificial intelligence is transforming communication."
    )

    assert (
        result.segments[0].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert result.segments[0].order == 0

    assert (
        result.segments[1].content
        == "This content passed through the complete ingestion pipeline."
    )

    assert (
        result.segments[1].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert result.segments[1].order == 1

    assert result.source.source_type == InputType.TXT

    assert len(result.provenance) > 0

    assert result.metadata["test"] is True


@pytest.mark.asyncio
async def test_markdown_pipeline() -> None:
    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.MARKDOWN,
        filename="pipeline-test.md",
        mime_type="text/markdown",
        content=(
            b"# Security Advisory\n\n"
            b"This advisory contains important guidance.\n\n"
            b"## Recommended Actions\n\n"
            b"Update affected systems immediately."
        ),
        metadata={
            "test": True,
            "format": "markdown",
        },
    )

    result = await pipeline.run(request)

    assert result.text == (
        "# Security Advisory\n\n"
        "This advisory contains important guidance.\n\n"
        "## Recommended Actions\n\n"
        "Update affected systems immediately."
    )

    assert len(result.segments) == 4

    assert result.segments[0].content == "Security Advisory"
    assert (
        result.segments[0].block_type
        == ContentBlockType.HEADING
    )
    assert result.segments[0].order == 0

    assert (
        result.segments[1].content
        == "This advisory contains important guidance."
    )
    assert (
        result.segments[1].block_type
        == ContentBlockType.PARAGRAPH
    )
    assert result.segments[1].order == 1

    assert result.segments[2].content == "Recommended Actions"
    assert (
        result.segments[2].block_type
        == ContentBlockType.HEADING
    )
    assert result.segments[2].order == 2

    assert (
        result.segments[3].content
        == "Update affected systems immediately."
    )
    assert (
        result.segments[3].block_type
        == ContentBlockType.PARAGRAPH
    )
    assert result.segments[3].order == 3

    assert result.source.source_type == InputType.MARKDOWN

    assert result.metadata["format"] == "markdown"


@pytest.mark.asyncio
async def test_pipeline_preserves_structure() -> None:
    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.MARKDOWN,
        filename="structure-test.md",
        mime_type="text/markdown",
        content=(
            b"# Main Heading\n\n"
            b"First paragraph.\n\n"
            b"## Sub Heading\n\n"
            b"Second paragraph."
        ),
    )

    result = await pipeline.run(request)

    # CanonicalContent intentionally preserves structural
    # ContentBlock information.
    assert len(result.segments) == 4

    assert result.segments[0].content == "Main Heading"
    assert (
        result.segments[0].block_type
        == ContentBlockType.HEADING
    )

    assert result.segments[1].content == "First paragraph."
    assert (
        result.segments[1].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert result.segments[2].content == "Sub Heading"
    assert (
        result.segments[2].block_type
        == ContentBlockType.HEADING
    )

    assert result.segments[3].content == "Second paragraph."
    assert (
        result.segments[3].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert [
        segment.order
        for segment in result.segments
    ] == [0, 1, 2, 3]