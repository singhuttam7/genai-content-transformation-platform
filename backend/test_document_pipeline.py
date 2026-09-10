import asyncio

from app.ingestion.factory import create_ingestion_pipeline
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


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

    assert result.segments[0] == (
        "Artificial intelligence is transforming communication."
    )

    assert result.segments[1] == (
        "This content passed through the complete ingestion pipeline."
    )

    assert result.source.source_type == InputType.TXT
    assert len(result.provenance) > 0
    assert result.metadata["test"] is True

    print("TXT complete pipeline: OK")


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

    assert result.segments[0] == "Security Advisory"
    assert result.segments[1] == (
        "This advisory contains important guidance."
    )
    assert result.segments[2] == "Recommended Actions"
    assert result.segments[3] == (
        "Update affected systems immediately."
    )

    assert result.source.source_type == InputType.MARKDOWN
    assert result.metadata["format"] == "markdown"

    print("Markdown complete pipeline: OK")


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

    # CanonicalContent contains semantic-ready strings,
    # while structural block information remains available
    # through the extraction stage.
    assert len(result.segments) == 4

    assert result.segments[0] == "Main Heading"
    assert result.segments[1] == "First paragraph."
    assert result.segments[2] == "Sub Heading"
    assert result.segments[3] == "Second paragraph."

    print("Pipeline structure preservation: OK")


async def main() -> None:
    print()
    print("==============================================")
    print("DOCUMENT PIPELINE INTEGRATION TESTS")
    print("==============================================")
    print()

    await test_txt_pipeline()
    await test_markdown_pipeline()
    await test_pipeline_preserves_structure()

    print()
    print("==============================================")
    print("DOCUMENT PIPELINE: ALL TESTS PASSED")
    print("==============================================")


if __name__ == "__main__":
    asyncio.run(main())