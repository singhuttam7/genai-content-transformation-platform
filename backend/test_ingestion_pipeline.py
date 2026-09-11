import asyncio

from app.ingestion.factory import create_ingestion_pipeline
from app.ingestion.schemas import IngestionRequest, InputType


async def test_pipeline() -> None:
    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.TEXT,
        title="AI Communication",
        filename="article.txt",
        mime_type="text/plain",
        content=(
            "Artificial intelligence is transforming communication.\n\n\n"
            "Organizations are using AI to create content faster."
        ),
        metadata={
            "source": "integration-test",
        },
    )

    result = await pipeline.run(request)

    # ---------------------------------------------------------
    # Pipeline creation
    # ---------------------------------------------------------
    assert type(pipeline).__name__ == "IngestionPipeline"

    # ---------------------------------------------------------
    # Input detection
    # ---------------------------------------------------------
    assert result.source.source_type == InputType.TEXT

    # ---------------------------------------------------------
    # Text processing
    # ---------------------------------------------------------
    expected_text = (
        "Artificial intelligence is transforming communication.\n\n"
        "Organizations are using AI to create content faster."
    )

    assert result.text == expected_text

    # ---------------------------------------------------------
    # Canonical content
    # ---------------------------------------------------------
    assert result.title == "AI Communication"

    assert len(result.segments) == 1

    assert result.segments[0].content == result.text
    assert result.segments[0].order == 0

    # ---------------------------------------------------------
    # Semantic placeholders
    # ---------------------------------------------------------
    # These will later be populated by the AI
    # Content Understanding subsystem.
    assert result.entities == []
    assert result.topics == []
    assert result.claims == []
    assert result.keywords == []

    # ---------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------
    assert result.metadata["source"] == "integration-test"

    # ---------------------------------------------------------
    # Provenance
    # ---------------------------------------------------------
    assert result.provenance["ingestion_normalizer"] == "default"
    assert result.provenance["normalization_version"] == "1.0"

    print("Pipeline creation: OK")
    print("Input detection: OK")
    print("Processor routing: OK")
    print("Text processing: OK")
    print("Content normalization: OK")
    print("Canonical content: OK")
    print("Metadata preservation: OK")
    print("Provenance generation: OK")
    print("Real ingestion pipeline: ALL TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(test_pipeline())