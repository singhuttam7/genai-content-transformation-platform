from __future__ import annotations

from io import BytesIO

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.ingestion.factory import create_ingestion_pipeline
from app.ingestion.schemas import (
    ContentBlockType,
    InputType,
    IngestionRequest,
)


def create_text_pdf() -> bytes:
    """Create a deterministic two-page text PDF."""

    output = BytesIO()

    pdf = canvas.Canvas(
        output,
        pagesize=A4,
    )

    pdf.drawString(
        72,
        780,
        "Threat intelligence report - Page One",
    )

    pdf.showPage()

    pdf.drawString(
        72,
        780,
        "Incident response advisory - Page Two",
    )

    pdf.save()

    return output.getvalue()


@pytest.mark.asyncio
async def test_pdf_complete_ingestion_pipeline():
    pdf_bytes = create_text_pdf()

    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.PDF,
        title="Security Report",
        filename="security-report.pdf",
        mime_type="application/pdf",
        content=pdf_bytes,
        metadata={
            "source": "integration-test",
        },
    )

    result = await pipeline.run(request)

    # ---------------------------------------------------------
    # Canonical content
    # ---------------------------------------------------------

    assert result.title == "Security Report"

    assert result.source.source_type == InputType.PDF

    assert result.source.filename == "security-report.pdf"

    # ---------------------------------------------------------
    # Full text
    # ---------------------------------------------------------

    assert (
        "Threat intelligence report - Page One"
        in result.text
    )

    assert (
        "Incident response advisory - Page Two"
        in result.text
    )

    # ---------------------------------------------------------
    # Normalized structured segments
    # ---------------------------------------------------------

    assert len(result.segments) == 2

    assert (
        "Threat intelligence report - Page One"
        in result.segments[0].content
    )

    assert (
        result.segments[0].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert result.segments[0].order == 0

    assert result.segments[0].page_number == 1

    assert (
        "Incident response advisory - Page Two"
        in result.segments[1].content
    )

    assert (
        result.segments[1].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert result.segments[1].order == 1

    assert result.segments[1].page_number == 2

    # ---------------------------------------------------------
    # Semantic fields should remain empty at this stage.
    #
    # Semantic enrichment belongs to the upcoming
    # Content Understanding subsystem.
    # ---------------------------------------------------------

    assert result.entities == []
    assert result.topics == []
    assert result.claims == []
    assert result.keywords == []

    # ---------------------------------------------------------
    # Normalization provenance
    # ---------------------------------------------------------

    assert (
        result.provenance["ingestion_normalizer"]
        == "default"
    )

    assert (
        result.provenance["normalization_version"]
        == "1.0"
    )

    # ---------------------------------------------------------
    # Original metadata preservation
    # ---------------------------------------------------------

    assert (
        result.metadata["source"]
        == "integration-test"
    )

    assert (
        result.metadata["processor"]
        == "pdf"
    )

    assert result.metadata["page_count"] == 2

    assert (
        result.metadata["extracted_page_count"]
        == 2
    )


@pytest.mark.asyncio
async def test_pdf_pipeline_detects_pdf_from_metadata():
    pdf_bytes = create_text_pdf()

    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="report.pdf",
        mime_type="application/pdf",
        content=pdf_bytes,
    )

    result = await pipeline.run(request)

    assert result.source.source_type == InputType.PDF

    assert "Page One" in result.text

    assert "Page Two" in result.text