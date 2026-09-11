from __future__ import annotations

from io import BytesIO

import pytest
from pypdf import PdfWriter
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from app.ingestion.parsers.pdf import PDFProcessor
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


class FakePDFResolver:
    """Test resolver that returns predefined PDF bytes."""

    def __init__(self, content: bytes) -> None:
        self.content = content

    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        return self.content


class FailingPDFResolver:
    """Test resolver that simulates a resolution failure."""

    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        raise RuntimeError("Resolver failure")


def create_text_pdf() -> bytes:
    """
    Create a deterministic two-page PDF containing
    extractable text for processor tests.
    """

    output = BytesIO()

    pdf = canvas.Canvas(
        output,
        pagesize=A4,
    )

    # Page 1
    pdf.drawString(
        72,
        780,
        "Threat intelligence report - Page One",
    )

    pdf.showPage()

    # Page 2
    pdf.drawString(
        72,
        780,
        "Incident response advisory - Page Two",
    )

    pdf.save()

    return output.getvalue()


def create_blank_pdf() -> bytes:
    """Create a valid PDF containing no extractable text."""

    writer = PdfWriter()

    writer.add_blank_page(
        width=595,
        height=842,
    )

    output = BytesIO()
    writer.write(output)

    return output.getvalue()


def create_encrypted_pdf() -> bytes:
    """Create a valid encrypted PDF."""

    writer = PdfWriter()

    writer.add_blank_page(
        width=595,
        height=842,
    )

    writer.encrypt("test-password")

    output = BytesIO()
    writer.write(output)

    return output.getvalue()


@pytest.mark.asyncio
async def test_pdf_processor_rejects_wrong_input_type():
    resolver = FakePDFResolver(b"dummy")
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.TXT,
        filename="test.txt",
        mime_type="text/plain",
        content=b"hello",
    )

    with pytest.raises(
        ValueError,
        match="only process PDF",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_pdf_processor_rejects_empty_pdf_bytes():
    resolver = FakePDFResolver(b"")
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="empty.pdf",
        mime_type="application/pdf",
        content=b"",
    )

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_pdf_processor_rejects_corrupted_pdf():
    corrupted_pdf = (
        b"This is not a valid PDF document."
    )

    resolver = FakePDFResolver(corrupted_pdf)
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="broken.pdf",
        mime_type="application/pdf",
        content=corrupted_pdf,
    )

    with pytest.raises(
        ValueError,
        match="Unable to read PDF|Failed to parse PDF",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_pdf_processor_detects_empty_or_scanned_pdf():
    pdf_bytes = create_blank_pdf()

    resolver = FakePDFResolver(pdf_bytes)
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="scanned.pdf",
        mime_type="application/pdf",
        content=pdf_bytes,
    )

    with pytest.raises(
        ValueError,
        match="no extractable text",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_pdf_processor_rejects_encrypted_pdf():
    pdf_bytes = create_encrypted_pdf()

    resolver = FakePDFResolver(pdf_bytes)
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="encrypted.pdf",
        mime_type="application/pdf",
        content=pdf_bytes,
    )

    with pytest.raises(
        ValueError,
        match="Encrypted PDF",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_pdf_processor_propagates_resolver_failure():
    resolver = FailingPDFResolver()
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.PDF,
        filename="test.pdf",
        mime_type="application/pdf",
        content=b"placeholder",
    )

    with pytest.raises(
        RuntimeError,
        match="Resolver failure",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_pdf_processor_extracts_text_with_page_provenance():
    pdf_bytes = create_text_pdf()

    resolver = FakePDFResolver(pdf_bytes)
    processor = PDFProcessor(resolver)

    request = IngestionRequest(
        input_type=InputType.PDF,
        title="Security Report",
        filename="security-report.pdf",
        mime_type="application/pdf",
        content=pdf_bytes,
        metadata={
            "source": "test",
        },
    )

    result = await processor.process(request)

    # ---------------------------------------------------------
    # Source metadata
    # ---------------------------------------------------------

    assert result.title == "Security Report"

    assert (
        result.source.source_type
        == InputType.PDF
    )

    assert (
        result.source.filename
        == "security-report.pdf"
    )

    assert (
        result.source.mime_type
        == "application/pdf"
    )

    # ---------------------------------------------------------
    # Complete extracted text
    # ---------------------------------------------------------

    assert (
        "Threat intelligence report - Page One"
        in result.text
    )

    assert (
        "Incident response advisory - Page Two"
        in result.text
    )

    # Both pages should be represented.
    assert result.text.count("\n\n") >= 1

    # ---------------------------------------------------------
    # Content blocks
    # ---------------------------------------------------------

    assert len(result.blocks) == 2

    first_block = result.blocks[0]
    second_block = result.blocks[1]

    assert (
        first_block.block_type
        == ContentBlockType.PARAGRAPH
    )

    assert (
        second_block.block_type
        == ContentBlockType.PARAGRAPH
    )

    # ---------------------------------------------------------
    # Page provenance
    # ---------------------------------------------------------

    assert first_block.page_number == 1
    assert second_block.page_number == 2

    # ---------------------------------------------------------
    # Deterministic block ordering
    # ---------------------------------------------------------

    assert first_block.order == 0
    assert second_block.order == 1

    # ---------------------------------------------------------
    # Processor metadata
    # ---------------------------------------------------------

    assert (
        first_block.metadata["processor"]
        == "pdf"
    )

    assert (
        second_block.metadata["processor"]
        == "pdf"
    )

    assert (
        result.metadata["processor"]
        == "pdf"
    )

    # ---------------------------------------------------------
    # PDF-level metadata
    # ---------------------------------------------------------

    assert result.metadata["page_count"] == 2

    assert (
        result.metadata["extracted_page_count"]
        == 2
    )

    # Original request metadata must be preserved.
    assert result.metadata["source"] == "test"


def test_pdf_processor_supported_type():
    resolver = FakePDFResolver(b"")
    processor = PDFProcessor(resolver)

    assert (
        processor.supported_types
        == frozenset({InputType.PDF})
    )