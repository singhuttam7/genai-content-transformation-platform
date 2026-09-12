from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image, ImageDraw, ImageFont

from app.ingestion.ocr.schemas import OCRRequest, OCRStatus
from app.ingestion.ocr.tesseract import TesseractOCRProvider


def create_test_image() -> bytes:
    """Create a deterministic image containing known text."""

    image = Image.new("RGB", (1200, 300), "white")
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype(
            "arial.ttf",
            48,
        )
    except OSError:
        font = ImageFont.load_default()

    draw.text(
        (50, 60),
        "Gen AI Content Transformation",
        fill="black",
        font=font,
    )

    draw.text(
        (50, 140),
        "Tesseract OCR Test",
        fill="black",
        font=font,
    )

    buffer = BytesIO()
    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


def test_tesseract_provider_discovers_engine() -> None:
    provider = TesseractOCRProvider()

    assert provider.executable_path is not None
    assert provider.executable_path.name.lower() == "tesseract.exe"


def test_tesseract_provider_recognizes_text() -> None:
    provider = TesseractOCRProvider()

    request = OCRRequest(
        image=create_test_image(),
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.COMPLETED
    assert result.provider == "tesseract"
    assert result.language == "eng"

    assert result.text

    normalized_text = " ".join(
        result.text.lower().split()
    )

    assert "content transformation" in normalized_text
    assert "tesseract" in normalized_text
    assert "ocr" in normalized_text


def test_tesseract_provider_returns_structured_ocr_blocks() -> None:
    provider = TesseractOCRProvider()

    request = OCRRequest(
        image=create_test_image(),
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.COMPLETED

    # At least one valid OCR region must be detected.
    assert result.blocks

    # Overall confidence must be available.
    assert result.confidence is not None
    assert 0.0 <= result.confidence <= 100.0

    for block in result.blocks:
        # Text must be present.
        assert block.text

        # Each OCR region must have confidence.
        assert block.confidence is not None
        assert 0.0 <= block.confidence <= 100.0

        # Each OCR region must have a bounding box.
        assert block.bounding_box is not None

        assert block.bounding_box.width > 0
        assert block.bounding_box.height > 0

        # Provider metadata must identify the OCR engine.
        assert block.metadata["engine"] == "tesseract"
        assert block.metadata["language"] == "eng"


def test_tesseract_provider_returns_block_count_metadata() -> None:
    provider = TesseractOCRProvider()

    request = OCRRequest(
        image=create_test_image(),
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.COMPLETED
    assert result.blocks

    assert result.metadata["engine"] == "tesseract"
    assert result.metadata["language"] == "eng"
    assert result.metadata["block_count"] == len(
        result.blocks
    )