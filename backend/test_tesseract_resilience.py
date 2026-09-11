from __future__ import annotations

import asyncio

from PIL import Image
from io import BytesIO

from app.ingestion.ocr.schemas import OCRRequest, OCRStatus
from app.ingestion.ocr.tesseract import TesseractOCRProvider


def create_blank_image() -> bytes:
    """Create a blank image containing no recognizable text."""

    image = Image.new("RGB", (800, 400), "white")

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


def test_blank_image_returns_no_text() -> None:
    provider = TesseractOCRProvider()

    request = OCRRequest(
        image=create_blank_image(),
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.NO_TEXT
    assert result.text == ""
    assert result.blocks == []
    assert result.provider == "tesseract"


def test_corrupt_image_returns_failed() -> None:
    provider = TesseractOCRProvider()

    request = OCRRequest(
        image=b"this-is-not-a-valid-image",
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.FAILED
    assert result.text == ""
    assert result.blocks == []
    assert result.provider == "tesseract"

    assert result.metadata["error_type"] == "UnidentifiedImageError"


def test_missing_executable_returns_failed() -> None:
    provider = TesseractOCRProvider(
        executable_path=r"C:\definitely-not-a-real\tesseract.exe"
    )

    request = OCRRequest(
        image=create_blank_image(),
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.FAILED
    assert result.provider == "tesseract"
    assert result.metadata["error_type"] == "FileNotFoundError"


def test_timeout_returns_failed() -> None:
    provider = TesseractOCRProvider(
        timeout_seconds=0.000001
    )

    request = OCRRequest(
        image=create_blank_image(),
        language="eng",
    )

    result = asyncio.run(
        provider.recognize(request)
    )

    assert result.status == OCRStatus.FAILED
    assert result.provider == "tesseract"
    assert result.metadata["error"] == "OCR operation timed out"