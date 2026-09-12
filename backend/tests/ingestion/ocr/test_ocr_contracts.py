from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.ocr.schemas import (
    OCRBoundingBox,
    OCRRequest,
    OCRResult,
    OCRStatus,
    OCRTextBlock,
)


def test_ocr_status_values() -> None:
    assert OCRStatus.NOT_REQUESTED.value == (
        "not_requested"
    )

    assert OCRStatus.PROCESSING.value == (
        "processing"
    )

    assert OCRStatus.COMPLETED.value == (
        "completed"
    )

    assert OCRStatus.NO_TEXT.value == "no_text"

    assert OCRStatus.FAILED.value == "failed"


def test_ocr_request_accepts_image_bytes() -> None:
    request = OCRRequest(
        image=b"test-image",
    )

    assert request.image == b"test-image"
    assert request.language is None
    assert request.metadata == {}


def test_ocr_request_accepts_language_and_metadata() -> None:
    request = OCRRequest(
        image=b"test-image",
        language="eng",
        metadata={
            "source": "image_processor",
        },
    )

    assert request.language == "eng"

    assert request.metadata == {
        "source": "image_processor",
    }


def test_ocr_request_rejects_non_bytes_image() -> None:
    with pytest.raises(ValidationError):
        OCRRequest(
            image="not-bytes",
        )


def test_ocr_bounding_box() -> None:
    box = OCRBoundingBox(
        x=10.0,
        y=20.0,
        width=100.0,
        height=40.0,
    )

    assert box.x == 10.0
    assert box.y == 20.0
    assert box.width == 100.0
    assert box.height == 40.0


def test_ocr_text_block() -> None:
    block = OCRTextBlock(
        text="Hello world",
        confidence=0.95,
    )

    assert block.text == "Hello world"
    assert block.confidence == 0.95
    assert block.bounding_box is None
    assert block.metadata == {}


def test_ocr_text_block_with_bounding_box() -> None:
    block = OCRTextBlock(
        text="Hello world",
        confidence=0.95,
        bounding_box=OCRBoundingBox(
            x=10,
            y=20,
            width=100,
            height=30,
        ),
    )

    assert block.bounding_box is not None
    assert block.bounding_box.x == 10
    assert block.bounding_box.y == 20
    assert block.bounding_box.width == 100
    assert block.bounding_box.height == 30


def test_ocr_result_completed() -> None:
    result = OCRResult(
        status=OCRStatus.COMPLETED,
        text="Hello world",
        confidence=0.95,
        language="eng",
        provider="test",
    )

    assert result.status == OCRStatus.COMPLETED
    assert result.text == "Hello world"
    assert result.confidence == 0.95
    assert result.language == "eng"
    assert result.provider == "test"
    assert result.blocks == []
    assert result.metadata == {}


def test_ocr_result_with_blocks() -> None:
    result = OCRResult(
        status=OCRStatus.COMPLETED,
        text="Hello world",
        blocks=[
            OCRTextBlock(
                text="Hello world",
                confidence=0.95,
            ),
        ],
        confidence=0.95,
        language="eng",
        provider="test",
    )

    assert len(result.blocks) == 1

    assert result.blocks[0].text == (
        "Hello world"
    )


def test_ocr_result_no_text() -> None:
    result = OCRResult(
        status=OCRStatus.NO_TEXT,
        provider="test",
    )

    assert result.status == OCRStatus.NO_TEXT
    assert result.text == ""
    assert result.blocks == []
    assert result.confidence is None


def test_ocr_result_failed() -> None:
    result = OCRResult(
        status=OCRStatus.FAILED,
        provider="test",
        metadata={
            "error": "OCR engine unavailable",
        },
    )

    assert result.status == OCRStatus.FAILED

    assert result.metadata["error"] == (
        "OCR engine unavailable"
    )


def test_ocr_result_preserves_provider_metadata() -> None:
    result = OCRResult(
        status=OCRStatus.COMPLETED,
        text="Security advisory",
        provider="tesseract",
        metadata={
            "engine_version": "5.x",
            "processing_time_ms": 120,
        },
    )

    assert result.metadata["engine_version"] == (
        "5.x"
    )

    assert result.metadata["processing_time_ms"] == 120