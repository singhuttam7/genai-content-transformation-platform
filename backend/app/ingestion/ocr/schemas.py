from __future__ import annotations

from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, StrictBytes

class OCRStatus(StrEnum):
    """Lifecycle states for an OCR operation."""

    NOT_REQUESTED = "not_requested"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NO_TEXT = "no_text"
    FAILED = "failed"


class OCRBoundingBox(BaseModel):
    """
    Bounding box for OCR-detected text.

    Coordinates are normalized to the image coordinate
    system used by the OCR provider.
    """

    x: float
    y: float
    width: float
    height: float


class OCRTextBlock(BaseModel):
    """
    A single OCR-detected text region.
    """

    text: str
    confidence: float | None = None
    bounding_box: OCRBoundingBox | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class OCRRequest(BaseModel):
    """
    Provider-neutral OCR request.

    The image itself is supplied as bytes at the internal
    ingestion boundary.
    """

    image: StrictBytes
    language: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class OCRResult(BaseModel):
    """
    Provider-neutral OCR result.

    Providers such as Tesseract, PaddleOCR, or a future
    cloud OCR service must normalize their output into
    this representation.
    """

    status: OCRStatus
    text: str = ""
    blocks: list[OCRTextBlock] = Field(
        default_factory=list
    )
    confidence: float | None = None
    language: str | None = None
    provider: str
    metadata: dict[str, Any] = Field(
        default_factory=dict
    )