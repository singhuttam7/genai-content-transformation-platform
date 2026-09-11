from __future__ import annotations

from typing import Protocol

from app.ingestion.ocr.schemas import (
    OCRRequest,
    OCRResult,
)


class OCRProvider(Protocol):
    """
    Provider-neutral contract for OCR engines.

    Implementations may use local OCR engines, GPU-based
    models, or remote OCR APIs.
    """

    name: str

    async def recognize(
        self,
        request: OCRRequest,
    ) -> OCRResult:
        """
        Perform OCR on the supplied image.

        Implementations must return a provider-neutral
        OCRResult.
        """
        ...