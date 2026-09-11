from __future__ import annotations

from app.core.config import settings
from app.ingestion.ocr.provider import OCRProvider
from app.ingestion.ocr.tesseract import TesseractOCRProvider


def create_ocr_provider() -> OCRProvider | None:
    """
    Create the configured OCR provider.

    Returns None when OCR is disabled.
    """

    if not settings.ocr_enabled:
        return None

    provider_name = settings.ocr_provider.strip().lower()

    if provider_name == "tesseract":
        return TesseractOCRProvider(
            executable_path=settings.tesseract_executable_path,
            timeout_seconds=settings.ocr_timeout_seconds,
            default_language=settings.ocr_default_language,
        )

    raise ValueError(
        f"Unsupported OCR provider: {settings.ocr_provider}"
    )