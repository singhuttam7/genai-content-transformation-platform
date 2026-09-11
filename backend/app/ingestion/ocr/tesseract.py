from __future__ import annotations

import asyncio
import shutil
from io import BytesIO
from pathlib import Path

import pytesseract
from PIL import Image
from pytesseract import Output

from app.ingestion.ocr.provider import OCRProvider
from app.ingestion.ocr.schemas import (
    OCRBoundingBox,
    OCRRequest,
    OCRResult,
    OCRStatus,
    OCRTextBlock,
)


class TesseractOCRProvider(OCRProvider):
    """
    OCR provider backed by the Tesseract OCR engine.

    Responsibilities:
    - Discover the Tesseract executable.
    - Accept validated image bytes through OCRRequest.
    - Execute OCR without blocking the async event loop.
    - Extract structured OCR regions.
    - Normalize Tesseract output into OCRResult.
    """

    name = "tesseract"

    def __init__(
        self,
        executable_path: str | Path | None = None,
        timeout_seconds: float = 30.0,
        default_language: str = "eng",
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.default_language = default_language
        self.executable_path = self._resolve_executable(
            executable_path
        )

    @staticmethod
    def _resolve_executable(
        executable_path: str | Path | None,
    ) -> Path | None:
        """
        Resolve the Tesseract executable.

        Priority:
        1. Explicit executable path.
        2. Tesseract available on PATH.
        """

        if executable_path is not None:
            path = Path(executable_path).expanduser()

            if path.is_file():
                return path.resolve()

            return None

        discovered = shutil.which("tesseract")

        if discovered:
            return Path(discovered).resolve()

        return None

    def _ensure_available(self) -> None:
        """Ensure that the Tesseract executable is available."""

        if self.executable_path is None:
            raise FileNotFoundError(
                "Tesseract OCR engine was not found. "
                "Install Tesseract or configure a valid executable path."
            )

        if not self.executable_path.is_file():
            raise FileNotFoundError(
                f"Tesseract executable was not found at: "
                f"{self.executable_path}"
            )

    @staticmethod
    def _open_image(image_bytes: bytes) -> Image.Image:
        """Open image bytes and force image loading."""

        image = Image.open(BytesIO(image_bytes))
        image.load()

        return image

    @staticmethod
    def _parse_ocr_data(
        data: dict,
        language: str,
    ) -> list[OCRTextBlock]:
        """
        Convert Tesseract's image_to_data() output into OCRTextBlock objects.

        Tesseract produces hierarchical rows. Rows with confidence -1
        represent non-text hierarchy levels and are ignored.
        """

        blocks: list[OCRTextBlock] = []

        texts = data.get("text", [])
        confidences = data.get("conf", [])
        lefts = data.get("left", [])
        tops = data.get("top", [])
        widths = data.get("width", [])
        heights = data.get("height", [])

        for index, raw_text in enumerate(texts):
            text = str(raw_text).strip()

            if not text:
                continue

            try:
                confidence = float(confidences[index])
            except (ValueError, TypeError, IndexError):
                continue

            if confidence < 0:
                continue

            try:
                x = float(lefts[index])
                y = float(tops[index])
                width = float(widths[index])
                height = float(heights[index])
            except (ValueError, TypeError, IndexError):
                continue

            blocks.append(
                OCRTextBlock(
                    text=text,
                    confidence=confidence,
                    bounding_box=OCRBoundingBox(
                        x=x,
                        y=y,
                        width=width,
                        height=height,
                    ),
                    metadata={
                        "engine": "tesseract",
                        "language": language,
                    },
                )
            )

        return blocks

    def _recognize_sync(
        self,
        request: OCRRequest,
    ) -> OCRResult:
        """Execute the blocking Tesseract operation."""

        self._ensure_available()

        image = self._open_image(request.image)

        language = request.language or self.default_language

        text = pytesseract.image_to_string(
            image,
            lang=language,
        )

        normalized_text = text.strip()

        data = pytesseract.image_to_data(
            image,
            lang=language,
            output_type=Output.DICT,
        )

        blocks = self._parse_ocr_data(
            data,
            language,
        )

        if not normalized_text or not blocks:
            return OCRResult(
                status=OCRStatus.NO_TEXT,
                text="",
                blocks=[],
                confidence=None,
                language=language,
                provider=self.name,
                metadata={
                    "engine": "tesseract",
                    "language": language,
                },
            )

        valid_confidences = [
            block.confidence
            for block in blocks
            if block.confidence is not None
        ]

        overall_confidence = (
            sum(valid_confidences) / len(valid_confidences)
            if valid_confidences
            else None
        )

        return OCRResult(
            status=OCRStatus.COMPLETED,
            text=normalized_text,
            blocks=blocks,
            confidence=overall_confidence,
            language=language,
            provider=self.name,
            metadata={
                "engine": "tesseract",
                "language": language,
                "block_count": len(blocks),
            },
        )

    async def recognize(
        self,
        request: OCRRequest,
    ) -> OCRResult:
        """Run OCR asynchronously without blocking the event loop."""

        language = request.language or self.default_language

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._recognize_sync,
                    request,
                ),
                timeout=self.timeout_seconds,
            )

        except asyncio.TimeoutError:
            return OCRResult(
                status=OCRStatus.FAILED,
                text="",
                blocks=[],
                confidence=None,
                language=language,
                provider=self.name,
                metadata={
                    "engine": "tesseract",
                    "error": "OCR operation timed out",
                    "timeout_seconds": self.timeout_seconds,
                },
            )

        except Exception as exc:
            return OCRResult(
                status=OCRStatus.FAILED,
                text="",
                blocks=[],
                confidence=None,
                language=language,
                provider=self.name,
                metadata={
                    "engine": "tesseract",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                },
            )