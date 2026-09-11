from __future__ import annotations

import hashlib
from io import BytesIO

from PIL import Image, UnidentifiedImageError

from app.ingestion.processor import ContentProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)


class ImageValidationError(ValueError):
    """Raised when image validation fails."""


class ImageDocumentProcessor(ContentProcessor):
    """
    Image ingestion processor.

    Responsibilities:
    - Validate image input.
    - Verify actual image format.
    - Validate image dimensions.
    - Calculate content hash.
    - Extract basic image metadata.
    - Produce an IMAGE ContentBlock.

    OCR and Vision processing are intentionally not performed here.
    They will be introduced as separate capabilities later.
    """

    supported_types = frozenset(
        {
            InputType.IMAGE,
        }
    )

    DEFAULT_MAX_SIZE_BYTES = 20 * 1024 * 1024
    DEFAULT_MAX_PIXELS = 25_000_000

    ALLOWED_FORMATS = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
        "GIF": "image/gif",
    }

    def __init__(
        self,
        *,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
        max_pixels: int = DEFAULT_MAX_PIXELS,
    ) -> None:
        if max_size_bytes <= 0:
            raise ValueError(
                "max_size_bytes must be greater than zero."
            )

        if max_pixels <= 0:
            raise ValueError(
                "max_pixels must be greater than zero."
            )

        self.max_size_bytes = max_size_bytes
        self.max_pixels = max_pixels

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        """
        Validate and process an image ingestion request.
        """

        if request.input_type != InputType.IMAGE:
            raise ValueError(
                "ImageDocumentProcessor only supports IMAGE input."
            )

        content = self._resolve_inline_content(
            request
        )

        validated = self._validate_image(
            content,
            declared_mime_type=request.mime_type,
        )

        metadata = dict(
            request.metadata
        )

        metadata["image"] = {
            "format": validated["format"],
            "mime_type": validated["mime_type"],
            "width": validated["width"],
            "height": validated["height"],
            "mode": validated["mode"],
            "size_bytes": validated["size_bytes"],
            "content_hash": validated["content_hash"],
        }

        # OCR and Vision are intentionally represented as
        # pending capabilities for the next stages.
        metadata["ocr"] = {
            "status": "not_processed",
        }

        metadata["vision"] = {
            "status": "not_processed",
        }

        source = SourceReference(
            source_id=request.source_id,
            source_type=InputType.IMAGE,
            title=request.title,
            filename=request.filename,
            mime_type=validated["mime_type"],
            content_hash=validated["content_hash"],
            storage_uri=request.storage_uri,
        )

        image_block = ContentBlock(
            block_type=ContentBlockType.IMAGE,
            content="",
            order=0,
            metadata={
                "format": validated["format"],
                "mime_type": validated["mime_type"],
                "width": validated["width"],
                "height": validated["height"],
                "mode": validated["mode"],
                "content_hash": validated["content_hash"],
            },
        )

        return ExtractedContent(
            source=source,
            title=request.title,
            text="",
            blocks=[
                image_block,
            ],
            language=None,
            metadata=metadata,
        )

    @staticmethod
    def _resolve_inline_content(
        request: IngestionRequest,
    ) -> bytes:
        """
        Resolve image bytes from the current ingestion boundary.

        Storage-reference resolution remains the responsibility of
        the existing ingestion/storage resolver layer.
        """

        if request.content is None:
            raise ImageValidationError(
                "Image content must be provided."
            )

        if not isinstance(
            request.content,
            bytes,
        ):
            raise ImageValidationError(
                "Image content must be binary bytes."
            )

        return request.content

    def _validate_image(
        self,
        content: bytes,
        *,
        declared_mime_type: str | None = None,
    ) -> dict[str, object]:
        """
        Validate image bytes and return normalized metadata.
        """

        if not content:
            raise ImageValidationError(
                "Image content cannot be empty."
            )

        size_bytes = len(content)

        if size_bytes > self.max_size_bytes:
            raise ImageValidationError(
                "Image exceeds the maximum allowed size."
            )

        try:
            # First pass verifies image structure without loading
            # pixel data.
            with Image.open(
                BytesIO(content)
            ) as image:
                image.verify()

            # verify() invalidates the image object for normal
            # loading, therefore the image is opened again.
            with Image.open(
                BytesIO(content)
            ) as image:
                image.load()

                image_format = image.format
                width, height = image.size
                mode = image.mode

        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            raise ImageValidationError(
                "Invalid or corrupted image."
            ) from exc

        if image_format not in self.ALLOWED_FORMATS:
            raise ImageValidationError(
                f"Unsupported image format: {image_format!r}."
            )

        if width <= 0 or height <= 0:
            raise ImageValidationError(
                "Image dimensions must be greater than zero."
            )

        pixel_count = width * height

        if pixel_count > self.max_pixels:
            raise ImageValidationError(
                "Image dimensions exceed the maximum allowed "
                "pixel count."
            )

        actual_mime_type = self.ALLOWED_FORMATS[
            image_format
        ]

        if declared_mime_type:
            normalized_mime = (
                declared_mime_type
                .split(";")[0]
                .strip()
                .lower()
            )

            if normalized_mime != actual_mime_type:
                raise ImageValidationError(
                    "Declared MIME type does not match "
                    "the actual image format."
                )

        content_hash = hashlib.sha256(
            content
        ).hexdigest()

        return {
            "format": image_format,
            "mime_type": actual_mime_type,
            "width": width,
            "height": height,
            "mode": mode,
            "size_bytes": size_bytes,
            "content_hash": content_hash,
        }