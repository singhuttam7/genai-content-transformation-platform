from __future__ import annotations

import hashlib
from io import BytesIO

import pytest
from PIL import Image
from pydantic import ValidationError

from app.ingestion.parsers.image import (
    ImageDocumentProcessor,
    ImageValidationError,
)
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


# ============================================================
# Helpers
# ============================================================

def create_image_bytes(
    image_format: str = "PNG",
    size: tuple[int, int] = (100, 100),
    color: str = "white",
) -> bytes:
    """Create a valid in-memory image for testing."""

    image = Image.new(
        "RGB",
        size,
        color=color,
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format=image_format,
    )

    return buffer.getvalue()


def create_request(
    content: bytes,
    *,
    mime_type: str | None = None,
    filename: str | None = None,
    title: str | None = None,
    metadata: dict | None = None,
) -> IngestionRequest:
    """Create a standard IMAGE ingestion request."""

    return IngestionRequest(
        input_type=InputType.IMAGE,
        content=content,
        mime_type=mime_type,
        filename=filename,
        title=title,
        metadata=metadata or {},
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def processor() -> ImageDocumentProcessor:
    return ImageDocumentProcessor()


@pytest.fixture
def png_bytes() -> bytes:
    return create_image_bytes("PNG")


@pytest.fixture
def jpeg_bytes() -> bytes:
    return create_image_bytes("JPEG")


@pytest.fixture
def webp_bytes() -> bytes:
    return create_image_bytes("WEBP")


@pytest.fixture
def gif_bytes() -> bytes:
    return create_image_bytes("GIF")


# ============================================================
# 1. Basic processor contract
# ============================================================

def test_processor_supports_image_only():
    processor = ImageDocumentProcessor()

    assert processor.supported_types == frozenset(
        {InputType.IMAGE}
    )


def test_processor_default_limits_are_positive():
    processor = ImageDocumentProcessor()

    assert processor.max_size_bytes > 0
    assert processor.max_pixels > 0


def test_processor_accepts_custom_size_limit():
    processor = ImageDocumentProcessor(
        max_size_bytes=12345
    )

    assert processor.max_size_bytes == 12345


def test_processor_accepts_custom_pixel_limit():
    processor = ImageDocumentProcessor(
        max_pixels=9999
    )

    assert processor.max_pixels == 9999


def test_processor_rejects_invalid_size_limit():
    with pytest.raises(ValueError):
        ImageDocumentProcessor(
            max_size_bytes=0
        )


def test_processor_rejects_invalid_pixel_limit():
    with pytest.raises(ValueError):
        ImageDocumentProcessor(
            max_pixels=0
        )


# ============================================================
# 2. Valid image formats
# ============================================================

@pytest.mark.asyncio
async def test_process_valid_png(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        mime_type="image/png",
        filename="test.png",
    )

    result = await processor.process(
        request
    )

    assert result.source.source_type == InputType.IMAGE
    assert result.source.mime_type == "image/png"
    assert result.source.filename == "test.png"

    assert len(result.blocks) == 1

    assert (
        result.blocks[0].block_type
        == ContentBlockType.IMAGE
    )


@pytest.mark.asyncio
async def test_process_valid_jpeg(
    processor,
    jpeg_bytes,
):
    request = create_request(
        jpeg_bytes,
        mime_type="image/jpeg",
        filename="test.jpg",
    )

    result = await processor.process(
        request
    )

    assert result.source.source_type == InputType.IMAGE
    assert result.source.mime_type == "image/jpeg"


@pytest.mark.asyncio
async def test_process_valid_webp(
    processor,
    webp_bytes,
):
    request = create_request(
        webp_bytes,
        mime_type="image/webp",
        filename="test.webp",
    )

    result = await processor.process(
        request
    )

    assert result.source.source_type == InputType.IMAGE
    assert result.source.mime_type == "image/webp"


@pytest.mark.asyncio
async def test_process_valid_gif(
    processor,
    gif_bytes,
):
    request = create_request(
        gif_bytes,
        mime_type="image/gif",
        filename="test.gif",
    )

    result = await processor.process(
        request
    )

    assert result.source.source_type == InputType.IMAGE
    assert result.source.mime_type == "image/gif"


# ============================================================
# 3. Input validation
# ============================================================

@pytest.mark.asyncio
async def test_rejects_non_image_input_type(
    processor,
    png_bytes,
):
    request = IngestionRequest(
        input_type=InputType.TEXT,
        content=png_bytes,
    )

    with pytest.raises(
        ValueError,
        match="only supports IMAGE",
    ):
        await processor.process(
            request
        )


def test_rejects_missing_content():
    with pytest.raises(
        ValidationError
    ):
        IngestionRequest(
            input_type=InputType.IMAGE,
            content=None,
        )


def test_rejects_text_content():
    with pytest.raises(
        ValidationError
    ):
        IngestionRequest(
            input_type=InputType.IMAGE,
            content="not image bytes",
        )


@pytest.mark.asyncio
async def test_rejects_empty_content(
    processor,
):
    request = create_request(
        b"",
        mime_type="image/png",
    )

    with pytest.raises(
        ImageValidationError
    ):
        await processor.process(
            request
        )


# ============================================================
# 4. Corrupted / invalid image data
# ============================================================

@pytest.mark.asyncio
async def test_rejects_corrupted_image(
    processor,
):
    corrupted = (
        b"This is definitely not "
        b"a valid image."
    )

    request = create_request(
        corrupted,
        mime_type="image/png",
        filename="corrupted.png",
    )

    with pytest.raises(
        ImageValidationError
    ):
        await processor.process(
            request
        )


@pytest.mark.asyncio
async def test_rejects_random_binary_data(
    processor,
):
    random_data = bytes(
        range(256)
    )

    request = create_request(
        random_data,
        mime_type="image/png",
    )

    with pytest.raises(
        ImageValidationError
    ):
        await processor.process(
            request
        )


# ============================================================
# 5. MIME type validation
# ============================================================

@pytest.mark.asyncio
async def test_rejects_mismatched_mime_type(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        mime_type="image/jpeg",
        filename="wrong.jpg",
    )

    with pytest.raises(
        ImageValidationError
    ):
        await processor.process(
            request
        )


@pytest.mark.asyncio
async def test_accepts_matching_mime_type(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        mime_type="image/png",
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    assert (
        result.source.mime_type
        == "image/png"
    )


@pytest.mark.asyncio
async def test_accepts_missing_mime_type(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    assert (
        result.source.mime_type
        == "image/png"
    )


# ============================================================
# 6. Image metadata
# ============================================================

@pytest.mark.asyncio
async def test_extracts_image_dimensions(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    metadata = result.blocks[0].metadata

    assert metadata["width"] == 100
    assert metadata["height"] == 100


@pytest.mark.asyncio
async def test_extracts_image_format(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    metadata = result.blocks[0].metadata

    assert metadata["format"] == "PNG"


@pytest.mark.asyncio
async def test_extracts_pixel_count(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    metadata = result.blocks[0].metadata

    assert (
        metadata["width"]
        * metadata["height"]
        == 10_000
    )


# ============================================================
# 7. Content hash
# ============================================================

@pytest.mark.asyncio
async def test_generates_sha256_content_hash(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    expected_hash = hashlib.sha256(
        png_bytes
    ).hexdigest()

    assert (
        result.source.content_hash
        == expected_hash
    )


@pytest.mark.asyncio
async def test_content_hash_is_deterministic(
    processor,
    png_bytes,
):
    request1 = create_request(
        png_bytes
    )

    request2 = create_request(
        png_bytes
    )

    result1 = await processor.process(
        request1
    )

    result2 = await processor.process(
        request2
    )

    assert (
        result1.source.content_hash
        == result2.source.content_hash
    )


# ============================================================
# 8. Content block
# ============================================================

@pytest.mark.asyncio
async def test_creates_single_image_content_block(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    assert len(result.blocks) == 1

    block = result.blocks[0]

    assert (
        block.block_type
        == ContentBlockType.IMAGE
    )

    assert block.order == 0


@pytest.mark.asyncio
async def test_image_block_contains_metadata(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="image.png",
    )

    result = await processor.process(
        request
    )

    block_metadata = (
        result.blocks[0].metadata
    )

    image_metadata = (
        result.metadata["image"]
    )

    assert "width" in block_metadata
    assert "height" in block_metadata
    assert "format" in block_metadata
    assert "mime_type" in block_metadata

    assert "size_bytes" in image_metadata
    assert "content_hash" in image_metadata


# ============================================================
# 9. OCR / Vision state
# ============================================================

@pytest.mark.asyncio
async def test_ocr_status_is_not_processed(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes
    )

    result = await processor.process(
        request
    )

    assert result.metadata["ocr"] == {
        "status": "not_processed"
    }


@pytest.mark.asyncio
async def test_vision_status_is_not_processed(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes
    )

    result = await processor.process(
        request
    )

    assert result.metadata["vision"] == {
        "status": "not_processed"
    }


# ============================================================
# 10. Configurable security limits
# ============================================================

@pytest.mark.asyncio
async def test_rejects_image_exceeding_size_limit():
    image_bytes = create_image_bytes(
        "PNG",
        size=(100, 100),
    )

    processor = ImageDocumentProcessor(
        max_size_bytes=10,
    )

    request = create_request(
        image_bytes,
        mime_type="image/png",
    )

    with pytest.raises(
        ImageValidationError
    ):
        await processor.process(
            request
        )


@pytest.mark.asyncio
async def test_rejects_image_exceeding_pixel_limit():
    image_bytes = create_image_bytes(
        "PNG",
        size=(100, 100),
    )

    processor = ImageDocumentProcessor(
        max_pixels=100,
    )

    request = create_request(
        image_bytes,
        mime_type="image/png",
    )

    with pytest.raises(
        ImageValidationError
    ):
        await processor.process(
            request
        )


# ============================================================
# 11. Request metadata
# ============================================================

@pytest.mark.asyncio
async def test_preserves_request_metadata(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="diagram.png",
        title="Network Diagram",
        metadata={
            "source": "user_upload",
            "category": "technical",
        },
    )

    result = await processor.process(
        request
    )

    assert (
        result.title
        == "Network Diagram"
    )

    assert (
        result.metadata["source"]
        == "user_upload"
    )

    assert (
        result.metadata["category"]
        == "technical"
    )


# ============================================================
# 12. Source reference
# ============================================================

@pytest.mark.asyncio
async def test_source_reference_contains_filename(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes,
        filename="diagram.png",
    )

    result = await processor.process(
        request
    )

    assert (
        result.source.filename
        == "diagram.png"
    )


@pytest.mark.asyncio
async def test_source_reference_contains_image_type(
    processor,
    png_bytes,
):
    request = create_request(
        png_bytes
    )

    result = await processor.process(
        request
    )

    assert (
        result.source.source_type
        == InputType.IMAGE
    )


# ============================================================
# 13. Custom limits - valid image
# ============================================================

@pytest.mark.asyncio
async def test_accepts_image_within_custom_limits():
    image_bytes = create_image_bytes(
        "PNG",
        size=(20, 20),
    )

    processor = ImageDocumentProcessor(
        max_size_bytes=len(image_bytes) + 100,
        max_pixels=500,
    )

    request = create_request(
        image_bytes,
        mime_type="image/png",
    )

    result = await processor.process(
        request
    )

    assert (
        result.blocks[0].metadata["width"]
        == 20
    )

    assert (
        result.blocks[0].metadata["height"]
        == 20
    )


# ============================================================
# End of image ingestion tests
# ============================================================