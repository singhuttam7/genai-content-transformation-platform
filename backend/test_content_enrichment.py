from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image

from app.ingestion.enrichment import ContentEnrichmentService
from app.ingestion.ocr.schemas import (
    OCRBoundingBox,
    OCRRequest,
    OCRResult,
    OCRStatus,
    OCRTextBlock,
)
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
)


# ============================================================
# Fake OCR Provider
# ============================================================


class FakeOCRProvider:
    """
    Deterministic OCR provider used for service-level testing.

    This test double allows us to test ContentEnrichmentService
    without depending on a real OCR engine such as Tesseract.
    """

    name = "fake-ocr"

    def __init__(
        self,
        result: OCRResult,
    ) -> None:
        self.result = result
        self.requests: list[OCRRequest] = []

    async def recognize(
        self,
        request: OCRRequest,
    ) -> OCRResult:
        self.requests.append(request)
        return self.result


# ============================================================
# Fake Content Resolver
# ============================================================


class FakeContentResolver:
    """
    Deterministic InputContentResolver implementation.

    It allows tests to verify that ContentEnrichmentService
    obtains binary content through the resolver abstraction
    rather than accessing request.content directly.
    """

    def __init__(
        self,
        content: bytes | None,
    ) -> None:
        self.content = content
        self.requests: list[IngestionRequest] = []

    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        self.requests.append(request)

        if self.content is None:
            raise ValueError(
                "No binary content or storage reference is available."
            )

        return self.content


# ============================================================
# Test Image
# ============================================================


def create_test_image() -> bytes:
    """
    Create a small valid PNG image for testing.
    """

    image = Image.new(
        "RGB",
        (400, 200),
        "white",
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# Test Request
# ============================================================


def create_image_request(
    *,
    metadata: dict | None = None,
) -> IngestionRequest:
    """
    Create a standard image ingestion request.
    """

    return IngestionRequest(
        input_type=InputType.IMAGE,
        filename="test.png",
        mime_type="image/png",
        content=create_test_image(),
        metadata=metadata or {},
    )


# ============================================================
# Test Extracted Content
# ============================================================


def create_image_content() -> ExtractedContent:
    """
    Create ExtractedContent representing an already-processed
    image input.
    """

    return ExtractedContent(
        source={
            "source_type": InputType.IMAGE,
            "filename": "test.png",
            "mime_type": "image/png",
        },
        title="Test Image",
        text="",
        blocks=[
            ContentBlock(
                block_type=ContentBlockType.IMAGE,
                content="",
                order=0,
                metadata={
                    "format": "PNG",
                    "mime_type": "image/png",
                    "width": 400,
                    "height": 200,
                },
            )
        ],
        language=None,
        metadata={
            "existing": "value",
        },
    )


# ============================================================
# Successful OCR Result
# ============================================================


def create_success_result() -> OCRResult:
    """
    Create deterministic successful OCR output.
    """

    return OCRResult(
        status=OCRStatus.COMPLETED,
        text="Gen AI Content Transformation",
        blocks=[
            OCRTextBlock(
                text="Gen AI Content Transformation",
                confidence=96.5,
                bounding_box=OCRBoundingBox(
                    x=10,
                    y=20,
                    width=300,
                    height=40,
                ),
                metadata={
                    "engine": "fake",
                    "language": "eng",
                },
            ),
            OCRTextBlock(
                text="OCR Test",
                confidence=94.0,
                bounding_box=OCRBoundingBox(
                    x=10,
                    y=70,
                    width=150,
                    height=40,
                ),
                metadata={
                    "engine": "fake",
                    "language": "eng",
                },
            ),
        ],
        confidence=95.25,
        language="eng",
        provider="fake-ocr",
        metadata={
            "engine": "fake",
            "block_count": 2,
        },
    )


# ============================================================
# 1. Non-image input
# ============================================================


def test_non_image_content_is_unchanged() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = IngestionRequest(
        input_type=InputType.TEXT,
        content="Original text",
    )

    content = ExtractedContent(
        source={
            "source_type": InputType.TEXT,
        },
        text="Original text",
        blocks=[
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content="Original text",
                order=0,
            )
        ],
    )

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert result.text == "Original text"

    assert len(result.blocks) == 1

    assert (
        result.blocks[0].content
        == "Original text"
    )

    assert provider.requests == []

    assert resolver.requests == []


# ============================================================
# 2. Image without OCR provider
# ============================================================


def test_image_without_ocr_provider_is_unchanged() -> None:
    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert result.text == content.text

    assert result.blocks == content.blocks

    assert result.metadata == content.metadata

    assert resolver.requests == []


# ============================================================
# 3. Successful OCR
# ============================================================


def test_successful_ocr_adds_text_and_blocks() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    image_bytes = create_test_image()

    resolver = FakeContentResolver(
        image_bytes
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert (
        result.text
        == "Gen AI Content Transformation"
    )

    assert len(result.blocks) == 3

    # Original IMAGE block.
    assert (
        result.blocks[0].block_type
        == ContentBlockType.IMAGE
    )

    # First OCR block.
    assert (
        result.blocks[1].block_type
        == ContentBlockType.PARAGRAPH
    )

    # Second OCR block.
    assert (
        result.blocks[2].block_type
        == ContentBlockType.PARAGRAPH
    )

    assert (
        result.blocks[1].content
        == "Gen AI Content Transformation"
    )

    assert (
        result.blocks[2].content
        == "OCR Test"
    )

    # Verify resolver was actually used.
    assert len(resolver.requests) == 1

    assert resolver.requests[0] is request

    # Verify OCR received resolved bytes.
    assert len(provider.requests) == 1

    assert (
        provider.requests[0].image
        == image_bytes
    )


# ============================================================
# 4. Original IMAGE block preservation
# ============================================================


def test_original_image_block_is_preserved() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    image_block = result.blocks[0]

    assert (
        image_block.block_type
        == ContentBlockType.IMAGE
    )

    assert image_block.content == ""

    assert (
        image_block.metadata["format"]
        == "PNG"
    )

    assert (
        image_block.metadata["width"]
        == 400
    )

    assert (
        image_block.metadata["height"]
        == 200
    )


# ============================================================
# 5. OCR metadata
# ============================================================


def test_ocr_metadata_is_recorded() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert result.metadata["existing"] == "value"

    ocr_metadata = result.metadata["ocr"]

    assert (
        ocr_metadata["status"]
        == "completed"
    )

    assert (
        ocr_metadata["provider"]
        == "fake-ocr"
    )

    assert (
        ocr_metadata["language"]
        == "eng"
    )

    assert (
        ocr_metadata["confidence"]
        == 95.25
    )

    assert (
        ocr_metadata["block_count"]
        == 2
    )

    assert (
        ocr_metadata["engine"]
        == "fake"
    )


# ============================================================
# 6. OCR block structure
# ============================================================


def test_ocr_block_metadata_contains_structure() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    block = result.blocks[1]

    assert (
        block.metadata["source"]
        == "ocr"
    )

    assert (
        block.metadata["provider"]
        == "fake-ocr"
    )

    assert (
        block.metadata["language"]
        == "eng"
    )

    assert (
        block.metadata["confidence"]
        == 96.5
    )

    assert (
        block.metadata["bounding_box"]
        == {
            "x": 10.0,
            "y": 20.0,
            "width": 300.0,
            "height": 40.0,
        }
    )

    assert (
        block.metadata["engine"]
        == "fake"
    )


# ============================================================
# 7. Existing text preservation
# ============================================================


def test_existing_text_is_preserved_when_ocr_succeeds() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content().model_copy(
        update={
            "text": "Existing contextual text"
        }
    )

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert (
        result.text
        == "Existing contextual text\n\n"
        "Gen AI Content Transformation"
    )


# ============================================================
# 8. OCR language propagation
# ============================================================


def test_ocr_language_is_passed_from_request_metadata() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request(
        metadata={
            "ocr_language": "fra",
        }
    )

    content = create_image_content()

    asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert len(provider.requests) == 1

    assert (
        provider.requests[0].language
        == "fra"
    )

    assert (
        provider.requests[0].metadata[
            "filename"
        ]
        == "test.png"
    )

    assert (
        provider.requests[0].metadata[
            "mime_type"
        ]
        == "image/png"
    )


# ============================================================
# 9. OCR NO_TEXT
# ============================================================


def test_ocr_no_text_is_non_fatal() -> None:
    provider = FakeOCRProvider(
        OCRResult(
            status=OCRStatus.NO_TEXT,
            text="",
            blocks=[],
            confidence=None,
            language="eng",
            provider="fake-ocr",
            metadata={
                "reason": "no text detected",
            },
        )
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert result.text == ""

    assert len(result.blocks) == 1

    assert (
        result.blocks[0].block_type
        == ContentBlockType.IMAGE
    )

    assert (
        result.metadata["ocr"]["status"]
        == "no_text"
    )

    assert (
        result.metadata["ocr"]["provider"]
        == "fake-ocr"
    )

    assert (
        result.metadata["ocr"]["reason"]
        == "no text detected"
    )


# ============================================================
# 10. OCR FAILED
# ============================================================


def test_ocr_failure_is_non_fatal() -> None:
    provider = FakeOCRProvider(
        OCRResult(
            status=OCRStatus.FAILED,
            text="",
            blocks=[],
            confidence=None,
            language="eng",
            provider="fake-ocr",
            metadata={
                "error": "OCR engine unavailable",
                "error_type": "FileNotFoundError",
            },
        )
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert len(result.blocks) == 1

    assert (
        result.blocks[0].block_type
        == ContentBlockType.IMAGE
    )

    assert result.text == ""

    assert (
        result.metadata["ocr"]["status"]
        == "failed"
    )

    assert (
        result.metadata["ocr"]["provider"]
        == "fake-ocr"
    )

    assert (
        result.metadata["ocr"]["error"]
        == "OCR engine unavailable"
    )

    assert (
        result.metadata["ocr"]["error_type"]
        == "FileNotFoundError"
    )


# ============================================================
# 11. Resolver failure
# ============================================================


def test_missing_image_bytes_is_non_fatal() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        None
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = IngestionRequest(
        input_type=InputType.IMAGE,
        filename="stored.png",
        mime_type="image/png",
        content=None,
        storage_key="sources/test.png",
    )

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert len(result.blocks) == 1

    assert (
        result.blocks[0].block_type
        == ContentBlockType.IMAGE
    )

    assert (
        result.metadata["ocr"]["status"]
        == "not_available"
    )

    assert (
        result.metadata["ocr"]["error_type"]
        == "ValueError"
    )

    assert (
        len(provider.requests)
        == 0
    )

    assert (
        len(resolver.requests)
        == 1
    )


# ============================================================
# 12. Deterministic block ordering
# ============================================================


def test_ocr_blocks_have_deterministic_order() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert [
        block.order
        for block in result.blocks
    ] == [0, 1, 2]


# ============================================================
# 13. Original content is not mutated
# ============================================================


def test_enrichment_does_not_mutate_original_content() -> None:
    provider = FakeOCRProvider(
        create_success_result()
    )

    resolver = FakeContentResolver(
        create_test_image()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=provider,
    )

    request = create_image_request()

    content = create_image_content()

    original_text = content.text

    original_block_count = len(
        content.blocks
    )

    original_metadata = (
        content.metadata.copy()
    )

    result = asyncio.run(
        service.enrich(
            request=request,
            content=content,
        )
    )

    assert result is not content

    assert (
        content.text
        == original_text
    )

    assert (
        len(content.blocks)
        == original_block_count
    )

    assert (
        content.metadata
        == original_metadata
    )

    assert len(result.blocks) == 3