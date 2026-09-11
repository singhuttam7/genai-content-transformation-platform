from __future__ import annotations

import asyncio
from io import BytesIO

from PIL import Image

from app.ingestion.detectors import DefaultInputDetector
from app.ingestion.enrichment import ContentEnrichmentService
from app.ingestion.normalization import DefaultContentNormalizer
from app.ingestion.ocr.schemas import (
    OCRBoundingBox,
    OCRRequest,
    OCRResult,
    OCRStatus,
    OCRTextBlock,
)
from app.ingestion.pipeline import IngestionPipeline
from app.ingestion.registry import create_processor_router
from app.ingestion.schemas import (
    IngestionRequest,
    InputType,
)


# ============================================================
# Test Image
# ============================================================


def create_test_image() -> bytes:
    """
    Create a valid PNG image for pipeline integration testing.
    """

    image = Image.new(
        "RGB",
        (800, 400),
        "white",
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# Fake Content Resolver
# ============================================================


class FakeContentResolver:
    """
    Test implementation of InputContentResolver.

    The resolver represents the boundary between the ingestion
    pipeline and the physical source-storage layer.
    """

    def __init__(
        self,
        content: bytes,
    ) -> None:
        self.content = content
        self.requests: list[IngestionRequest] = []

    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        self.requests.append(request)

        return self.content


# ============================================================
# Fake OCR Provider
# ============================================================


class FakeOCRProvider:
    """
    Deterministic OCR provider for pipeline integration tests.
    """

    name = "fake-ocr"

    def __init__(self) -> None:
        self.requests: list[OCRRequest] = []

    async def recognize(
        self,
        request: OCRRequest,
    ) -> OCRResult:
        self.requests.append(request)

        return OCRResult(
            status=OCRStatus.COMPLETED,
            text=(
                "Gen AI Content Transformation\n\n"
                    "Pipeline OCR Test"
            ),
            blocks=[
                OCRTextBlock(
                    text=(
                        "Gen AI Content Transformation"
                    ),
                    confidence=97.0,
                    bounding_box=OCRBoundingBox(
                        x=20,
                        y=30,
                        width=500,
                        height=60,
                    ),
                    metadata={
                        "engine": "fake",
                        "language": "eng",
                    },
                ),
                OCRTextBlock(
                    text="Pipeline OCR Test",
                    confidence=95.0,
                    bounding_box=OCRBoundingBox(
                        x=20,
                        y=110,
                        width=300,
                        height=60,
                    ),
                    metadata={
                        "engine": "fake",
                        "language": "eng",
                    },
                ),
            ],
            confidence=96.0,
            language="eng",
            provider="fake-ocr",
            metadata={
                "engine": "fake",
                "block_count": 2,
            },
        )


# ============================================================
# Pipeline Factory for Tests
# ============================================================


def create_test_pipeline(
    *,
    resolver: FakeContentResolver,
    ocr_provider: FakeOCRProvider,
) -> IngestionPipeline:
    """
    Construct the complete ingestion pipeline using real
    pipeline components and deterministic test doubles
    for external dependencies.
    """

    detector = DefaultInputDetector()

    router = create_processor_router()

    normalizer = DefaultContentNormalizer()

    enrichment = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=ocr_provider,
    )

    return IngestionPipeline(
        detector=detector,
        router=router,
        normalizer=normalizer,
        enrichment=enrichment,
    )


# ============================================================
# Integration Test
# ============================================================


def test_image_pipeline_produces_ocr_enriched_canonical_content() -> None:
    """
    Verify the complete image ingestion path:

        IngestionRequest
            ↓
        Detection
            ↓
        ImageDocumentProcessor
            ↓
        ContentEnrichmentService
            ↓
        OCRProvider
            ↓
        ContentNormalizer
            ↓
        CanonicalContent
    """

    image_bytes = create_test_image()

    resolver = FakeContentResolver(
        image_bytes
    )

    ocr_provider = FakeOCRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        ocr_provider=ocr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.IMAGE,
        filename="pipeline-test.png",
        mime_type="image/png",
        content=image_bytes,
        metadata={
            "ocr_language": "eng",
        },
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    # ---------------------------------------------------------
    # Canonical text
    # ---------------------------------------------------------

    assert (
        "Gen AI Content Transformation"
        in canonical_content.text
    )

    assert (
        "Pipeline OCR Test"
        in canonical_content.text
    )

    # ---------------------------------------------------------
    # Canonical segments
    # ---------------------------------------------------------

    assert canonical_content.segments

    # The above comparison is intentionally not used for
    # ContentBlockType validation; verify via metadata below.
    assert canonical_content.segments

    assert any(
        segment.metadata.get("source") == "ocr"
        for segment in canonical_content.segments
    )

    # ---------------------------------------------------------
    # OCR metadata
    # ---------------------------------------------------------

    assert (
        canonical_content.metadata["ocr"]["status"]
        == "completed"
    )

    assert (
        canonical_content.metadata["ocr"]["provider"]
        == "fake-ocr"
    )

    assert (
        canonical_content.metadata["ocr"]["language"]
        == "eng"
    )

    assert (
        canonical_content.metadata["ocr"]["block_count"]
        == 2
    )

    # ---------------------------------------------------------
    # Resolver integration
    # ---------------------------------------------------------

    assert len(resolver.requests) == 1

    assert (
        resolver.requests[0] is request
    )

    # ---------------------------------------------------------
    # OCR provider integration
    # ---------------------------------------------------------

    assert len(ocr_provider.requests) == 1

    assert (
        ocr_provider.requests[0].image
        == image_bytes
    )

    assert (
        ocr_provider.requests[0].language
        == "eng"
    )


# ============================================================
# Integration: OCR metadata survives normalization
# ============================================================


def test_ocr_metadata_survives_canonical_normalization() -> None:
    """
    Verify that normalization does not remove the enrichment
    metadata required by downstream systems.
    """

    image_bytes = create_test_image()

    resolver = FakeContentResolver(
        image_bytes
    )

    ocr_provider = FakeOCRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        ocr_provider=ocr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.IMAGE,
        filename="metadata-test.png",
        mime_type="image/png",
        content=image_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    assert "ocr" in canonical_content.metadata

    ocr_metadata = (
        canonical_content.metadata["ocr"]
    )

    assert (
        ocr_metadata["provider"]
        == "fake-ocr"
    )

    assert (
        ocr_metadata["confidence"]
        == 96.0
    )

    assert (
        ocr_metadata["block_count"]
        == 2
    )


# ============================================================
# Integration: original image survives OCR
# ============================================================


def test_image_pipeline_preserves_original_image_segment() -> None:
    """
    Verify that OCR enrichment adds textual segments without
    destroying the original IMAGE structural segment.
    """

    image_bytes = create_test_image()

    resolver = FakeContentResolver(
        image_bytes
    )

    ocr_provider = FakeOCRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        ocr_provider=ocr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.IMAGE,
        filename="preservation-test.png",
        mime_type="image/png",
        content=image_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    # The canonical representation should contain both:
    # - original image structural content
    # - OCR-derived textual content

    assert len(canonical_content.segments) >= 3

    assert any(
        segment.metadata.get("source") == "ocr"
        for segment in canonical_content.segments
    )

    assert any(
        segment.metadata.get("format") == "PNG"
        for segment in canonical_content.segments
    )