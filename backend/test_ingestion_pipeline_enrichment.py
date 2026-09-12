from __future__ import annotations

import asyncio
import io
import wave
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
    ContentBlockType,
    IngestionRequest,
    InputType,
)
from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRStatus,
)


# ============================================================
# TEST FIXTURES
# ============================================================


def create_test_image(
    *,
    width: int = 320,
    height: int = 200,
    image_format: str = "PNG",
) -> bytes:
    """
    Create a small valid image for integration tests.
    """

    image = Image.new(
        "RGB",
        (width, height),
        color="white",
    )

    buffer = BytesIO()

    image.save(
        buffer,
        format=image_format,
    )

    return buffer.getvalue()


def create_test_wav(
    duration_seconds: int = 1,
    sample_rate: int = 16000,
    channels: int = 1,
) -> bytes:
    """
    Create a valid silent WAV file for integration tests.

    A real WAV container is required because the production
    AudioDocumentProcessor performs FFprobe validation.
    """

    buffer = io.BytesIO()

    sample_width = 2
    frame_count = sample_rate * duration_seconds

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)

        wav.writeframes(
            b"\x00\x00"
            * channels
            * frame_count
        )

    return buffer.getvalue()


# ============================================================
# FAKE CONTENT RESOLVER
# ============================================================


class FakeContentResolver:
    """
    Deterministic content resolver used by enrichment tests.

    It returns predefined bytes and records every request.
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
# FAKE OCR PROVIDER
# ============================================================


class FakeOCRProvider:
    """
    Deterministic OCR provider used for pipeline integration
    tests.
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
                "Artificial intelligence is transforming "
                "communication.\n\n"
                "Organizations are using AI to create "
                "content faster."
            ),
            blocks=[
                OCRTextBlock(
                    text=(
                        "Artificial intelligence is "
                        "transforming communication."
                    ),
                    confidence=0.96,
                    bounding_box=OCRBoundingBox(
                        x=10,
                        y=20,
                        width=250,
                        height=30,
                    ),
                ),
                OCRTextBlock(
                    text=(
                        "Organizations are using AI to "
                        "create content faster."
                    ),
                    confidence=0.93,
                    bounding_box=OCRBoundingBox(
                        x=10,
                        y=60,
                        width=280,
                        height=30,
                    ),
                ),
            ],
            confidence=0.945,
            language="en",
            provider=self.name,
            metadata={
                "engine": "fake",
                "model": "fake-ocr-model",
                "block_count": 2,
            },
        )


# ============================================================
# FAKE ASR PROVIDER
# ============================================================


class FakeASRProvider:
    """
    Deterministic ASR provider used for pipeline integration
    tests.

    This avoids downloading/running a real Whisper model while
    still exercising the complete ASR integration contract.
    """

    name = "fake-asr"

    def __init__(self) -> None:
        self.requests: list[ASRRequest] = []

    async def transcribe(
        self,
        request: ASRRequest,
    ) -> ASRResult:
        self.requests.append(request)

        return ASRResult(
            status=ASRStatus.COMPLETED,
            text=(
                "Artificial intelligence is transforming "
                "communication. "
                "Organizations are using AI to create "
                "content faster."
            ),
            segments=[
                ASRSegment(
                    text=(
                        "Artificial intelligence is "
                        "transforming communication."
                    ),
                    start_time=0.0,
                    end_time=3.5,
                    confidence=0.94,
                    metadata={
                        "engine": "fake",
                    },
                ),
                ASRSegment(
                    text=(
                        "Organizations are using AI to "
                        "create content faster."
                    ),
                    start_time=3.5,
                    end_time=7.2,
                    confidence=0.91,
                    metadata={
                        "engine": "fake",
                    },
                ),
            ],
            language="en",
            confidence=0.925,
            provider=self.name,
            metadata={
                "engine": "fake",
                "model": "fake-whisper",
                "segment_count": 2,
            },
        )


# ============================================================
# TEST PIPELINE FACTORY
# ============================================================


def create_test_pipeline(
    *,
    resolver: FakeContentResolver,
    ocr_provider: FakeOCRProvider | None = None,
    asr_provider: FakeASRProvider | None = None,
) -> IngestionPipeline:
    """
    Build the real ingestion pipeline while injecting fake
    enrichment providers.
    """

    detector = DefaultInputDetector()

    router = create_processor_router()

    normalizer = DefaultContentNormalizer()

    enrichment = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=ocr_provider,
        asr_provider=asr_provider,
    )

    return IngestionPipeline(
        detector=detector,
        router=router,
        normalizer=normalizer,
        enrichment=enrichment,
    )


# ============================================================
# OCR INTEGRATION TESTS
# ============================================================


def test_image_pipeline_produces_ocr_enriched_canonical_content() -> None:
    """
    Verify the complete IMAGE ingestion path:

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
        filename="test-image.png",
        mime_type="image/png",
        content=image_bytes,
        metadata={
            "ocr_language": "eng",
        },
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    assert (
        "Artificial intelligence is transforming "
        "communication."
        in canonical_content.text
    )

    assert (
        "Organizations are using AI to create "
        "content faster."
        in canonical_content.text
    )

    paragraph_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.PARAGRAPH
    ]

    assert len(paragraph_blocks) == 2

    assert paragraph_blocks[0].content == (
        "Artificial intelligence is transforming "
        "communication."
    )

    assert paragraph_blocks[1].content == (
        "Organizations are using AI to create "
        "content faster."
    )

    assert paragraph_blocks[0].metadata["source"] == "ocr"
    assert paragraph_blocks[0].metadata["provider"] == "fake-ocr"
    assert paragraph_blocks[0].metadata["confidence"] == 0.96

    assert paragraph_blocks[1].metadata["source"] == "ocr"
    assert paragraph_blocks[1].metadata["provider"] == "fake-ocr"
    assert paragraph_blocks[1].metadata["confidence"] == 0.93

    assert canonical_content.language == "en"

    assert canonical_content.metadata["ocr"]["status"] == (
        "completed"
    )

    assert canonical_content.metadata["ocr"]["provider"] == (
        "fake-ocr"
    )

    assert canonical_content.metadata["ocr"]["block_count"] == 2

    assert len(resolver.requests) == 1
    assert resolver.requests[0] == request

    assert len(ocr_provider.requests) == 1
    assert ocr_provider.requests[0].image == image_bytes
    assert ocr_provider.requests[0].language == "eng"


def test_ocr_metadata_survives_canonical_normalization() -> None:
    """
    Verify OCR metadata survives the normalization stage.
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

    ocr_metadata = canonical_content.metadata["ocr"]

    assert ocr_metadata["status"] == "completed"
    assert ocr_metadata["provider"] == "fake-ocr"
    assert ocr_metadata["language"] == "en"
    assert ocr_metadata["confidence"] == 0.945
    assert ocr_metadata["block_count"] == 2
    assert ocr_metadata["engine"] == "fake"
    assert ocr_metadata["model"] == "fake-ocr-model"


def test_image_pipeline_preserves_original_image_segment() -> None:
    """
    Verify OCR enrichment does not destroy the original
    IMAGE structural segment.
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

    image_segments = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.IMAGE
    ]

    paragraph_segments = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.PARAGRAPH
    ]

    # --------------------------------------------------------
    # Original IMAGE structural segment survives.
    # --------------------------------------------------------

    assert len(image_segments) == 1

    assert image_segments[0].content == ""

    # --------------------------------------------------------
    # OCR metadata is stored at canonical level.
    # --------------------------------------------------------

    assert "ocr" in canonical_content.metadata

    assert (
        canonical_content.metadata["ocr"]["status"]
        == "completed"
    )

    assert (
        canonical_content.metadata["ocr"]["provider"]
        == "fake-ocr"
    )

    # --------------------------------------------------------
    # OCR-generated text blocks survive normalization.
    # --------------------------------------------------------

    assert len(paragraph_segments) == 2

    assert paragraph_segments[0].content == (
        "Artificial intelligence is transforming "
        "communication."
    )

    assert paragraph_segments[1].content == (
        "Organizations are using AI to create "
        "content faster."
    )


# ============================================================
# ASR INTEGRATION TESTS — A3.6
# ============================================================


def test_audio_pipeline_produces_asr_enriched_canonical_content() -> None:
    """
    Verify the complete AUDIO ingestion path:

        IngestionRequest
            ↓
        Detection
            ↓
        AudioDocumentProcessor
            ↓
        FFprobe Media Inspection
            ↓
        ContentEnrichmentService
            ↓
        ASRProvider
            ↓
        ContentNormalizer
            ↓
        CanonicalContent
    """

    # Use a real WAV file because the production audio
    # processor performs FFprobe validation.
    audio_bytes = create_test_wav()

    resolver = FakeContentResolver(
        audio_bytes
    )

    asr_provider = FakeASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        asr_provider=asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="pipeline-test.wav",
        mime_type="audio/wav",
        content=audio_bytes,
        metadata={
            "asr_language": "en",
        },
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    # --------------------------------------------------------
    # Canonical transcript
    # --------------------------------------------------------

    assert (
        "Artificial intelligence is transforming "
        "communication."
        in canonical_content.text
    )

    assert (
        "Organizations are using AI to create "
        "content faster."
        in canonical_content.text
    )

    # --------------------------------------------------------
    # Transcript blocks
    # --------------------------------------------------------

    transcript_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 2

    assert transcript_blocks[0].content == (
        "Artificial intelligence is transforming "
        "communication."
    )

    assert transcript_blocks[1].content == (
        "Organizations are using AI to create "
        "content faster."
    )

    # --------------------------------------------------------
    # Timestamps
    # --------------------------------------------------------

    assert transcript_blocks[0].start_time == 0.0
    assert transcript_blocks[0].end_time == 3.5

    assert transcript_blocks[1].start_time == 3.5
    assert transcript_blocks[1].end_time == 7.2

    # --------------------------------------------------------
    # Transcript metadata
    # --------------------------------------------------------

    assert transcript_blocks[0].metadata["source"] == "asr"
    assert transcript_blocks[0].metadata["provider"] == "fake-asr"
    assert transcript_blocks[0].metadata["language"] == "en"
    assert transcript_blocks[0].metadata["confidence"] == 0.94

    assert transcript_blocks[1].metadata["source"] == "asr"
    assert transcript_blocks[1].metadata["provider"] == "fake-asr"
    assert transcript_blocks[1].metadata["language"] == "en"
    assert transcript_blocks[1].metadata["confidence"] == 0.91

    # --------------------------------------------------------
    # Canonical ASR metadata
    # --------------------------------------------------------

    assert "asr" in canonical_content.metadata

    asr_metadata = canonical_content.metadata["asr"]

    assert asr_metadata["status"] == "completed"
    assert asr_metadata["provider"] == "fake-asr"
    assert asr_metadata["language"] == "en"
    assert asr_metadata["confidence"] == 0.925
    assert asr_metadata["segment_count"] == 2
    assert asr_metadata["engine"] == "fake"
    assert asr_metadata["model"] == "fake-whisper"

    # --------------------------------------------------------
    # Resolver contract
    # --------------------------------------------------------

    assert len(resolver.requests) == 1
    assert resolver.requests[0] == request

    # --------------------------------------------------------
    # ASR provider contract
    # --------------------------------------------------------

    assert len(asr_provider.requests) == 1

    assert (
        asr_provider.requests[0].audio
        == audio_bytes
    )

    assert (
        asr_provider.requests[0].language
        == "en"
    )


def test_audio_pipeline_preserves_original_audio_segment() -> None:
    """
    Verify ASR enrichment adds transcript segments without
    destroying the original AUDIO structural segment.
    """

    audio_bytes = create_test_wav()

    resolver = FakeContentResolver(
        audio_bytes
    )

    asr_provider = FakeASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        asr_provider=asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="preservation-test.wav",
        mime_type="audio/wav",
        content=audio_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    audio_segments = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.AUDIO
    ]

    transcript_segments = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    # --------------------------------------------------------
    # Original AUDIO segment survives.
    # --------------------------------------------------------

    assert len(audio_segments) == 1

    assert audio_segments[0].content == ""

    assert (
        audio_segments[0].metadata["transcription_status"]
        == "not_requested"
    )

    # --------------------------------------------------------
    # ASR transcript segments are added.
    # --------------------------------------------------------

    assert len(transcript_segments) == 2

    assert transcript_segments[0].content == (
        "Artificial intelligence is transforming "
        "communication."
    )

    assert transcript_segments[1].content == (
        "Organizations are using AI to create "
        "content faster."
    )


def test_asr_metadata_survives_canonical_normalization() -> None:
    """
    Verify ASR metadata survives the canonical normalization
    stage.
    """

    audio_bytes = create_test_wav()

    resolver = FakeContentResolver(
        audio_bytes
    )

    asr_provider = FakeASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        asr_provider=asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="metadata-test.wav",
        mime_type="audio/wav",
        content=audio_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    # --------------------------------------------------------
    # ASR metadata
    # --------------------------------------------------------

    assert "asr" in canonical_content.metadata

    asr_metadata = canonical_content.metadata["asr"]

    assert asr_metadata["status"] == "completed"
    assert asr_metadata["provider"] == "fake-asr"
    assert asr_metadata["language"] == "en"
    assert asr_metadata["confidence"] == 0.925
    assert asr_metadata["segment_count"] == 2
    assert asr_metadata["engine"] == "fake"
    assert asr_metadata["model"] == "fake-whisper"

    # --------------------------------------------------------
    # Transcript metadata survives normalization.
    # --------------------------------------------------------

    transcript_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 2

    assert transcript_blocks[0].metadata["source"] == "asr"
    assert transcript_blocks[0].metadata["provider"] == "fake-asr"

    assert transcript_blocks[1].metadata["source"] == "asr"
    assert transcript_blocks[1].metadata["provider"] == "fake-asr"