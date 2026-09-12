from __future__ import annotations

import asyncio
import io
import wave
from io import BytesIO
from pathlib import Path

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
from app.ingestion.video.asr import VideoASRService
from app.ingestion.video.ffmpeg import FFmpegAudioExtractor


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


def get_sample_video() -> bytes:
    """
    Load the repository's real video integration fixture.

    Expected location:

        backend/test_data/video/sample_with_audio.mp4
    """

    project_root = Path(
        __file__
    ).resolve().parents[3]

    video_path = (
        project_root
        / "test_data"
        / "video"
        / "sample_with_audio.mp4"
    )

    assert video_path.is_file(), (
        f"Video fixture not found: {video_path}"
    )

    return video_path.read_bytes()


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

    def __init__(
        self,
        result: OCRResult | None = None,
    ) -> None:
        self.result = result or self._default_result()
        self.requests: list[OCRRequest] = []

    async def recognize(
        self,
        request: OCRRequest,
    ) -> OCRResult:
        self.requests.append(request)

        return self.result

    @staticmethod
    def _default_result() -> OCRResult:
        return OCRResult(
            status=OCRStatus.COMPLETED,
            text=(
                "Gen AI Content Transformation\n"
                "Pipeline OCR Test"
            ),
            blocks=[
                OCRTextBlock(
                    text="Gen AI Content Transformation",
                    confidence=95.25,
                    bounding_box=OCRBoundingBox(
                        x=10.0,
                        y=20.0,
                        width=300.0,
                        height=40.0,
                    ),
                    metadata={
                        "engine": "fake",
                    },
                ),
                OCRTextBlock(
                    text="Pipeline OCR Test",
                    confidence=96.5,
                    bounding_box=OCRBoundingBox(
                        x=10.0,
                        y=70.0,
                        width=300.0,
                        height=40.0,
                    ),
                    metadata={
                        "engine": "fake",
                    },
                ),
            ],
            confidence=95.25,
            language="eng",
            provider="fake-ocr",
            metadata={
                "engine": "fake",
            },
        )


# ============================================================
# FAKE ASR PROVIDER
# ============================================================


class FakeASRProvider:
    """
    Deterministic ASR provider used for pipeline integration
    tests.
    """

    name = "fake-asr"

    def __init__(
        self,
        result: ASRResult | None = None,
    ) -> None:
        self.result = result or self._default_result()
        self.requests: list[ASRRequest] = []

    async def transcribe(
        self,
        request: ASRRequest,
    ) -> ASRResult:
        self.requests.append(request)

        return self.result

    @staticmethod
    def _default_result() -> ASRResult:
        return ASRResult(
            status=ASRStatus.COMPLETED,
            text=(
                "The security incident affected "
                "several systems."
            ),
            segments=[
                ASRSegment(
                    text=(
                        "The security incident affected"
                    ),
                    start_time=0.0,
                    end_time=2.1,
                    confidence=0.95,
                    metadata={
                        "avg_logprob": -0.12,
                    },
                ),
                ASRSegment(
                    text="several systems.",
                    start_time=2.1,
                    end_time=3.8,
                    confidence=0.95,
                    metadata={
                        "avg_logprob": -0.12,
                    },
                ),
            ],
            language="eng",
            confidence=0.94,
            provider="fake-asr",
            metadata={
                "model": "fake-model",
            },
        )


# ============================================================
# FAKE VIDEO ASR PROVIDER
# ============================================================


class FakeVideoASRProvider:
    """
    Deterministic ASR provider used by the real
    VideoASRService integration tests.

    VideoASRService is real.
    FFmpegAudioExtractor is real.

    Only the final speech-recognition provider is replaced
    with this deterministic test double.
    """

    name = "fake-video-asr"

    def __init__(
        self,
        result: ASRResult | None = None,
    ) -> None:
        self.result = result or self._default_result()
        self.requests: list[ASRRequest] = []

    async def transcribe(
        self,
        request: ASRRequest,
    ) -> ASRResult:
        self.requests.append(request)

        return self.result

    @staticmethod
    def _default_result() -> ASRResult:
        return ASRResult(
            status=ASRStatus.COMPLETED,
            text="This is a test video transcript.",
            segments=[
                ASRSegment(
                    text=(
                        "This is a test video transcript."
                    ),
                    start_time=0.5,
                    end_time=2.5,
                    confidence=0.97,
                    metadata={
                        "source": "fake-video-asr",
                    },
                ),
            ],
            language="en",
            confidence=0.97,
            provider="fake-video-asr",
            metadata={
                "test": True,
            },
        )


# ============================================================
# PIPELINE FACTORY FOR TESTS
# ============================================================


def create_test_pipeline(
    *,
    resolver: FakeContentResolver,
    ocr_provider: FakeOCRProvider | None = None,
    asr_provider: FakeASRProvider | None = None,
    video_asr_provider: FakeVideoASRProvider | None = None,
) -> IngestionPipeline:
    """
    Construct the complete ingestion pipeline using real
    pipeline components and deterministic test doubles
    for external dependencies.

    Video integration additionally uses the real:

        VideoASRService
            ↓
        FFmpegAudioExtractor

    while replacing only the final ASR provider.
    """

    detector = DefaultInputDetector()

    router = create_processor_router()

    normalizer = DefaultContentNormalizer()

    # ---------------------------------------------------------
    # Optional VIDEO ASR service
    # ---------------------------------------------------------

    video_asr_service = None

    if video_asr_provider is not None:
        video_asr_service = VideoASRService(
            audio_extractor=FFmpegAudioExtractor(),
            asr_provider=video_asr_provider,
        )

    enrichment = ContentEnrichmentService(
        content_resolver=resolver,
        ocr_provider=ocr_provider,
        asr_provider=asr_provider,
        video_asr_service=video_asr_service,
    )

    return IngestionPipeline(
        detector=detector,
        router=router,
        normalizer=normalizer,
        enrichment=enrichment,
    )


# ============================================================
# IMAGE INTEGRATION TESTS
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

    assert (
        "Gen AI Content Transformation"
        in canonical_content.text
    )

    assert (
        "Pipeline OCR Test"
        in canonical_content.text
    )

    assert canonical_content.segments

    assert any(
        segment.metadata.get("source") == "ocr"
        for segment in canonical_content.segments
    )

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

    assert len(resolver.requests) == 1

    assert resolver.requests[0] is request

    assert len(ocr_provider.requests) == 1

    assert (
        ocr_provider.requests[0].image
        == image_bytes
    )


def test_ocr_metadata_survives_canonical_normalization() -> None:
    """
    Verify OCR metadata survives the complete pipeline and
    canonical normalization.
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
    assert ocr_metadata["language"] == "eng"
    assert ocr_metadata["confidence"] == 95.25
    assert ocr_metadata["block_count"] == 2
    assert ocr_metadata["engine"] == "fake"


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

    assert len(image_segments) == 1

    assert image_segments[0].content == ""

    assert "ocr" in canonical_content.metadata

    assert (
        canonical_content.metadata["ocr"]["status"]
        == "completed"
    )

    assert (
        canonical_content.metadata["ocr"]["provider"]
        == "fake-ocr"
    )

    assert len(paragraph_segments) == 2

    assert paragraph_segments[0].content == (
        "Gen AI Content Transformation"
    )

    assert paragraph_segments[1].content == (
        "Pipeline OCR Test"
    )


# ============================================================
# AUDIO INTEGRATION TESTS
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

    assert (
        canonical_content.text
        == "The security incident affected "
        "several systems."
    )

    transcript_segments = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_segments) == 2

    assert (
        transcript_segments[0].content
        == "The security incident affected"
    )

    assert (
        transcript_segments[1].content
        == "several systems."
    )

    assert (
        transcript_segments[0].start_time
        == 0.0
    )

    assert (
        transcript_segments[0].end_time
        == 2.1
    )

    assert (
        transcript_segments[1].start_time
        == 2.1
    )

    assert (
        transcript_segments[1].end_time
        == 3.8
    )

    assert len(resolver.requests) == 1

    assert resolver.requests[0] is request

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
    Verify ASR enrichment adds transcript segments
    without destroying the original AUDIO structural segment.
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

    assert len(audio_segments) == 1

    assert audio_segments[0].content == ""

    assert len(transcript_segments) == 2

    assert (
        canonical_content.metadata["asr"]["status"]
        == "completed"
    )


def test_asr_metadata_survives_canonical_normalization() -> None:
    """
    Verify ASR metadata survives the complete pipeline and
    canonical normalization.
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

    assert "asr" in canonical_content.metadata

    asr_metadata = canonical_content.metadata["asr"]

    assert asr_metadata["status"] == "completed"
    assert asr_metadata["provider"] == "fake-asr"
    assert asr_metadata["language"] == "eng"
    assert asr_metadata["confidence"] == 0.94
    assert asr_metadata["segment_count"] == 2
    assert asr_metadata["model"] == "fake-model"


# ============================================================
# VIDEO INTEGRATION TESTS — A4.5.7.5
# ============================================================


def test_video_pipeline_produces_asr_enriched_canonical_content() -> None:
    """
    Verify the complete VIDEO ingestion path:

        IngestionRequest
            ↓
        Input Detection
            ↓
        VideoDocumentProcessor
            ↓
        VideoInfo / FFprobe
            ↓
        ContentEnrichmentService
            ↓
        VideoASRService
            ↓
        FFmpegAudioExtractor
            ↓
        ASRProvider
            ↓
        TRANSCRIPT blocks
            ↓
        ContentNormalizer
            ↓
        CanonicalContent
    """

    video_bytes = get_sample_video()

    resolver = FakeContentResolver(
        video_bytes
    )

    video_asr_provider = FakeVideoASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        video_asr_provider=video_asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.VIDEO,
        title="Test Video",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
        metadata={
            "asr_language": "en",
        },
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    # ---------------------------------------------------------
    # Canonical title
    # ---------------------------------------------------------

    assert canonical_content.title == "Test Video"

    # ---------------------------------------------------------
    # Original VIDEO structural block
    # ---------------------------------------------------------

    video_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.VIDEO
    ]

    assert len(video_blocks) == 1

    assert video_blocks[0].content == ""

    # ---------------------------------------------------------
    # Generated transcript block
    # ---------------------------------------------------------

    transcript_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 1

    transcript = transcript_blocks[0]

    assert (
        transcript.content
        == "This is a test video transcript."
    )

    assert transcript.start_time == 0.5
    assert transcript.end_time == 2.5

    assert (
    transcript.metadata["source"]
    == "fake-video-asr"
)

    assert (
        transcript.metadata["provider"]
        == "fake-video-asr"
    )

    assert (
        transcript.metadata["language"]
        == "en"
    )

    assert (
        transcript.metadata["confidence"]
        == 0.97
    )

    # ---------------------------------------------------------
    # Canonical text
    # ---------------------------------------------------------

    assert (
        canonical_content.text
        == "This is a test video transcript."
    )

    # ---------------------------------------------------------
    # Video ASR metadata
    # ---------------------------------------------------------

    assert "video_asr" in canonical_content.metadata

    video_asr_metadata = (
        canonical_content.metadata["video_asr"]
    )

    assert (
        video_asr_metadata["status"]
        == "completed"
    )

    # ---------------------------------------------------------
    # Resolver integration
    # ---------------------------------------------------------

    assert len(resolver.requests) == 1

    assert resolver.requests[0] is request

    # ---------------------------------------------------------
    # ASR provider integration
    # ---------------------------------------------------------

    assert len(video_asr_provider.requests) == 1

    asr_request = video_asr_provider.requests[0]

    assert asr_request.audio

    assert (
        asr_request.language
        == "en"
    )

    assert (
        asr_request.metadata["source"]
        == "video"
    )

    assert (
        asr_request.metadata["video_filename"]
        == "sample_with_audio.mp4"
    )


def test_video_pipeline_preserves_original_video_segment() -> None:
    """
    Verify video ASR enrichment does not destroy the original
    VIDEO structural segment.
    """

    video_bytes = get_sample_video()

    resolver = FakeContentResolver(
        video_bytes
    )

    video_asr_provider = FakeVideoASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        video_asr_provider=video_asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.VIDEO,
        title="Video Preservation Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    video_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.VIDEO
    ]

    assert len(video_blocks) == 1

    video_block = video_blocks[0]

    assert (
        video_block.block_type
        == ContentBlockType.VIDEO
    )

    assert video_block.content == ""

    assert (
        video_block.metadata["media_type"]
        == "video"
    )

    assert video_block.metadata["format_name"]

    assert (
        video_block.metadata["duration_seconds"]
        > 0
    )

    assert (
        video_block.metadata["size_bytes"]
        == len(video_bytes)
    )

    assert video_block.metadata["content_hash"]

    # Transcript should be added separately.
    transcript_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type
        == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 1


def test_video_pipeline_preserves_video_metadata_after_normalization() -> None:
    """
    Verify video inspection metadata and video-ASR metadata
    survive canonical normalization.
    """

    video_bytes = get_sample_video()

    resolver = FakeContentResolver(
        video_bytes
    )

    video_asr_provider = FakeVideoASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        video_asr_provider=video_asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.VIDEO,
        title="Video Metadata Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    # ---------------------------------------------------------
    # Video metadata
    # ---------------------------------------------------------

    assert "video" in canonical_content.metadata

    video_metadata = (
        canonical_content.metadata["video"]
    )

    assert (
        video_metadata["format_name"]
    )

    assert (
        video_metadata["duration_seconds"]
        > 0
    )

    assert (
        video_metadata["size_bytes"]
        == len(video_bytes)
    )

    assert (
        video_metadata["content_hash"]
    )

    assert (
        video_metadata["inspection_provider"]
        == "ffprobe"
    )

    assert (
        video_metadata["video_info"]
    )

    # ---------------------------------------------------------
    # Video ASR metadata
    # ---------------------------------------------------------

    assert (
        canonical_content.metadata["video_asr"]["status"]
        == "completed"
    )

    audio_extraction = (
        canonical_content.metadata["video_asr"][
            "audio_extraction"
        ]
    )

    assert (
        audio_extraction["status"]
        == "completed"
    )

    assert (
        audio_extraction["provider"]
        == "ffmpeg"
    )

    assert (
        audio_extraction["sample_rate"]
        == 16_000
    )

    assert (
        audio_extraction["channels"]
        == 1
    )

    assert (
        audio_extraction["audio_format"]
        == "wav"
    )

    assert (
        audio_extraction["size_bytes"]
        > 0
    )


def test_video_pipeline_preserves_asr_timestamps_after_normalization() -> None:
    """
    Verify video ASR timestamps survive:

        VideoASRService
            ↓
        ContentEnrichmentService
            ↓
        ContentNormalizer
    """

    video_bytes = get_sample_video()

    resolver = FakeContentResolver(
        video_bytes
    )

    video_asr_provider = FakeVideoASRProvider(
        ASRResult(
            status=ASRStatus.COMPLETED,
            text="Timestamp preservation test.",
            segments=[
                ASRSegment(
                    text="Timestamp preservation test.",
                    start_time=1.25,
                    end_time=4.75,
                    confidence=0.96,
                    metadata={
                        "test": "timestamp",
                    },
                ),
            ],
            language="en",
            confidence=0.96,
            provider="fake-video-asr",
            metadata={
                "model": "fake-video-model",
            },
        )
    )

    pipeline = create_test_pipeline(
        resolver=resolver,
        video_asr_provider=video_asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.VIDEO,
        title="Timestamp Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    transcript_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type
        == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 1

    transcript = transcript_blocks[0]

    assert transcript.start_time == 1.25
    assert transcript.end_time == 4.75

    assert (
        transcript.metadata["test"]
        == "timestamp"
    )

    assert (
        transcript.metadata["provider"]
        == "fake-video-asr"
    )


def test_video_pipeline_passes_extracted_audio_to_asr() -> None:
    """
    Verify that the real FFmpeg audio extraction layer feeds
    actual extracted WAV bytes into the ASR provider.

    This test does not invoke a real speech model.
    """

    video_bytes = get_sample_video()

    resolver = FakeContentResolver(
        video_bytes
    )

    video_asr_provider = FakeVideoASRProvider()

    pipeline = create_test_pipeline(
        resolver=resolver,
        video_asr_provider=video_asr_provider,
    )

    request = IngestionRequest(
        input_type=InputType.VIDEO,
        title="Audio Extraction Test",
        filename="sample_with_audio.mp4",
        mime_type="video/mp4",
        content=video_bytes,
    )

    canonical_content = asyncio.run(
        pipeline.run(request)
    )

    assert (
        canonical_content.metadata[
            "video_asr"
        ]["status"]
        == "completed"
    )

    assert len(video_asr_provider.requests) == 1

    extracted_audio = (
        video_asr_provider.requests[0].audio
    )

    assert extracted_audio

    # WAV RIFF/WAVE signature.
    assert extracted_audio[:4] == b"RIFF"
    assert extracted_audio[8:12] == b"WAVE"

    # The extracted audio should be materially different
    # from the original MP4 container.
    assert extracted_audio != video_bytes