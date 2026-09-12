from __future__ import annotations

from typing import Any

import pytest

from app.ingestion.enrichment.service import ContentEnrichmentService
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)
from app.ingestion.speech.schemas import (
    ASRResult,
    ASRSegment,
    ASRStatus,
)
from app.ingestion.video.schemas import (
    VideoASRResult,
    VideoASRStatus,
    VideoInfo,
)


# ============================================================
# Test Doubles
# ============================================================


class FakeContentResolver:
    def __init__(
        self,
        content: bytes = b"fake-video-bytes",
    ):
        self.content = content
        self.calls = 0

    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        self.calls += 1
        return self.content


class FailingContentResolver:
    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        raise RuntimeError(
            "video source unavailable"
        )


class FakeVideoASRService:
    def __init__(
        self,
        result: VideoASRResult,
    ):
        self.result = result
        self.calls: list[
            dict[str, Any]
        ] = []

    async def transcribe(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: Any = None,
        filename: str | None = None,
    ) -> VideoASRResult:
        self.calls.append(
            {
                "media": media,
                "video_info": video_info,
                "request": request,
                "filename": filename,
            }
        )

        return self.result


class RaisingVideoASRService:
    async def transcribe(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: Any = None,
        filename: str | None = None,
    ) -> VideoASRResult:
        raise RuntimeError(
            "video ASR crashed"
        )


# ============================================================
# Fixtures / Helpers
# ============================================================


def make_video_info() -> VideoInfo:
    """
    Build a valid VideoInfo according to the actual
    project schema.

    VideoInfo contains:
        - format_name
        - duration_seconds
        - size_bytes
        - bitrate
        - video
        - audio
        - metadata
    """

    return VideoInfo(
        format_name="mp4",
        duration_seconds=8.153,
        size_bytes=123456,
        bitrate=256000,
        video={
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "frame_rate": 30.0,
            "bitrate": 200000,
            "pixel_format": "yuv420p",
            "metadata": {},
        },
        audio={
            "codec_name": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
            "bitrate": 56000,
            "metadata": {},
        },
        metadata={},
    )


def make_video_content() -> ExtractedContent:
    video_info = make_video_info()

    source = SourceReference(
        source_id=None,
        source_type=InputType.VIDEO,
        title="sample video",
        filename="sample.mp4",
        mime_type="video/mp4",
        content_hash="abc123",
        storage_uri=None,
    )

    return ExtractedContent(
        source=source,
        title="sample video",
        language=None,
        text="",
        blocks=[
            ContentBlock(
                block_type=ContentBlockType.VIDEO,
                content="",
                order=0,
                metadata={
                    "media_type": "video",
                    "format_name": "mp4",
                    "duration_seconds": 8.153,
                },
            )
        ],
        metadata={
            "video": {
                "format_name": "mp4",
                "duration_seconds": 8.153,
                "size_bytes": 123456,
                "content_hash": "abc123",
                "inspection_provider": "ffprobe",
                "video_stream_count": 1,
                "audio_stream_count": 1,
                "video_info": video_info.model_dump(
                    mode="json"
                ),
            }
        },
    )


def make_video_request() -> IngestionRequest:
    return IngestionRequest(
        input_type=InputType.VIDEO,
        title="sample video",
        filename="sample.mp4",
        mime_type="video/mp4",
        content=b"fake-video-bytes",
        metadata={
            "asr_language": "en",
        },
    )


def make_completed_asr_result() -> ASRResult:
    return ASRResult(
        status=ASRStatus.COMPLETED,
        text="hello from the video",
        segments=[
            ASRSegment(
                text="hello",
                start_time=0.0,
                end_time=1.25,
                confidence=0.95,
            ),
            ASRSegment(
                text="from the video",
                start_time=1.25,
                end_time=3.40,
                confidence=0.91,
            ),
        ],
        language="en",
        confidence=0.93,
        provider="faster-whisper",
        metadata={
            "model": "small",
        },
    )


def make_completed_video_asr_result() -> VideoASRResult:
    return VideoASRResult(
        status=VideoASRStatus.COMPLETED,
        asr_result=make_completed_asr_result(),
        audio_extraction={
            "status": "completed",
            "sample_rate": 16000,
            "channels": 1,
        },
        metadata={
            "stage": "video_asr",
            "provider": "faster-whisper",
        },
    )


def make_no_audio_video_asr_result() -> VideoASRResult:
    return VideoASRResult(
        status=VideoASRStatus.NO_AUDIO,
        asr_result=None,
        metadata={
            "stage": "video_asr",
        },
    )


def make_extraction_failure_result() -> VideoASRResult:
    return VideoASRResult(
        status=VideoASRStatus.AUDIO_EXTRACTION_FAILED,
        asr_result=None,
        errors=[
            "audio extraction failed"
        ],
        metadata={
            "stage": "audio_extraction",
            "retryable": False,
        },
    )


def make_no_speech_result() -> VideoASRResult:
    asr_result = ASRResult(
        status=ASRStatus.NO_SPEECH,
        text="",
        segments=[],
        language="en",
        confidence=None,
        provider="faster-whisper",
        metadata={
            "reason": "no speech detected",
        },
    )

    return VideoASRResult(
        status=VideoASRStatus.NO_SPEECH,
        asr_result=asr_result,
        metadata={
            "stage": "video_asr",
        },
    )


# ============================================================
# Successful Integration
# ============================================================


@pytest.mark.asyncio
async def test_video_enrichment_calls_video_asr_service():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    request = make_video_request()
    content = make_video_content()

    result = await service.enrich(
        request=request,
        content=content,
    )

    assert resolver.calls == 1
    assert len(video_asr.calls) == 1

    call = video_asr.calls[0]

    assert call["media"] == (
        b"fake-video-bytes"
    )

    assert call["filename"] == (
        "sample.mp4"
    )

    assert isinstance(
        call["video_info"],
        VideoInfo,
    )

    assert result is not None


@pytest.mark.asyncio
async def test_video_enrichment_passes_requested_language():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    request = make_video_request()
    content = make_video_content()

    await service.enrich(
        request=request,
        content=content,
    )

    video_asr_request = (
        video_asr.calls[0]["request"]
    )

    assert (
        video_asr_request.language
        == "en"
    )


@pytest.mark.asyncio
async def test_video_enrichment_creates_transcript_blocks():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    transcript_blocks = [
        block
        for block in result.blocks
        if block.block_type
        == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 2

    assert (
        transcript_blocks[0].content
        == "hello"
    )

    assert (
        transcript_blocks[1].content
        == "from the video"
    )


@pytest.mark.asyncio
async def test_video_enrichment_preserves_asr_timestamps():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    transcript_blocks = [
        block
        for block in result.blocks
        if block.block_type
        == ContentBlockType.TRANSCRIPT
    ]

    assert (
        transcript_blocks[0].start_time
        == 0.0
    )

    assert (
        transcript_blocks[0].end_time
        == 1.25
    )

    assert (
        transcript_blocks[1].start_time
        == 1.25
    )

    assert (
        transcript_blocks[1].end_time
        == 3.40
    )


@pytest.mark.asyncio
async def test_video_enrichment_combines_transcript_text():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    assert (
        result.text
        == "hello from the video"
    )


# ============================================================
# Metadata Preservation
# ============================================================


@pytest.mark.asyncio
async def test_video_asr_metadata_is_preserved():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    assert "video_asr" in result.metadata

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "completed"
    )

    assert (
        result.metadata["video_asr"][
            "audio_extraction"
        ]["sample_rate"]
        == 16000
    )

    assert (
        result.metadata["asr"]["provider"]
        == "faster-whisper"
    )


@pytest.mark.asyncio
async def test_original_video_block_is_preserved():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    video_blocks = [
        block
        for block in result.blocks
        if block.metadata.get(
            "media_type"
        )
        == "video"
    ]

    assert len(video_blocks) == 1

    # ---------------------------------------------------------
    # Original video block type
    # ---------------------------------------------------------

    assert (
        video_blocks[0].block_type
        == ContentBlockType.VIDEO
    )

    # ---------------------------------------------------------
    # Original video block ordering
    # ---------------------------------------------------------

    assert (
        video_blocks[0].order
        == 0
    )


# ============================================================
# NO AUDIO
# ============================================================


@pytest.mark.asyncio
async def test_video_with_no_audio_does_not_create_transcript():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_no_audio_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    transcript_blocks = [
        block
        for block in result.blocks
        if block.block_type
        == ContentBlockType.TRANSCRIPT
    ]

    assert transcript_blocks == []

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "no_audio"
    )


# ============================================================
# Audio Extraction Failure
# ============================================================


@pytest.mark.asyncio
async def test_video_audio_extraction_failure_is_non_fatal():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_extraction_failure_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "audio_extraction_failed"
    )

    assert (
        result.blocks[0].metadata[
            "media_type"
        ]
        == "video"
    )


# ============================================================
# NO SPEECH
# ============================================================


@pytest.mark.asyncio
async def test_video_no_speech_is_preserved():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_no_speech_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "no_speech"
    )

    assert (
        result.metadata["asr"]["status"]
        == "no_speech"
    )

    transcript_blocks = [
        block
        for block in result.blocks
        if block.block_type
        == ContentBlockType.TRANSCRIPT
    ]

    assert transcript_blocks == []


# ============================================================
# Resolver Failure
# ============================================================


@pytest.mark.asyncio
async def test_video_resolution_failure_is_non_fatal():
    resolver = FailingContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    original = make_video_content()

    result = await service.enrich(
        request=make_video_request(),
        content=original,
    )

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "asr_failed"
    )

    assert (
        result.metadata["video_asr"][
            "stage"
        ]
        == "video_resolution"
    )

    assert len(result.blocks) == len(
        original.blocks
    )


# ============================================================
# Video ASR Service Exception
# ============================================================


@pytest.mark.asyncio
async def test_video_asr_exception_is_non_fatal():
    resolver = FakeContentResolver()

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=(
            RaisingVideoASRService()
        ),
    )

    result = await service.enrich(
        request=make_video_request(),
        content=make_video_content(),
    )

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "asr_failed"
    )

    assert (
        result.metadata["video_asr"][
            "error_type"
        ]
        == "RuntimeError"
    )


# ============================================================
# Missing VideoInfo
# ============================================================


@pytest.mark.asyncio
async def test_missing_video_info_is_non_fatal():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    content = make_video_content()

    content.metadata["video"].pop(
        "video_info"
    )

    result = await service.enrich(
        request=make_video_request(),
        content=content,
    )

    assert (
        result.metadata["video_asr"][
            "status"
        ]
        == "asr_failed"
    )

    assert (
        result.metadata["video_asr"][
            "error_type"
        ]
        == "MissingVideoInfo"
    )

    assert len(result.blocks) == 1


# ============================================================
# Optional Dependency
# ============================================================


@pytest.mark.asyncio
async def test_video_enrichment_is_noop_when_service_not_configured():
    resolver = FakeContentResolver()

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=None,
    )

    original = make_video_content()

    result = await service.enrich(
        request=make_video_request(),
        content=original,
    )

    assert result == original
    assert resolver.calls == 0


# ============================================================
# Input Isolation / Immutability
# ============================================================


@pytest.mark.asyncio
async def test_video_enrichment_does_not_mutate_original_content():
    resolver = FakeContentResolver()

    video_asr = FakeVideoASRService(
        make_completed_video_asr_result()
    )

    service = ContentEnrichmentService(
        content_resolver=resolver,
        video_asr_service=video_asr,
    )

    original = make_video_content()

    original_metadata = (
        original.metadata.copy()
    )

    original_block_count = len(
        original.blocks
    )

    result = await service.enrich(
        request=make_video_request(),
        content=original,
    )

    assert result is not original

    assert len(original.blocks) == (
        original_block_count
    )

    assert "video_asr" not in (
        original.metadata
    )

    assert original.metadata == (
        original_metadata
    )