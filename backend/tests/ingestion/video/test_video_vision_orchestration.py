from __future__ import annotations

import pytest

from app.ingestion.video.schemas import (
    FrameExtractionRequest,
    FrameExtractionResult,
    FrameExtractionStatus,
    VideoFrame,
    VideoInfo,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from app.ingestion.video.vision import (
    VisionService,
)
from app.ingestion.video.vision_orchestration import (
    VideoVisionService,
)


# ============================================================
# Test doubles
# ============================================================


class FakeFrameExtractor:
    name = "fake-frame-extractor"

    def __init__(
        self,
        result: FrameExtractionResult,
    ) -> None:
        self.result = result
        self.calls: list[dict[str, object]] = []

    async def extract(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: FrameExtractionRequest | None = None,
    ) -> FrameExtractionResult:
        self.calls.append(
            {
                "media": media,
                "video_info": video_info,
                "request": request,
            }
        )

        return self.result


class RaisingFrameExtractor:
    name = "raising-frame-extractor"

    async def extract(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: FrameExtractionRequest | None = None,
    ) -> FrameExtractionResult:
        raise RuntimeError(
            "frame extraction crashed"
        )


class FakeVisionService:
    def __init__(
        self,
        result: VisionResult,
    ) -> None:
        self.result = result
        self.calls: list[VisionRequest] = []

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        self.calls.append(request)

        return self.result


class RaisingVisionService:
    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        raise RuntimeError(
            "vision service crashed"
        )


# ============================================================
# Helpers
# ============================================================


def make_video_info() -> VideoInfo:
    return VideoInfo(
        format_name="mp4",
        duration_seconds=8.0,
        size_bytes=1000,
        video={
            "codec_name": "h264",
            "width": 1280,
            "height": 720,
            "frame_rate": 30.0,
        },
        audio={
            "codec_name": "aac",
            "sample_rate": 16000,
            "channels": 1,
        },
    )


def make_frame(
    *,
    frame_index: int = 0,
    timestamp_seconds: float = 1.0,
) -> VideoFrame:
    return VideoFrame(
        frame_index=frame_index,
        timestamp_seconds=timestamp_seconds,
        image=b"\xff\xd8\xff\xd9",
        width=1280,
        height=720,
    )


def make_completed_extraction(
    *,
    frame_count: int = 2,
) -> FrameExtractionResult:
    frames = [
        make_frame(
            frame_index=index,
            timestamp_seconds=float(
                index * 2
            ),
        )
        for index in range(frame_count)
    ]

    return FrameExtractionResult(
        status=FrameExtractionStatus.COMPLETED,
        frames=frames,
        requested_interval_seconds=2.0,
        actual_interval_seconds=2.0,
        start_time_seconds=0.0,
        end_time_seconds=8.0,
        total_frames=len(frames),
        metadata={
            "provider": "fake-frame-extractor",
            "requested_frame_count": len(frames),
            "extracted_frame_count": len(frames),
        },
    )


def make_completed_vision_result(
    *,
    frame_count: int = 2,
) -> VisionResult:
    return VisionResult(
        status=VisionStatus.COMPLETED,
        observations=[],
        requested_frames=frame_count,
        processed_frames=frame_count,
        failed_frames=0,
        metadata={
            "provider": "fake-vision-provider",
        },
    )


# ============================================================
# Constructor
# ============================================================


def test_video_vision_requires_vision_service() -> None:
    with pytest.raises(
        ValueError,
        match="Vision service must be provided",
    ):
        VideoVisionService(
            frame_extractor=FakeFrameExtractor(
                make_completed_extraction()
            ),
            vision_service=None,  # type: ignore[arg-type]
        )


def test_video_vision_uses_injected_frame_extractor() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    assert service.frame_extractor is extractor
    assert service.vision_service is vision_service


# ============================================================
# Frame extraction
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_extracts_frames_before_vision() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction(
            frame_count=3
        )
    )

    vision_service = FakeVisionService(
        make_completed_vision_result(
            frame_count=3
        )
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    await service.analyze(
        b"video-bytes",
        video_info=make_video_info(),
    )

    assert len(extractor.calls) == 1
    assert len(vision_service.calls) == 1


@pytest.mark.asyncio
async def test_video_vision_passes_original_media_to_extractor() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    await service.analyze(
        b"original-video",
        video_info=make_video_info(),
    )

    assert extractor.calls[0]["media"] == (
        b"original-video"
    )


@pytest.mark.asyncio
async def test_video_vision_passes_video_info_to_extractor() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    video_info = make_video_info()

    await service.analyze(
        b"video",
        video_info=video_info,
    )

    assert extractor.calls[0]["video_info"] is video_info


@pytest.mark.asyncio
async def test_video_vision_forwards_frame_extraction_request() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    extraction_request = FrameExtractionRequest(
        interval_seconds=1.5,
        max_frames=5,
    )

    await service.analyze(
        b"video",
        video_info=make_video_info(),
        frame_extraction_request=extraction_request,
    )

    assert (
        extractor.calls[0]["request"]
        is extraction_request
    )


# ============================================================
# Vision request
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_builds_vision_request_from_extracted_frames() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction(
            frame_count=3
        )
    )

    vision_service = FakeVisionService(
        make_completed_vision_result(
            frame_count=3
        )
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    request = vision_service.calls[0]

    assert len(request.frames) == 3

    assert [
        frame.frame_index
        for frame in request.frames
    ] == [0, 1, 2]


@pytest.mark.asyncio
async def test_video_vision_preserves_custom_vision_request() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction(
            frame_count=2
        )
    )

    vision_service = FakeVisionService(
        make_completed_vision_result(
            frame_count=2
        )
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    request = VisionRequest(
        prompt="Identify security-relevant objects.",
        detail_level="detailed",
        metadata={
            "source": "si h-video",
        },
    )

    await service.analyze(
        b"video",
        video_info=make_video_info(),
        vision_request=request,
    )

    forwarded = vision_service.calls[0]

    assert (
        forwarded.prompt
        == "Identify security-relevant objects."
    )

    assert forwarded.detail_level == (
        "detailed"
    )

    assert forwarded.metadata["source"] == (
        "si h-video"
    )

    assert len(forwarded.frames) == 2


# ============================================================
# Result propagation
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_returns_vision_result() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    expected = make_completed_vision_result()

    vision_service = FakeVisionService(
        expected
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    assert result.status == (
        VisionStatus.COMPLETED
    )


@pytest.mark.asyncio
async def test_video_vision_preserves_vision_metadata() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    expected = make_completed_vision_result()

    vision_service = FakeVisionService(
        expected
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    assert result.metadata["provider"] == (
        "fake-vision-provider"
    )


@pytest.mark.asyncio
async def test_video_vision_attaches_frame_extraction_metadata() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction(
            frame_count=3
        )
    )

    vision_service = FakeVisionService(
        make_completed_vision_result(
            frame_count=3
        )
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    extraction_metadata = result.metadata[
        "video_vision"
    ]["frame_extraction"]

    assert extraction_metadata["status"] == (
        "completed"
    )

    assert extraction_metadata["total_frames"] == 3

    assert extraction_metadata["provider"] == (
        "fake-frame-extractor"
    )


# ============================================================
# Empty / failed extraction
# ============================================================


@pytest.mark.asyncio
async def test_empty_video_returns_structured_failure() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction()
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"",
        video_info=make_video_info(),
    )

    assert result.status == (
        VisionStatus.FAILED
    )

    assert vision_service.calls == []


@pytest.mark.asyncio
async def test_frame_extraction_failure_does_not_call_vision() -> None:
    extractor = FakeFrameExtractor(
        FrameExtractionResult(
            status=FrameExtractionStatus.FAILED,
            errors=[
                "FFmpeg failed"
            ],
            metadata={
                "provider": "fake-frame-extractor",
                "requested_frame_count": 5,
            },
        )
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    assert result.status == (
        VisionStatus.FAILED
    )

    assert vision_service.calls == []

    assert result.errors == [
        "FFmpeg failed"
    ]


@pytest.mark.asyncio
async def test_no_frames_does_not_call_vision() -> None:
    extractor = FakeFrameExtractor(
        FrameExtractionResult(
            status=FrameExtractionStatus.NO_FRAMES,
            total_frames=0,
            metadata={
                "provider": "fake-frame-extractor",
                "requested_frame_count": 0,
            },
        )
    )

    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    assert result.status == (
        VisionStatus.NO_FRAMES
    )

    assert vision_service.calls == []


# ============================================================
# Exception boundaries
# ============================================================


@pytest.mark.asyncio
async def test_frame_extractor_exception_becomes_structured_failure() -> None:
    vision_service = FakeVisionService(
        make_completed_vision_result()
    )

    service = VideoVisionService(
        frame_extractor=RaisingFrameExtractor(),
        vision_service=vision_service,  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    assert result.status == (
        VisionStatus.FAILED
    )

    assert result.metadata[
        "error_type"
    ] == "RuntimeError"

    assert result.errors == [
        "frame extraction crashed"
    ]

    assert vision_service.calls == []


@pytest.mark.asyncio
async def test_vision_service_exception_becomes_structured_failure() -> None:
    extractor = FakeFrameExtractor(
        make_completed_extraction(
            frame_count=2
        )
    )

    service = VideoVisionService(
        frame_extractor=extractor,
        vision_service=RaisingVisionService(),  # type: ignore[arg-type]
    )

    result = await service.analyze(
        b"video",
        video_info=make_video_info(),
    )

    assert result.status == (
        VisionStatus.FAILED
    )

    assert result.metadata[
        "stage"
    ] == "vision"

    assert result.metadata[
        "error_type"
    ] == "RuntimeError"

    assert result.requested_frames == 2
    assert result.failed_frames == 2