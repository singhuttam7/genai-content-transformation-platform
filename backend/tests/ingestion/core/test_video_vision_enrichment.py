from __future__ import annotations

from typing import Any

import pytest

from app.ingestion.enrichment.service import (
    ContentEnrichmentService,
)
from app.ingestion.schemas import (
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)
from app.ingestion.video.schemas import (
    AudioStreamInfo,
    VideoInfo,
    VideoStreamInfo,
    VisionObservation,
    VisionRequest,
    VisionResult,
    VisionStatus,
)


# ============================================================
# Fake dependencies
# ============================================================


class FakeContentResolver:
    """Minimal async content resolver used by enrichment tests."""

    async def resolve(
        self,
        request: IngestionRequest,
    ) -> bytes:
        return b"fake-video-content"


class FakeVideoVisionService:
    """Deterministic async fake VideoVisionService."""

    def __init__(
        self,
        result: VisionResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    async def analyze(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        vision_request: VisionRequest | None = None,
        frame_extraction_request: Any | None = None,
    ) -> VisionResult:
        self.calls.append(
            {
                "media": media,
                "video_info": video_info,
                "vision_request": vision_request,
                "frame_extraction_request": (
                    frame_extraction_request
                ),
            }
        )

        if self.error is not None:
            raise self.error

        if self.result is None:
            raise RuntimeError(
                "FakeVideoVisionService has no configured result."
            )

        return self.result


# ============================================================
# Fixtures / helpers
# ============================================================


def make_video_info() -> VideoInfo:
    """Create deterministic video metadata for tests."""

    return VideoInfo(
        format_name="mp4",
        duration_seconds=12.5,
        size_bytes=1_000_000,
        bitrate=500_000,
        video=VideoStreamInfo(
            codec_name="h264",
            width=1920,
            height=1080,
            frame_rate=30.0,
            bitrate=450_000,
            pixel_format="yuv420p",
        ),
        audio=AudioStreamInfo(
            codec_name="aac",
            sample_rate=48_000,
            channels=2,
            bitrate=50_000,
        ),
        metadata={
            "source": "test",
        },
    )


def make_video_content() -> ExtractedContent:
    """Create valid video ExtractedContent."""

    return ExtractedContent(
        source=SourceReference(
            source_type=InputType.VIDEO,
            filename="vision-test.mp4",
            mime_type="video/mp4",
        ),
        text="",
        blocks=[
            {
                "block_type": ContentBlockType.VIDEO,
                "content": "",
                "order": 0,
                "metadata": {
                    "media_type": "video",
                },
            }
        ],
        metadata={
            "video": {
                "video_info": make_video_info(),
            }
        },
    )


def make_request() -> IngestionRequest:
    """Create a deterministic video ingestion request."""

    return IngestionRequest(
        input_type=InputType.VIDEO,
        filename="vision-test.mp4",
        mime_type="video/mp4",
        content=b"fake-video-content",
    )


def make_vision_result(
    *,
    status: VisionStatus = VisionStatus.COMPLETED,
) -> VisionResult:
    """Create a deterministic vision result."""

    observation = VisionObservation(
        timestamp_seconds=2.5,
        frame_index=3,
        description=(
            "A person is standing in front of "
            "a computer screen."
        ),
        objects=[
            "person",
            "computer",
        ],
        entities=[
            "person",
        ],
        actions=[
            "standing",
        ],
        scene="office",
        visible_text="Security Alert",
        confidence=0.92,
        metadata={
            "frame_source": "test-frame-003",
        },
    )

    if status == VisionStatus.NO_FRAMES:
        return VisionResult(
            status=status,
            observations=[],
            requested_frames=4,
            processed_frames=0,
            failed_frames=0,
            errors=[],
            metadata={
                "reason": "no_frames_available",
            },
        )

    if status == VisionStatus.PARTIAL:
        return VisionResult(
            status=status,
            observations=[observation],
            requested_frames=4,
            processed_frames=3,
            failed_frames=1,
            errors=[
                "Frame 4 could not be processed.",
            ],
            metadata={
                "partial": True,
            },
        )

    if status == VisionStatus.FAILED:
        return VisionResult(
            status=status,
            observations=[],
            requested_frames=4,
            processed_frames=0,
            failed_frames=4,
            errors=[
                "Vision provider failed.",
            ],
            metadata={
                "failure": "provider_error",
            },
        )

    return VisionResult(
        status=status,
        observations=[observation],
        requested_frames=4,
        processed_frames=4,
        failed_frames=0,
        errors=[],
        metadata={
            "provider": "local",
            "model": "gemma3:4b",
        },
    )


def make_service(
    *,
    result: VisionResult | None = None,
    error: Exception | None = None,
) -> tuple[
    ContentEnrichmentService,
    FakeVideoVisionService,
]:
    """Create enrichment service with fake vision dependency."""

    video_vision_service = FakeVideoVisionService(
        result=result,
        error=error,
    )

    service = ContentEnrichmentService(
        content_resolver=FakeContentResolver(),
        video_vision_service=video_vision_service,
    )

    return service, video_vision_service


# ============================================================
# Basic integration
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_service_is_called_for_video_content() -> None:
    result = make_vision_result()

    service, fake_vision = make_service(
        result=result,
    )

    content = make_video_content()
    request = make_request()

    enriched = await service.enrich(
        request=request,
        content=content,
    )

    assert len(fake_vision.calls) == 1

    call = fake_vision.calls[0]

    assert call["media"] == b"fake-video-content"
    assert call["video_info"] == make_video_info()

    assert enriched is not content


@pytest.mark.asyncio
async def test_video_block_is_preserved() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    content = make_video_content()

    enriched = await service.enrich(
        request=make_request(),
        content=content,
    )

    video_blocks = [
        block
        for block in enriched.blocks
        if block.block_type == ContentBlockType.VIDEO
    ]

    assert len(video_blocks) == 1


@pytest.mark.asyncio
async def test_original_content_is_not_mutated() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    content = make_video_content()

    original_block_count = len(content.blocks)

    enriched = await service.enrich(
        request=make_request(),
        content=content,
    )

    assert len(content.blocks) == original_block_count
    assert len(enriched.blocks) > original_block_count


# ============================================================
# Vision observation -> canonical blocks
# ============================================================


@pytest.mark.asyncio
async def test_vision_observation_creates_canonical_block() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    assert len(vision_blocks) == 1


@pytest.mark.asyncio
async def test_visual_block_uses_paragraph_block_type() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    assert vision_blocks[0].block_type == (
        ContentBlockType.PARAGRAPH
    )


@pytest.mark.asyncio
async def test_visual_block_contains_observation_description() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    assert (
        "A person is standing in front of "
        "a computer screen."
        in vision_blocks[0].content
    )


@pytest.mark.asyncio
async def test_visual_block_preserves_timestamp() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    block = vision_blocks[0]

    assert block.start_time == 2.5
    assert block.end_time == 2.5


@pytest.mark.asyncio
async def test_visual_block_preserves_frame_context() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    metadata = vision_blocks[0].metadata

    assert metadata["frame_index"] == 3
    assert metadata["timestamp_seconds"] == 2.5


@pytest.mark.asyncio
async def test_visual_block_preserves_structured_observation_data() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    metadata = vision_blocks[0].metadata

    assert metadata["objects"] == [
        "person",
        "computer",
    ]

    assert metadata["entities"] == [
        "person",
    ]

    assert metadata["actions"] == [
        "standing",
    ]

    assert metadata["scene"] == "office"

    assert metadata["visible_text"] == "Security Alert"

    assert metadata["confidence"] == 0.92


# ============================================================
# Video vision metadata
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_metadata_is_stored() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    assert "video_vision" in enriched.metadata


@pytest.mark.asyncio
async def test_video_vision_metadata_contains_status() -> None:
    result = make_vision_result(
        status=VisionStatus.COMPLETED,
    )

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    metadata = enriched.metadata["video_vision"]

    assert metadata["status"] == (
        VisionStatus.COMPLETED.value
    )


@pytest.mark.asyncio
async def test_video_vision_metadata_contains_processing_counts() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    metadata = enriched.metadata["video_vision"]

    assert metadata["requested_frames"] == 4
    assert metadata["processed_frames"] == 4
    assert metadata["failed_frames"] == 0


# ============================================================
# Request construction
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_request_contains_video_context() -> None:
    result = make_vision_result()

    service, fake_vision = make_service(
        result=result,
    )

    await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_request = fake_vision.calls[0]["vision_request"]

    assert vision_request is not None
    assert vision_request.metadata


@pytest.mark.asyncio
async def test_video_vision_request_contains_video_metadata() -> None:
    result = make_vision_result()

    service, fake_vision = make_service(
        result=result,
    )

    await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_request = fake_vision.calls[0]["vision_request"]

    assert vision_request is not None

    metadata = vision_request.metadata

    assert metadata["source_id"] is None
    assert metadata["filename"] == "vision-test.mp4"
    assert metadata["mime_type"] == "video/mp4"


@pytest.mark.asyncio
async def test_video_info_is_passed_to_video_vision_service() -> None:
    result = make_vision_result()

    service, fake_vision = make_service(
        result=result,
    )

    await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    call = fake_vision.calls[0]

    assert call["video_info"].duration_seconds == 12.5
    assert call["video_info"].video is not None
    assert call["video_info"].video.width == 1920
    assert call["video_info"].video.height == 1080


# ============================================================
# Non-success results
# ============================================================


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status",
    [
        VisionStatus.FAILED,
        VisionStatus.PARTIAL,
        VisionStatus.NO_FRAMES,
    ],
)
async def test_non_success_vision_results_are_non_fatal(
    status: VisionStatus,
) -> None:
    result = make_vision_result(
        status=status,
    )

    service, _ = make_service(
        result=result,
    )

    content = make_video_content()

    enriched = await service.enrich(
        request=make_request(),
        content=content,
    )

    assert enriched is not None
    assert "video_vision" in enriched.metadata


@pytest.mark.asyncio
async def test_partial_result_preserves_available_observations() -> None:
    result = make_vision_result(
        status=VisionStatus.PARTIAL,
    )

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    assert len(vision_blocks) == 1

    metadata = enriched.metadata["video_vision"]

    assert metadata["status"] == (
        VisionStatus.PARTIAL.value
    )
    assert metadata["processed_frames"] == 3
    assert metadata["failed_frames"] == 1


@pytest.mark.asyncio
async def test_no_frames_result_does_not_create_visual_blocks() -> None:
    result = make_vision_result(
        status=VisionStatus.NO_FRAMES,
    )

    service, _ = make_service(
        result=result,
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    vision_blocks = [
        block
        for block in enriched.blocks
        if block.metadata.get("source") == "vision"
    ]

    assert vision_blocks == []


# ============================================================
# Missing video metadata
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_missing_video_info_is_non_fatal() -> None:
    service, fake_vision = make_service(
        result=make_vision_result(),
    )

    content = ExtractedContent(
        source=SourceReference(
            source_type=InputType.VIDEO,
            filename="vision-test.mp4",
            mime_type="video/mp4",
        ),
        text="",
        blocks=[
            {
                "block_type": ContentBlockType.VIDEO,
                "content": "",
                "order": 0,
            }
        ],
        metadata={},
    )

    enriched = await service.enrich(
        request=make_request(),
        content=content,
    )

    assert enriched is not None
    assert len(fake_vision.calls) == 0

    assert "video_vision" in enriched.metadata


# ============================================================
# Provider failure handling
# ============================================================


@pytest.mark.asyncio
async def test_video_vision_provider_failure_is_non_fatal() -> None:
    service, fake_vision = make_service(
        error=RuntimeError(
            "Vision provider unavailable."
        ),
    )

    enriched = await service.enrich(
        request=make_request(),
        content=make_video_content(),
    )

    assert len(fake_vision.calls) == 1

    assert "video_vision" in enriched.metadata

    metadata = enriched.metadata["video_vision"]

    assert metadata["status"] == (
        VisionStatus.FAILED.value
    )


# ============================================================
# Metadata preservation
# ============================================================


@pytest.mark.asyncio
async def test_existing_metadata_is_preserved() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    content = make_video_content()

    content.metadata["custom"] = {
        "important": True,
    }

    enriched = await service.enrich(
        request=make_request(),
        content=content,
    )

    assert enriched.metadata["custom"] == {
        "important": True,
    }


@pytest.mark.asyncio
async def test_existing_blocks_are_preserved() -> None:
    result = make_vision_result()

    service, _ = make_service(
        result=result,
    )

    content = make_video_content()

    original_blocks = list(content.blocks)

    enriched = await service.enrich(
        request=make_request(),
        content=content,
    )

    assert len(enriched.blocks) >= len(
        original_blocks
    )

    for original_block in original_blocks:
        assert original_block in enriched.blocks