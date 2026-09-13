from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.video.schemas import (
    FrameExtractionRequest,
    FrameExtractionResult,
    FrameExtractionStatus,
    VideoFrame,
)


# ============================================================
# FrameExtractionRequest
# ============================================================


def test_frame_extraction_request_defaults() -> None:
    request = FrameExtractionRequest()

    assert request.interval_seconds == 2.0
    assert request.max_frames == 300
    assert request.start_time_seconds == 0.0
    assert request.end_time_seconds is None


def test_frame_extraction_request_rejects_invalid_interval() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            interval_seconds=0,
        )

    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            interval_seconds=-1,
        )


def test_frame_extraction_request_accepts_valid_interval() -> None:
    request = FrameExtractionRequest(
        interval_seconds=1.5,
    )

    assert request.interval_seconds == 1.5


def test_frame_extraction_request_rejects_invalid_max_frames() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            max_frames=0,
        )


def test_frame_extraction_request_accepts_valid_max_frames() -> None:
    request = FrameExtractionRequest(
        max_frames=5000,
    )

    assert request.max_frames == 5000


def test_frame_extraction_request_rejects_negative_start_time() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            start_time_seconds=-1,
        )


def test_frame_extraction_request_accepts_end_time() -> None:
    request = FrameExtractionRequest(
        start_time_seconds=2.0,
        end_time_seconds=10.0,
    )

    assert request.start_time_seconds == 2.0
    assert request.end_time_seconds == 10.0


def test_frame_extraction_request_rejects_invalid_end_time() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            end_time_seconds=0,
        )

    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            end_time_seconds=-1,
        )


def test_frame_extraction_request_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            interval_seconds=1.0,
            unknown_field="not-allowed",
        )


# ============================================================
# VideoFrame
# ============================================================


def test_video_frame_contract() -> None:
    frame = VideoFrame(
        frame_index=0,
        timestamp_seconds=1.5,
        image=b"fake-image",
        width=1280,
        height=720,
    )

    assert frame.frame_index == 0
    assert frame.timestamp_seconds == 1.5
    assert frame.image == b"fake-image"
    assert frame.width == 1280
    assert frame.height == 720
    assert frame.metadata == {}


def test_video_frame_allows_missing_dimensions() -> None:
    frame = VideoFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        image=b"fake-image",
    )

    assert frame.width is None
    assert frame.height is None


def test_video_frame_rejects_negative_index() -> None:
    with pytest.raises(ValidationError):
        VideoFrame(
            frame_index=-1,
            timestamp_seconds=0.0,
            image=b"fake-image",
        )


def test_video_frame_rejects_negative_timestamp() -> None:
    with pytest.raises(ValidationError):
        VideoFrame(
            frame_index=0,
            timestamp_seconds=-1.0,
            image=b"fake-image",
        )


def test_video_frame_rejects_invalid_dimensions() -> None:
    with pytest.raises(ValidationError):
        VideoFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            image=b"fake-image",
            width=0,
        )

    with pytest.raises(ValidationError):
        VideoFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            image=b"fake-image",
            height=0,
        )


def test_video_frame_requires_binary_image() -> None:
    with pytest.raises(ValidationError):
        VideoFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            image="not-bytes",
        )


def test_video_frame_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        VideoFrame(
            frame_index=0,
            timestamp_seconds=0.0,
            image=b"fake-image",
            unknown_field="not-allowed",
        )


# ============================================================
# FrameExtractionResult
# ============================================================


def test_frame_extraction_result_defaults() -> None:
    result = FrameExtractionResult(
        status=FrameExtractionStatus.NO_FRAMES,
    )

    assert result.status == FrameExtractionStatus.NO_FRAMES
    assert result.frames == []
    assert result.total_frames == 0
    assert result.errors == []
    assert result.metadata == {}


def test_frame_extraction_result_with_frames() -> None:
    frame = VideoFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        image=b"fake-image",
        width=640,
        height=360,
    )

    result = FrameExtractionResult(
        status=FrameExtractionStatus.COMPLETED,
        frames=[frame],
        requested_interval_seconds=2.0,
        actual_interval_seconds=2.0,
        start_time_seconds=0.0,
        end_time_seconds=10.0,
        total_frames=1,
    )

    assert result.status == FrameExtractionStatus.COMPLETED
    assert len(result.frames) == 1
    assert result.total_frames == 1
    assert result.requested_interval_seconds == 2.0
    assert result.actual_interval_seconds == 2.0
    assert result.start_time_seconds == 0.0
    assert result.end_time_seconds == 10.0


def test_frame_extraction_result_supports_partial_results() -> None:
    frame = VideoFrame(
        frame_index=0,
        timestamp_seconds=0.0,
        image=b"fake-image",
        width=640,
        height=360,
    )

    result = FrameExtractionResult(
        status=FrameExtractionStatus.PARTIAL,
        frames=[frame],
        total_frames=1,
        errors=[
            "Frame extraction stopped after resource limit."
        ],
    )

    assert result.status == FrameExtractionStatus.PARTIAL
    assert len(result.frames) == 1
    assert result.total_frames == 1
    assert len(result.errors) == 1


def test_frame_extraction_result_rejects_negative_total_frames() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionResult(
            status=FrameExtractionStatus.FAILED,
            total_frames=-1,
        )


def test_frame_extraction_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        FrameExtractionResult(
            status=FrameExtractionStatus.COMPLETED,
            unknown_field="not-allowed",
        )