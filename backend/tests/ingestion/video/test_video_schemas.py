import pytest
from pydantic import ValidationError

from app.ingestion.video.schemas import (
    AudioExtractionRequest,
    AudioStreamInfo,
    FrameExtractionRequest,
    VideoFrame,
    VideoInfo,
    VideoProcessingResult,
    VideoProcessingStatus,
    VideoStreamInfo,
)


def test_video_stream_info_accepts_valid_metadata():
    stream = VideoStreamInfo(
        codec_name="h264",
        width=1920,
        height=1080,
        frame_rate=30.0,
        bitrate=2_000_000,
        pixel_format="yuv420p",
    )

    assert stream.codec_name == "h264"
    assert stream.width == 1920
    assert stream.height == 1080
    assert stream.frame_rate == 30.0


def test_audio_stream_info_accepts_valid_metadata():
    stream = AudioStreamInfo(
        codec_name="aac",
        sample_rate=16_000,
        channels=1,
        bitrate=128_000,
    )

    assert stream.codec_name == "aac"
    assert stream.sample_rate == 16_000
    assert stream.channels == 1


def test_video_info_contains_video_and_audio_streams():
    info = VideoInfo(
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=10.5,
        size_bytes=500_000,
        video=VideoStreamInfo(
            codec_name="h264",
            width=1280,
            height=720,
            frame_rate=30,
        ),
        audio=AudioStreamInfo(
            codec_name="aac",
            sample_rate=16_000,
            channels=1,
        ),
    )

    assert info.duration_seconds == 10.5
    assert info.video is not None
    assert info.audio is not None


def test_video_frame_preserves_timestamp_and_bytes():
    frame = VideoFrame(
        timestamp_seconds=4.5,
        frame_index=135,
        image=b"fake-image",
        width=1280,
        height=720,
    )

    assert frame.timestamp_seconds == 4.5
    assert frame.frame_index == 135
    assert frame.image == b"fake-image"


def test_frame_extraction_request_defaults():
    request = FrameExtractionRequest()

    assert request.interval_seconds == 2.0
    assert request.max_frames == 300
    assert request.start_time_seconds == 0.0
    assert request.end_time_seconds is None


def test_audio_extraction_request_defaults():
    request = AudioExtractionRequest()

    assert request.sample_rate == 16_000
    assert request.channels == 1
    assert request.audio_format == "wav"


def test_video_processing_result_supports_partial_results():
    result = VideoProcessingResult(
        status=VideoProcessingStatus.PARTIAL,
        video_info=VideoInfo(
            duration_seconds=10,
        ),
        errors=["Vision analysis failed."],
    )

    assert result.status == VideoProcessingStatus.PARTIAL
    assert result.video_info is not None
    assert len(result.errors) == 1


def test_video_frame_rejects_negative_timestamp():
    with pytest.raises(ValidationError):
        VideoFrame(
            timestamp_seconds=-1,
            frame_index=0,
            image=b"image",
        )


def test_video_frame_rejects_negative_frame_index():
    with pytest.raises(ValidationError):
        VideoFrame(
            timestamp_seconds=0,
            frame_index=-1,
            image=b"image",
        )


def test_frame_extraction_request_rejects_zero_interval():
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            interval_seconds=0,
        )


def test_frame_extraction_request_rejects_zero_max_frames():
    with pytest.raises(ValidationError):
        FrameExtractionRequest(
            max_frames=0,
        )


def test_audio_extraction_request_rejects_invalid_sample_rate():
    with pytest.raises(ValidationError):
        AudioExtractionRequest(
            sample_rate=0,
        )


def test_video_stream_rejects_invalid_dimensions():
    with pytest.raises(ValidationError):
        VideoStreamInfo(
            width=0,
        )


def test_video_info_rejects_negative_duration():
    with pytest.raises(ValidationError):
        VideoInfo(
            duration_seconds=-1,
        )