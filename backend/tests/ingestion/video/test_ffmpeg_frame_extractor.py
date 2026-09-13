from pathlib import Path

import pytest

from app.ingestion.video.ffmpeg import (
    FFmpegFrameExtractor,
)
from app.ingestion.video.schemas import (
    FrameExtractionRequest,
    FrameExtractionStatus,
    VideoInfo,
    VideoStreamInfo,
)


SAMPLE_VIDEO = (
    Path(__file__).resolve().parents[3]
    / "test_data"
    / "video"
    / "sample_with_audio.mp4"
)


@pytest.fixture
def video_bytes() -> bytes:
    return SAMPLE_VIDEO.read_bytes()


@pytest.fixture
def video_info() -> VideoInfo:
    return VideoInfo(
        format_name="mov,mp4,m4a,3gp,3g2,mj2",
        duration_seconds=8.133333,
        size_bytes=None,
        bitrate=None,
        video=VideoStreamInfo(
            codec_name="h264",
            width=1280,
            height=720,
            frame_rate=30.0,
        ),
    )


@pytest.mark.asyncio
async def test_extract_default_frames(
    video_bytes: bytes,
    video_info: VideoInfo,
) -> None:
    extractor = FFmpegFrameExtractor()

    result = await extractor.extract(
        video_bytes,
        video_info=video_info,
        filename="sample_with_audio.mp4",
    )

    assert result.status == FrameExtractionStatus.COMPLETED

    assert result.total_frames == 5

    assert len(result.frames) == 5

    assert [
        frame.timestamp_seconds
        for frame in result.frames
    ] == [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
    ]


@pytest.mark.asyncio
async def test_extracted_frames_are_jpeg(
    video_bytes: bytes,
    video_info: VideoInfo,
) -> None:
    extractor = FFmpegFrameExtractor()

    result = await extractor.extract(
        video_bytes,
        video_info=video_info,
        request=FrameExtractionRequest(
            interval_seconds=2.0,
            max_frames=3,
        ),
        filename="sample_with_audio.mp4",
    )

    assert result.status == FrameExtractionStatus.COMPLETED

    for frame in result.frames:
        assert frame.image.startswith(b"\xff\xd8")
        assert frame.image.endswith(b"\xff\xd9")
        assert frame.width == 1280
        assert frame.height == 720


@pytest.mark.asyncio
async def test_max_frames_is_respected(
    video_bytes: bytes,
    video_info: VideoInfo,
) -> None:
    extractor = FFmpegFrameExtractor()

    result = await extractor.extract(
        video_bytes,
        video_info=video_info,
        request=FrameExtractionRequest(
            interval_seconds=1.0,
            max_frames=3,
        ),
        filename="sample_with_audio.mp4",
    )

    assert result.total_frames == 3

    assert [
        frame.timestamp_seconds
        for frame in result.frames
    ] == [
        0.0,
        1.0,
        2.0,
    ]


@pytest.mark.asyncio
async def test_custom_time_range(
    video_bytes: bytes,
    video_info: VideoInfo,
) -> None:
    extractor = FFmpegFrameExtractor()

    result = await extractor.extract(
        video_bytes,
        video_info=video_info,
        request=FrameExtractionRequest(
            interval_seconds=2.0,
            max_frames=10,
            start_time_seconds=2.0,
            end_time_seconds=6.0,
        ),
        filename="sample_with_audio.mp4",
    )

    assert result.status == FrameExtractionStatus.COMPLETED

    assert [
        frame.timestamp_seconds
        for frame in result.frames
    ] == [
        2.0,
        4.0,
        6.0,
    ]


@pytest.mark.asyncio
async def test_empty_video_is_rejected(
    video_info: VideoInfo,
) -> None:
    extractor = FFmpegFrameExtractor()

    with pytest.raises(ValueError):
        await extractor.extract(
            b"",
            video_info=video_info,
        )


@pytest.mark.asyncio
async def test_video_without_video_stream_returns_no_video(
) -> None:
    extractor = FFmpegFrameExtractor()

    result = await extractor.extract(
        b"not-empty",
        video_info=VideoInfo(
            duration_seconds=8.0,
            video=None,
        ),
    )

    assert result.status == FrameExtractionStatus.NO_VIDEO
    assert result.frames == []


def test_invalid_timeout_is_rejected() -> None:
    with pytest.raises(ValueError):
        FFmpegFrameExtractor(
            timeout_seconds=0,
        )