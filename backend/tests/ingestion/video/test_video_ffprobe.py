import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.ingestion.video.ffprobe import FFprobeVideoInspector


def ffprobe_payload(
    *,
    include_audio: bool = True,
) -> dict:
    streams = [
        {
            "index": 0,
            "codec_type": "video",
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30000/1001",
            "avg_frame_rate": "30000/1001",
            "bit_rate": "2500000",
            "pix_fmt": "yuv420p",
        }
    ]

    if include_audio:
        streams.append(
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "48000",
                "channels": 2,
                "bit_rate": "128000",
            }
        )

    return {
        "format": {
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "duration": "12.345",
            "bit_rate": "2700000",
            "size": "1234567",
        },
        "streams": streams,
    }


def test_parse_video_with_audio():
    result = FFprobeVideoInspector._parse_video_info(
        ffprobe_payload(),
        size_bytes=1234567,
        mime_type="video/mp4",
    )

    assert result.format_name == "mov,mp4,m4a,3gp,3g2,mj2"
    assert result.duration_seconds == pytest.approx(12.345)
    assert result.size_bytes == 1234567
    assert result.bitrate == 2700000

    assert result.video is not None
    assert result.video.codec_name == "h264"
    assert result.video.width == 1920
    assert result.video.height == 1080
    assert result.video.frame_rate == pytest.approx(29.97002997)
    assert result.video.bitrate == 2500000
    assert result.video.pixel_format == "yuv420p"

    assert result.audio is not None
    assert result.audio.codec_name == "aac"
    assert result.audio.sample_rate == 48000
    assert result.audio.channels == 2
    assert result.audio.bitrate == 128000


def test_parse_video_without_audio():
    result = FFprobeVideoInspector._parse_video_info(
        ffprobe_payload(include_audio=False),
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.video is not None
    assert result.audio is None


def test_video_stream_is_required():
    payload = ffprobe_payload()
    payload["streams"] = [
        {
            "index": 0,
            "codec_type": "audio",
            "codec_name": "aac",
        }
    ]

    with pytest.raises(ValueError, match="video stream"):
        FFprobeVideoInspector._parse_video_info(
            payload,
            size_bytes=100,
            mime_type="video/mp4",
        )


def test_missing_audio_stream_is_allowed():
    result = FFprobeVideoInspector._parse_video_info(
        ffprobe_payload(include_audio=False),
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.audio is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("30000/1001", 30000 / 1001),
        ("25/1", 25.0),
        ("30", 30.0),
        ("N/A", None),
        ("0/0", None),
        ("invalid", None),
        (None, None),
    ],
)
def test_frame_rate_parsing(value, expected):
    result = FFprobeVideoInspector._parse_frame_rate(value)

    if expected is None:
        assert result is None
    else:
        assert result == pytest.approx(expected)


def test_avg_frame_rate_is_preferred():
    payload = ffprobe_payload()

    payload["streams"][0]["avg_frame_rate"] = "24/1"
    payload["streams"][0]["r_frame_rate"] = "30/1"

    result = FFprobeVideoInspector._parse_video_info(
        payload,
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.video is not None
    assert result.video.frame_rate == 24.0


def test_r_frame_rate_is_used_when_avg_is_missing():
    payload = ffprobe_payload()

    payload["streams"][0]["avg_frame_rate"] = "N/A"
    payload["streams"][0]["r_frame_rate"] = "25/1"

    result = FFprobeVideoInspector._parse_video_info(
        payload,
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.video is not None
    assert result.video.frame_rate == 25.0


def test_na_values_are_normalized():
    payload = ffprobe_payload()

    payload["format"]["duration"] = "N/A"
    payload["format"]["bit_rate"] = "N/A"
    payload["streams"][0]["width"] = "N/A"
    payload["streams"][0]["height"] = "N/A"
    payload["streams"][0]["bit_rate"] = "N/A"
    payload["streams"][0]["pix_fmt"] = "N/A"

    result = FFprobeVideoInspector._parse_video_info(
        payload,
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.duration_seconds is None
    assert result.bitrate is None
    assert result.video is not None
    assert result.video.width is None
    assert result.video.height is None
    assert result.video.bitrate is None
    assert result.video.pixel_format is None


def test_multiple_video_streams_selects_lowest_index():
    payload = ffprobe_payload()

    payload["streams"].insert(
        0,
        {
            "index": 2,
            "codec_type": "video",
            "codec_name": "hevc",
            "width": 3840,
            "height": 2160,
            "r_frame_rate": "60/1",
        },
    )

    result = FFprobeVideoInspector._parse_video_info(
        payload,
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.video is not None
    assert result.video.codec_name == "h264"
    assert result.video.width == 1920


def test_metadata_contains_mime_type_and_stream_count():
    result = FFprobeVideoInspector._parse_video_info(
        ffprobe_payload(),
        size_bytes=100,
        mime_type="video/mp4",
    )

    assert result.metadata["mime_type"] == "video/mp4"
    assert result.metadata["stream_count"] == 2


@pytest.mark.asyncio
async def test_inspect_rejects_non_bytes():
    inspector = FFprobeVideoInspector()

    with pytest.raises(TypeError, match="bytes"):
        await inspector.inspect(
            "not-bytes",  # type: ignore[arg-type]
            filename="sample.mp4",
        )


@pytest.mark.asyncio
async def test_inspect_rejects_empty_media():
    inspector = FFprobeVideoInspector()

    with pytest.raises(ValueError, match="empty"):
        await inspector.inspect(
            b"",
            filename="sample.mp4",
        )


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        FFprobeVideoInspector(timeout_seconds=0)

    with pytest.raises(ValueError):
        FFprobeVideoInspector(timeout_seconds=float("nan"))


def test_missing_executable_is_rejected():
    inspector = FFprobeVideoInspector(
        executable_path="C:\\does-not-exist\\ffprobe.exe",
    )

    with pytest.raises(FileNotFoundError):
        inspector._resolve_executable()


@pytest.mark.asyncio
async def test_invalid_json_from_ffprobe_is_rejected():
    inspector = FFprobeVideoInspector()

    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (
        b"not-json",
        b"",
    )

    with patch(
        "app.ingestion.video.ffprobe.asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(
            ValueError,
            match="invalid JSON",
        ):
            await inspector.inspect(
                b"fake-video",
                filename="sample.mp4",
            )


@pytest.mark.asyncio
async def test_ffprobe_failure_is_rejected():
    inspector = FFprobeVideoInspector()

    process = AsyncMock()
    process.returncode = 1
    process.communicate.return_value = (
        b"",
        b"Invalid data found when processing input",
    )

    with patch(
        "app.ingestion.video.ffprobe.asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(
            ValueError,
            match="could not inspect",
        ):
            await inspector.inspect(
                b"fake-video",
                filename="sample.mp4",
            )


@pytest.mark.asyncio
async def test_temporary_file_is_removed():
    inspector = FFprobeVideoInspector()

    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (
        json.dumps(ffprobe_payload()).encode(),
        b"",
    )

    created_path: Path | None = None

    original_named_temporary_file = (
        __import__(
            "app.ingestion.video.ffprobe",
            fromlist=["tempfile"],
        ).tempfile.NamedTemporaryFile
    )

    def capture_temp_file(*args, **kwargs):
        nonlocal created_path

        temporary_file = original_named_temporary_file(
            *args,
            **kwargs,
        )

        created_path = Path(temporary_file.name)

        return temporary_file

    with patch(
        "app.ingestion.video.ffprobe.asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with patch(
            "app.ingestion.video.ffprobe.tempfile.NamedTemporaryFile",
            side_effect=capture_temp_file,
        ):
            result = await inspector.inspect(
                b"fake-video",
                filename="sample.mp4",
            )

    assert result.video is not None
    assert created_path is not None
    assert not created_path.exists()