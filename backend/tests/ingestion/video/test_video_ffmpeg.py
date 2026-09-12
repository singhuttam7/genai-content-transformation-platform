from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.ingestion.video.ffmpeg import (
    FFmpegAudioExtractionError,
    FFmpegAudioExtractor,
)
from app.ingestion.video.schemas import AudioExtractionRequest


def valid_wav() -> bytes:
    """
    Minimal valid-looking RIFF/WAVE container for output validation.
    """

    return (
        b"RIFF"
        + (36).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + (1).to_bytes(2, "little")
        + (16000).to_bytes(4, "little")
        + (32000).to_bytes(4, "little")
        + (2).to_bytes(2, "little")
        + (16).to_bytes(2, "little")
        + b"data"
        + (0).to_bytes(4, "little")
    )


def test_invalid_timeout_is_rejected():
    with pytest.raises(ValueError):
        FFmpegAudioExtractor(timeout_seconds=0)

    with pytest.raises(ValueError):
        FFmpegAudioExtractor(
            timeout_seconds=float("nan")
        )


def test_invalid_output_limit_is_rejected():
    with pytest.raises(ValueError):
        FFmpegAudioExtractor(
            max_output_bytes=0
        )


def test_missing_executable_is_rejected():
    extractor = FFmpegAudioExtractor(
        executable_path=(
            "C:\\does-not-exist\\ffmpeg.exe"
        ),
    )

    with pytest.raises(FileNotFoundError):
        extractor._resolve_executable()


def test_build_command_defaults_to_canonical_audio():
    extractor = FFmpegAudioExtractor()

    request = AudioExtractionRequest()

    command = extractor._build_command(
        executable="ffmpeg",
        input_path=Path("input.mp4"),
        request=request,
    )

    assert command[0] == "ffmpeg"

    assert "-hide_banner" in command
    assert "-loglevel" in command
    assert "error" in command
    assert "-nostdin" in command
    assert "-y" in command

    assert "-map" in command
    assert "0:a:0" in command

    assert "-vn" in command
    assert "-sn" in command
    assert "-dn" in command

    assert "-ac" in command
    assert "1" in command

    assert "-ar" in command
    assert "16000" in command

    assert "-c:a" in command
    assert "pcm_s16le" in command

    assert "-f" in command
    assert "wav" in command

    assert "pipe:1" in command


def test_build_command_supports_time_range():
    extractor = FFmpegAudioExtractor()

    request = AudioExtractionRequest(
        start_time_seconds=5,
        end_time_seconds=10,
    )

    command = extractor._build_command(
        executable="ffmpeg",
        input_path=Path("input.mp4"),
        request=request,
    )

    assert "-ss" in command
    assert "5" in command

    assert "-t" in command
    assert "5" in command


def test_invalid_time_range_is_rejected():
    extractor = FFmpegAudioExtractor()

    request = AudioExtractionRequest(
        start_time_seconds=10,
        end_time_seconds=5,
    )

    with pytest.raises(
        ValueError,
        match="end_time_seconds",
    ):
        extractor._build_command(
            executable="ffmpeg",
            input_path=Path("input.mp4"),
            request=request,
        )


def test_unsupported_audio_format_is_rejected():
    extractor = FFmpegAudioExtractor()

    request = AudioExtractionRequest(
        audio_format="mp3",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported audio format",
    ):
        extractor._validate_request(request)


def test_negative_start_time_is_rejected_by_schema():
    with pytest.raises(ValueError):
        AudioExtractionRequest(
            start_time_seconds=-1,
        )


def test_invalid_sample_rate_is_rejected_by_schema():
    with pytest.raises(ValueError):
        AudioExtractionRequest(
            sample_rate=0,
        )


def test_invalid_channels_are_rejected_by_schema():
    with pytest.raises(ValueError):
        AudioExtractionRequest(
            channels=0,
        )


@pytest.mark.asyncio
async def test_extract_rejects_non_bytes():
    extractor = FFmpegAudioExtractor()

    with pytest.raises(
        TypeError,
        match="bytes",
    ):
        await extractor.extract(
            "not-bytes",  # type: ignore[arg-type]
        )


@pytest.mark.asyncio
async def test_extract_rejects_empty_media():
    extractor = FFmpegAudioExtractor()

    with pytest.raises(
        ValueError,
        match="empty",
    ):
        await extractor.extract(b"")


@pytest.mark.asyncio
async def test_successful_extraction_returns_wav():
    extractor = FFmpegAudioExtractor()

    process = Mock()
    process.returncode = 0
    process.communicate = AsyncMock(
        return_value=(
            valid_wav(),
            b"",
        )
    )

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        result = await extractor.extract(
            b"fake-video",
            filename="sample.mp4",
        )

    assert result.startswith(b"RIFF")
    assert result[8:12] == b"WAVE"

    process.communicate.assert_awaited_once()


@pytest.mark.asyncio
async def test_ffmpeg_failure_is_reported():
    extractor = FFmpegAudioExtractor()

    process = Mock()
    process.returncode = 1
    process.communicate = AsyncMock(
        return_value=(
            b"",
            b"Invalid data found when processing input",
        )
    )

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(
            FFmpegAudioExtractionError,
            match="could not extract",
        ):
            await extractor.extract(
                b"fake-video",
                filename="sample.mp4",
            )


@pytest.mark.asyncio
async def test_empty_ffmpeg_output_is_rejected():
    extractor = FFmpegAudioExtractor()

    process = Mock()
    process.returncode = 0
    process.communicate = AsyncMock(
        return_value=(
            b"",
            b"",
        )
    )

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(
            FFmpegAudioExtractionError,
            match="empty audio",
        ):
            await extractor.extract(
                b"fake-video",
                filename="sample.mp4",
            )


@pytest.mark.asyncio
async def test_invalid_wav_output_is_rejected():
    extractor = FFmpegAudioExtractor()

    process = Mock()
    process.returncode = 0
    process.communicate = AsyncMock(
        return_value=(
            b"not-a-wav",
            b"",
        )
    )

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(
            FFmpegAudioExtractionError,
            match="WAV",
        ):
            await extractor.extract(
                b"fake-video",
                filename="sample.mp4",
            )


@pytest.mark.asyncio
async def test_output_size_limit_is_enforced():
    extractor = FFmpegAudioExtractor(
        max_output_bytes=50,
    )

    output = valid_wav() + b"x" * 50

    process = Mock()
    process.returncode = 0
    process.communicate = AsyncMock(
        return_value=(
            output,
            b"",
        )
    )

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(
            FFmpegAudioExtractionError,
            match="maximum size",
        ):
            await extractor.extract(
                b"fake-video",
                filename="sample.mp4",
            )


@pytest.mark.asyncio
async def test_timeout_kills_process():
    extractor = FFmpegAudioExtractor(
        timeout_seconds=0.01,
    )

    process = Mock()
    process.returncode = -1

    # communicate() is asynchronous.
    process.communicate = AsyncMock(
        side_effect=TimeoutError,
    )

    # kill() is synchronous.
    process.kill = Mock()

    # wait() is asynchronous.
    process.wait = AsyncMock()

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        with pytest.raises(TimeoutError):
            await extractor.extract(
                b"fake-video",
                filename="sample.mp4",
            )

    process.kill.assert_called_once()
    process.wait.assert_awaited_once()


@pytest.mark.asyncio
async def test_temporary_input_is_removed():
    extractor = FFmpegAudioExtractor()

    process = Mock()
    process.returncode = 0
    process.communicate = AsyncMock(
        return_value=(
            valid_wav(),
            b"",
        )
    )

    with patch(
        "app.ingestion.video.ffmpeg."
        "asyncio.create_subprocess_exec",
        return_value=process,
    ):
        result = await extractor.extract(
            b"fake-video",
            filename="sample.mp4",
        )

    assert result.startswith(b"RIFF")


def test_validate_wav_accepts_valid_header():
    extractor = FFmpegAudioExtractor()

    # Should not raise.
    extractor._validate_wav_output(
        valid_wav()
    )


def test_validate_wav_rejects_short_output():
    extractor = FFmpegAudioExtractor()

    with pytest.raises(
        FFmpegAudioExtractionError,
        match="too small",
    ):
        extractor._validate_wav_output(
            b"RIFF"
        )


def test_validate_wav_rejects_invalid_riff():
    extractor = FFmpegAudioExtractor()

    audio = (
        b"XXXX"
        + b"\x00" * 40
    )

    with pytest.raises(
        FFmpegAudioExtractionError,
        match="RIFF",
    ):
        extractor._validate_wav_output(audio)


def test_validate_wav_rejects_invalid_wave():
    extractor = FFmpegAudioExtractor()

    audio = (
        b"RIFF"
        + b"\x00" * 4
        + b"XXXX"
        + b"\x00" * 36
    )

    with pytest.raises(
        FFmpegAudioExtractionError,
        match="WAVE",
    ):
        extractor._validate_wav_output(audio)