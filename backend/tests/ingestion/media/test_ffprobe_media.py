from __future__ import annotations

import io
import shutil
import wave

import pytest

from app.ingestion.media import FFprobeMediaInspector


def create_test_wav(
    *,
    duration_seconds: int = 1,
    sample_rate: int = 16_000,
    channels: int = 1,
) -> bytes:
    """Create a small valid PCM WAV entirely in memory."""

    buffer = io.BytesIO()

    sample_width = 2
    frame_count = sample_rate * duration_seconds

    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)

        # Silence.
        wav.writeframes(
            b"\x00\x00" * channels * frame_count
        )

    return buffer.getvalue()


@pytest.mark.asyncio
async def test_ffprobe_inspects_valid_wav():
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")

    media = create_test_wav()

    inspector = FFprobeMediaInspector()

    result = await inspector.inspect(
        media,
        filename="test.wav",
        mime_type="audio/wav",
    )

    assert result.format_name is not None
    assert "wav" in result.format_name

    assert result.codec_name is not None
    assert result.duration_seconds is not None
    assert result.duration_seconds == pytest.approx(1.0, abs=0.05)

    assert result.sample_rate == 16_000
    assert result.channels == 1

    assert result.size_bytes == len(media)
    assert result.metadata["mime_type"] == "audio/wav"


@pytest.mark.asyncio
async def test_ffprobe_supports_stereo_audio():
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")

    media = create_test_wav(
        duration_seconds=2,
        sample_rate=44_100,
        channels=2,
    )

    inspector = FFprobeMediaInspector()

    result = await inspector.inspect(
        media,
        filename="stereo.wav",
        mime_type="audio/wav",
    )

    assert result.duration_seconds == pytest.approx(
        2.0,
        abs=0.05,
    )
    assert result.sample_rate == 44_100
    assert result.channels == 2


@pytest.mark.asyncio
async def test_ffprobe_rejects_empty_media():
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")

    inspector = FFprobeMediaInspector()

    with pytest.raises(ValueError, match="must not be empty"):
        await inspector.inspect(b"")


@pytest.mark.asyncio
async def test_ffprobe_rejects_invalid_media():
    if shutil.which("ffprobe") is None:
        pytest.skip("ffprobe is not installed")

    inspector = FFprobeMediaInspector()

    with pytest.raises(ValueError, match="FFprobe could not inspect"):
        await inspector.inspect(
            b"this is not a real media file",
            filename="fake.wav",
            mime_type="audio/wav",
        )


@pytest.mark.asyncio
async def test_ffprobe_requires_bytes():
    inspector = FFprobeMediaInspector()

    with pytest.raises(TypeError, match="media must be bytes"):
        await inspector.inspect("not bytes")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_ffprobe_missing_executable():
    inspector = FFprobeMediaInspector(
        executable_path="C:\\definitely\\missing\\ffprobe.exe",
    )

    with pytest.raises(
        FileNotFoundError,
        match="FFprobe executable not found",
    ):
        await inspector.inspect(
            create_test_wav(),
            filename="test.wav",
        )


def test_ffprobe_name():
    inspector = FFprobeMediaInspector()

    assert inspector.name == "ffprobe"