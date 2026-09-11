from __future__ import annotations

import io
import wave

import pytest

from app.ingestion.parsers.audio import AudioDocumentProcessor
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)


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

        wav.writeframes(
            b"\x00\x00" * channels * frame_count
        )

    return buffer.getvalue()


@pytest.mark.asyncio
async def test_audio_processor_creates_audio_block():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="test.wav",
        mime_type="audio/wav",
        content=create_test_wav(),
        title="Test Audio",
    )

    result = await processor.process(request)

    assert result.text == ""
    assert len(result.blocks) == 1

    block = result.blocks[0]

    assert block.block_type == ContentBlockType.AUDIO
    assert block.content == ""
    assert block.order == 0

    assert block.metadata["format_name"] is not None
    assert block.metadata["codec_name"] is not None

    assert block.metadata["duration_seconds"] == pytest.approx(
        1.0,
        abs=0.05,
    )

    assert block.metadata["sample_rate"] == 16_000
    assert block.metadata["channels"] == 1

    assert (
        block.metadata["transcription_status"]
        == "not_requested"
    )


@pytest.mark.asyncio
async def test_audio_processor_preserves_title():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="lecture.wav",
        mime_type="audio/wav",
        content=create_test_wav(),
        title="Operating Systems Lecture",
    )

    result = await processor.process(request)

    assert result.title == "Operating Systems Lecture"


@pytest.mark.asyncio
async def test_audio_processor_rejects_empty_audio():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="empty.wav",
        mime_type="audio/wav",
        content=b"",
    )

    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_audio_processor_rejects_oversized_audio():
    processor = AudioDocumentProcessor(
        max_size_bytes=100,
    )

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="large.wav",
        mime_type="audio/wav",
        content=create_test_wav(),
    )

    with pytest.raises(
        ValueError,
        match="exceeds maximum allowed size",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_audio_processor_rejects_non_audio_input_type():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.TEXT,
        content="hello",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported input type",
    ):
        await processor.process(request)


def test_audio_request_rejects_non_bytes_content():
    with pytest.raises(
        ValueError,
        match=(
            "Binary and document inputs require inline bytes "
            "or a storage reference"
        ),
    ):
        IngestionRequest(
            input_type=InputType.AUDIO,
            filename="test.wav",
            mime_type="audio/wav",
            content="not bytes",
        )


@pytest.mark.asyncio
async def test_audio_processor_rejects_mismatched_extension():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="recording.mp3",
        mime_type="audio/mpeg",
        content=create_test_wav(),
    )

    with pytest.raises(
        ValueError,
        match="extension does not match",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_audio_processor_allows_filename_without_extension():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="recording",
        mime_type="audio/wav",
        content=create_test_wav(),
    )

    result = await processor.process(request)

    assert result.blocks[0].block_type == ContentBlockType.AUDIO


@pytest.mark.asyncio
async def test_audio_processor_rejects_invalid_media():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="fake.wav",
        mime_type="audio/wav",
        content=b"this is not valid audio",
    )

    with pytest.raises(
        ValueError,
        match="FFprobe could not inspect",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_audio_processor_enforces_minimum_duration():
    processor = AudioDocumentProcessor(
        min_duration_seconds=2.0,
    )

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="short.wav",
        mime_type="audio/wav",
        content=create_test_wav(
            duration_seconds=1,
        ),
    )

    with pytest.raises(
        ValueError,
        match="shorter than the minimum allowed duration",
    ):
        await processor.process(request)


@pytest.mark.asyncio
async def test_audio_processor_accepts_stereo_audio():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="stereo.wav",
        mime_type="audio/wav",
        content=create_test_wav(
            duration_seconds=2,
            sample_rate=44_100,
            channels=2,
        ),
    )

    result = await processor.process(request)

    block = result.blocks[0]

    assert block.metadata["sample_rate"] == 44_100
    assert block.metadata["channels"] == 2
    assert block.metadata["duration_seconds"] == pytest.approx(
        2.0,
        abs=0.05,
    )


@pytest.mark.asyncio
async def test_audio_processor_does_not_transcribe():
    processor = AudioDocumentProcessor()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="lecture.wav",
        mime_type="audio/wav",
        content=create_test_wav(),
    )

    result = await processor.process(request)

    assert result.text == ""

    assert (
        result.metadata["asr"]["status"]
        == "not_requested"
    )

    assert (
        result.blocks[0].metadata["transcription_status"]
        == "not_requested"
    )