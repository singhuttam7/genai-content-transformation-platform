from __future__ import annotations

import io
import wave

import pytest

from app.ingestion.factory import create_ingestion_pipeline
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
async def test_audio_flows_through_ingestion_pipeline():
    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="lecture.wav",
        mime_type="audio/wav",
        content=create_test_wav(),
        title="Test Lecture",
    )

    result = await pipeline.run(request)

    assert result.title == "Test Lecture"

    assert len(result.segments) == 1

    segment = result.segments[0]

    assert segment.block_type == ContentBlockType.AUDIO

    assert segment.metadata["format_name"] is not None
    assert segment.metadata["codec_name"] is not None

    assert segment.metadata["sample_rate"] == 16_000
    assert segment.metadata["channels"] == 1

    assert (
        segment.metadata["transcription_status"]
        == "not_requested"
    )


@pytest.mark.asyncio
async def test_audio_pipeline_does_not_generate_transcript():
    pipeline = create_ingestion_pipeline()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="lecture.wav",
        mime_type="audio/wav",
        content=create_test_wav(),
    )

    result = await pipeline.run(request)

    assert result.text == ""

    transcript_blocks = [
        block
        for block in result.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    assert transcript_blocks == []