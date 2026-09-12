from pathlib import Path

import pytest

from app.ingestion.content_resolver import InputContentResolver
from app.ingestion.factory import create_ingestion_pipeline
from app.ingestion.schemas import (
    ContentBlockType,
    IngestionRequest,
    InputType,
)
from app.ingestion.speech.faster_whisper import FasterWhisperASRProvider


class FileContentResolver:
    """Test resolver that returns inline request content."""

    async def resolve(self, request: IngestionRequest) -> bytes:
        if not isinstance(request.content, bytes):
            raise TypeError("Expected inline binary content.")

        return request.content


@pytest.mark.real_asr
@pytest.mark.asyncio
async def test_real_audio_flows_into_canonical_content():
    audio_path = Path(r".\test_data\asr\sample_en.wav")
    audio = audio_path.read_bytes()

    request = IngestionRequest(
        input_type=InputType.AUDIO,
        filename="sample_en.wav",
        mime_type="audio/wav",
        content=audio,
        metadata={
            "asr_language": "en",
        },
    )

    resolver: InputContentResolver = FileContentResolver()

    asr_provider = FasterWhisperASRProvider(
        model_name="small",
        device="cpu",
        compute_type="int8",
    )

    pipeline = create_ingestion_pipeline(
        content_resolver=resolver,
        asr_provider=asr_provider,
    )

    canonical_content = await pipeline.run(request)

    expected_text = (
        "Artificial intelligence is transforming communication. "
        "Organizations are using AI to create content faster."
    )

    assert canonical_content.text.strip() == expected_text

    assert canonical_content.language == "en"

    transcript_blocks = [
        block
        for block in canonical_content.segments
        if block.block_type == ContentBlockType.TRANSCRIPT
    ]

    assert len(transcript_blocks) == 2

    assert transcript_blocks[0].content.strip() == (
        "Artificial intelligence is transforming communication."
    )

    assert transcript_blocks[1].content.strip() == (
        "Organizations are using AI to create content faster."
    )

    assert transcript_blocks[0].start_time is not None
    assert transcript_blocks[0].end_time is not None

    assert transcript_blocks[1].start_time is not None
    assert transcript_blocks[1].end_time is not None

    assert (
        transcript_blocks[0].start_time
        < transcript_blocks[0].end_time
    )

    assert (
        transcript_blocks[1].start_time
        < transcript_blocks[1].end_time
    )

    assert canonical_content.metadata["asr"]["status"] == "completed"

    assert (
        canonical_content.metadata["asr"]["provider"]
        == "faster_whisper"
    )

    assert (
        canonical_content.metadata["asr"]["segment_count"]
        == 2
    )