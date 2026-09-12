from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.speech.schemas import (
    ASRResult,
    ASRStatus,
)
from app.ingestion.video.schemas import (
    AudioExtractionRequest,
    VideoASRRequest,
    VideoASRResult,
    VideoASRStatus,
)


def test_video_asr_status_values():
    assert VideoASRStatus.NOT_REQUESTED.value == "not_requested"
    assert VideoASRStatus.PROCESSING.value == "processing"
    assert VideoASRStatus.COMPLETED.value == "completed"
    assert VideoASRStatus.NO_AUDIO.value == "no_audio"
    assert VideoASRStatus.NO_SPEECH.value == "no_speech"
    assert (
        VideoASRStatus.AUDIO_EXTRACTION_FAILED.value
        == "audio_extraction_failed"
    )
    assert VideoASRStatus.ASR_FAILED.value == "asr_failed"


def test_video_asr_request_defaults():
    request = VideoASRRequest()

    assert request.language is None
    assert request.audio_extraction.sample_rate == 16_000
    assert request.audio_extraction.channels == 1
    assert request.audio_extraction.audio_format == "wav"
    assert request.metadata == {}


def test_video_asr_request_custom_values():
    request = VideoASRRequest(
        language="en",
        audio_extraction=AudioExtractionRequest(
            sample_rate=16_000,
            channels=1,
            audio_format="wav",
        ),
        metadata={
            "source": "video",
        },
    )

    assert request.language == "en"
    assert request.audio_extraction.sample_rate == 16_000
    assert request.metadata["source"] == "video"


def test_video_asr_request_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VideoASRRequest(
            unexpected_field="value",
        )


def test_video_asr_result_defaults():
    result = VideoASRResult(
        status=VideoASRStatus.PROCESSING,
    )

    assert result.status == VideoASRStatus.PROCESSING
    assert result.asr_result is None
    assert result.audio_extraction == {}
    assert result.errors == []
    assert result.metadata == {}


def test_video_asr_result_preserves_asr_result():
    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Hello from the video.",
        language="en",
        confidence=0.95,
        provider="faster_whisper",
    )

    result = VideoASRResult(
        status=VideoASRStatus.COMPLETED,
        asr_result=asr_result,
    )

    assert result.asr_result is not None
    assert result.asr_result.status == ASRStatus.COMPLETED
    assert result.asr_result.text == "Hello from the video."
    assert result.asr_result.language == "en"
    assert result.asr_result.confidence == 0.95
    assert result.asr_result.provider == "faster_whisper"


def test_video_asr_result_supports_audio_failure():
    result = VideoASRResult(
        status=VideoASRStatus.AUDIO_EXTRACTION_FAILED,
        errors=["FFmpeg failed to extract audio."],
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED
    assert result.asr_result is None
    assert result.errors == ["FFmpeg failed to extract audio."]


def test_video_asr_result_supports_no_audio():
    result = VideoASRResult(
        status=VideoASRStatus.NO_AUDIO,
    )

    assert result.status == VideoASRStatus.NO_AUDIO
    assert result.asr_result is None


def test_video_asr_result_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VideoASRResult(
            status=VideoASRStatus.COMPLETED,
            unexpected_field="value",
        )


def test_video_asr_request_has_independent_metadata():
    first = VideoASRRequest()
    second = VideoASRRequest()

    first.metadata["key"] = "value"

    assert second.metadata == {}


def test_video_asr_result_has_independent_mutable_fields():
    first = VideoASRResult(
        status=VideoASRStatus.COMPLETED,
    )
    second = VideoASRResult(
        status=VideoASRStatus.COMPLETED,
    )

    first.errors.append("failure")
    first.metadata["key"] = "value"

    assert second.errors == []
    assert second.metadata == {}