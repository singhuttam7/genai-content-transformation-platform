from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRStatus,
)


def test_asr_status_values() -> None:
    assert ASRStatus.NOT_REQUESTED.value == "not_requested"
    assert ASRStatus.PROCESSING.value == "processing"
    assert ASRStatus.COMPLETED.value == "completed"
    assert ASRStatus.NO_SPEECH.value == "no_speech"
    assert ASRStatus.FAILED.value == "failed"


def test_asr_request_accepts_bytes() -> None:
    request = ASRRequest(
        audio=b"fake-audio",
        language="eng",
    )

    assert request.audio == b"fake-audio"
    assert request.language == "eng"


def test_asr_request_rejects_string_audio() -> None:
    with pytest.raises(ValidationError):
        ASRRequest(
            audio="fake-audio",
        )


def test_asr_request_defaults_metadata() -> None:
    request = ASRRequest(
        audio=b"audio",
    )

    assert request.metadata == {}


def test_asr_segment() -> None:
    segment = ASRSegment(
        text="Hello world.",
        start_time=0.0,
        end_time=2.5,
        confidence=0.95,
    )

    assert segment.text == "Hello world."
    assert segment.start_time == 0.0
    assert segment.end_time == 2.5
    assert segment.confidence == 0.95


def test_asr_segment_supports_speaker() -> None:
    segment = ASRSegment(
        text="Welcome.",
        start_time=1.0,
        end_time=2.0,
        speaker="speaker_1",
    )

    assert segment.speaker == "speaker_1"


def test_asr_segment_rejects_negative_start_time() -> None:
    with pytest.raises(ValidationError):
        ASRSegment(
            text="Hello",
            start_time=-1,
            end_time=2,
        )


def test_asr_segment_rejects_invalid_confidence() -> None:
    with pytest.raises(ValidationError):
        ASRSegment(
            text="Hello",
            start_time=0,
            end_time=2,
            confidence=1.5,
        )


def test_asr_result_completed() -> None:
    result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Hello world.",
        language="eng",
        confidence=0.94,
        provider="test",
    )

    assert result.status == ASRStatus.COMPLETED
    assert result.text == "Hello world."
    assert result.language == "eng"
    assert result.confidence == 0.94
    assert result.provider == "test"


def test_asr_result_contains_segments() -> None:
    result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Hello world.",
        provider="test",
        segments=[
            ASRSegment(
                text="Hello world.",
                start_time=0,
                end_time=2,
                confidence=0.93,
            )
        ],
    )

    assert len(result.segments) == 1
    assert result.segments[0].text == "Hello world."


def test_asr_result_no_speech() -> None:
    result = ASRResult(
        status=ASRStatus.NO_SPEECH,
        provider="test",
    )

    assert result.status == ASRStatus.NO_SPEECH
    assert result.text == ""
    assert result.segments == []


def test_asr_result_failed() -> None:
    result = ASRResult(
        status=ASRStatus.FAILED,
        provider="test",
        metadata={
            "reason": "provider unavailable",
        },
    )

    assert result.status == ASRStatus.FAILED
    assert result.metadata["reason"] == "provider unavailable"


def test_asr_result_confidence_validation() -> None:
    with pytest.raises(ValidationError):
        ASRResult(
            status=ASRStatus.COMPLETED,
            provider="test",
            confidence=-0.1,
        )