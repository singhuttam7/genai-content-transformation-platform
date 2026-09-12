from unittest.mock import AsyncMock, Mock

import pytest

from app.ingestion.speech.schemas import ASRResult, ASRStatus
from app.ingestion.video.asr import VideoASRService
from app.ingestion.video.schemas import (
    AudioStreamInfo,
    VideoASRRequest,
    VideoASRStatus,
    VideoInfo,
    VideoStreamInfo,
)


def create_video_info() -> VideoInfo:
    return VideoInfo(
        format_name="mp4",
        duration_seconds=8.0,
        video=VideoStreamInfo(
            codec_name="h264",
            width=1280,
            height=720,
            frame_rate=30.0,
        ),
        audio=AudioStreamInfo(
            codec_name="aac",
            sample_rate=22050,
            channels=1,
        ),
    )


def create_service():
    audio_extractor = Mock()
    audio_extractor.name = "ffmpeg"
    audio_extractor.extract = AsyncMock(
        return_value=b"valid-audio"
    )

    asr_provider = Mock()
    asr_provider.name = "faster-whisper"
    asr_provider.transcribe = AsyncMock()

    service = VideoASRService(
        audio_extractor=audio_extractor,
        asr_provider=asr_provider,
    )

    return service, audio_extractor, asr_provider


@pytest.mark.asyncio
async def test_asr_exception_returns_structured_failure():
    service, audio_extractor, asr_provider = create_service()

    asr_provider.transcribe.side_effect = RuntimeError(
        "ASR provider unavailable"
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.ASR_FAILED
    assert result.asr_result is None

    assert result.errors == ["ASR provider unavailable"]

    assert result.metadata["stage"] == "asr"
    assert result.metadata["provider"] == "faster-whisper"
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["retryable"] is False

    assert result.audio_extraction["status"] == "completed"
    assert result.audio_extraction["size_bytes"] == len(b"valid-audio")

    audio_extractor.extract.assert_awaited_once()
    asr_provider.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_asr_failed_status_maps_to_video_asr_failed():
    service, _, asr_provider = create_service()

    asr_provider.transcribe.return_value = ASRResult(
        status=ASRStatus.FAILED,
        text="",
        language="en",
        provider="faster-whisper",
        metadata={
            "reason": "model inference failed",
        },
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.ASR_FAILED

    # Provider returned a valid structured result,
    # so preserve it rather than replacing it with None.
    assert result.asr_result is not None
    assert result.asr_result.status == ASRStatus.FAILED
    assert result.asr_result.metadata["reason"] == "model inference failed"

    assert result.errors == []


@pytest.mark.asyncio
async def test_no_speech_is_not_treated_as_asr_failure():
    service, _, asr_provider = create_service()

    asr_provider.transcribe.return_value = ASRResult(
        status=ASRStatus.NO_SPEECH,
        text="",
        language="en",
        provider="faster-whisper",
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.NO_SPEECH

    assert result.asr_result is not None
    assert result.asr_result.status == ASRStatus.NO_SPEECH

    assert result.errors == []


@pytest.mark.asyncio
async def test_processing_status_is_preserved():
    service, _, asr_provider = create_service()

    asr_provider.transcribe.return_value = ASRResult(
        status=ASRStatus.PROCESSING,
        text="",
        language="en",
        provider="faster-whisper",
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.PROCESSING

    assert result.asr_result is not None
    assert result.asr_result.status == ASRStatus.PROCESSING

    assert result.errors == []


@pytest.mark.asyncio
async def test_successful_asr_preserves_full_result():
    service, _, asr_provider = create_service()

    asr_provider.transcribe.return_value = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Hello from the video",
        language="en",
        confidence=0.96,
        provider="faster-whisper",
        metadata={
            "model": "small",
        },
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.COMPLETED

    assert result.asr_result is not None
    assert result.asr_result.text == "Hello from the video"
    assert result.asr_result.language == "en"
    assert result.asr_result.confidence == 0.96
    assert result.asr_result.provider == "faster-whisper"
    assert result.asr_result.metadata["model"] == "small"

    assert result.errors == []