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
        return_value=b"audio-bytes"
    )

    asr_provider = Mock()
    asr_provider.name = "faster-whisper"
    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.COMPLETED,
            text="Hello world",
            language="en",
            provider="faster-whisper",
        )
    )

    service = VideoASRService(
        audio_extractor=audio_extractor,
        asr_provider=asr_provider,
    )

    return service, audio_extractor, asr_provider


@pytest.mark.asyncio
async def test_audio_extraction_failure_contains_structured_metadata():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract.side_effect = RuntimeError(
        "FFmpeg extraction failed"
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED

    assert result.metadata["stage"] == "audio_extraction"
    assert result.metadata["provider"] == "ffmpeg"
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["retryable"] is False

    assert result.errors == ["FFmpeg extraction failed"]
    assert result.asr_result is None

    asr_provider.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_asr_failure_contains_structured_metadata():
    service, _, asr_provider = create_service()

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

    assert result.metadata["stage"] == "asr"
    assert result.metadata["provider"] == "faster-whisper"
    assert result.metadata["error_type"] == "RuntimeError"
    assert result.metadata["retryable"] is False

    assert result.errors == ["ASR provider unavailable"]
    assert result.asr_result is None


@pytest.mark.asyncio
async def test_empty_audio_has_structured_failure_metadata():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract.return_value = b""

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED

    assert result.metadata["stage"] == "audio_extraction"
    assert result.metadata["provider"] == "ffmpeg"
    assert result.metadata["error_type"] == "EmptyAudioOutput"
    assert result.metadata["retryable"] is False

    assert result.asr_result is None
    asr_provider.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_no_audio_is_not_reported_as_failure():
    service, audio_extractor, asr_provider = create_service()

    video_info = create_video_info()
    video_info.audio = None

    result = await service.transcribe(
        b"video-bytes",
        video_info=video_info,
        filename="silent-video.mp4",
    )

    assert result.status == VideoASRStatus.NO_AUDIO

    assert result.errors == []

    assert result.metadata["stage"] == "audio_capability"
    assert result.metadata["capability"] == "no_audio"

    audio_extractor.extract.assert_not_awaited()
    asr_provider.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_successful_processing_contains_provider_metadata():
    service, _, _ = create_service()

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.COMPLETED

    assert result.metadata["stage"] == "asr"
    assert result.metadata["provider"] == "faster-whisper"
    assert result.metadata["language"] == "en"

    assert result.errors == []

    assert result.audio_extraction["status"] == "completed"
    assert result.audio_extraction["provider"] == "ffmpeg"
    assert result.audio_extraction["size_bytes"] == len(b"audio-bytes")