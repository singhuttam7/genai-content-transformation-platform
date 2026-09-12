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
    audio_extractor.extract = AsyncMock()

    asr_provider = Mock()
    asr_provider.name = "faster-whisper"
    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.COMPLETED,
            text="Should never be called",
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
async def test_extraction_exception_stops_asr_execution():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract.side_effect = RuntimeError(
        "decoder process failed"
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED
    assert result.asr_result is None

    asr_provider.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_empty_extraction_stops_asr_execution():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract.return_value = b""

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED
    assert result.asr_result is None

    asr_provider.transcribe.assert_not_awaited()


@pytest.mark.asyncio
async def test_extracted_audio_is_not_exposed_in_result():
    service, audio_extractor, _ = create_service()

    audio_bytes = b"private-audio-data"
    audio_extractor.extract.side_effect = RuntimeError(
        "extraction failed"
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED

    # VideoASRResult intentionally contains metadata only.
    assert not hasattr(result, "extracted_audio")
    assert audio_bytes not in result.errors


@pytest.mark.asyncio
async def test_extraction_success_is_required_before_asr():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract.return_value = b"valid-audio"

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.COMPLETED

    audio_extractor.extract.assert_awaited_once()
    asr_provider.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_extraction_failure_preserves_failure_context():
    service, audio_extractor, _ = create_service()

    audio_extractor.extract.side_effect = ValueError(
        "unsupported audio codec"
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED

    assert result.errors == ["unsupported audio codec"]

    assert result.metadata["stage"] == "audio_extraction"
    assert result.metadata["provider"] == "ffmpeg"
    assert result.metadata["error_type"] == "ValueError"
    assert result.metadata["retryable"] is False

    assert result.audio_extraction["status"] == "failed"
    assert result.audio_extraction["provider"] == "ffmpeg"