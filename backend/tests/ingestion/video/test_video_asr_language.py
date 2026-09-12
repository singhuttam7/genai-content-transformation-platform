from unittest.mock import AsyncMock, Mock

import pytest

from app.ingestion.speech.schemas import ASRResult, ASRStatus
from app.ingestion.video.asr import VideoASRService
from app.ingestion.video.schemas import (
    AudioStreamInfo,
    VideoASRRequest,
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
    audio_extractor.extract = AsyncMock(return_value=b"audio-bytes")

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
async def test_explicit_language_is_propagated_to_asr():
    service, _, asr_provider = create_service()

    request = VideoASRRequest(language="en")

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=request,
        filename="sample.mp4",
    )

    assert result.status.value == "completed"

    asr_provider.transcribe.assert_awaited_once()

    asr_request = asr_provider.transcribe.await_args.args[0]

    assert asr_request.language == "en"


@pytest.mark.asyncio
async def test_hindi_language_is_propagated():
    service, _, asr_provider = create_service()

    request = VideoASRRequest(language="hi")

    await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=request,
        filename="sample.mp4",
    )

    asr_request = asr_provider.transcribe.await_args.args[0]

    assert asr_request.language == "hi"


@pytest.mark.asyncio
async def test_none_language_allows_provider_auto_detection():
    service, _, asr_provider = create_service()

    request = VideoASRRequest(language=None)

    await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=request,
        filename="sample.mp4",
    )

    asr_request = asr_provider.transcribe.await_args.args[0]

    assert asr_request.language is None


@pytest.mark.asyncio
async def test_asr_result_language_is_preserved():
    service, _, asr_provider = create_service()

    asr_provider.transcribe.return_value = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Namaste",
        language="hi",
        provider="faster-whisper",
    )

    request = VideoASRRequest(language="hi")

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=request,
        filename="sample.mp4",
    )

    assert result.asr_result is not None
    assert result.asr_result.language == "hi"