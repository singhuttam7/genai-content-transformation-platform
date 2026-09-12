from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest

from app.ingestion.speech.schemas import (
    ASRResult,
    ASRSegment,
    ASRStatus,
)
from app.ingestion.video.asr import VideoASRService
from app.ingestion.video.schemas import (
    AudioStreamInfo,
    VideoASRRequest,
    VideoASRStatus,
    VideoInfo,
    VideoStreamInfo,
)


def create_video_info(
    *,
    audio: AudioStreamInfo | None = None,
) -> VideoInfo:
    return VideoInfo(
        format_name="mp4",
        duration_seconds=8.15,
        video=VideoStreamInfo(
            codec_name="h264",
            width=1280,
            height=720,
            frame_rate=30.0,
        ),
        audio=audio,
    )


def create_service(
    *,
    audio_extractor: Mock | None = None,
    asr_provider: Mock | None = None,
) -> tuple[VideoASRService, Mock, Mock]:
    audio_extractor = audio_extractor or Mock()
    asr_provider = asr_provider or Mock()

    audio_extractor.name = "ffmpeg"
    asr_provider.name = "faster_whisper"

    service = VideoASRService(
        audio_extractor=audio_extractor,
        asr_provider=asr_provider,
    )

    return service, audio_extractor, asr_provider


@pytest.mark.asyncio
async def test_no_audio_skips_extraction_and_asr():
    service, audio_extractor, asr_provider = create_service()

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(audio=None),
    )

    assert result.status == VideoASRStatus.NO_AUDIO
    assert result.asr_result is None

    audio_extractor.extract.assert_not_called()
    asr_provider.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_successful_video_to_asr_flow():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"wav-audio"
    )

    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.COMPLETED,
            text="Hello from the video.",
            language="en",
            confidence=0.95,
            provider="faster_whisper",
            segments=[
                ASRSegment(
                    text="Hello from the video.",
                    start_time=0.0,
                    end_time=2.5,
                    confidence=0.95,
                )
            ],
        )
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(
                codec_name="aac",
                sample_rate=22050,
                channels=1,
            )
        ),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.COMPLETED
    assert result.asr_result is not None
    assert result.asr_result.text == "Hello from the video."
    assert len(result.asr_result.segments) == 1
    assert result.asr_result.segments[0].start_time == 0.0
    assert result.asr_result.segments[0].end_time == 2.5

    audio_extractor.extract.assert_awaited_once()
    asr_provider.transcribe.assert_awaited_once()


@pytest.mark.asyncio
async def test_language_is_propagated_to_asr():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"wav-audio"
    )

    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.COMPLETED,
            text="Hello.",
            language="en",
            provider="faster_whisper",
        )
    )

    await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
        request=VideoASRRequest(
            language="en",
        ),
    )

    asr_request = asr_provider.transcribe.call_args.args[0]

    assert asr_request.language == "en"


@pytest.mark.asyncio
async def test_metadata_is_propagated_to_asr():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"wav-audio"
    )

    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.COMPLETED,
            text="Hello.",
            language="en",
            provider="faster_whisper",
        )
    )

    await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
        request=VideoASRRequest(
            language="en",
            metadata={
                "request_id": "test-123",
            },
        ),
    )

    asr_request = asr_provider.transcribe.call_args.args[0]

    assert asr_request.metadata["request_id"] == "test-123"
    assert asr_request.metadata["source"] == "video"
    assert asr_request.metadata["audio_extractor"] == "ffmpeg"


@pytest.mark.asyncio
async def test_audio_extraction_failure():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        side_effect=RuntimeError("FFmpeg extraction failed")
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED
    assert result.asr_result is None
    assert "FFmpeg extraction failed" in result.errors[0]

    asr_provider.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_empty_audio_extraction_is_rejected():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b""
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
    )

    assert result.status == VideoASRStatus.AUDIO_EXTRACTION_FAILED
    assert "empty audio" in result.errors[0].lower()

    asr_provider.transcribe.assert_not_called()


@pytest.mark.asyncio
async def test_asr_exception_returns_asr_failed():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"wav-audio"
    )

    asr_provider.transcribe = AsyncMock(
        side_effect=RuntimeError("ASR engine failed")
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
    )

    assert result.status == VideoASRStatus.ASR_FAILED
    assert "ASR engine failed" in result.errors[0]


@pytest.mark.asyncio
async def test_no_speech_is_mapped_correctly():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"wav-audio"
    )

    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.NO_SPEECH,
            provider="faster_whisper",
        )
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
    )

    assert result.status == VideoASRStatus.NO_SPEECH
    assert result.asr_result is not None
    assert result.asr_result.status == ASRStatus.NO_SPEECH


@pytest.mark.asyncio
async def test_asr_failure_status_is_mapped_correctly():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"wav-audio"
    )

    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.FAILED,
            provider="faster_whisper",
        )
    )

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
    )

    assert result.status == VideoASRStatus.ASR_FAILED


@pytest.mark.asyncio
async def test_empty_media_is_rejected():
    service, _, _ = create_service()

    with pytest.raises(ValueError, match="must not be empty"):
        await service.transcribe(
            b"",
            video_info=create_video_info(
                audio=AudioStreamInfo(codec_name="aac")
            ),
        )


@pytest.mark.asyncio
async def test_non_bytes_media_is_rejected():
    service, _, _ = create_service()

    with pytest.raises(TypeError, match="must be bytes"):
        await service.transcribe(
            "not-bytes",  # type: ignore[arg-type]
            video_info=create_video_info(
                audio=AudioStreamInfo(codec_name="aac")
            ),
        )


@pytest.mark.asyncio
async def test_audio_extraction_metadata_is_recorded():
    service, audio_extractor, asr_provider = create_service()

    audio_extractor.extract = AsyncMock(
        return_value=b"12345"
    )

    asr_provider.transcribe = AsyncMock(
        return_value=ASRResult(
            status=ASRStatus.COMPLETED,
            text="test",
            provider="faster_whisper",
        )
    )

    result = await service.transcribe(
        b"video",
        video_info=create_video_info(
            audio=AudioStreamInfo(codec_name="aac")
        ),
    )

    assert result.audio_extraction["status"] == "completed"
    assert result.audio_extraction["provider"] == "ffmpeg"
    assert result.audio_extraction["size_bytes"] == 5