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


def create_video_info() -> VideoInfo:
    return VideoInfo(
        format_name="mp4",
        duration_seconds=30.0,
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


def create_service(asr_result: ASRResult):
    audio_extractor = Mock()
    audio_extractor.name = "ffmpeg"
    audio_extractor.extract = AsyncMock(
        return_value=b"audio-bytes"
    )

    asr_provider = Mock()
    asr_provider.name = "faster-whisper"
    asr_provider.transcribe = AsyncMock(
        return_value=asr_result
    )

    service = VideoASRService(
        audio_extractor=audio_extractor,
        asr_provider=asr_provider,
    )

    return service, audio_extractor, asr_provider


@pytest.mark.asyncio
async def test_single_segment_timestamps_are_preserved():
    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Threat detected",
        language="en",
        provider="faster-whisper",
        segments=[
            ASRSegment(
                text="Threat detected",
                start_time=12.4,
                end_time=17.8,
            )
        ],
    )

    service, _, _ = create_service(asr_result)

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        request=VideoASRRequest(language="en"),
        filename="sample.mp4",
    )

    assert result.status == VideoASRStatus.COMPLETED
    assert result.asr_result is not None
    assert len(result.asr_result.segments) == 1

    segment = result.asr_result.segments[0]

    assert segment.start_time == 12.4
    assert segment.end_time == 17.8


@pytest.mark.asyncio
async def test_multiple_segment_timestamps_are_preserved_in_order():
    segments = [
        ASRSegment(
            text="First event",
            start_time=2.15,
            end_time=5.73,
        ),
        ASRSegment(
            text="Second event",
            start_time=8.42,
            end_time=11.96,
        ),
        ASRSegment(
            text="Third event",
            start_time=20.01,
            end_time=24.88,
        ),
    ]

    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="First event Second event Third event",
        language="en",
        provider="faster-whisper",
        segments=segments,
    )

    service, _, _ = create_service(asr_result)

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.asr_result is not None

    returned_segments = result.asr_result.segments

    assert [
        (segment.start_time, segment.end_time)
        for segment in returned_segments
    ] == [
        (2.15, 5.73),
        (8.42, 11.96),
        (20.01, 24.88),
    ]


@pytest.mark.asyncio
async def test_zero_based_timestamp_is_preserved():
    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Beginning",
        language="en",
        provider="faster-whisper",
        segments=[
            ASRSegment(
                text="Beginning",
                start_time=0.0,
                end_time=2.75,
            )
        ],
    )

    service, _, _ = create_service(asr_result)

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.asr_result is not None

    segment = result.asr_result.segments[0]

    assert segment.start_time == 0.0
    assert segment.end_time == 2.75


@pytest.mark.asyncio
async def test_fractional_timestamps_are_not_rounded():
    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Precise timing",
        language="en",
        provider="faster-whisper",
        segments=[
            ASRSegment(
                text="Precise timing",
                start_time=12.345678,
                end_time=18.901234,
            )
        ],
    )

    service, _, _ = create_service(asr_result)

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="sample.mp4",
    )

    assert result.asr_result is not None

    segment = result.asr_result.segments[0]

    assert segment.start_time == 12.345678
    assert segment.end_time == 18.901234


@pytest.mark.asyncio
async def test_timestamp_metadata_survives_with_segment_metadata():
    asr_result = ASRResult(
        status=ASRStatus.COMPLETED,
        text="Incident confirmed",
        language="en",
        provider="faster-whisper",
        segments=[
            ASRSegment(
                text="Incident confirmed",
                start_time=42.5,
                end_time=47.25,
                confidence=0.94,
                speaker="speaker_1",
                metadata={
                    "channel": 0,
                    "model_segment_id": 17,
                },
            )
        ],
    )

    service, _, _ = create_service(asr_result)

    result = await service.transcribe(
        b"video-bytes",
        video_info=create_video_info(),
        filename="incident.mp4",
    )

    assert result.asr_result is not None

    segment = result.asr_result.segments[0]

    assert segment.start_time == 42.5
    assert segment.end_time == 47.25
    assert segment.confidence == 0.94
    assert segment.speaker == "speaker_1"
    assert segment.metadata["channel"] == 0
    assert segment.metadata["model_segment_id"] == 17