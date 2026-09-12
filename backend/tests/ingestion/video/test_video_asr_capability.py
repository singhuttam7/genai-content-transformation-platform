from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.video.asr_capability import (
    VideoASRCapability,
    VideoASRCapabilityChecker,
    VideoASRCapabilityResult,
)
from app.ingestion.video.schemas import (
    AudioStreamInfo,
    VideoInfo,
    VideoStreamInfo,
)


@pytest.fixture
def checker() -> VideoASRCapabilityChecker:
    return VideoASRCapabilityChecker()


def test_audio_available_when_audio_stream_exists(
    checker: VideoASRCapabilityChecker,
):
    video_info = VideoInfo(
        format_name="mp4",
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

    result = checker.check(video_info)

    assert result.capability == VideoASRCapability.AUDIO_AVAILABLE
    assert result.metadata["audio_present"] is True


def test_no_audio_when_audio_stream_is_missing(
    checker: VideoASRCapabilityChecker,
):
    video_info = VideoInfo(
        format_name="mp4",
        video=VideoStreamInfo(
            codec_name="h264",
            width=1280,
            height=720,
            frame_rate=30.0,
        ),
        audio=None,
    )

    result = checker.check(video_info)

    assert result.capability == VideoASRCapability.NO_AUDIO
    assert result.metadata["audio_present"] is False


def test_audio_metadata_is_preserved(
    checker: VideoASRCapabilityChecker,
):
    video_info = VideoInfo(
        audio=AudioStreamInfo(
            codec_name="aac",
            sample_rate=22050,
            channels=1,
            bitrate=62265,
        ),
    )

    result = checker.check(video_info)

    assert result.metadata["codec_name"] == "aac"
    assert result.metadata["sample_rate"] == 22050
    assert result.metadata["channels"] == 1


def test_audio_available_does_not_require_complete_audio_metadata(
    checker: VideoASRCapabilityChecker,
):
    video_info = VideoInfo(
        audio=AudioStreamInfo()
    )

    result = checker.check(video_info)

    assert result.capability == VideoASRCapability.AUDIO_AVAILABLE
    assert result.metadata["audio_present"] is True


def test_result_has_expected_reason_for_audio_available(
    checker: VideoASRCapabilityChecker,
):
    video_info = VideoInfo(
        audio=AudioStreamInfo(
            codec_name="aac",
        ),
    )

    result = checker.check(video_info)

    assert result.reason == "The video contains an audio stream."


def test_result_has_expected_reason_for_no_audio(
    checker: VideoASRCapabilityChecker,
):
    video_info = VideoInfo(
        audio=None,
    )

    result = checker.check(video_info)

    assert result.reason == "The video does not contain an audio stream."


def test_capability_result_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        VideoASRCapabilityResult(
            capability=VideoASRCapability.NO_AUDIO,
            reason="No audio.",
            unexpected_field="value",
        )


def test_capability_result_metadata_is_independent():
    first = VideoASRCapabilityResult(
        capability=VideoASRCapability.AUDIO_AVAILABLE,
        reason="Audio exists.",
    )

    second = VideoASRCapabilityResult(
        capability=VideoASRCapability.AUDIO_AVAILABLE,
        reason="Audio exists.",
    )

    first.metadata["test"] = True

    assert second.metadata == {}


def test_capability_enum_values():
    assert VideoASRCapability.AUDIO_AVAILABLE.value == "audio_available"
    assert VideoASRCapability.NO_AUDIO.value == "no_audio"