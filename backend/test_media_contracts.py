from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.ingestion.media.schemas import MediaInfo


def test_media_info_defaults() -> None:
    info = MediaInfo()

    assert info.format_name is None
    assert info.codec_name is None
    assert info.duration_seconds is None
    assert info.sample_rate is None
    assert info.channels is None
    assert info.bitrate is None
    assert info.size_bytes is None
    assert info.metadata == {}


def test_media_info_audio_metadata() -> None:
    info = MediaInfo(
        format_name="wav",
        codec_name="pcm_s16le",
        duration_seconds=12.5,
        sample_rate=16000,
        channels=1,
        bitrate=256000,
        size_bytes=400000,
    )

    assert info.format_name == "wav"
    assert info.codec_name == "pcm_s16le"
    assert info.duration_seconds == 12.5
    assert info.sample_rate == 16000
    assert info.channels == 1
    assert info.bitrate == 256000
    assert info.size_bytes == 400000


def test_media_info_supports_extra_metadata() -> None:
    info = MediaInfo(
        format_name="wav",
        metadata={
            "encoder": "test",
            "container": "RIFF",
        },
    )

    assert info.metadata["encoder"] == "test"
    assert info.metadata["container"] == "RIFF"


def test_media_info_rejects_negative_duration() -> None:
    with pytest.raises(ValidationError):
        MediaInfo(
            duration_seconds=-1,
        )


def test_media_info_rejects_zero_sample_rate() -> None:
    with pytest.raises(ValidationError):
        MediaInfo(
            sample_rate=0,
        )


def test_media_info_rejects_zero_channels() -> None:
    with pytest.raises(ValidationError):
        MediaInfo(
            channels=0,
        )


def test_media_info_rejects_negative_size() -> None:
    with pytest.raises(ValidationError):
        MediaInfo(
            size_bytes=-1,
        )


def test_media_info_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MediaInfo(
            format_name="wav",
            unexpected="value",
        )