from __future__ import annotations

import math
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from app.ingestion.speech.faster_whisper import (
    FasterWhisperASRProvider,
)
from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRStatus,
)


def make_segment(
    text: str,
    start: float,
    end: float,
    avg_logprob: float = -0.2,
    no_speech_prob: float = 0.01,
) -> SimpleNamespace:
    return SimpleNamespace(
        text=text,
        start=start,
        end=end,
        avg_logprob=avg_logprob,
        no_speech_prob=no_speech_prob,
    )


# =========================================================
# Basic transcription behavior
# =========================================================


@pytest.mark.asyncio
async def test_transcribe_returns_segments_and_text() -> None:
    provider = FasterWhisperASRProvider(
        model_name="small",
        device="cpu",
        compute_type="int8",
    )

    fake_model = MagicMock()

    info = SimpleNamespace(
        language="en",
        language_probability=0.98,
    )

    fake_model.transcribe.return_value = (
        iter(
            [
                make_segment(
                    " Hello",
                    0.0,
                    1.2,
                ),
                make_segment(
                    " world",
                    1.2,
                    2.4,
                ),
            ]
        ),
        info,
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.COMPLETED
    assert result.text == "Hello world"
    assert result.language == "en"

    assert len(result.segments) == 2

    assert result.segments[0].start_time == 0.0
    assert result.segments[0].end_time == 1.2

    assert result.segments[1].start_time == 1.2
    assert result.segments[1].end_time == 2.4

    assert result.segments[0].confidence is None

    assert "avg_logprob" in result.segments[0].metadata
    assert "no_speech_prob" in result.segments[0].metadata


@pytest.mark.asyncio
async def test_transcribe_passes_language_to_whisper() -> None:
    provider = FasterWhisperASRProvider(
        default_language="en",
    )

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(
            language="en",
            language_probability=0.9,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio",
                language="hi",
            )
        )

    fake_model.transcribe.assert_called_once()

    kwargs = fake_model.transcribe.call_args.kwargs

    assert kwargs["language"] == "hi"
    assert kwargs["beam_size"] == 5
    assert kwargs["vad_filter"] is True


@pytest.mark.asyncio
async def test_default_language_is_used() -> None:
    provider = FasterWhisperASRProvider(
        default_language="en",
    )

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(
            language="en",
            language_probability=0.9,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    kwargs = fake_model.transcribe.call_args.kwargs

    assert kwargs["language"] == "en"


@pytest.mark.asyncio
async def test_request_language_takes_priority_over_default() -> None:
    provider = FasterWhisperASRProvider(
        default_language="en",
    )

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(
            language="hi",
            language_probability=0.95,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio",
                language="hi",
            )
        )

    kwargs = fake_model.transcribe.call_args.kwargs

    assert kwargs["language"] == "hi"


# =========================================================
# No speech / empty audio
# =========================================================


@pytest.mark.asyncio
async def test_no_speech_result() -> None:
    provider = FasterWhisperASRProvider()

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter([]),
        SimpleNamespace(
            language="en",
            language_probability=0.5,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.NO_SPEECH
    assert result.text == ""
    assert result.segments == []


@pytest.mark.asyncio
async def test_empty_audio_fails_without_loading_model() -> None:
    provider = FasterWhisperASRProvider()

    with patch(
        "faster_whisper.WhisperModel"
    ) as model_class:
        result = await provider.transcribe(
            ASRRequest(
                audio=b""
            )
        )

    assert result.status == ASRStatus.FAILED
    assert result.metadata["reason"] == "empty_audio"

    model_class.assert_not_called()


# =========================================================
# Lazy model loading
# =========================================================


def test_model_is_loaded_lazily() -> None:
    provider = FasterWhisperASRProvider()

    assert provider._model is None


# =========================================================
# Configuration validation
# =========================================================


def test_invalid_model_name_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="model_name",
    ):
        FasterWhisperASRProvider(
            model_name="   "
        )


def test_invalid_device_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="device",
    ):
        FasterWhisperASRProvider(
            device="   "
        )


def test_invalid_compute_type_is_rejected() -> None:
    with pytest.raises(
        ValueError,
        match="compute_type",
    ):
        FasterWhisperASRProvider(
            compute_type="   "
        )


@pytest.mark.parametrize(
    "timeout",
    [
        0,
        -1,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_invalid_timeout_is_rejected(
    timeout: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="timeout_seconds",
    ):
        FasterWhisperASRProvider(
            timeout_seconds=timeout,
        )


# =========================================================
# Provider error handling
# =========================================================


@pytest.mark.asyncio
async def test_provider_error_is_returned_as_failed() -> None:
    provider = FasterWhisperASRProvider()

    with patch(
        "faster_whisper.WhisperModel",
        side_effect=RuntimeError(
            "model loading failed"
        ),
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.FAILED
    assert result.metadata["reason"] == "provider_error"
    assert result.metadata["error_type"] == "RuntimeError"


@pytest.mark.asyncio
async def test_malformed_segment_returns_failed_result() -> None:
    provider = FasterWhisperASRProvider()

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter(
            [
                make_segment(
                    "hello",
                    math.nan,
                    1.0,
                ),
            ]
        ),
        SimpleNamespace(
            language="en",
            language_probability=0.9,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.FAILED
    assert result.metadata["reason"] == "provider_error"
    assert result.metadata["error_type"] == "ValueError"


@pytest.mark.asyncio
async def test_invalid_no_speech_probability_returns_failed_result() -> None:
    provider = FasterWhisperASRProvider()

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter(
            [
                make_segment(
                    "hello",
                    0.0,
                    1.0,
                    no_speech_prob=1.5,
                ),
            ]
        ),
        SimpleNamespace(
            language="en",
            language_probability=0.9,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.FAILED
    assert result.metadata["reason"] == "provider_error"


# =========================================================
# Metadata validation
# =========================================================


@pytest.mark.asyncio
async def test_invalid_language_probability_returns_failed_result() -> None:
    provider = FasterWhisperASRProvider()

    fake_model = MagicMock()

    fake_model.transcribe.return_value = (
        iter(
            [
                make_segment(
                    "hello",
                    0.0,
                    1.0,
                ),
            ]
        ),
        SimpleNamespace(
            language="en",
            language_probability=math.nan,
        ),
    )

    with patch(
        "faster_whisper.WhisperModel",
        return_value=fake_model,
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.FAILED
    assert result.metadata["reason"] == "provider_error"
    assert result.metadata["error_type"] == "ValueError"


# =========================================================
# Timeout behavior
# =========================================================


@pytest.mark.asyncio
async def test_transcription_timeout_returns_failed_result() -> None:
    provider = FasterWhisperASRProvider(
        timeout_seconds=0.05,
    )

    with patch.object(
        provider,
        "_transcribe_sync",
        side_effect=lambda request: time.sleep(0.2),
    ):
        result = await provider.transcribe(
            ASRRequest(
                audio=b"fake-audio"
            )
        )

    assert result.status == ASRStatus.FAILED

    assert result.metadata["reason"] == "timeout"

    assert (
        result.metadata["timeout_seconds"]
        == 0.05
    )