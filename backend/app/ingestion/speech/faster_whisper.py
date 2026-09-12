from __future__ import annotations

import asyncio
import math
from functools import partial
from threading import Lock
from typing import Any

from app.ingestion.speech.provider import ASRProvider
from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRResult,
    ASRSegment,
    ASRStatus,
)


class FasterWhisperASRProvider:
    """
    Local ASR provider backed by faster-whisper.

    The Whisper model is loaded lazily on the first transcription request.
    Model inference runs in a worker thread so the async event loop remains
    responsive.
    """

    name = "faster_whisper"

    def __init__(
        self,
        *,
        model_name: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        timeout_seconds: float = 300.0,
        default_language: str | None = None,
    ) -> None:
        model_name = model_name.strip()
        device = device.strip().lower()
        compute_type = compute_type.strip().lower()

        if not model_name:
            raise ValueError("model_name must not be empty")

        if not device:
            raise ValueError("device must not be empty")

        if not compute_type:
            raise ValueError("compute_type must not be empty")

        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be a finite value greater than zero"
            )

        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.timeout_seconds = timeout_seconds

        self.default_language = (
            default_language.strip()
            if default_language and default_language.strip()
            else None
        )

        self._model: Any | None = None
        self._model_lock = Lock()

    def _load_model(self) -> Any:
        """
        Lazily load the Whisper model.

        Model construction is protected by a lock so concurrent first
        requests cannot initialize multiple model instances.
        """

        if self._model is not None:
            return self._model

        with self._model_lock:
            if self._model is not None:
                return self._model

            try:
                from faster_whisper import WhisperModel
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is not installed"
                ) from exc

            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )

        return self._model

    @staticmethod
    def _safe_float(
        value: Any,
        *,
        field_name: str,
    ) -> float:
        """
        Convert a value to a finite float.

        Whisper metadata should contain finite numeric values. Invalid
        values are rejected explicitly instead of silently propagating
        NaN or infinity into the API response.
        """

        try:
            converted = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid {field_name}: expected a numeric value"
            ) from exc

        if not math.isfinite(converted):
            raise ValueError(
                f"Invalid {field_name}: value must be finite"
            )

        return converted

    def _build_segment(
        self,
        segment: Any,
    ) -> ASRSegment | None:
        """
        Convert a faster-whisper segment into our stable ASR contract.

        Empty transcript segments are ignored.
        """

        text = str(getattr(segment, "text", "")).strip()

        if not text:
            return None

        raw_start = getattr(segment, "start", None)
        raw_end = getattr(segment, "end", None)

        start_time = max(
            self._safe_float(
                raw_start,
                field_name="segment.start",
            ),
            0.0,
        )

        end_time = max(
            self._safe_float(
                raw_end,
                field_name="segment.end",
            ),
            start_time,
        )

        metadata: dict[str, object] = {}

        if hasattr(segment, "avg_logprob"):
            metadata["avg_logprob"] = self._safe_float(
                segment.avg_logprob,
                field_name="segment.avg_logprob",
            )

        if hasattr(segment, "no_speech_prob"):
            no_speech_prob = self._safe_float(
                segment.no_speech_prob,
                field_name="segment.no_speech_prob",
            )

            if not 0.0 <= no_speech_prob <= 1.0:
                raise ValueError(
                    "Invalid segment.no_speech_prob: "
                    "value must be between 0 and 1"
                )

            metadata["no_speech_prob"] = no_speech_prob

        return ASRSegment(
            text=text,
            start_time=start_time,
            end_time=end_time,
            confidence=None,
            metadata=metadata,
        )

    def _transcribe_sync(
        self,
        request: ASRRequest,
    ) -> ASRResult:
        """
        Execute synchronous faster-whisper inference.
        """

        model = self._load_model()

        language = request.language or self.default_language

        segments, info = model.transcribe(
            request.audio,
            language=language,
            beam_size=5,
            vad_filter=True,
        )

        result_segments: list[ASRSegment] = []
        text_parts: list[str] = []

        for segment in segments:
            result_segment = self._build_segment(segment)

            if result_segment is None:
                continue

            result_segments.append(result_segment)
            text_parts.append(result_segment.text)

        detected_language = getattr(info, "language", None)

        language_probability = getattr(
            info,
            "language_probability",
            None,
        )

        metadata: dict[str, object] = {
            "model": self.model_name,
            "device": self.device,
            "compute_type": self.compute_type,
            "segment_count": len(result_segments),
        }

        if language_probability is not None:
            metadata["language_probability"] = self._safe_float(
                language_probability,
                field_name="language_probability",
            )

        if not result_segments:
            return ASRResult(
                status=ASRStatus.NO_SPEECH,
                text="",
                segments=[],
                language=detected_language,
                confidence=None,
                provider=self.name,
                metadata=metadata,
            )

        return ASRResult(
            status=ASRStatus.COMPLETED,
            text=" ".join(text_parts),
            segments=result_segments,
            language=detected_language,
            confidence=None,
            provider=self.name,
            metadata=metadata,
        )

    async def transcribe(
        self,
        request: ASRRequest,
    ) -> ASRResult:
        """
        Transcribe audio asynchronously.

        The timeout limits how long the caller waits for the operation.
        The underlying worker thread cannot be forcibly terminated by
        asyncio once inference has started.
        """

        if not request.audio:
            return ASRResult(
                status=ASRStatus.FAILED,
                provider=self.name,
                metadata={
                    "reason": "empty_audio",
                },
            )

        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    partial(self._transcribe_sync, request)
                ),
                timeout=self.timeout_seconds,
            )

        except asyncio.TimeoutError:
            return ASRResult(
                status=ASRStatus.FAILED,
                provider=self.name,
                metadata={
                    "reason": "timeout",
                    "timeout_seconds": self.timeout_seconds,
                },
            )

        except Exception as exc:
            return ASRResult(
                status=ASRStatus.FAILED,
                provider=self.name,
                metadata={
                    "reason": "provider_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )