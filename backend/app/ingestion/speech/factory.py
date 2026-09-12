from __future__ import annotations

from app.core.config import settings
from app.ingestion.speech.provider import ASRProvider


def create_asr_provider() -> ASRProvider | None:
    """
    Create the configured ASR provider.

    Returns None when ASR is disabled.

    Provider implementations are imported lazily so optional ASR
    dependencies are not required when ASR is disabled.
    """

    if not settings.asr_enabled:
        return None

    provider_name = settings.asr_provider.strip().lower()

    if provider_name == "faster_whisper":
        from app.ingestion.speech.faster_whisper import (
            FasterWhisperASRProvider,
        )

        return FasterWhisperASRProvider(
            model_name=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            timeout_seconds=settings.asr_timeout_seconds,
            default_language=settings.asr_default_language,
        )

    raise ValueError(
        f"Unsupported ASR provider: {settings.asr_provider}"
    )