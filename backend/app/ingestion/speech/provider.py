from __future__ import annotations

from typing import Protocol

from app.ingestion.speech.schemas import (
    ASRRequest,
    ASRResult,
)


class ASRProvider(Protocol):
    """
    Provider abstraction for automatic speech recognition.

    Implementations may use local or cloud-based ASR engines.
    """

    name: str

    async def transcribe(
        self,
        request: ASRRequest,
    ) -> ASRResult:
        """
        Transcribe the supplied audio.
        """
        ...