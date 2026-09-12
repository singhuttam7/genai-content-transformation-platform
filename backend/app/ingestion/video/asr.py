from __future__ import annotations

from app.ingestion.speech.provider import ASRProvider
from app.ingestion.speech.schemas import ASRRequest, ASRStatus
from app.ingestion.video.asr_capability import (
    VideoASRCapability,
    VideoASRCapabilityChecker,
)
from app.ingestion.video.provider import AudioExtractor
from app.ingestion.video.schemas import (
    VideoASRRequest,
    VideoASRResult,
    VideoASRStatus,
    VideoInfo,
)


class VideoASRError(RuntimeError):
    """Base exception for video-to-ASR processing failures."""


class VideoASRService:
    """Coordinates video audio extraction and speech recognition."""

    def __init__(
        self,
        *,
        audio_extractor: AudioExtractor,
        asr_provider: ASRProvider,
        capability_checker: VideoASRCapabilityChecker | None = None,
    ) -> None:
        self.audio_extractor = audio_extractor
        self.asr_provider = asr_provider
        self.capability_checker = (
            capability_checker or VideoASRCapabilityChecker()
        )

    async def transcribe(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: VideoASRRequest | None = None,
        filename: str | None = None,
    ) -> VideoASRResult:
        if not isinstance(media, bytes):
            raise TypeError("Video media must be bytes.")

        if not media:
            raise ValueError("Video media must not be empty.")

        request = request or VideoASRRequest()

        capability = self.capability_checker.check(video_info)

        if capability.capability == VideoASRCapability.NO_AUDIO:
            return VideoASRResult(
                status=VideoASRStatus.NO_AUDIO,
                audio_extraction={
                    "status": "not_requested",
                    "provider": self.audio_extractor.name,
                },
                metadata={
                    "stage": "audio_capability",
                    "capability": capability.capability.value,
                    "reason": capability.reason,
                },
            )

        extraction_metadata = {
            "status": "processing",
            "provider": self.audio_extractor.name,
            "sample_rate": request.audio_extraction.sample_rate,
            "channels": request.audio_extraction.channels,
            "audio_format": request.audio_extraction.audio_format,
        }

        try:
            audio_bytes = await self.audio_extractor.extract(
                media,
                request=request.audio_extraction,
                filename=filename,
            )
        except Exception as exc:
            return VideoASRResult(
                status=VideoASRStatus.AUDIO_EXTRACTION_FAILED,
                audio_extraction={
                    **extraction_metadata,
                    "status": "failed",
                },
                errors=[str(exc)],
                metadata={
                    "stage": "audio_extraction",
                    "provider": self.audio_extractor.name,
                    "error_type": type(exc).__name__,
                    "retryable": False,
                },
            )

        if not audio_bytes:
            return VideoASRResult(
                status=VideoASRStatus.AUDIO_EXTRACTION_FAILED,
                audio_extraction={
                    **extraction_metadata,
                    "status": "failed",
                },
                errors=["Audio extraction returned empty audio data."],
                metadata={
                    "stage": "audio_extraction",
                    "provider": self.audio_extractor.name,
                    "error_type": "EmptyAudioOutput",
                    "retryable": False,
                },
            )

        extraction_metadata["status"] = "completed"
        extraction_metadata["size_bytes"] = len(audio_bytes)

        asr_metadata = {
            **request.metadata,
            "source": "video",
            "audio_extractor": self.audio_extractor.name,
            "video_filename": filename,
        }

        asr_request = ASRRequest(
            audio=audio_bytes,
            language=request.language,
            metadata=asr_metadata,
        )

        try:
            asr_result = await self.asr_provider.transcribe(asr_request)
        except Exception as exc:
            return VideoASRResult(
                status=VideoASRStatus.ASR_FAILED,
                asr_result=None,
                audio_extraction=extraction_metadata,
                errors=[str(exc)],
                metadata={
                    "stage": "asr",
                    "provider": self.asr_provider.name,
                    "error_type": type(exc).__name__,
                    "retryable": False,
                },
            )

        status = self._map_asr_status(asr_result.status)

        return VideoASRResult(
            status=status,
            asr_result=asr_result,
            audio_extraction=extraction_metadata,
            errors=[],
            metadata={
                "stage": "asr",
                "provider": self.asr_provider.name,
                "language": asr_result.language,
            },
        )

    @staticmethod
    def _map_asr_status(status: ASRStatus) -> VideoASRStatus:
        mapping = {
            ASRStatus.COMPLETED: VideoASRStatus.COMPLETED,
            ASRStatus.NO_SPEECH: VideoASRStatus.NO_SPEECH,
            ASRStatus.FAILED: VideoASRStatus.ASR_FAILED,
            ASRStatus.PROCESSING: VideoASRStatus.PROCESSING,
        }

        return mapping.get(
            status,
            VideoASRStatus.ASR_FAILED,
        )