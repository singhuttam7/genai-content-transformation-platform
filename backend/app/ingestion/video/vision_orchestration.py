from __future__ import annotations

from typing import Protocol

from app.ingestion.video.frame_extractor import (
    FFmpegFrameExtractor,
)
from app.ingestion.video.schemas import (
    FrameExtractionRequest,
    FrameExtractionResult,
    FrameExtractionStatus,
    VideoInfo,
    VisionRequest,
    VisionResult,
    VisionStatus,
)
from app.ingestion.video.vision import (
    VisionService,
)


class VideoFrameExtractor(Protocol):
    """
    Provider-independent contract for extracting frames
    from a video asset.

    The orchestration layer depends on this protocol rather
    than directly depending on FFmpeg implementation details.
    """

    name: str

    async def extract(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: FrameExtractionRequest | None = None,
    ) -> FrameExtractionResult:
        """
        Extract timestamped VideoFrame objects from video bytes.
        """
        ...


class VideoVisionService:
    """
    Application-level orchestration service for video vision.

    Responsibilities:

        Video bytes
            ↓
        Frame extraction
            ↓
        Vision analysis
            ↓
        Provider-independent VisionResult

    This service deliberately does not know about:
        - Ollama
        - Gemma
        - any specific vision provider
        - FFmpeg command construction
        - OCR
        - ASR
        - content normalization

    Frame extraction and visual analysis remain independent
    capabilities connected through provider-independent contracts.
    """

    def __init__(
        self,
        *,
        frame_extractor: VideoFrameExtractor | None = None,
        vision_service: VisionService,
    ) -> None:
        """
        Initialize video-vision orchestration.

        Args:
            frame_extractor:
                Component responsible for extracting VideoFrame
                objects. FFmpegFrameExtractor is used by default.

            vision_service:
                Existing application-level VisionService.

        Raises:
            ValueError:
                If vision_service is not provided.
        """

        if vision_service is None:
            raise ValueError(
                "Vision service must be provided."
            )

        self.frame_extractor = (
            frame_extractor
            or FFmpegFrameExtractor()
        )

        self.vision_service = vision_service

    async def analyze(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        vision_request: VisionRequest | None = None,
        frame_extraction_request: (
            FrameExtractionRequest | None
        ) = None,
    ) -> VisionResult:
        """
        Extract representative video frames and analyze them
        using the configured VisionService.

        The orchestration boundary preserves frame-extraction
        failures as structured VisionResult values and never
        exposes provider-specific failures to callers.
        """

        if not media:
            return VisionResult(
                status=VisionStatus.FAILED,
                requested_frames=0,
                processed_frames=0,
                failed_frames=0,
                errors=[
                    "Video content must not be empty."
                ],
                metadata={
                    "service": "video_vision",
                    "stage": "frame_extraction",
                    "error_type": "EmptyVideoContent",
                },
            )

        if video_info is None:
            raise ValueError(
                "VideoInfo must be provided."
            )

        vision_request = (
            vision_request
            or VisionRequest()
        )

        try:
            extraction_result = (
                await self.frame_extractor.extract(
                    media,
                    video_info=video_info,
                    request=frame_extraction_request,
                )
            )

        except Exception as exc:
            return self._build_extraction_failure(
                exception=exc,
            )

        if extraction_result.status == (
            FrameExtractionStatus.NO_VIDEO
        ):
            return self._build_no_video_result(
                extraction_result
            )

        if extraction_result.status == (
            FrameExtractionStatus.NO_FRAMES
        ):
            return self._build_no_frames_result(
                extraction_result
            )

        if extraction_result.status == (
            FrameExtractionStatus.FAILED
        ):
            return self._build_extraction_result_failure(
                extraction_result
            )

        if not extraction_result.frames:
            return self._build_no_frames_result(
                extraction_result
            )

        prepared_request = vision_request.model_copy(
            update={
                "frames": extraction_result.frames,
            },
            deep=True,
        )

        try:
            vision_result = (
                await self.vision_service.analyze(
                    prepared_request
                )
            )

        except Exception as exc:
            return self._build_vision_failure(
                extraction_result=extraction_result,
                exception=exc,
            )

        return self._attach_orchestration_metadata(
            result=vision_result,
            extraction_result=extraction_result,
        )

    # =========================================================
    # Metadata
    # =========================================================

    @staticmethod
    def _attach_orchestration_metadata(
        *,
        result: VisionResult,
        extraction_result: FrameExtractionResult,
    ) -> VisionResult:
        """
        Attach frame-extraction metadata without replacing
        VisionService or provider metadata.
        """

        metadata = dict(
            result.metadata
        )

        metadata["video_vision"] = {
            "frame_extraction": {
                "status": (
                    extraction_result.status.value
                ),
                "total_frames": (
                    extraction_result.total_frames
                ),
                "requested_interval_seconds": (
                    extraction_result
                    .requested_interval_seconds
                ),
                "actual_interval_seconds": (
                    extraction_result
                    .actual_interval_seconds
                ),
                "start_time_seconds": (
                    extraction_result
                    .start_time_seconds
                ),
                "end_time_seconds": (
                    extraction_result
                    .end_time_seconds
                ),
                "provider": (
                    extraction_result
                    .metadata
                    .get("provider")
                ),
                "errors": list(
                    extraction_result.errors
                ),
            },
        }

        return result.model_copy(
            update={
                "metadata": metadata,
            },
        )

    # =========================================================
    # Frame extraction failures
    # =========================================================

    @staticmethod
    def _build_extraction_failure(
        *,
        exception: Exception,
    ) -> VisionResult:
        """
        Convert an unexpected frame-extraction exception into
        a structured VisionResult.
        """

        return VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=0,
            processed_frames=0,
            failed_frames=0,
            errors=[
                str(exception)
            ],
            metadata={
                "service": "video_vision",
                "stage": "frame_extraction",
                "error_type": type(
                    exception
                ).__name__,
            },
        )

    @staticmethod
    def _build_extraction_result_failure(
        extraction_result: FrameExtractionResult,
    ) -> VisionResult:
        """
        Convert a structured frame-extraction failure into
        a structured VisionResult.
        """

        return VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=(
                extraction_result.metadata.get(
                    "requested_frame_count",
                    extraction_result.total_frames,
                )
                or 0
            ),
            processed_frames=0,
            failed_frames=(
                extraction_result.metadata.get(
                    "requested_frame_count",
                    extraction_result.total_frames,
                )
                or 0
            ),
            errors=list(
                extraction_result.errors
            ),
            metadata={
                "service": "video_vision",
                "stage": "frame_extraction",
                "frame_extraction": {
                    "status": (
                        extraction_result.status.value
                    ),
                    "provider": (
                        extraction_result.metadata.get(
                            "provider"
                        )
                    ),
                },
            },
        )

    @staticmethod
    def _build_no_frames_result(
        extraction_result: FrameExtractionResult,
    ) -> VisionResult:
        """
        Convert an empty frame-extraction result into
        a provider-independent NO_FRAMES result.
        """

        return VisionResult(
            status=VisionStatus.NO_FRAMES,
            requested_frames=(
                extraction_result.metadata.get(
                    "requested_frame_count",
                    0,
                )
                or 0
            ),
            processed_frames=0,
            failed_frames=0,
            errors=list(
                extraction_result.errors
            ),
            metadata={
                "service": "video_vision",
                "stage": "frame_extraction",
                "frame_extraction": {
                    "status": (
                        extraction_result.status.value
                    ),
                    "provider": (
                        extraction_result.metadata.get(
                            "provider"
                        )
                    ),
                    "total_frames": (
                        extraction_result.total_frames
                    ),
                },
            },
        )

    @staticmethod
    def _build_no_video_result(
        extraction_result: FrameExtractionResult,
    ) -> VisionResult:
        """
        Convert a NO_VIDEO extraction result into a structured
        vision result.
        """

        return VisionResult(
            status=VisionStatus.NO_FRAMES,
            requested_frames=0,
            processed_frames=0,
            failed_frames=0,
            errors=list(
                extraction_result.errors
            ),
            metadata={
                "service": "video_vision",
                "stage": "frame_extraction",
                "frame_extraction": {
                    "status": (
                        extraction_result.status.value
                    ),
                    "provider": (
                        extraction_result.metadata.get(
                            "provider"
                        )
                    ),
                },
            },
        )

    # =========================================================
    # Vision failures
    # =========================================================

    @staticmethod
    def _build_vision_failure(
        *,
        extraction_result: FrameExtractionResult,
        exception: Exception,
    ) -> VisionResult:
        """
        Convert an unexpected VisionService failure into a
        structured VisionResult.
        """

        frame_count = (
            extraction_result.total_frames
        )

        return VisionResult(
            status=VisionStatus.FAILED,
            requested_frames=frame_count,
            processed_frames=0,
            failed_frames=frame_count,
            errors=[
                str(exception)
            ],
            metadata={
                "service": "video_vision",
                "stage": "vision",
                "error_type": type(
                    exception
                ).__name__,
                "frame_extraction": {
                    "status": (
                        extraction_result.status.value
                    ),
                    "provider": (
                        extraction_result.metadata.get(
                            "provider"
                        )
                    ),
                    "total_frames": frame_count,
                },
            },
        )