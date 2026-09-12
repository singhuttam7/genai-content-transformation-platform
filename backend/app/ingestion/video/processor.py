from __future__ import annotations

import hashlib

from app.ingestion.processor import ContentProcessor
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)

from app.ingestion.video.ffprobe import FFprobeVideoInspector
from app.ingestion.video.provider import VideoInspector
from app.ingestion.video.schemas import VideoInfo
from app.ingestion.video.validator import VideoSecurityValidator


class VideoProcessingError(ValueError):
    """Raised when video processing fails."""


class VideoDocumentProcessor(ContentProcessor):
    """
    Video ingestion processor.

    Responsibilities:
    - Validate the ingestion input type.
    - Validate video security constraints.
    - Resolve inline video bytes.
    - Inspect the actual video using FFprobe.
    - Build provider-independent VideoInfo.
    - Calculate the source content hash.
    - Produce an ExtractedContent representation.
    - Preserve video metadata for downstream enrichment.

    This processor does NOT:
    - perform speech recognition,
    - extract audio for ASR,
    - perform OCR,
    - perform computer vision,
    - generate captions,
    - perform multimodal fusion.

    Those capabilities belong to later video-processing stages.
    """

    supported_types = frozenset(
        {
            InputType.VIDEO,
        }
    )

    DEFAULT_MAX_SIZE_BYTES = 500 * 1024 * 1024

    def __init__(
        self,
        *,
        video_inspector: VideoInspector | None = None,
        security_validator: VideoSecurityValidator | None = None,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if max_size_bytes <= 0:
            raise ValueError(
                "max_size_bytes must be greater than zero."
            )

        self.video_inspector = (
            video_inspector
            or FFprobeVideoInspector()
        )

        self.security_validator = (
            security_validator
            or VideoSecurityValidator(
                max_size_bytes=max_size_bytes,
            )
        )

        self.max_size_bytes = max_size_bytes

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        """
        Validate, inspect, and convert video input into
        provider-independent ExtractedContent.
        """

        # ---------------------------------------------------------
        # 1. Validate input type
        # ---------------------------------------------------------

        if request.input_type != InputType.VIDEO:
            raise ValueError(
                "VideoDocumentProcessor only supports VIDEO input."
            )

        # ---------------------------------------------------------
        # 2. Resolve inline video bytes
        # ---------------------------------------------------------

        media = self._resolve_media_bytes(
            request
        )

        # ---------------------------------------------------------
        # 3. Basic size validation
        # ---------------------------------------------------------

        if not media:
            raise VideoProcessingError(
                "Video content must not be empty."
            )

        if len(media) > self.max_size_bytes:
            raise VideoProcessingError(
                "Video exceeds the maximum allowed size."
            )

        # ---------------------------------------------------------
        # 4. Security validation
        # ---------------------------------------------------------

        self._validate_security(
            media=media,
            request=request,
        )

        # ---------------------------------------------------------
        # 5. Inspect actual video
        # ---------------------------------------------------------

        video_info = await self.video_inspector.inspect(
            media,
            filename=request.filename,
            mime_type=request.mime_type,
        )

        # ---------------------------------------------------------
        # 6. Validate inspected video
        # ---------------------------------------------------------

        self._validate_video_info(
            video_info
        )

        # ---------------------------------------------------------
        # 7. Calculate source hash
        # ---------------------------------------------------------

        content_hash = hashlib.sha256(
            media
        ).hexdigest()

        # ---------------------------------------------------------
        # 8. Build source reference
        # ---------------------------------------------------------

        source = SourceReference(
            source_id=request.source_id,
            source_type=InputType.VIDEO,
            title=request.title,
            filename=request.filename,
            mime_type=(
                request.mime_type
                or self._infer_mime_type(
                    video_info
                )
            ),
            content_hash=content_hash,
            storage_uri=request.storage_uri,
        )

        # ---------------------------------------------------------
        # 9. Build video stream metadata
        # ---------------------------------------------------------

        video_stream_metadata = None

        if video_info.video is not None:
            video_stream_metadata = (
                video_info.video.model_dump(
                    mode="json"
                )
            )

        # ---------------------------------------------------------
        # 10. Build audio stream metadata
        # ---------------------------------------------------------

        audio_stream_metadata = None

        if video_info.audio is not None:
            audio_stream_metadata = (
                video_info.audio.model_dump(
                    mode="json"
                )
            )

        # ---------------------------------------------------------
        # 11. Build video content block
        # ---------------------------------------------------------

        video_block = ContentBlock(
            block_type=ContentBlockType.VIDEO,
            content="",
            order=0,
            metadata={
                "media_type": "video",
                "format_name": video_info.format_name,
                "duration_seconds": (
                    video_info.duration_seconds
                ),
                "size_bytes": len(media),
                "content_hash": content_hash,
                "inspection_provider": (
                    self.video_inspector.name
                ),
                "video_stream": video_stream_metadata,
                "audio_stream": audio_stream_metadata,
            },
        )

        # ---------------------------------------------------------
        # 12. Build extracted-content metadata
        # ---------------------------------------------------------

        metadata = dict(
            request.metadata
        )

        metadata["video"] = {
            "format_name": video_info.format_name,
            "duration_seconds": (
                video_info.duration_seconds
            ),
            "size_bytes": len(media),
            "content_hash": content_hash,
            "inspection_provider": (
                self.video_inspector.name
            ),
            "video_stream_count": (
                1
                if video_info.video is not None
                else 0
            ),
            "audio_stream_count": (
                1
                if video_info.audio is not None
                else 0
            ),
            "video_info": video_info.model_dump(
                mode="json"
            ),
        }

        metadata["asr"] = {
            "status": "not_requested",
        }

        metadata["vision"] = {
            "status": "not_requested",
        }

        metadata["ocr"] = {
            "status": "not_requested",
        }

        # ---------------------------------------------------------
        # 13. Return provider-independent content
        # ---------------------------------------------------------

        return ExtractedContent(
            source=source,
            title=(
                request.title
                or request.filename
            ),
            text="",
            blocks=[
                video_block,
            ],
            language=None,
            metadata=metadata,
        )

    # =========================================================
    # Input resolution
    # =========================================================

    @staticmethod
    def _resolve_media_bytes(
        request: IngestionRequest,
    ) -> bytes:
        """
        Resolve video bytes from the current ingestion boundary.

        Storage-backed resolution remains the responsibility of
        the existing ingestion/storage resolver layer.
        """

        content = request.content

        if content is None:
            raise VideoProcessingError(
                "Video content must be provided."
            )

        if not isinstance(
            content,
            bytes,
        ):
            raise VideoProcessingError(
                "Video content must be binary bytes."
            )

        return content

    # =========================================================
    # Security validation
    # =========================================================

    def _validate_security(
        self,
        *,
        media: bytes,
        request: IngestionRequest,
    ) -> None:
        """
        Apply the existing video security validation boundary.
        """

        self.security_validator.validate(
            media,
            filename=request.filename,
            mime_type=request.mime_type,
        )

    # =========================================================
    # Video metadata validation
    # =========================================================

    @staticmethod
    def _validate_video_info(
        video_info: VideoInfo,
    ) -> None:
        """
        Validate normalized metadata returned by the inspector.
        """

        if not video_info.format_name:
            raise VideoProcessingError(
                "Unable to determine video format."
            )

        if (
            video_info.duration_seconds is not None
            and video_info.duration_seconds < 0
        ):
            raise VideoProcessingError(
                "Video duration cannot be negative."
            )

        # A valid video asset must contain a primary video stream.
        if video_info.video is None:
            raise VideoProcessingError(
                "No video stream was detected."
            )

        if video_info.video.width is None:
            raise VideoProcessingError(
                "Video stream width could not be determined."
            )

        if video_info.video.height is None:
            raise VideoProcessingError(
                "Video stream height could not be determined."
            )

        if video_info.video.width <= 0:
            raise VideoProcessingError(
                "Video stream width must be positive."
            )

        if video_info.video.height <= 0:
            raise VideoProcessingError(
                "Video stream height must be positive."
            )

    # =========================================================
    # MIME fallback
    # =========================================================

    @staticmethod
    def _infer_mime_type(
        video_info: VideoInfo,
    ) -> str | None:
        """
        Infer a basic MIME type from the inspected container.

        The declared request MIME type remains authoritative when
        supplied and validated by the security layer.
        """

        format_name = (
            video_info.format_name or ""
        ).lower()

        if "mp4" in format_name:
            return "video/mp4"

        if "webm" in format_name:
            return "video/webm"

        if (
            "matroska" in format_name
            or "mkv" in format_name
        ):
            return "video/x-matroska"

        if (
            "mov" in format_name
            or "quicktime" in format_name
        ):
            return "video/quicktime"

        if "avi" in format_name:
            return "video/x-msvideo"

        return None