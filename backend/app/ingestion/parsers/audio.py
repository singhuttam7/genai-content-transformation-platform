from __future__ import annotations

from pathlib import Path

from app.ingestion.media import FFprobeMediaInspector, MediaInspector
from app.ingestion.schemas import (
    ContentBlock,
    ContentBlockType,
    ExtractedContent,
    IngestionRequest,
    InputType,
    SourceReference,
)


class AudioDocumentProcessor:
    """
    Validates and inspects audio input and converts it into
    provider-independent ExtractedContent.

    Transcription is intentionally handled by the ASR layer.
    """

    supported_types = {
        InputType.AUDIO,
    }

    def __init__(
        self,
        *,
        media_inspector: MediaInspector | None = None,
        max_size_bytes: int = 100 * 1024 * 1024,
        min_duration_seconds: float = 0.0,
    ) -> None:
        self.media_inspector = (
            media_inspector
            or FFprobeMediaInspector()
        )

        self.max_size_bytes = max_size_bytes
        self.min_duration_seconds = min_duration_seconds

    async def process(
        self,
        request: IngestionRequest,
    ) -> ExtractedContent:
        # ---------------------------------------------------------
        # 1. Validate input type
        # ---------------------------------------------------------
        if request.input_type not in self.supported_types:
            raise ValueError(
                f"Unsupported input type: {request.input_type}"
            )

        # ---------------------------------------------------------
        # 2. Resolve inline media bytes
        # ---------------------------------------------------------
        media = self._resolve_media_bytes(request)

        if not media:
            raise ValueError(
                "Audio content must not be empty."
            )

        # ---------------------------------------------------------
        # 3. Validate file size
        # ---------------------------------------------------------
        if len(media) > self.max_size_bytes:
            raise ValueError(
                f"Audio exceeds maximum allowed size "
                f"of {self.max_size_bytes} bytes."
            )

        # ---------------------------------------------------------
        # 4. Inspect actual media
        # ---------------------------------------------------------
        media_info = await self.media_inspector.inspect(
            media,
            filename=request.filename,
            mime_type=request.mime_type,
        )

        # ---------------------------------------------------------
        # 5. Validate actual media information
        # ---------------------------------------------------------
        self._validate_media_info(media_info)

        # ---------------------------------------------------------
        # 6. Validate filename extension when supplied
        # ---------------------------------------------------------
        self._validate_extension(
            filename=request.filename,
            format_name=media_info.format_name,
        )

        # ---------------------------------------------------------
        # 7. Build source reference
        # ---------------------------------------------------------
        source = SourceReference(
            source_id=request.source_id,
            source_type=InputType.AUDIO,
            title=request.title,
            filename=request.filename,
            mime_type=request.mime_type,
            content_hash=None,
            storage_uri=request.storage_uri,
        )

        # ---------------------------------------------------------
        # 8. Create canonical audio block
        # ---------------------------------------------------------
        block = ContentBlock(
            block_type=ContentBlockType.AUDIO,
            content="",
            order=0,
            metadata={
                "format_name": media_info.format_name,
                "codec_name": media_info.codec_name,
                "duration_seconds": media_info.duration_seconds,
                "sample_rate": media_info.sample_rate,
                "channels": media_info.channels,
                "bitrate": media_info.bitrate,
                "size_bytes": len(media),
                "mime_type": request.mime_type,
                "filename": request.filename,
                "inspection_provider": self.media_inspector.name,
                "transcription_status": "not_requested",
            },
        )

        # ---------------------------------------------------------
        # 9. Return extracted content
        # ---------------------------------------------------------
        return ExtractedContent(
            source=source,
            text="",
            blocks=[block],
            language=None,
            title=request.title or request.filename,
            metadata={
                "media": {
                    "format_name": media_info.format_name,
                    "codec_name": media_info.codec_name,
                    "duration_seconds": media_info.duration_seconds,
                    "sample_rate": media_info.sample_rate,
                    "channels": media_info.channels,
                    "bitrate": media_info.bitrate,
                    "size_bytes": len(media),
                    "inspection_provider": self.media_inspector.name,
                },
                "asr": {
                    "status": "not_requested",
                },
            },
        )

    def _validate_media_info(
        self,
        media_info,
    ) -> None:
        """
        Validate metadata discovered from the actual media stream.

        Filename and declared MIME type are not treated as proof
        that the uploaded object is valid audio.
        """

        if not media_info.format_name:
            raise ValueError(
                "Unable to determine audio format."
            )

        if not media_info.codec_name:
            raise ValueError(
                "No audio stream was detected."
            )

        if media_info.duration_seconds is None:
            raise ValueError(
                "Unable to determine audio duration."
            )

        if media_info.duration_seconds < 0:
            raise ValueError(
                "Audio duration cannot be negative."
            )

        if (
            media_info.duration_seconds
            < self.min_duration_seconds
        ):
            raise ValueError(
                f"Audio duration is shorter than the "
                f"minimum allowed duration of "
                f"{self.min_duration_seconds} seconds."
            )

        if (
            media_info.sample_rate is not None
            and media_info.sample_rate <= 0
        ):
            raise ValueError(
                "Audio sample rate must be positive."
            )

        if (
            media_info.channels is not None
            and media_info.channels <= 0
        ):
            raise ValueError(
                "Audio channel count must be positive."
            )

    @staticmethod
    def _validate_extension(
        *,
        filename: str | None,
        format_name: str | None,
    ) -> None:
        """
        Perform a consistency check between the filename and
        the media format detected by FFprobe.

        We intentionally do not require an extension because
        valid uploaded media can have no filename extension.
        """

        if not filename or not format_name:
            return

        extension = Path(filename).suffix.lower()

        if not extension:
            return

        extension_map = {
            ".wav": {"wav"},
            ".mp3": {"mp3"},
            ".flac": {"flac"},
            ".ogg": {"ogg", "oga"},
            ".oga": {"ogg", "oga"},
            ".m4a": {"mov", "mp4", "m4a"},
            ".mp4": {"mov", "mp4"},
            ".webm": {"webm"},
        }

        expected_formats = extension_map.get(extension)

        if expected_formats is None:
            return

        detected_formats = {
            item.strip().lower()
            for item in format_name.split(",")
        }

        if not detected_formats.intersection(
            expected_formats
        ):
            raise ValueError(
                "Audio filename extension does not match "
                "the detected media format."
            )

    @staticmethod
    def _resolve_media_bytes(
        request: IngestionRequest,
    ) -> bytes:
        content = request.content

        if isinstance(content, bytes):
            return content

        raise ValueError(
            "Audio processor requires audio bytes in request.content."
        )