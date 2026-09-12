from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath

from app.ingestion.video.schemas import VideoProcessingStatus


@dataclass(frozen=True)
class VideoValidationResult:
    """
    Result of pre-processing video security validation.
    """

    valid: bool
    status: VideoProcessingStatus
    reason: str | None = None
    extension: str | None = None
    mime_type: str | None = None
    size_bytes: int = 0


class VideoValidationError(ValueError):
    """
    Raised when video input fails security validation.
    """


class VideoSecurityValidator:
    """
    Performs security and structural checks that can be completed
    before invoking a media-processing tool such as FFprobe/FFmpeg.

    This validator deliberately does not attempt to determine
    duration, codecs, resolution, or stream structure. Those checks
    belong to the media-inspection stage.
    """

    DEFAULT_MAX_SIZE_BYTES = 500 * 1024 * 1024

    ALLOWED_EXTENSIONS = frozenset(
        {
            ".mp4",
            ".mov",
            ".m4v",
            ".mkv",
            ".webm",
            ".avi",
        }
    )

    EXTENSION_MIME_TYPES = {
        ".mp4": frozenset(
            {
                "video/mp4",
                "application/mp4",
            }
        ),
        ".mov": frozenset(
            {
                "video/quicktime",
            }
        ),
        ".m4v": frozenset(
            {
                "video/x-m4v",
                "video/mp4",
            }
        ),
        ".mkv": frozenset(
            {
                "video/x-matroska",
            }
        ),
        ".webm": frozenset(
            {
                "video/webm",
            }
        ),
        ".avi": frozenset(
            {
                "video/x-msvideo",
                "video/avi",
            }
        ),
    }

    def __init__(
        self,
        *,
        max_size_bytes: int = DEFAULT_MAX_SIZE_BYTES,
    ) -> None:
        if max_size_bytes <= 0:
            raise ValueError("max_size_bytes must be greater than zero.")

        self.max_size_bytes = max_size_bytes

    def validate(
        self,
        media: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> VideoValidationResult:
        """
        Validate raw video input.

        Returns a structured validation result rather than exposing
        implementation-specific exceptions to callers.
        """

        size_bytes = len(media)

        if size_bytes == 0:
            return self._invalid(
                reason="Video input is empty.",
                size_bytes=size_bytes,
            )

        if size_bytes > self.max_size_bytes:
            return self._invalid(
                reason=(
                    "Video exceeds the maximum allowed size "
                    f"of {self.max_size_bytes} bytes."
                ),
                size_bytes=size_bytes,
            )

        try:
            extension = self._validate_filename(filename)
        except VideoValidationError as exc:
            return self._invalid(
                reason=str(exc),
                size_bytes=size_bytes,
            )

        try:
            normalized_mime = self._validate_mime_type(mime_type)
        except VideoValidationError as exc:
            return self._invalid(
                reason=str(exc),
                extension=extension,
                size_bytes=size_bytes,
            )

        if normalized_mime is not None:
            allowed_mimes = self.EXTENSION_MIME_TYPES.get(extension, frozenset())

            if normalized_mime not in allowed_mimes:
                return self._invalid(
                    reason=(
                        f"MIME type '{normalized_mime}' is not compatible "
                        f"with extension '{extension}'."
                    ),
                    extension=extension,
                    mime_type=normalized_mime,
                    size_bytes=size_bytes,
                )

        if not self._matches_file_signature(
            media,
            extension=extension,
        ):
            return self._invalid(
                reason=(
                    "Video content does not match the expected file "
                    f"signature for '{extension}'."
                ),
                extension=extension,
                mime_type=normalized_mime,
                size_bytes=size_bytes,
            )

        return VideoValidationResult(
            valid=True,
            status=VideoProcessingStatus.COMPLETED,
            extension=extension,
            mime_type=normalized_mime,
            size_bytes=size_bytes,
        )

    def _validate_filename(
        self,
        filename: str | None,
    ) -> str:
        if filename is None or not filename.strip():
            raise VideoValidationError(
                "Video filename is required."
            )

        normalized = filename.strip()

        if "\x00" in normalized:
            raise VideoValidationError(
                "Video filename contains a null byte."
            )

        if "/" in normalized or "\\" in normalized:
            raise VideoValidationError(
                "Video filename must not contain path separators."
            )

        path = PurePath(normalized)

        if path.name != normalized:
            raise VideoValidationError(
                "Video filename contains an invalid path component."
            )

        if normalized in {".", ".."}:
            raise VideoValidationError(
                "Video filename is invalid."
            )

        extension = path.suffix.lower()

        if extension not in self.ALLOWED_EXTENSIONS:
            raise VideoValidationError(
                f"Unsupported video extension '{extension}'."
            )

        return extension

    @staticmethod
    def _validate_mime_type(
        mime_type: str | None,
    ) -> str | None:
        if mime_type is None:
            return None

        normalized = mime_type.strip().lower()

        if not normalized:
            raise VideoValidationError(
                "MIME type cannot be empty."
            )

        if ";" in normalized:
            normalized = normalized.split(";", 1)[0].strip()

        if not normalized.startswith("video/") and normalized != "application/mp4":
            raise VideoValidationError(
                f"Unsupported video MIME type '{normalized}'."
            )

        return normalized

    @staticmethod
    def _matches_file_signature(
        media: bytes,
        *,
        extension: str,
    ) -> bool:
        """
        Perform lightweight magic-byte validation.

        This is intentionally not treated as complete media validation.
        FFprobe performs authoritative stream/container inspection later.
        """

        if extension in {".mp4", ".mov", ".m4v"}:
            return VideoSecurityValidator._has_iso_base_media_signature(media)

        if extension in {".mkv", ".webm"}:
            return media.startswith(b"\x1A\x45\xDF\xA3")

        if extension == ".avi":
            return (
                len(media) >= 12
                and media[:4] == b"RIFF"
                and media[8:12] == b"AVI "
            )

        return False

    @staticmethod
    def _has_iso_base_media_signature(
        media: bytes,
    ) -> bool:
        """
        MP4/MOV/M4V containers contain an `ftyp` box near the beginning.

        Allow a small bounded search because the box does not have to
        start at byte zero.
        """

        search_window = media[:64]

        return b"ftyp" in search_window

    @staticmethod
    def _invalid(
        *,
        reason: str,
        size_bytes: int,
        extension: str | None = None,
        mime_type: str | None = None,
    ) -> VideoValidationResult:
        return VideoValidationResult(
            valid=False,
            status=VideoProcessingStatus.FAILED,
            reason=reason,
            extension=extension,
            mime_type=mime_type,
            size_bytes=size_bytes,
        )