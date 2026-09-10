from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

from app.ingestion.detector import InputDetector
from app.ingestion.schemas import IngestionRequest, InputType


class DefaultInputDetector(InputDetector):
    """Detect and validate the input type for an ingestion request."""

    MIME_MAP: dict[str, InputType] = {
        "text/plain": InputType.TEXT,
        "text/markdown": InputType.MARKDOWN,
        "application/pdf": InputType.PDF,
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": (
            InputType.DOCX
        ),
        "image/jpeg": InputType.IMAGE,
        "image/png": InputType.IMAGE,
        "image/webp": InputType.IMAGE,
        "image/gif": InputType.IMAGE,
        "audio/mpeg": InputType.AUDIO,
        "audio/wav": InputType.AUDIO,
        "audio/x-wav": InputType.AUDIO,
        "audio/mp4": InputType.AUDIO,
        "video/mp4": InputType.VIDEO,
        "video/webm": InputType.VIDEO,
        "video/quicktime": InputType.VIDEO,
    }

    EXTENSION_MAP: dict[str, InputType] = {
        ".txt": InputType.TXT,
        ".md": InputType.MARKDOWN,
        ".markdown": InputType.MARKDOWN,
        ".pdf": InputType.PDF,
        ".docx": InputType.DOCX,
        ".jpg": InputType.IMAGE,
        ".jpeg": InputType.IMAGE,
        ".png": InputType.IMAGE,
        ".webp": InputType.IMAGE,
        ".gif": InputType.IMAGE,
        ".mp3": InputType.AUDIO,
        ".wav": InputType.AUDIO,
        ".m4a": InputType.AUDIO,
        ".mp4": InputType.VIDEO,
        ".webm": InputType.VIDEO,
        ".mov": InputType.VIDEO,
    }

    FILE_INPUT_TYPES = {
        InputType.PDF,
        InputType.DOCX,
        InputType.TXT,
        InputType.MARKDOWN,
        InputType.IMAGE,
        InputType.AUDIO,
        InputType.VIDEO,
    }

    def detect(self, request: IngestionRequest) -> InputType:
        """Return the validated input type for a request."""

        input_type = request.input_type

        # ---------------------------------------------------------
        # TEXT / PROMPT
        # ---------------------------------------------------------

        if input_type in {
            InputType.TEXT,
            InputType.PROMPT,
        }:
            self._validate_text_input(request)
            return input_type

        # ---------------------------------------------------------
        # URL
        # ---------------------------------------------------------

        if input_type == InputType.URL:
            self._validate_url(request)
            return InputType.URL

        # ---------------------------------------------------------
        # FILE / BINARY INPUT
        # ---------------------------------------------------------

        if input_type in self.FILE_INPUT_TYPES:
            self._validate_filename(request)

            mime_type = self._normalize_mime_type(
                request.mime_type
            )

            extension = self._normalize_extension(
                request.filename
            )

            detected_from_mime = (
                self.MIME_MAP.get(mime_type)
                if mime_type
                else None
            )

            detected_from_extension = (
                self.EXTENSION_MAP.get(extension)
                if extension
                else None
            )

            # At least one reliable file indicator must exist.
            if (
                detected_from_mime is None
                and detected_from_extension is None
            ):
                raise ValueError(
                    "File input requires a supported MIME type "
                    "or supported file extension."
                )

            # MIME type validation.
            if detected_from_mime is not None:
                if not self._compatible(
                    input_type,
                    detected_from_mime,
                ):
                    raise ValueError(
                        f"Input type '{input_type.value}' is inconsistent "
                        f"with MIME type '{mime_type}'."
                    )

            # Extension validation.
            if detected_from_extension is not None:
                if not self._compatible(
                    input_type,
                    detected_from_extension,
                ):
                    raise ValueError(
                        f"Input type '{input_type.value}' is inconsistent "
                        f"with file extension '{extension}'."
                    )

            # MIME and extension must agree when both are recognized.
            if (
                detected_from_mime is not None
                and detected_from_extension is not None
                and not self._compatible(
                    detected_from_mime,
                    detected_from_extension,
                )
            ):
                raise ValueError(
                    "MIME type and file extension represent "
                    "different content types."
                )

            return input_type

        raise ValueError(
            f"Unsupported input type: {input_type.value}"
        )

    @staticmethod
    def _validate_text_input(
        request: IngestionRequest,
    ) -> None:
        """Validate text and prompt inputs."""

        if request.content is None:
            raise ValueError(
                f"{request.input_type.value} input requires content."
            )

        if isinstance(request.content, bytes):
            try:
                text = request.content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ValueError(
                    "Text and prompt content must be valid UTF-8."
                ) from exc
        else:
            text = str(request.content)

        if not text.strip():
            raise ValueError(
                f"{request.input_type.value} content cannot be empty."
            )

    @staticmethod
    def _validate_url(
        request: IngestionRequest,
    ) -> None:
        """Validate an external HTTP(S) URL."""

        if not request.url:
            raise ValueError(
                "URL input requires a URL."
            )

        value = request.url.strip()

        parsed = urlparse(value)

        if parsed.scheme.lower() not in {
            "http",
            "https",
        }:
            raise ValueError(
                "URL must use HTTP or HTTPS."
            )

        if not parsed.netloc:
            raise ValueError(
                "URL must contain a valid host."
            )

        if any(
            character.isspace()
            for character in value
        ):
            raise ValueError(
                "URL cannot contain whitespace."
            )

    @staticmethod
    def _validate_filename(
        request: IngestionRequest,
    ) -> None:
        """Validate that the filename is a filename, not a path."""

        if not request.filename:
            return

        filename = request.filename.replace(
            "\\",
            "/",
        )

        parts = filename.split("/")

        if len(parts) != 1:
            raise ValueError(
                "Directory paths are not allowed in filenames."
            )

        if parts[0] in {
            "",
            ".",
            "..",
        }:
            raise ValueError(
                "Invalid filename."
            )

        # Reject Windows drive-style paths.
        if len(parts[0]) >= 2 and parts[0][1] == ":":
            raise ValueError(
                "Absolute paths are not allowed."
            )

    @classmethod
    def _normalize_mime_type(
        cls,
        mime_type: str | None,
    ) -> str | None:
        if not mime_type:
            return None

        return mime_type.split(";", 1)[0].strip().lower()

    @staticmethod
    def _normalize_extension(
        filename: str | None,
    ) -> str | None:
        if not filename:
            return None

        return Path(filename).suffix.lower()

    @staticmethod
    def _compatible(
        declared_type: InputType,
        detected_type: InputType,
    ) -> bool:
        """Check whether declared and detected types are compatible."""

        if declared_type == detected_type:
            return True

        # TEXT and TXT both represent textual source content.
        if {
            declared_type,
            detected_type,
        } == {
            InputType.TEXT,
            InputType.TXT,
        }:
            return True

        return False