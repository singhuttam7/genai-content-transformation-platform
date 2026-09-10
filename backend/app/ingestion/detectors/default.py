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

    def detect(self, request: IngestionRequest) -> InputType:
        """Return the validated input type for a request."""

        input_type = request.input_type

        # Text and prompt are explicit logical input types.
        if input_type in {InputType.TEXT, InputType.PROMPT}:
            if request.content is None:
                raise ValueError(
                    f"{input_type.value} input requires content."
                )
            return input_type

        # URL must contain a valid HTTP(S) URL.
        if input_type == InputType.URL:
            if not request.storage_uri:
                raise ValueError("URL input requires storage_uri to contain the URL.")

            parsed = urlparse(request.storage_uri)

            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Invalid URL input.")

            return InputType.URL

        # MIME type validation.
        if request.mime_type:
            mime_type = request.mime_type.lower().strip()

            detected_from_mime = self.MIME_MAP.get(mime_type)

            if detected_from_mime is not None:
                if not self._compatible(input_type, detected_from_mime):
                    raise ValueError(
                        f"Input type '{input_type.value}' is inconsistent "
                        f"with MIME type '{mime_type}'."
                    )

                return input_type

        # Filename extension validation.
        if request.filename:
            suffix = Path(request.filename).suffix.lower()

            detected_from_extension = self.EXTENSION_MAP.get(suffix)

            if detected_from_extension is not None:
                if not self._compatible(input_type, detected_from_extension):
                    raise ValueError(
                        f"Input type '{input_type.value}' is inconsistent "
                        f"with file extension '{suffix}'."
                    )

        return input_type

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
        } == {InputType.TEXT, InputType.TXT}:
            return True

        return False