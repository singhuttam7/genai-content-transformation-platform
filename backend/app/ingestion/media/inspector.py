from __future__ import annotations

from typing import Protocol

from app.ingestion.media.schemas import MediaInfo


class MediaInspector(Protocol):
    """
    Provider abstraction for inspecting media files.

    Implementations may use FFprobe or another media inspection
    backend without exposing backend-specific details to the
    ingestion layer.
    """

    name: str

    async def inspect(
        self,
        media: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> MediaInfo:
        """
        Inspect media bytes and return normalized technical metadata.
        """
        ...