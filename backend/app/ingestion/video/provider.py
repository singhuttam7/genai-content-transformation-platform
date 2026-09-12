from __future__ import annotations

from typing import Protocol

from app.ingestion.video.schemas import (
    AudioExtractionRequest,
    VideoFrame,
    VideoInfo,
)


class VideoInspector(Protocol):
    """
    Provider abstraction for inspecting video media.
    """

    name: str

    async def inspect(
        self,
        media: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> VideoInfo:
        """
        Inspect a video and return structured media information.
        """
        ...


class AudioExtractor(Protocol):
    """
    Provider abstraction for extracting audio from video.
    """

    name: str

    async def extract(
        self,
        media: bytes,
        *,
        request: AudioExtractionRequest | None = None,
        filename: str | None = None,
    ) -> bytes:
        """
        Extract normalized audio bytes from a video.
        """
        ...


class FrameExtractor(Protocol):
    """
    Provider abstraction for extracting frames from video.
    """

    name: str

    async def extract(
        self,
        media: bytes,
        *,
        interval_seconds: float = 2.0,
        max_frames: int = 300,
        start_time_seconds: float = 0.0,
        end_time_seconds: float | None = None,
        filename: str | None = None,
    ) -> list[VideoFrame]:
        """
        Extract timestamped frames from a video.
        """
        ...