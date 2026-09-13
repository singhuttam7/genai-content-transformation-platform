from __future__ import annotations

from typing import Protocol

from app.ingestion.video.schemas import (
    AudioExtractionRequest,
    FrameExtractionRequest,
    FrameExtractionResult,
    VideoInfo,
    VisionRequest,
    VisionResult,
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

    Implementations receive a provider-independent frame
    extraction request and return a provider-independent
    extraction result.

    This abstraction keeps FFmpeg-specific behavior out of
    the higher-level video processing pipeline.
    """

    name: str

    async def extract(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: FrameExtractionRequest | None = None,
        filename: str | None = None,
    ) -> FrameExtractionResult:
        """
        Extract timestamped frames from a video.

        Args:
            media:
                Raw video bytes.

            video_info:
                Previously inspected video metadata.

            request:
                Frame extraction parameters. If omitted,
                the implementation uses the contract defaults.

            filename:
                Optional original filename. Implementations
                may use its extension when creating temporary
                input files.

        Returns:
            Provider-independent FrameExtractionResult.
        """
        ...


class VisionProvider(Protocol):
    """
    Provider abstraction for visual analysis.

    Implementations receive provider-independent vision
    requests and return provider-independent vision results.

    The higher-level vision service depends only on this
    contract and therefore remains independent from any
    specific vision model, runtime, SDK, or API.
    """

    name: str

    async def analyze(
        self,
        request: VisionRequest,
    ) -> VisionResult:
        """
        Analyze extracted video frames.

        Args:
            request:
                Provider-independent visual analysis request
                containing the frames and analysis options.

        Returns:
            Provider-independent VisionResult.
        """
        ...