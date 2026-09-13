from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Protocol

from app.ingestion.video.schemas import VideoFrame


@dataclass(frozen=True, slots=True)
class SerializedFrame:
    """
    Provider-independent serialized representation of a video frame.

    The original frame timing and index are preserved so downstream
    vision processing can correlate observations with the source video.
    """

    frame_index: int
    timestamp_seconds: float
    mime_type: str
    data: str

    @property
    def is_empty(self) -> bool:
        return not bool(self.data)


class FrameSerializer(Protocol):
    """
    Protocol for converting VideoFrame objects into a representation
    suitable for a local vision runtime.
    """

    name: str

    def serialize(
        self,
        frame: VideoFrame,
    ) -> SerializedFrame:
        """
        Serialize one video frame without modifying the original frame.
        """
        ...


class Base64FrameSerializer:
    """
    Serialize video-frame image bytes into Base64.

    Base64 is intentionally confined to this boundary. The core
    VideoFrame contract continues to use raw bytes and therefore
    remains independent of any particular model runtime.
    """

    name = "base64"

    def serialize(
        self,
        frame: VideoFrame,
    ) -> SerializedFrame:
        if frame is None:
            raise ValueError("Video frame must be provided.")

        if not frame.image:
            raise ValueError("Video frame image must not be empty.")

        image_bytes = bytes(frame.image)

        if not self._is_supported_image(image_bytes):
            raise ValueError(
                "Video frame image must contain a supported JPEG or PNG image."
            )

        mime_type = self._detect_mime_type(image_bytes)

        encoded = base64.b64encode(image_bytes).decode("ascii")

        return SerializedFrame(
            frame_index=frame.frame_index,
            timestamp_seconds=frame.timestamp_seconds,
            mime_type=mime_type,
            data=encoded,
        )

    @staticmethod
    def _is_supported_image(image: bytes) -> bool:
        return (
            Base64FrameSerializer._is_jpeg(image)
            or Base64FrameSerializer._is_png(image)
        )

    @staticmethod
    def _is_jpeg(image: bytes) -> bool:
        return (
            len(image) >= 4
            and image[:2] == b"\xff\xd8"
            and image[-2:] == b"\xff\xd9"
        )

    @staticmethod
    def _is_png(image: bytes) -> bool:
        return (
            len(image) >= 8
            and image[:8]
            == b"\x89PNG\r\n\x1a\n"
        )

    @staticmethod
    def _detect_mime_type(image: bytes) -> str:
        if Base64FrameSerializer._is_jpeg(image):
            return "image/jpeg"

        if Base64FrameSerializer._is_png(image):
            return "image/png"

        raise ValueError("Unsupported image format.")

    @staticmethod
    def decode(
        serialized: SerializedFrame,
    ) -> bytes:
        """
        Decode a serialized frame.

        This helper exists primarily for provider/runtime adapters and
        tests. It does not modify the SerializedFrame.
        """

        if serialized is None:
            raise ValueError("Serialized frame must be provided.")

        if not serialized.data:
            raise ValueError("Serialized frame data must not be empty.")

        try:
            return base64.b64decode(
                serialized.data,
                validate=True,
            )
        except (ValueError, binascii.Error) as exc:
            raise ValueError(
                "Serialized frame contains invalid Base64 data."
            ) from exc