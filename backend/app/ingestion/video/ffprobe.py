from __future__ import annotations

import asyncio
import json
import math
import shutil
import tempfile
from pathlib import Path
from typing import Any

from app.ingestion.video.provider import VideoInspector
from app.ingestion.video.schemas import (
    AudioStreamInfo,
    VideoInfo,
    VideoStreamInfo,
)


class FFprobeVideoInspector:
    """
    VideoInspector implementation backed by FFprobe.

    This implementation is intentionally separate from the generic
    MediaInspector implementation because video inspection requires
    video-stream-specific metadata such as resolution, frame rate,
    pixel format, and video codec.
    """

    name = "ffprobe"

    def __init__(
        self,
        *,
        executable_path: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be a finite value greater than zero."
            )

        self.executable_path = executable_path
        self.timeout_seconds = timeout_seconds

    def _resolve_executable(self) -> str:
        if self.executable_path:
            executable = Path(self.executable_path)

            if not executable.is_file():
                raise FileNotFoundError(
                    f"FFprobe executable not found: {executable}"
                )

            return str(executable)

        executable = shutil.which("ffprobe")

        if executable is None:
            raise FileNotFoundError(
                "FFprobe executable was not found on PATH."
            )

        return executable

    async def inspect(
        self,
        media: bytes,
        *,
        filename: str | None = None,
        mime_type: str | None = None,
    ) -> VideoInfo:
        """
        Inspect a video and return provider-independent VideoInfo.
        """

        if not isinstance(media, bytes):
            raise TypeError("media must be bytes")

        if not media:
            raise ValueError("media must not be empty")

        executable = self._resolve_executable()

        suffix = Path(filename or "").suffix

        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(media)
                temporary_path = Path(temporary_file.name)

            command = [
                executable,
                "-v",
                "error",
                "-show_entries",
                (
                    "format="
                    "format_name,"
                    "duration,"
                    "bit_rate,"
                    "size"
                    ":stream="
                    "index,"
                    "codec_type,"
                    "codec_name,"
                    "width,"
                    "height,"
                    "r_frame_rate,"
                    "avg_frame_rate,"
                    "bit_rate,"
                    "pix_fmt,"
                    "sample_rate,"
                    "channels"
                ),
                "-of",
                "json",
                str(temporary_path),
            ]

            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()

                raise TimeoutError(
                    "FFprobe video inspection timed out."
                )

            if process.returncode != 0:
                error_message = stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                raise ValueError(
                    "FFprobe could not inspect the video."
                    + (
                        f" {error_message}"
                        if error_message
                        else ""
                    )
                )

            try:
                payload = json.loads(
                    stdout.decode(
                        "utf-8",
                        errors="replace",
                    )
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "FFprobe returned invalid JSON."
                ) from exc

            return self._parse_video_info(
                payload,
                size_bytes=len(media),
                mime_type=mime_type,
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True,
                )

    @classmethod
    def _parse_video_info(
        cls,
        payload: dict[str, Any],
        *,
        size_bytes: int,
        mime_type: str | None,
    ) -> VideoInfo:
        format_data = payload.get("format")

        if not isinstance(format_data, dict):
            raise ValueError(
                "FFprobe response does not contain format information."
            )

        streams = payload.get("streams", [])

        if not isinstance(streams, list):
            streams = []

        video_stream = cls._select_stream(
            streams,
            codec_type="video",
        )

        if video_stream is None:
            raise ValueError(
                "FFprobe response does not contain a video stream."
            )

        audio_stream = cls._select_stream(
            streams,
            codec_type="audio",
        )

        video_info = cls._parse_video_stream(
            video_stream,
        )

        audio_info = (
            cls._parse_audio_stream(audio_stream)
            if audio_stream is not None
            else None
        )

        return VideoInfo(
            format_name=cls._string_value(
                format_data.get("format_name")
            ),
            duration_seconds=cls._non_negative_float(
                format_data.get("duration")
            ),
            size_bytes=size_bytes,
            bitrate=cls._positive_int(
                format_data.get("bit_rate")
            ),
            video=video_info,
            audio=audio_info,
            metadata={
                "mime_type": mime_type,
                "stream_count": len(streams),
            },
        )

    @staticmethod
    def _select_stream(
        streams: list[object],
        *,
        codec_type: str,
    ) -> dict[str, Any] | None:
        candidates = [
            stream
            for stream in streams
            if isinstance(stream, dict)
            and stream.get("codec_type") == codec_type
        ]

        if not candidates:
            return None

        def stream_index(stream: dict[str, Any]) -> int:
            value = stream.get("index")

            try:
                return int(value)
            except (TypeError, ValueError):
                return 0

        return min(
            candidates,
            key=stream_index,
        )

    @classmethod
    def _parse_video_stream(
        cls,
        stream: dict[str, Any],
    ) -> VideoStreamInfo:
        frame_rate = cls._parse_frame_rate(
            stream.get("avg_frame_rate")
        )

        if frame_rate is None:
            frame_rate = cls._parse_frame_rate(
                stream.get("r_frame_rate")
            )

        return VideoStreamInfo(
            codec_name=cls._string_value(
                stream.get("codec_name")
            ),
            width=cls._positive_int(
                stream.get("width")
            ),
            height=cls._positive_int(
                stream.get("height")
            ),
            frame_rate=frame_rate,
            bitrate=cls._positive_int(
                stream.get("bit_rate")
            ),
            pixel_format=cls._string_value(
                stream.get("pix_fmt")
            ),
            metadata={
                "stream_index": stream.get("index"),
            },
        )

    @classmethod
    def _parse_audio_stream(
        cls,
        stream: dict[str, Any],
    ) -> AudioStreamInfo:
        return AudioStreamInfo(
            codec_name=cls._string_value(
                stream.get("codec_name")
            ),
            sample_rate=cls._positive_int(
                stream.get("sample_rate")
            ),
            channels=cls._positive_int(
                stream.get("channels")
            ),
            bitrate=cls._positive_int(
                stream.get("bit_rate")
            ),
            metadata={
                "stream_index": stream.get("index"),
            },
        )

    @staticmethod
    def _string_value(
        value: object,
    ) -> str | None:
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized or normalized.upper() == "N/A":
            return None

        return normalized

    @staticmethod
    def _non_negative_float(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(parsed) or parsed < 0:
            return None

        return parsed

    @staticmethod
    def _positive_int(
        value: object,
    ) -> int | None:
        if value is None:
            return None

        try:
            parsed = int(float(value))
        except (TypeError, ValueError):
            return None

        if parsed <= 0:
            return None

        return parsed

    @staticmethod
    def _parse_frame_rate(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        normalized = str(value).strip()

        if not normalized or normalized.upper() == "N/A":
            return None

        if "/" in normalized:
            numerator, denominator = normalized.split(
                "/",
                maxsplit=1,
            )

            try:
                numerator_value = float(numerator)
                denominator_value = float(denominator)
            except (TypeError, ValueError):
                return None

            if denominator_value == 0:
                return None

            parsed = numerator_value / denominator_value

        else:
            try:
                parsed = float(normalized)
            except (TypeError, ValueError):
                return None

        if not math.isfinite(parsed) or parsed <= 0:
            return None

        return parsed