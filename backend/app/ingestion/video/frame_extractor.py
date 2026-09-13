from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from app.ingestion.video.sampling import FrameSamplingPolicy
from app.ingestion.video.schemas import (
    FrameExtractionRequest,
    FrameExtractionResult,
    FrameExtractionStatus,
    VideoFrame,
    VideoInfo,
)


class FrameExtractionError(RuntimeError):
    """Raised when FFmpeg frame extraction fails."""


class FFmpegFrameExtractor:
    """
    FFmpeg-backed video frame extractor.

    Responsibilities:
    - Generate deterministic frame timestamps.
    - Extract frames using FFmpeg.
    - Preserve the requested video timeline.
    - Decode extracted frames into bytes.
    - Return provider-independent VideoFrame objects.
    - Enforce subprocess timeout and frame limits.

    This class does not perform:
    - Vision analysis
    - OCR
    - ASR
    - scene detection
    - multimodal fusion
    """

    name = "ffmpeg"

    DEFAULT_TIMEOUT_SECONDS = 120.0

    def __init__(
        self,
        *,
        sampling_policy: FrameSamplingPolicy | None = None,
        ffmpeg_binary: str = "ffmpeg",
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be greater than zero."
            )

        self.sampling_policy = (
            sampling_policy
            or FrameSamplingPolicy()
        )

        self.ffmpeg_binary = ffmpeg_binary
        self.timeout_seconds = timeout_seconds

    async def extract(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: FrameExtractionRequest | None = None,
    ) -> FrameExtractionResult:
        """
        Extract representative frames from video bytes.
        """

        if not media:
            raise FrameExtractionError(
                "Video content must not be empty."
            )

        if video_info.video is None:
            return FrameExtractionResult(
                status=FrameExtractionStatus.NO_VIDEO,
                metadata={
                    "provider": self.name,
                },
            )

        duration = video_info.duration_seconds

        if duration is None:
            raise FrameExtractionError(
                "Video duration is required for frame extraction."
            )

        request = (
            request
            or FrameExtractionRequest()
        )

        timestamps = (
            self.sampling_policy.generate_timestamps(
                duration_seconds=duration,
                request=request,
            )
        )

        if not timestamps:
            return FrameExtractionResult(
                status=FrameExtractionStatus.NO_FRAMES,
                requested_interval_seconds=(
                    request.interval_seconds
                ),
                start_time_seconds=(
                    request.start_time_seconds
                ),
                end_time_seconds=(
                    request.end_time_seconds
                ),
                total_frames=0,
                metadata={
                    "provider": self.name,
                },
            )

        try:
            frames = await self._extract_frames(
                media=media,
                timestamps=timestamps,
                video_info=video_info,
            )

        except asyncio.TimeoutError:
            return FrameExtractionResult(
                status=FrameExtractionStatus.FAILED,
                requested_interval_seconds=(
                    request.interval_seconds
                ),
                start_time_seconds=(
                    request.start_time_seconds
                ),
                end_time_seconds=(
                    request.end_time_seconds
                ),
                errors=[
                    "FFmpeg frame extraction timed out."
                ],
                metadata={
                    "provider": self.name,
                    "timeout_seconds": (
                        self.timeout_seconds
                    ),
                },
            )

        except Exception as exc:
            return FrameExtractionResult(
                status=FrameExtractionStatus.FAILED,
                requested_interval_seconds=(
                    request.interval_seconds
                ),
                start_time_seconds=(
                    request.start_time_seconds
                ),
                end_time_seconds=(
                    request.end_time_seconds
                ),
                errors=[
                    str(exc)
                ],
                metadata={
                    "provider": self.name,
                    "error_type": type(exc).__name__,
                },
            )

        if not frames:
            return FrameExtractionResult(
                status=FrameExtractionStatus.NO_FRAMES,
                requested_interval_seconds=(
                    request.interval_seconds
                ),
                start_time_seconds=(
                    request.start_time_seconds
                ),
                end_time_seconds=(
                    request.end_time_seconds
                ),
                total_frames=0,
                metadata={
                    "provider": self.name,
                },
            )

        status = (
            FrameExtractionStatus.COMPLETED
            if len(frames) == len(timestamps)
            else FrameExtractionStatus.PARTIAL
        )

        actual_interval = None

        if len(frames) >= 2:
            actual_interval = (
                frames[1].timestamp_seconds
                - frames[0].timestamp_seconds
            )

        return FrameExtractionResult(
            status=status,
            frames=frames,
            requested_interval_seconds=(
                request.interval_seconds
            ),
            actual_interval_seconds=actual_interval,
            start_time_seconds=(
                request.start_time_seconds
            ),
            end_time_seconds=(
                request.end_time_seconds
            ),
            total_frames=len(frames),
            metadata={
                "provider": self.name,
                "requested_frame_count": len(
                    timestamps
                ),
                "extracted_frame_count": len(
                    frames
                ),
            },
        )

    async def _extract_frames(
        self,
        *,
        media: bytes,
        timestamps: list[float],
        video_info: VideoInfo,
    ) -> list[VideoFrame]:
        """
        Execute FFmpeg and decode timestamped frames.

        FFmpeg outputs one JPEG image per requested timestamp
        using a temporary output directory.
        """

        with tempfile.TemporaryDirectory(
            prefix="video_frames_"
        ) as temp_dir:

            temp_path = Path(temp_dir)

            input_path = (
                temp_path / "input_video"
            )

            output_pattern = (
                temp_path / "frame_%06d.jpg"
            )

            input_path.write_bytes(
                media
            )

            filter_expression = self._build_select_filter(
                timestamps=timestamps
            )

            command = [
                self.ffmpeg_binary,
                "-hide_banner",
                "-loglevel",
                "error",
                "-i",
                str(input_path),
                "-vf",
                filter_expression,
                "-vsync",
                "vfr",
                "-frames:v",
                str(len(timestamps)),
                "-q:v",
                "2",
                str(output_pattern),
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

                await process.communicate()

                raise

            if process.returncode != 0:
                error_message = (
                    stderr.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                )

                raise FrameExtractionError(
                    "FFmpeg frame extraction failed: "
                    + (
                        error_message
                        or "unknown FFmpeg error"
                    )
                )

            output_files = sorted(
                temp_path.glob(
                    "frame_*.jpg"
                )
            )

            if not output_files:
                return []

            frames: list[VideoFrame] = []

            for index, output_file in enumerate(
                output_files
            ):
                if index >= len(timestamps):
                    break

                image_bytes = output_file.read_bytes()

                if not image_bytes:
                    continue

                width, height = (
                    self._read_jpeg_dimensions(
                        image_bytes
                    )
                )

                frames.append(
                    VideoFrame(
                        frame_index=index,
                        timestamp_seconds=timestamps[
                            index
                        ],
                        image=image_bytes,
                        width=width,
                        height=height,
                        metadata={
                            "provider": self.name,
                            "format": "jpeg",
                            "source_timestamp_seconds": (
                                timestamps[index]
                            ),
                        },
                    )
                )

            return frames

    @staticmethod
    def _build_select_filter(
        *,
        timestamps: list[float],
    ) -> str:
        """
        Build a deterministic FFmpeg timestamp selection filter.
        """

        expressions = [
            f"eq(t\\,{timestamp:.9f})"
            for timestamp in timestamps
        ]

        return (
            "select='"
            + "+".join(expressions)
            + "'"
        )

    @staticmethod
    def _read_jpeg_dimensions(
        image_bytes: bytes,
    ) -> tuple[int | None, int | None]:
        """
        Read JPEG dimensions without requiring Pillow.

        This is intentionally lightweight. Full image validation
        belongs to the frame-validation stage.
        """

        if not image_bytes.startswith(
            b"\xff\xd8"
        ):
            return None, None

        index = 2
        length = len(image_bytes)

        while index < length:
            while (
                index < length
                and image_bytes[index] != 0xFF
            ):
                index += 1

            if index + 1 >= length:
                break

            marker = image_bytes[
                index + 1
            ]

            index += 2

            if marker in (
                0xD8,
                0xD9,
            ):
                continue

            if index + 2 > length:
                break

            segment_length = int.from_bytes(
                image_bytes[
                    index:index + 2
                ],
                byteorder="big",
            )

            if segment_length < 2:
                break

            if marker in {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }:
                if index + 7 > length:
                    break

                height = int.from_bytes(
                    image_bytes[
                        index + 3:index + 5
                    ],
                    byteorder="big",
                )

                width = int.from_bytes(
                    image_bytes[
                        index + 5:index + 7
                    ],
                    byteorder="big",
                )

                return width, height

            index += segment_length

        return None, None