from __future__ import annotations

import asyncio
import math
import shutil
import tempfile
from pathlib import Path

from app.ingestion.video.sampling import FrameSamplingPolicy
from app.ingestion.video.schemas import (
    AudioExtractionRequest,
    FrameExtractionRequest,
    FrameExtractionResult,
    FrameExtractionStatus,
    VideoFrame,
    VideoInfo,
)


# ============================================================
# AUDIO EXTRACTION
# ============================================================


class FFmpegAudioExtractionError(RuntimeError):
    """Raised when FFmpeg cannot extract audio from a video."""


class FFmpegAudioExtractor:
    """
    AudioExtractor implementation backed by FFmpeg.

    The extractor converts the first available audio stream into
    canonical WAV audio suitable for downstream speech processing.

    Default output:
        - WAV
        - PCM signed 16-bit
        - mono
        - 16 kHz
    """

    name = "ffmpeg"

    SUPPORTED_FORMATS = frozenset({"wav"})

    DEFAULT_MAX_OUTPUT_BYTES = 100 * 1024 * 1024

    def __init__(
        self,
        *,
        executable_path: str | None = None,
        timeout_seconds: float = 120.0,
        max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be a finite value greater than zero."
            )

        if max_output_bytes <= 0:
            raise ValueError(
                "max_output_bytes must be greater than zero."
            )

        self.executable_path = executable_path
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def _resolve_executable(self) -> str:
        """
        Resolve FFmpeg executable from an explicit path or PATH.
        """

        if self.executable_path:
            executable = Path(self.executable_path)

            if not executable.is_file():
                raise FileNotFoundError(
                    f"FFmpeg executable not found: {executable}"
                )

            return str(executable)

        executable = shutil.which("ffmpeg")

        if executable is None:
            raise FileNotFoundError(
                "FFmpeg executable was not found on PATH."
            )

        return executable

    async def extract(
        self,
        media: bytes,
        *,
        request: AudioExtractionRequest | None = None,
        filename: str | None = None,
    ) -> bytes:
        """
        Extract canonical WAV audio from video bytes.
        """

        if not isinstance(media, bytes):
            raise TypeError("media must be bytes")

        if not media:
            raise ValueError("media must not be empty")

        request = request or AudioExtractionRequest()

        self._validate_request(request)

        executable = self._resolve_executable()

        input_suffix = Path(filename or "").suffix

        temporary_input: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=input_suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(media)
                temporary_input = Path(temporary_file.name)

            command = self._build_command(
                executable=executable,
                input_path=temporary_input,
                request=request,
            )

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
                    "FFmpeg audio extraction timed out."
                )

            if process.returncode != 0:
                error_message = stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                raise FFmpegAudioExtractionError(
                    "FFmpeg could not extract audio."
                    + (
                        f" {error_message}"
                        if error_message
                        else ""
                    )
                )

            if not stdout:
                raise FFmpegAudioExtractionError(
                    "FFmpeg produced empty audio output."
                )

            if len(stdout) > self.max_output_bytes:
                raise FFmpegAudioExtractionError(
                    "FFmpeg audio output exceeds the configured "
                    f"maximum size of {self.max_output_bytes} bytes."
                )

            self._validate_wav_output(stdout)

            return stdout

        finally:
            if temporary_input is not None:
                temporary_input.unlink(
                    missing_ok=True,
                )

    def _build_command(
        self,
        *,
        executable: str,
        input_path: Path,
        request: AudioExtractionRequest,
    ) -> list[str]:
        """
        Construct an explicit FFmpeg argument list.

        No shell is used, preventing shell interpretation of
        user-controlled filenames or paths.
        """

        command = [
            executable,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
        ]

        if request.start_time_seconds > 0:
            command.extend(
                [
                    "-ss",
                    self._format_seconds(
                        request.start_time_seconds
                    ),
                ]
            )

        command.extend(
            [
                "-i",
                str(input_path),
                "-map",
                "0:a:0",
                "-vn",
                "-sn",
                "-dn",
                "-ac",
                str(request.channels),
                "-ar",
                str(request.sample_rate),
            ]
        )

        if request.end_time_seconds is not None:
            duration = (
                request.end_time_seconds
                - request.start_time_seconds
            )

            if duration <= 0:
                raise ValueError(
                    "end_time_seconds must be greater than "
                    "start_time_seconds."
                )

            command.extend(
                [
                    "-t",
                    self._format_seconds(duration),
                ]
            )

        if request.audio_format == "wav":
            command.extend(
                [
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    "pipe:1",
                ]
            )
        else:
            raise ValueError(
                f"Unsupported audio format: {request.audio_format}"
            )

        return command

    @staticmethod
    def _validate_request(
        request: AudioExtractionRequest,
    ) -> None:
        audio_format = request.audio_format.strip().lower()

        if audio_format not in FFmpegAudioExtractor.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported audio format: {request.audio_format}"
            )

        if request.end_time_seconds is not None:
            if request.end_time_seconds <= request.start_time_seconds:
                raise ValueError(
                    "end_time_seconds must be greater than "
                    "start_time_seconds."
                )

    @staticmethod
    def _format_seconds(
        value: float,
    ) -> str:
        return f"{value:.6f}".rstrip("0").rstrip(".")

    @staticmethod
    def _validate_wav_output(
        audio: bytes,
    ) -> None:
        """
        Perform lightweight WAV container validation.

        FFmpeg remains responsible for producing the actual audio.
        """

        if len(audio) < 44:
            raise FFmpegAudioExtractionError(
                "FFmpeg output is too small to be a valid WAV file."
            )

        if audio[:4] != b"RIFF":
            raise FFmpegAudioExtractionError(
                "FFmpeg output is not a RIFF WAV file."
            )

        if audio[8:12] != b"WAVE":
            raise FFmpegAudioExtractionError(
                "FFmpeg output does not contain a WAVE header."
            )


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================


class FFmpegFrameExtractionError(RuntimeError):
    """Raised when FFmpeg cannot extract video frames."""


class FFmpegFrameExtractor:
    """
    FrameExtractor implementation backed by FFmpeg.

    Responsibilities:
        - Generate deterministic sampling timestamps.
        - Extract JPEG frames using FFmpeg.
        - Preserve the requested source timeline.
        - Respect the maximum frame limit.
        - Enforce subprocess timeout.
        - Validate extracted JPEG data.
        - Avoid permanent intermediate files.
        - Return provider-independent frame contracts.

    This class does not perform:
        - Vision inference
        - OCR
        - ASR
        - scene detection
        - temporal fusion
    """

    name = "ffmpeg"

    DEFAULT_TIMEOUT_SECONDS = 120.0

    DEFAULT_JPEG_QUALITY = 2

    def __init__(
        self,
        *,
        executable_path: str | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        sampling_policy: FrameSamplingPolicy | None = None,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds must be a finite value greater than zero."
            )

        self.executable_path = executable_path
        self.timeout_seconds = timeout_seconds
        self.sampling_policy = (
            sampling_policy
            or FrameSamplingPolicy()
        )

    # --------------------------------------------------------
    # EXECUTABLE
    # --------------------------------------------------------

    def _resolve_executable(self) -> str:
        """
        Resolve FFmpeg executable from an explicit path or PATH.
        """

        if self.executable_path:
            executable = Path(self.executable_path)

            if not executable.is_file():
                raise FileNotFoundError(
                    f"FFmpeg executable not found: {executable}"
                )

            return str(executable)

        executable = shutil.which("ffmpeg")

        if executable is None:
            raise FileNotFoundError(
                "FFmpeg executable was not found on PATH."
            )

        return executable

    # --------------------------------------------------------
    # PUBLIC EXTRACTION API
    # --------------------------------------------------------

    async def extract(
        self,
        media: bytes,
        *,
        video_info: VideoInfo,
        request: FrameExtractionRequest | None = None,
        filename: str | None = None,
    ) -> FrameExtractionResult:
        """
        Extract timestamped JPEG frames from video bytes.
        """

        if not isinstance(media, bytes):
            raise TypeError("media must be bytes")

        if not media:
            raise ValueError("media must not be empty")

        request = (
            request
            or FrameExtractionRequest()
        )

        if video_info.video is None:
            return FrameExtractionResult(
                status=FrameExtractionStatus.NO_VIDEO,
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

        duration = video_info.duration_seconds

        if duration is None:
            raise ValueError(
                "video_info.duration_seconds is required "
                "for frame extraction."
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

        executable = self._resolve_executable()

        input_suffix = Path(
            filename or ""
        ).suffix

        temporary_input: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                suffix=input_suffix,
                delete=False,
            ) as temporary_file:
                temporary_file.write(media)
                temporary_input = Path(
                    temporary_file.name
                )

            return await self._extract_frames(
                executable=executable,
                input_path=temporary_input,
                timestamps=timestamps,
                video_info=video_info,
                request=request,
            )

        finally:
            if temporary_input is not None:
                temporary_input.unlink(
                    missing_ok=True,
                )

    # --------------------------------------------------------
    # FFMPEG EXECUTION
    # --------------------------------------------------------

    async def _extract_frames(
        self,
        *,
        executable: str,
        input_path: Path,
        timestamps: list[float],
        video_info: VideoInfo,
        request: FrameExtractionRequest,
    ) -> FrameExtractionResult:
        """
        Execute FFmpeg and collect the extracted JPEG frames.

        Timestamp windows are used instead of exact floating-point
        timestamp equality because encoded video timestamps do not
        necessarily match requested timestamps exactly.
        """

        with tempfile.TemporaryDirectory(
            prefix="video_frames_",
        ) as temporary_directory:

            output_directory = Path(
                temporary_directory
            )

            output_pattern = (
                output_directory
                / "frame_%06d.jpg"
            )

            select_expression = (
                self._build_select_expression(
                    timestamps=timestamps,
                    video_info=video_info,
                )
            )

            command = [
                executable,
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-i",
                str(input_path),
                "-map",
                "0:v:0",
                "-an",
                "-sn",
                "-dn",
                "-vf",
                select_expression,

                # FFmpeg 9.x uses fps_mode instead of
                # the deprecated/removed vsync option.
                "-fps_mode",
                "vfr",

                "-frames:v",
                str(len(timestamps)),
                "-q:v",
                str(self.DEFAULT_JPEG_QUALITY),
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

                try:
                    await process.wait()
                except ProcessLookupError:
                    pass

                raise TimeoutError(
                    "FFmpeg frame extraction timed out."
                )

            if process.returncode != 0:
                error_message = stderr.decode(
                    "utf-8",
                    errors="replace",
                ).strip()

                raise FFmpegFrameExtractionError(
                    "FFmpeg could not extract video frames."
                    + (
                        f" {error_message}"
                        if error_message
                        else ""
                    )
                )

            output_files = sorted(
                output_directory.glob(
                    "frame_*.jpg"
                )
            )

            if not output_files:
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
                    errors=[
                        "FFmpeg completed successfully "
                        "but produced no frames."
                    ],
                    metadata={
                        "provider": self.name,
                    },
                )

            frames: list[VideoFrame] = []

            extraction_count = min(
                len(output_files),
                len(timestamps),
                request.max_frames,
            )

            for index in range(extraction_count):
                output_file = output_files[index]

                image_bytes = output_file.read_bytes()

                self._validate_jpeg_output(
                    image_bytes
                )

                width, height = (
                    self._read_jpeg_dimensions(
                        image_bytes
                    )
                )

                frames.append(
                    VideoFrame(
                        timestamp_seconds=(
                            timestamps[index]
                        ),
                        frame_index=index,
                        image=image_bytes,
                        width=width,
                        height=height,
                        metadata={
                            "provider": self.name,
                            "format": "jpeg",
                            "requested_timestamp_seconds": (
                                timestamps[index]
                            ),
                        },
                    )
                )

            if len(frames) == len(timestamps):
                status = (
                    FrameExtractionStatus.COMPLETED
                )
                errors: list[str] = []

            else:
                status = (
                    FrameExtractionStatus.PARTIAL
                )
                errors = [
                    "FFmpeg extracted fewer frames "
                    "than requested."
                ]

            actual_interval = None

            if len(frames) >= 2:
                intervals = [
                    frames[index].timestamp_seconds
                    - frames[index - 1].timestamp_seconds
                    for index in range(
                        1,
                        len(frames),
                    )
                ]

                if intervals:
                    actual_interval = (
                        sum(intervals)
                        / len(intervals)
                    )

            return FrameExtractionResult(
                status=status,
                frames=frames,
                requested_interval_seconds=(
                    request.interval_seconds
                ),
                actual_interval_seconds=(
                    actual_interval
                ),
                start_time_seconds=(
                    request.start_time_seconds
                ),
                end_time_seconds=(
                    request.end_time_seconds
                ),
                total_frames=len(frames),
                errors=errors,
                metadata={
                    "provider": self.name,
                    "requested_frame_count": (
                        len(timestamps)
                    ),
                    "extracted_frame_count": (
                        len(frames)
                    ),
                    "max_frames": (
                        request.max_frames
                    ),
                },
            )

    # --------------------------------------------------------
    # TIMESTAMP SELECTION
    # --------------------------------------------------------

    @staticmethod
    def _build_select_expression(
        *,
        timestamps: list[float],
        video_info: VideoInfo,
    ) -> str:
        """
        Build a tolerant FFmpeg timestamp selection expression.

        Exact floating-point equality is intentionally avoided.

        For a 30 FPS source, half a frame duration is approximately
        16.67 milliseconds. This allows FFmpeg to select the frame
        nearest to the requested timeline timestamp.
        """

        frame_rate = None

        if video_info.video is not None:
            frame_rate = (
                video_info.video.frame_rate
            )

        if frame_rate is None or frame_rate <= 0:
            frame_rate = 30.0

        frame_duration = (
            1.0 / frame_rate
        )

        tolerance = max(
            frame_duration / 2.0,
            0.001,
        )

        expressions: list[str] = []

        for timestamp in timestamps:
            start = max(
                timestamp - tolerance,
                0.0,
            )

            end = (
                timestamp + tolerance
            )

            expressions.append(
                "between("
                f"t\\,{start:.9f}\\,{end:.9f}"
                ")"
            )

        return (
            "select='"
            + "+".join(expressions)
            + "'"
        )

    # --------------------------------------------------------
    # JPEG VALIDATION
    # --------------------------------------------------------

    @staticmethod
    def _validate_jpeg_output(
        image: bytes,
    ) -> None:
        """
        Perform lightweight JPEG validation.

        This deliberately does not depend on Pillow.
        """

        if len(image) < 4:
            raise FFmpegFrameExtractionError(
                "FFmpeg produced an invalid JPEG frame."
            )

        if not image.startswith(
            b"\xff\xd8"
        ):
            raise FFmpegFrameExtractionError(
                "FFmpeg output is not a JPEG image."
            )

        if not image.endswith(
            b"\xff\xd9"
        ):
            raise FFmpegFrameExtractionError(
                "FFmpeg produced an incomplete JPEG frame."
            )

    # --------------------------------------------------------
    # JPEG DIMENSIONS
    # --------------------------------------------------------

    @staticmethod
    def _read_jpeg_dimensions(
        image_bytes: bytes,
    ) -> tuple[int | None, int | None]:
        """
        Read JPEG dimensions without requiring Pillow.

        Returns:
            (width, height)

        If dimensions cannot be determined:
            (None, None)
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

            if marker in {
                0xD8,
                0xD9,
            }:
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