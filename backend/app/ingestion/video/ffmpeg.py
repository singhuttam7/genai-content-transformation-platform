from __future__ import annotations

import asyncio
import math
import shutil
import tempfile
from pathlib import Path

from app.ingestion.video.schemas import AudioExtractionRequest


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