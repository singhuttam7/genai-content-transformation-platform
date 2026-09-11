from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from pathlib import Path

from app.ingestion.media.schemas import MediaInfo


class FFprobeMediaInspector:
    """
    MediaInspector implementation backed by FFprobe.

    FFprobe is used only for technical media inspection.
    The rest of the application receives the provider-independent
    MediaInfo model.
    """

    name = "ffprobe"

    def __init__(
        self,
        *,
        executable_path: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
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
    ) -> MediaInfo:
        """
        Inspect media bytes using FFprobe.

        A temporary file is used because FFprobe operates reliably
        on filesystem media paths across supported platforms.
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
                    "codec_type,"
                    "codec_name,"
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
                    "FFprobe inspection timed out."
                )

            if process.returncode != 0:
                error_message = (
                    stderr.decode(
                        "utf-8",
                        errors="replace",
                    ).strip()
                )

                raise ValueError(
                    "FFprobe could not inspect the media."
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

            return self._parse_media_info(
                payload,
                size_bytes=len(media),
                mime_type=mime_type,
            )

        finally:
            if temporary_path is not None:
                temporary_path.unlink(
                    missing_ok=True,
                )

    @staticmethod
    def _parse_media_info(
        payload: dict[str, object],
        *,
        size_bytes: int,
        mime_type: str | None,
    ) -> MediaInfo:
        format_data = payload.get("format")

        if not isinstance(format_data, dict):
            raise ValueError(
                "FFprobe response does not contain format information."
            )

        format_name = FFprobeMediaInspector._string_value(
            format_data.get("format_name")
        )

        duration = FFprobeMediaInspector._float_value(
            format_data.get("duration")
        )

        bitrate = FFprobeMediaInspector._int_value(
            format_data.get("bit_rate")
        )

        streams = payload.get("streams", [])

        if not isinstance(streams, list):
            streams = []

        audio_stream = next(
            (
                stream
                for stream in streams
                if isinstance(stream, dict)
                and stream.get("codec_type") == "audio"
            ),
            None,
        )

        codec_name = None
        sample_rate = None
        channels = None

        if isinstance(audio_stream, dict):
            codec_name = (
                FFprobeMediaInspector._string_value(
                    audio_stream.get("codec_name")
                )
            )

            sample_rate = (
                FFprobeMediaInspector._int_value(
                    audio_stream.get("sample_rate")
                )
            )

            channels = (
                FFprobeMediaInspector._int_value(
                    audio_stream.get("channels")
                )
            )

        return MediaInfo(
            format_name=format_name,
            codec_name=codec_name,
            duration_seconds=duration,
            sample_rate=sample_rate,
            channels=channels,
            bitrate=bitrate,
            size_bytes=size_bytes,
            metadata={
                "mime_type": mime_type,
            },
        )

    @staticmethod
    def _string_value(
        value: object,
    ) -> str | None:
        if value is None:
            return None

        value = str(value).strip()

        return value or None

    @staticmethod
    def _float_value(
        value: object,
    ) -> float | None:
        if value is None:
            return None

        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _int_value(
        value: object,
    ) -> int | None:
        if value is None:
            return None

        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None