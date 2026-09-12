from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictBytes

from app.ingestion.speech.schemas import ASRResult


class VideoProcessingStatus(StrEnum):
    """
    High-level status of video processing.
    """

    NOT_REQUESTED = "not_requested"
    PROCESSING = "processing"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"


class VideoStreamInfo(BaseModel):
    """
    Metadata describing the primary video stream.
    """

    model_config = ConfigDict(extra="forbid")

    codec_name: str | None = None

    width: int | None = Field(
        default=None,
        ge=1,
    )

    height: int | None = Field(
        default=None,
        ge=1,
    )

    frame_rate: float | None = Field(
        default=None,
        ge=0,
    )

    bitrate: int | None = Field(
        default=None,
        ge=0,
    )

    pixel_format: str | None = None

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class AudioStreamInfo(BaseModel):
    """
    Metadata describing the primary audio stream.
    """

    model_config = ConfigDict(extra="forbid")

    codec_name: str | None = None

    sample_rate: int | None = Field(
        default=None,
        ge=1,
    )

    channels: int | None = Field(
        default=None,
        ge=1,
    )

    bitrate: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class VideoInfo(BaseModel):
    """
    Complete media information for a video asset.
    """

    model_config = ConfigDict(extra="forbid")

    format_name: str | None = None

    duration_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    size_bytes: int | None = Field(
        default=None,
        ge=0,
    )

    bitrate: int | None = Field(
        default=None,
        ge=0,
    )

    video: VideoStreamInfo | None = None

    audio: AudioStreamInfo | None = None

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class VideoFrame(BaseModel):
    """
    A single extracted video frame.

    The timestamp identifies where the frame belongs in the
    original video timeline.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp_seconds: float = Field(
        ge=0,
    )

    frame_index: int = Field(
        ge=0,
    )

    image: StrictBytes

    width: int | None = Field(
        default=None,
        ge=1,
    )

    height: int | None = Field(
        default=None,
        ge=1,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class FrameExtractionRequest(BaseModel):
    """
    Parameters controlling video frame extraction.
    """

    model_config = ConfigDict(extra="forbid")

    interval_seconds: float = Field(
        default=2.0,
        gt=0,
    )

    max_frames: int = Field(
        default=300,
        ge=1,
    )

    start_time_seconds: float = Field(
        default=0.0,
        ge=0,
    )

    end_time_seconds: float | None = Field(
        default=None,
        gt=0,
    )


class AudioExtractionRequest(BaseModel):
    """
    Parameters controlling extraction of audio from video.
    """

    model_config = ConfigDict(extra="forbid")

    sample_rate: int = Field(
        default=16_000,
        ge=1,
    )

    channels: int = Field(
        default=1,
        ge=1,
    )

    audio_format: str = "wav"

    start_time_seconds: float = Field(
        default=0.0,
        ge=0,
    )

    end_time_seconds: float | None = Field(
        default=None,
        gt=0,
    )


class VideoASRStatus(StrEnum):
    """
    High-level status of video-to-ASR processing.

    This status represents the complete video ASR workflow,
    including audio availability, audio extraction, and
    downstream speech recognition.
    """

    NOT_REQUESTED = "not_requested"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NO_AUDIO = "no_audio"
    NO_SPEECH = "no_speech"
    AUDIO_EXTRACTION_FAILED = "audio_extraction_failed"
    ASR_FAILED = "asr_failed"


class VideoASRRequest(BaseModel):
    """
    Parameters controlling video-to-ASR processing.

    The request describes video-level ASR behavior while
    reusing the existing audio extraction contract.
    """

    model_config = ConfigDict(extra="forbid")

    language: str | None = None

    audio_extraction: AudioExtractionRequest = Field(
        default_factory=AudioExtractionRequest,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class VideoASRResult(BaseModel):
    """
    Provider-independent result of video-to-ASR processing.

    The underlying ASRResult is preserved without duplicating
    transcript, segment, language, or confidence fields.
    """

    model_config = ConfigDict(extra="forbid")

    status: VideoASRStatus

    asr_result: ASRResult | None = None

    audio_extraction: dict[str, object] = Field(
        default_factory=dict,
    )

    errors: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class VideoProcessingResult(BaseModel):
    """
    Aggregated result of video processing.

    Individual stages may succeed or fail independently.
    This allows partial results to survive non-critical failures.
    """

    model_config = ConfigDict(extra="forbid")

    status: VideoProcessingStatus

    video_info: VideoInfo | None = None

    extracted_audio: StrictBytes | None = None

    frames: list[VideoFrame] = Field(
        default_factory=list,
    )

    errors: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )