from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class MediaInfo(BaseModel):
    """
    Provider-independent technical information about a media file.

    Fields are intentionally optional because different media formats
    and inspection backends may expose different metadata.
    """

    model_config = ConfigDict(extra="forbid")

    format_name: str | None = None
    codec_name: str | None = None

    duration_seconds: float | None = Field(
        default=None,
        ge=0,
    )

    sample_rate: int | None = Field(
        default=None,
        gt=0,
    )

    channels: int | None = Field(
        default=None,
        gt=0,
    )

    bitrate: int | None = Field(
        default=None,
        gt=0,
    )

    size_bytes: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, object] = Field(
        default_factory=dict,
    )