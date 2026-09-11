from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, StrictBytes


class ASRStatus(StrEnum):
    NOT_REQUESTED = "not_requested"
    PROCESSING = "processing"
    COMPLETED = "completed"
    NO_SPEECH = "no_speech"
    FAILED = "failed"


class ASRSegment(BaseModel):
    """
    A temporal segment of an audio transcript.
    """

    model_config = ConfigDict(extra="forbid")

    text: str
    start_time: float = Field(ge=0)
    end_time: float = Field(ge=0)
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    speaker: str | None = None
    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class ASRRequest(BaseModel):
    """
    Request passed from the enrichment layer to an ASR provider.
    """

    model_config = ConfigDict(extra="forbid")

    audio: StrictBytes
    language: str | None = None
    metadata: dict[str, object] = Field(
        default_factory=dict,
    )


class ASRResult(BaseModel):
    """
    Provider-independent ASR result.
    """

    model_config = ConfigDict(extra="forbid")

    status: ASRStatus
    text: str = ""
    segments: list[ASRSegment] = Field(
        default_factory=list,
    )
    language: str | None = None
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    provider: str
    metadata: dict[str, object] = Field(
        default_factory=dict,
    )