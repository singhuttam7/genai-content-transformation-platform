from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class TransformationType(StrEnum):
    """Supported content transformation families."""

    ADVISORY = "advisory"
    EXECUTIVE_SUMMARY = "executive_summary"
    SOCIAL_MEDIA = "social_media"
    INFOGRAPHIC = "infographic"
    PRESENTATION = "presentation"
    VIDEO = "video"


class TransformationStatus(StrEnum):
    """Lifecycle states for a transformation operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ArtifactEnvelope(BaseModel):
    """
    Provider-independent representation of a generated artifact.

    This is an application-layer contract. Persistence is handled by the
    existing Artifact database model and must not be coupled to this object.
    """

    model_config = ConfigDict(extra="forbid")

    artifact_type: str = Field(min_length=1)
    title: str | None = None
    content: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: dict[str, Any] = Field(default_factory=dict)


class TransformationRequest(BaseModel):
    """
    Application-layer request for a content transformation.

    The request intentionally mirrors the transformation capabilities already
    represented by the persistence model without exposing SQLAlchemy models
    to transformation agents.
    """

    model_config = ConfigDict(extra="forbid")

    transformation_type: TransformationType
    input: Any
    objective: str | None = None
    audience: str | None = None
    tone: str | None = None
    language: str = Field(default="English", min_length=1)
    detail_level: str | None = None
    style: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TransformationResult(BaseModel):
    """
    Result produced by a transformation agent.

    A successful transformation may produce one or multiple artifacts.
    """

    model_config = ConfigDict(extra="forbid")

    status: TransformationStatus
    artifacts: list[ArtifactEnvelope] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)

        if self.status == TransformationStatus.FAILED:
            if not self.error:
                raise ValueError(
                    "Failed transformation results must include an error."
                )
        elif self.error is not None:
            raise ValueError(
                "Only failed transformation results may include an error."
            )