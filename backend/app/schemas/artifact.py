from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ArtifactCreateRequest(BaseModel):
    """Request to persist a generated artifact."""

    model_config = ConfigDict(
        extra="forbid",
    )

    transformation_id: UUID
    execution_id: UUID

    artifact_type: str = Field(
        min_length=1,
        max_length=100,
    )

    title: str | None = Field(
        default=None,
        max_length=300,
    )

    content: Any = None

    storage_uri: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    status: str = Field(
        default="GENERATED",
        min_length=1,
        max_length=50,
    )


class ArtifactResponse(BaseModel):
    """Public API representation of an artifact."""

    model_config = ConfigDict(
        extra="forbid",
    )

    id: UUID
    transformation_id: UUID
    execution_id: UUID
    artifact_type: str
    title: str | None
    content: str | None
    storage_uri: str | None
    content_hash: str | None
    metadata: dict[str, Any]
    status: str
    created_at: datetime
    updated_at: datetime


class ArtifactListResponse(BaseModel):
    """Collection of artifacts."""

    model_config = ConfigDict(
        extra="forbid",
    )

    items: list[ArtifactResponse]
    total: int