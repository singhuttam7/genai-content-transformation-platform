from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.agents.transformation.contracts import (
    TransformationStatus,
    TransformationType,
)


class TransformationCreateRequest(BaseModel):
    """HTTP request for creating a transformation."""

    model_config = ConfigDict(extra="forbid")

    project_id: UUID
    source_id: UUID
    transformation_type: TransformationType

    objective: str | None = None
    audience: str | None = None
    tone: str | None = None
    language: str = Field(default="English", min_length=1)
    detail_level: str | None = None
    style: str | None = None

    requested_outputs: list[str] = Field(
        default_factory=list,
    )

    configuration: dict[str, Any] = Field(
        default_factory=dict,
    )


class TransformationResponse(BaseModel):
    """HTTP representation of a persisted transformation."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    project_id: UUID
    source_id: UUID

    objective: str | None
    audience: str | None
    tone: str | None
    language: str
    detail_level: str | None
    style: str | None

    requested_outputs: list[Any]
    configuration: dict[str, Any]

    status: str

    created_at: datetime
    updated_at: datetime


class TransformationListResponse(BaseModel):
    """Paginated transformation collection."""

    items: list[TransformationResponse]
    total: int