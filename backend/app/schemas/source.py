from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.schemas import (
    ContentBlock,
    InputType,
    ProcessingStatus,
)


class SourceCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_id: UUID | None = None
    source_id: UUID | None = None

    input_type: InputType

    title: str | None = None
    filename: str | None = None
    mime_type: str | None = None

    content: str | None = None
    url: str | None = None

    storage_key: str | None = None
    storage_uri: str | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class SourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_id: UUID
    status: ProcessingStatus

    storage_key: str | None = None
    storage_uri: str | None = None
    content_hash: str | None = None

    title: str | None = None
    source_type: InputType | None = None

    canonical_text: str | None = None

    segments: list[ContentBlock] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    provenance: dict[str, Any] = Field(
        default_factory=dict,
    )