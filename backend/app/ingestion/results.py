from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field

from app.ingestion.schemas import (
    CanonicalContent,
    ProcessingStatus,
)


class IngestionResult(BaseModel):
    """Result returned by the unified ingestion application service."""

    source_id: UUID

    canonical_content: CanonicalContent

    storage_key: str | None = None

    storage_uri: str | None = None

    content_hash: str | None = None

    status: ProcessingStatus = ProcessingStatus.COMPLETED

    metadata: dict = Field(default_factory=dict)