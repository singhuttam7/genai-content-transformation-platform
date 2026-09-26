from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class ExecutionCreateRequest(BaseModel):
    """Request to create a queued workflow execution."""

    model_config = ConfigDict(extra="forbid")

    transformation_id: UUID
    workflow_id: UUID
    execution_context: dict[str, Any] = Field(
        default_factory=dict,
    )


class ExecutionResponse(BaseModel):
    """HTTP representation of an execution job."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    transformation_id: UUID
    workflow_id: UUID
    workflow_version: int
    status: str
    started_at: datetime | None
    completed_at: datetime | None
    error: str | None
    execution_context: dict[str, Any]
    metrics: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ExecutionListResponse(BaseModel):
    """Paginated-style execution collection."""

    items: list[ExecutionResponse]
    total: int