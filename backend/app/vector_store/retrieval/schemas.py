from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models_ai.embeddings.schemas import EmbeddingModelInfo


class VectorRetrievalRequest(BaseModel):
    """Request for semantic vector retrieval."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query_vector: list[float] = Field(min_length=1)
    model: EmbeddingModelInfo
    top_k: int = Field(default=5, ge=1)
    similarity_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    project_id: UUID | None = None
    metadata_filter: dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_vector")
    @classmethod
    def validate_query_vector(
        cls,
        value: list[float],
    ) -> list[float]:
        if not value:
            raise ValueError("query_vector must not be empty.")

        for item in value:
            if not isinstance(item, (int, float)):
                raise ValueError(
                    "query_vector must contain only numeric values."
                )

        return value


class VectorRetrievalMatch(BaseModel):
    """A single vector retrieval match."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk_id: UUID
    similarity: float
    model: EmbeddingModelInfo
    metadata: dict[str, Any] = Field(default_factory=dict)


class VectorRetrievalResult(BaseModel):
    """Result of a semantic vector retrieval operation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    matches: list[VectorRetrievalMatch]
    query_model: EmbeddingModelInfo
    count: int
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("count")
    @classmethod
    def validate_count(cls, value: int) -> int:
        if value < 0:
            raise ValueError("count must not be negative.")

        return value