from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.models_ai.embeddings.schemas import EmbeddingModelInfo


class VectorRecord(BaseModel):
    """
    Provider-independent representation of a persisted embedding.

    A vector record connects an embedding to the knowledge chunk that
    produced it while retaining the embedding model identity and vector
    metadata required for future retrieval and re-indexing.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID

    values: list[float] = Field(
        min_length=1,
    )

    model: EmbeddingModelInfo

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("values")
    @classmethod
    def validate_values(cls, value: list[float]) -> list[float]:
        """Reject non-finite vector values."""

        import math

        if any(not math.isfinite(item) for item in value):
            raise ValueError(
                "Vector values must be finite."
            )

        return list(value)


class VectorPersistenceRequest(BaseModel):
    """
    Request to persist one vector for one knowledge chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID

    embedding: VectorRecord


class VectorBatchPersistenceRequest(BaseModel):
    """
    Request to persist multiple vector records.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    records: list[VectorRecord] = Field(
        min_length=1,
    )


class VectorQuery(BaseModel):
    """
    Provider-independent vector lookup query.

    The initial contract intentionally supports exact chunk lookup.
    Similarity search will be introduced by the retrieval subsystem
    rather than prematurely coupling persistence to a particular
    vector-database implementation.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID


class VectorDeleteRequest(BaseModel):
    """
    Request to delete persisted vectors for a knowledge chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID


class VectorPersistenceResult(BaseModel):
    """
    Result of a persistence operation.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    records: list[VectorRecord] = Field(
        min_length=1,
    )

    count: int = Field(
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class VectorDeleteResult(BaseModel):
    """
    Result of a vector deletion operation.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_id: UUID

    deleted_count: int = Field(
        ge=0,
    )