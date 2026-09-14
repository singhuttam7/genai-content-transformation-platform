from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EmbeddingModelConfig(BaseModel):
    """
    Provider-independent configuration for an embedding model.

    This configuration describes how an embedding adapter should be
    initialized without exposing provider-specific implementation
    details to the RAG layer.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    provider: str = Field(
        min_length=1,
        max_length=100,
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
    )

    expected_dimension: int | None = Field(
        default=None,
        ge=1,
    )

    normalized: bool = False

    batch_size: int = Field(
        default=32,
        ge=1,
    )

    options: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("provider", "model_name")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        """Reject values containing only whitespace."""

        normalized = value.strip()

        if not normalized:
            raise ValueError("Value cannot be blank.")

        return normalized


class EmbeddingModelInfo(BaseModel):
    """Describe the embedding model used to produce vectors."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    provider: str = Field(
        min_length=1,
        max_length=100,
    )

    model_name: str = Field(
        min_length=1,
        max_length=200,
    )

    dimension: int = Field(
        ge=1,
    )

    normalized: bool = False

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class EmbeddingRequest(BaseModel):
    """Represent one embedding input."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    text: str

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class EmbeddingVector(BaseModel):
    """Represent one generated embedding vector."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    values: list[float] = Field(
        min_length=1,
    )

    model: EmbeddingModelInfo

    input_index: int = Field(
        ge=0,
    )


class EmbeddingBatchRequest(BaseModel):
    """Represent a batch embedding request."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    requests: list[EmbeddingRequest] = Field(
        min_length=1,
    )


class EmbeddingBatchResult(BaseModel):
    """Represent the result of a batch embedding operation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    embeddings: list[EmbeddingVector] = Field(
        min_length=1,
    )

    model: EmbeddingModelInfo

    input_count: int = Field(
        ge=1,
    )

    dimension: int = Field(
        ge=1,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )