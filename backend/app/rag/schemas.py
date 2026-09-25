from __future__ import annotations

from math import isfinite
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.knowledge.provenance.schemas import KnowledgeChunkProvenance
from app.models_ai.embeddings.schemas import EmbeddingModelInfo


class RAGQuery(BaseModel):
    """Application-level query contract for retrieval-augmented generation."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    text: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1)
    similarity_threshold: float | None = Field(
        default=None,
        ge=-1.0,
        le=1.0,
    )
    project_id: UUID | None = None
    metadata_filter: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank.")
        return value

    @field_validator("similarity_threshold")
    @classmethod
    def validate_similarity_threshold(
        cls,
        value: float | None,
    ) -> float | None:
        if value is not None and not isfinite(value):
            raise ValueError("similarity_threshold must be finite.")
        return value


class RAGRetrievedChunk(BaseModel):
    """A retrieved knowledge chunk ready for RAG context assembly."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    chunk_id: UUID
    text: str = Field(min_length=1)
    similarity: float
    model: EmbeddingModelInfo
    provenance: KnowledgeChunkProvenance | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text cannot be blank.")
        return value

    @field_validator("similarity")
    @classmethod
    def validate_similarity(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("similarity must be finite.")
        return value


class RAGContext(BaseModel):
    """Immutable context package consumed by downstream intelligence layers."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    query: RAGQuery
    chunks: list[RAGRetrievedChunk]
    context_text: str
    metadata: dict[str, Any] = Field(default_factory=dict)