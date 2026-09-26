from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.models_ai.embeddings.schemas import EmbeddingModelInfo
from app.rag.schemas import (
    RAGQuery,
    RAGRetrievedChunk,
)


class RAGQueryRequest(RAGQuery):
    """
    HTTP request contract for RAG retrieval.

    The underlying RAGQuery contract remains the single source of truth
    for validation of query text and retrieval constraints.
    """

    model_config = ConfigDict(
        extra="forbid",
    )


class RAGRetrievedChunkResponse(BaseModel):
    """HTTP representation of one retrieved knowledge chunk."""

    model_config = ConfigDict(
        from_attributes=True,
    )

    chunk_id: UUID
    text: str
    similarity: float
    model: EmbeddingModelInfo
    provenance: Any | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class RAGQueryResponse(BaseModel):
    """HTTP response containing retrieved RAG context."""

    model_config = ConfigDict(
        extra="forbid",
    )

    query: RAGQuery
    chunks: list[RAGRetrievedChunkResponse]
    context_text: str
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @classmethod
    def from_context(
        cls,
        *,
        context: Any,
    ) -> "RAGQueryResponse":
        return cls(
            query=context.query,
            chunks=[
                RAGRetrievedChunkResponse(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    similarity=chunk.similarity,
                    model=chunk.model,
                    provenance=chunk.provenance,
                    metadata=chunk.metadata,
                )
                for chunk in context.chunks
            ],
            context_text=context.context_text,
            metadata=context.metadata,
        )