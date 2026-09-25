from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RAGContextConfig(BaseModel):
    """
    Deterministic limits applied while assembling RAG context.
    """

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
    )

    max_chunks: int = Field(
        default=8,
        ge=1,
    )

    max_context_characters: int = Field(
        default=12_000,
        ge=1,
    )