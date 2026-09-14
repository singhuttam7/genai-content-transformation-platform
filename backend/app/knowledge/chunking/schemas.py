from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.schemas import ContentBlockType


class ChunkSourceReference(BaseModel):
    """
    Provenance information describing which normalized source
    elements contributed to a knowledge chunk.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    element_orders: list[int] = Field(
        default_factory=list,
        description=(
            "Canonical/normalized element order values contributing "
            "to this chunk."
        ),
    )

    block_types: list[ContentBlockType] = Field(
        default_factory=list,
        description=(
            "Structural block types contributing to this chunk."
        ),
    )

    page_numbers: list[int] = Field(
        default_factory=list,
        description=(
            "Source document page numbers represented by this chunk."
        ),
    )

    start_time: float | None = Field(
        default=None,
        description=(
            "Earliest media timestamp represented by this chunk."
        ),
    )

    end_time: float | None = Field(
        default=None,
        description=(
            "Latest media timestamp represented by this chunk."
        ),
    )

    section_path: list[str] = Field(
        default_factory=list,
        description=(
            "Hierarchical heading/section context inherited by the chunk."
        ),
    )


class KnowledgeChunkDraft(BaseModel):
    """
    Provider-independent, persistence-independent representation
    of a knowledge chunk.

    This is intentionally not the SQLAlchemy KnowledgeChunk model.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_index: int = Field(
        ge=0,
        description="Deterministic zero-based position of the chunk.",
    )

    text: str = Field(
        min_length=1,
        description="Textual content of the knowledge chunk.",
    )

    content_hash: str = Field(
        min_length=64,
        max_length=64,
        description="SHA-256 hash of the deterministic chunk representation.",
    )

    token_count: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Optional tokenizer-derived token count. "
            "Character-based chunking does not require this."
        ),
    )

    source: ChunkSourceReference = Field(
        description="Source provenance for this chunk.",
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional deterministic metadata associated with the chunk."
        ),
    )


class ChunkingStatistics(BaseModel):
    """Deterministic summary information about a chunking operation."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    input_element_count: int = Field(
        ge=0,
    )

    output_chunk_count: int = Field(
        ge=0,
    )

    input_character_count: int = Field(
        ge=0,
    )

    output_character_count: int = Field(
        ge=0,
    )

    largest_chunk_characters: int = Field(
        ge=0,
    )

    smallest_chunk_characters: int = Field(
        ge=0,
    )


class ChunkingResult(BaseModel):
    """
    Complete result returned by the structure-aware chunker.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunks: list[KnowledgeChunkDraft] = Field(
        default_factory=list,
    )

    statistics: ChunkingStatistics

    strategy: str = Field(
        min_length=1,
    )

    strategy_version: str = Field(
        min_length=1,
    )