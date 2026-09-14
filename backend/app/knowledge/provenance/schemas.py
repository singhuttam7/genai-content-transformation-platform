from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.ingestion.schemas import ContentBlockType


class KnowledgeSourceProvenance(BaseModel):
    """
    Identifies the original source associated with knowledge content.

    This contract is intentionally provider-independent and can be used
    by ingestion, chunking, embedding, retrieval, and context assembly.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    project_id: UUID
    source_id: UUID

    source_type: str = Field(
        min_length=1,
        max_length=50,
    )

    source_uri: str | None = Field(
        default=None,
        max_length=2048,
    )

    title: str | None = Field(
        default=None,
        max_length=500,
    )

    language: str | None = Field(
        default=None,
        max_length=20,
    )


class KnowledgeDocumentProvenance(BaseModel):
    """
    Identifies the exact knowledge-document version from which
    a chunk originated.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    document_id: UUID
    document_version: int = Field(
        ge=1,
    )

    content_hash: str = Field(
        min_length=64,
        max_length=64,
    )


class KnowledgeLocationProvenance(BaseModel):
    """
    Describes the physical or temporal location of knowledge content
    inside its original source.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    element_orders: list[int] = Field(
        default_factory=list,
    )

    block_types: list[ContentBlockType] = Field(
        default_factory=list,
    )

    page_numbers: list[int] = Field(
        default_factory=list,
    )

    section_path: list[str] = Field(
        default_factory=list,
    )

    start_time: float | None = Field(
        default=None,
        ge=0,
    )

    end_time: float | None = Field(
        default=None,
        ge=0,
    )


class KnowledgeChunkProvenance(BaseModel):
    """
    Complete provenance contract for a knowledge chunk.

    This object answers:

        Which project?
        Which source?
        Which document version?
        Which exact content?
        Where inside the source?
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source: KnowledgeSourceProvenance
    document: KnowledgeDocumentProvenance
    location: KnowledgeLocationProvenance

    chunk_index: int = Field(
        ge=0,
    )


class KnowledgeChunkMetadata(BaseModel):
    """
    Metadata associated with a knowledge chunk.

    Provenance is kept as a first-class field instead of being buried
    inside an unstructured metadata dictionary.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    chunk_index: int = Field(
        ge=0,
    )

    character_count: int = Field(
        ge=1,
    )

    token_count: int | None = Field(
        default=None,
        ge=0,
    )

    content_hash: str = Field(
        min_length=64,
        max_length=64,
    )

    chunking_strategy: str = Field(
        min_length=1,
        max_length=100,
    )

    chunking_strategy_version: str = Field(
        min_length=1,
        max_length=50,
    )

    provenance: KnowledgeChunkProvenance

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )