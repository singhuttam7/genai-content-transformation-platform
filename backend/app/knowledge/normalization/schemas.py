from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.ingestion.schemas import ContentBlockType


class NormalizedKnowledgeElement(BaseModel):
    """
    Structure-preserving knowledge element derived from an
    A4 CanonicalContent segment.
    """

    content: str = ""

    block_type: ContentBlockType

    order: int

    page_number: int | None = None

    start_time: float | None = None

    end_time: float | None = None

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class NormalizedKnowledgeDocument(BaseModel):
    """
    Deterministic knowledge representation produced from
    CanonicalContent.

    This representation is intentionally independent of
    embeddings, vector stores, LLMs, and chunking.
    """

    source: Any

    title: str | None = None

    language: str | None = None

    text: str

    elements: list[NormalizedKnowledgeElement] = Field(
        default_factory=list,
    )

    entities: list[str] = Field(
        default_factory=list,
    )

    topics: list[str] = Field(
        default_factory=list,
    )

    claims: list[str] = Field(
        default_factory=list,
    )

    keywords: list[str] = Field(
        default_factory=list,
    )

    context: dict[str, Any] = Field(
        default_factory=dict,
    )

    provenance: dict[str, Any] = Field(
        default_factory=dict,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    content_hash: str