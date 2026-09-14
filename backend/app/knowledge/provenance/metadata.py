from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SourceMetadata(BaseModel):
    """
    Deterministic source-specific metadata attached to a
    knowledge chunk.

    The fields are intentionally generic at this layer so that
    A4-specific metadata can be preserved without coupling the
    provenance system to individual ingestion implementations.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_type: str = Field(
        min_length=1,
        max_length=50,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class EnrichedKnowledgeMetadata(BaseModel):
    """
    Complete metadata enrichment result for a knowledge chunk.

    This contract separates source-specific metadata from the
    core provenance information produced by A5.5.2.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    source_metadata: SourceMetadata

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )