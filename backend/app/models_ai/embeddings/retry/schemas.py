from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.models_ai.embeddings.exceptions import (
    EmbeddingErrorCategory,
)


class RetryDecision(BaseModel):
    """Provider-independent decision about whether an operation may retry."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    retryable: bool
    category: EmbeddingErrorCategory
    reason: str = Field(
        min_length=1,
        max_length=1000,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

class EmbeddingFailureDecision(BaseModel):
    """Combined provider failure information and retry decision."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    category: EmbeddingErrorCategory
    retryable: bool
    reason: str = Field(
        min_length=1,
        max_length=1000,
    )
    error_message: str = Field(
        min_length=1,
        max_length=5000,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )