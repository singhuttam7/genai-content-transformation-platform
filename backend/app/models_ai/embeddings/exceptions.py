from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EmbeddingErrorCategory(StrEnum):
    """Provider-independent categories for embedding failures."""

    CONFIGURATION = "configuration"
    INPUT = "input"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    NETWORK = "network"
    SERVICE_UNAVAILABLE = "service_unavailable"
    MODEL_NOT_FOUND = "model_not_found"
    DIMENSION = "dimension"
    BATCH = "batch"
    UNKNOWN = "unknown"


class EmbeddingOperation(StrEnum):
    """Embedding operations that can produce an error."""

    EMBED = "embed"
    EMBED_BATCH = "embed_batch"


class EmbeddingErrorContext(BaseModel):
    """Provider-independent context attached to an embedding error."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    operation: EmbeddingOperation
    provider: str | None = Field(
        default=None,
        min_length=1,
        max_length=100,
    )
    model: str | None = Field(
        default=None,
        min_length=1,
        max_length=200,
    )
    batch_index: int | None = Field(
        default=None,
        ge=0,
    )
    input_count: int | None = Field(
        default=None,
        ge=0,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class EmbeddingErrorClassification(BaseModel):
    """Provider-independent classification of an embedding error."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    category: EmbeddingErrorCategory
    retryable: bool
    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )
    reason: str | None = Field(
        default=None,
        max_length=1000,
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


class EmbeddingError(Exception):
    """Base exception for embedding subsystem failures."""

    category = EmbeddingErrorCategory.UNKNOWN


class EmbeddingConfigurationError(EmbeddingError):
    """Raised when embedding configuration is invalid."""

    category = EmbeddingErrorCategory.CONFIGURATION


class EmbeddingInputError(EmbeddingError):
    """Raised when embedding input is invalid."""

    category = EmbeddingErrorCategory.INPUT


class EmbeddingDimensionError(EmbeddingError):
    """Raised when embedding dimensions are inconsistent."""

    category = EmbeddingErrorCategory.DIMENSION


class EmbeddingProviderError(EmbeddingError):
    """Provider-independent error raised for provider failures."""

    category = EmbeddingErrorCategory.UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        category: EmbeddingErrorCategory = EmbeddingErrorCategory.UNKNOWN,
        context: EmbeddingErrorContext | None = None,
        classification: EmbeddingErrorClassification | None = None,
    ) -> None:
        super().__init__(message)

        self.category = category
        self.context = context
        self.classification = classification

        self.batch_index = (
            context.batch_index
            if context is not None
            else None
        )
class EmbeddingBatchError(EmbeddingError):
    """Raised when an embedding batch cannot be completed."""

    category = EmbeddingErrorCategory.BATCH

    def __init__(
        self,
        message: str,
        *,
        batch_index: int | None = None,
        context: EmbeddingErrorContext | None = None,
    ) -> None:
        super().__init__(message)

        if batch_index is not None and batch_index < 0:
            raise ValueError(
                "batch_index cannot be negative."
            )

        if (
            batch_index is not None
            and context is not None
            and context.batch_index is not None
            and batch_index != context.batch_index
        ):
            raise ValueError(
                "batch_index must match context.batch_index."
            )

        self.batch_index = batch_index
        self.context = context