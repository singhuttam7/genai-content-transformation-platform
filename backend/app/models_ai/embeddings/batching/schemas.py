from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from app.models_ai.embeddings.schemas import EmbeddingRequest


class EmbeddingBatchPlan(BaseModel):
    """Deterministic plan describing how inputs are divided into batches."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    batches: list[list[EmbeddingRequest]] = Field(
        min_length=1,
    )

    batch_size: int = Field(
        ge=1,
    )

    input_count: int = Field(
        ge=1,
    )

    batch_count: int = Field(
        ge=1,
    )