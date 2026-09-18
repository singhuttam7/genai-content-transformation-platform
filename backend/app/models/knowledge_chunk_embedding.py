from __future__ import annotations

from typing import Any
from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class KnowledgeChunkEmbedding(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """
    Persistent embedding generated for a knowledge chunk.

    A chunk may have multiple embeddings produced by different
    embedding models. Therefore, the logical uniqueness boundary
    is (chunk_id, provider, model_name).
    """

    __tablename__ = "knowledge_chunk_embeddings"

    __table_args__ = (
        UniqueConstraint(
            "chunk_id",
            "provider",
            "model_name",
            name="uq_knowledge_chunk_embeddings_chunk_provider_model",
        ),
    )

    chunk_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "knowledge_chunks.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    provider: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    model_name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
    )

    dimension: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    normalized: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
    )

    embedding: Mapped[list[float]] = mapped_column(
        Vector(384),
        nullable=False,
    )

    # "metadata" is reserved by SQLAlchemy's Declarative API,
    # therefore the Python attribute is named vector_metadata
    # while the actual database column remains "metadata".
    vector_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    chunk: Mapped["KnowledgeChunk"] = relationship(
        "KnowledgeChunk",
        back_populates="embeddings",
    )