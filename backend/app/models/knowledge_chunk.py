from typing import TYPE_CHECKING
from uuid import UUID

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.knowledge_chunk_embedding import KnowledgeChunkEmbedding

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from app.models.knowledge_document import KnowledgeDocument


class KnowledgeChunk(
    UUIDPrimaryKeyMixin,
    TimestampMixin,
    Base,
):
    """Retrievable semantic unit belonging to a knowledge document."""

    __tablename__ = "knowledge_chunks"

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey(
            "knowledge_documents.id",
            ondelete="CASCADE",
        ),
        nullable=False,
        index=True,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    content_hash: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
    )

    token_count: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
    )

    chunk_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    document: Mapped["KnowledgeDocument"] = relationship(
        back_populates="chunks",
    )

    embeddings: Mapped[list["KnowledgeChunkEmbedding"]] = relationship(
    "KnowledgeChunkEmbedding",
    back_populates="chunk",
    cascade="all, delete-orphan",
)