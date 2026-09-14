
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import (
    Base,
    TimestampMixin,
    UUIDPrimaryKeyMixin,
)

if TYPE_CHECKING:
    from app.models.knowledge_document import KnowledgeDocument
    from app.models.project import Project
    from app.models.transformation import Transformation


class Source(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    """Input source supplied for content transformation."""

    __tablename__ = "sources"

    project_id: Mapped[UUID] = mapped_column(
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    source_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        index=True,
    )

    title: Mapped[str | None] = mapped_column(
        String(300),
        nullable=True,
    )

    original_filename: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
    )

    mime_type: Mapped[str | None] = mapped_column(
        String(150),
        nullable=True,
    )

    storage_uri: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    content_hash: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        index=True,
    )

    source_metadata: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        default=dict,
    )

    status: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="PENDING",
        index=True,
    )

    project: Mapped["Project"] = relationship(
        back_populates="sources",
    )

    knowledge_documents: Mapped[list["KnowledgeDocument"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )

    transformations: Mapped[list["Transformation"]] = relationship(
        back_populates="source",
        cascade="all, delete-orphan",
    )
